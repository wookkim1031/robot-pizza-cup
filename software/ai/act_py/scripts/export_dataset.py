# Copyright 2026 Norma Core contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export tagged takes from the station into a dataset, then publish it.

Pipeline:
  1. Read station tags (`<task> | <session> | R<n> | E<n> | <OP> | START/END/FAIL ...`,
     as written by st3215-remote-teleop-py/tag.py) and keep START→END takes.
     `FAIL RETRO` cancels the previous END of the same round/episode.
  2. Export each take with dataset-generator; takes it discards (frame gaps,
     too few commands) are dropped.
  3. Canonicalize camera order per take (cam0 = scene, cam1 = wrist; the wrist
     view has a dark gripper band at the bottom). The station's order can swap
     between restarts.
  4. Merge into <out>/<name>.parquet + .manifest.json + .sha256.
  5. Convert to a LeRobot dataset at <out>/lerobot/<name>.
  6. Push the LeRobot dataset to the Hugging Face Hub (private by default), with
     the raw parquet + manifest under raw/.
  7. Optionally copy everything into the Jupyter containers on the GPU VMs.

    uv run python scripts/export_dataset.py --tag-task cheese --session s1 --min-round 3 \\
        --task "put the cheese on the pizza" --name cheese-fixed-s1-2026-10-24 \\
        --vm 81.85.4.50 --vm 81.85.0.200
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]
sys.path.insert(0, str(_REPO))

from software.station.shared.station_py import new_station_client  # noqa: E402

TAG_RE = re.compile(
    r"^(?P<task>[^|]+?) \| (?P<session>\S+) \| R(?P<round>\d+) \| E(?P<ep>\d+) \| "
    # Layout fields written by the final-collection tag.py; absent in older tags.
    # From session s2 on, the quadrant is left out and follows from the episode.
    r"(?:(?:(?P<quadrant>TL|TR|BR|BL) \| )?(?P<pos>P\w+) \| (?P<boxes>AB|BA|3BOWL|[CKP]{3})\+?K? \| )?"
    r"(?P<op>[^|]+?) \| (?P<marker>.*)$"
)
def quadrant_order(args, rnd: int) -> list[str]:
    order = args.quadrant_order
    for spec in sorted(args.quadrant_order_from, key=lambda x: int(x.split("=")[0])):
        start, o = spec.split("=")
        if rnd >= int(start):
            order = o
    return order.split(",")


def quadrant_from_episode(rnd: int, ep: int, per_round: int, order: list[str]) -> str:
    """Same plan as tag.py. 4/8-episode rounds (tag.py --only): TL TR BL BR, repeated. 10-episode
    rounds: 4 cheese (TL TR BL BR), then 6 pepperoni (TL TR BL BR + 2 extras),
    pepperoni first every third round; wrong if `tag.py order` overrode it."""
    cheese = ["TL", "TR", "BL", "BR"]
    if per_round in (4, 6, 8):
        return order[(ep - 1) % len(order)]
    pepperoni = cheese + [["TL", "BR"], ["TR", "BL"]][(rnd - 1) % 2]
    plan = cheese + pepperoni if rnd % 3 else pepperoni + cheese
    return plan[ep - 1]


QUADRANT_NAMES = {"TL": "top-left", "TR": "top-right", "BR": "bottom-right", "BL": "bottom-left",
                  "CENTER": "center"}
WRIST_BAND_MAX = 70  # mean brightness of the bottom 40 rows below this = wrist camera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag-task", required=True, help="First tag field to select, e.g. 'cheese'")
    p.add_argument("--session", required=True, help="Second tag field, e.g. 's1'")
    p.add_argument("--min-round", type=int, default=1, help="Ignore rounds below this (unexportable early takes)")
    p.add_argument("--max-round", type=int, default=0, help="Ignore rounds above this (0 = no limit)")
    p.add_argument("--skip", type=int, action="append", default=[], metavar="START_FRAME",
                   help="Leave out the take that starts at this frame (a bad take already closed with END); repeatable")
    p.add_argument("--rescue", type=int, action="append", default=[], metavar="START_FRAME",
                   help="Keep the take that starts at this frame although it was closed with FAIL by mistake "
                        "(the FAIL frame becomes its end); repeatable")
    p.add_argument("--episodes-per-round", type=int, default=10, choices=[4, 6, 8, 10],
                   help="4, 6 or 8 for rounds recorded with tag.py --only (used to infer the quadrant)")
    p.add_argument("--quadrant-order", default="TL,TR,BR,BL",
                   help="with --episodes-per-round 4/8: quadrant per episode (s2 R1-R2: TL,TR,BL,BR; "
                        "R3+: TL,TR,BR,BL, see --quadrant-order-from)")
    p.add_argument("--quadrant-order-from", action="append", default=[], metavar="ROUND=ORDER",
                   help="use ORDER from round ROUND on, e.g. 3=TL,TR,BR,BL (repeatable)")
    p.add_argument("--task", required=True,
                   help="Natural-language task written into every frame. '{quadrant}' is replaced "
                        "per take from its tag, e.g. 'put one pepperoni slice on the {quadrant} quarter'")
    p.add_argument("--quadrant-task", action="append", default=[], metavar="QUADRANT=TEXT",
                   help="full task text for one quadrant instead of --task, e.g. "
                        "'GAP=put the pepperoni where the pizza has the most empty space' (repeatable)")
    p.add_argument("--name", required=True, help="Dataset name, used for files and the Hub repo")
    p.add_argument("--out-dir", type=Path, default=Path.home() / "datasets")
    p.add_argument("--generator", type=Path, default=Path.home() / "Downloads" / "dataset-generator")
    p.add_argument("--server", default="localhost")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hf-user", default="johan1031", help="Hub namespace; repo is <hf-user>/<name>")
    p.add_argument("--public", action="store_true", help="Create the Hub repo public (default: private)")
    p.add_argument("--no-hub", action="store_true", help="Skip the Hugging Face upload")
    p.add_argument("--vm", action="append", default=[], help="GPU VM to copy into (nebius@<ip>); repeatable")
    p.add_argument("--dry-run", action="store_true", help="Only list the takes that would be exported")
    p.add_argument("--existing", action="store_true",
                   help="Skip steps 1-4 and publish <out-dir>/<name>.parquet from an earlier export "
                        "(the station keeps only ~2 GB of frames, so old takes can't be re-exported)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Tags

def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]; i += 1; r |= (x & 0x7F) << s; s += 7
        if not x & 0x80:
            return r, i


def _decode_rx_envelope(b: bytes) -> dict[int, int | bytes]:
    """inference_tags.RxEnvelope (no Python binding is generated for it)."""
    out, i = {}, 0
    while i < len(b):
        key, i = _varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            out[field], i = _varint(b, i)
        elif wire == 2:
            n, i = _varint(b, i); out[field] = b[i:i + n]; i += n
        else:
            raise ValueError(f"unexpected wire type {wire}")
    return out


async def read_tags(server: str, limit: int = 5000) -> list[tuple[int, str]]:
    client = await new_station_client(server, logging.getLogger("export"))
    qr = client.read_from_tail("inference-tags/rx", limit.to_bytes(8, "little"), limit, 1, 256)
    tags = []
    while (entry := await qr.data.get()) is not None:
        d = _decode_rx_envelope(bytes(entry.Data))
        if d.get(10, 0) == 1:  # CT_REMOVE_TAG
            continue
        tags.append((int.from_bytes(d.get(11, b""), "little"), d.get(12, b"").decode(errors="replace")))
    return tags


def select_takes(tags, args) -> list[dict]:
    takes, current = [], None
    for ptr, text in tags:
        m = TAG_RE.match(text.strip())
        if not m or m["task"] != args.tag_task or m["session"] != args.session:
            continue
        rnd, ep, marker = int(m["round"]), int(m["ep"]), m["marker"].strip()
        if marker == "START":
            current = {"round": rnd, "ep": ep, "op": m["op"], "from": ptr}
            if m["pos"]:
                quadrant = m["quadrant"] or quadrant_from_episode(rnd, ep, args.episodes_per_round, quadrant_order(args, rnd))
                current.update(quadrant=quadrant, pos=m["pos"], boxes=m["boxes"])
        elif marker == "END" and current and (current["round"], current["ep"]) == (rnd, ep):
            takes.append({**current, "to": ptr}); current = None
        elif marker.startswith("FAIL RETRO"):
            for t in reversed(takes):
                if (t["round"], t["ep"]) == (rnd, ep):
                    takes.remove(t); break
            current = None
        elif marker.startswith("FAIL"):
            if current and current["from"] in args.rescue and (current["round"], current["ep"]) == (rnd, ep):
                takes.append({**current, "to": ptr})
            current = None
    takes = [t for t in takes if t["round"] >= args.min_round and (not args.max_round or t["round"] <= args.max_round)
             and t["from"] not in args.skip]
    for i, t in enumerate(takes):
        t["name"] = f"R{t['round']:02d}E{t['ep']}_{i:03d}"
    return takes


# ---------------------------------------------------------------------------
# 2-4. Export, camera order, merge

def task_text(take: dict, args) -> str:
    """--task, with {quadrant} filled in per take (e.g. "top-left")."""
    overrides = dict(spec.split("=", 1) for spec in args.quadrant_task)
    if take.get("quadrant") in overrides:
        return overrides[take["quadrant"]]
    if "{quadrant}" not in args.task:
        return args.task
    if "quadrant" not in take:
        raise SystemExit(f"--task uses {{quadrant}} but {take['name']} has no quadrant in its tag")
    return args.task.format(quadrant=QUADRANT_NAMES[take["quadrant"]])


def export_take(take: dict, args, workdir: Path) -> Path | None:
    out = workdir / take["name"]
    log = subprocess.run(
        [str(args.generator), "-no-tui", "-episode.duration", "3", "-episode.min-commands", "10",
         "-from", str(take["from"]), "-to", str(take["to"]), "-robot", args.server,
         "-output", str(out), "-task", task_text(take, args)],
        capture_output=True, text=True,
    )
    path = out.with_suffix(".parquet")
    if not path.exists():
        reason = re.findall(r"discarded[^\n]*", re.sub(r"\x1b\[[0-9;]*m", "", log.stdout + log.stderr))
        print(f"  {take['name']}: DROPPED ({reason[-1] if reason else 'no frames'})")
        return None
    return path


def wrist_index(images: list[list[dict]]) -> int | None:
    sample = images[:: max(1, len(images) // 10)]
    is_wrist = []
    for cam in range(len(images[0])):
        mean = np.mean([np.asarray(Image.open(io.BytesIO(fr[cam]["jpeg"])).convert("L"), dtype=np.float32)
                        for fr in sample], axis=0)
        is_wrist.append(mean[-40:].mean() < WRIST_BAND_MAX)
    return is_wrist.index(True) if is_wrist.count(True) == 1 else None


def merge(takes: list[dict], parts: dict[str, Path], args) -> tuple[Path, Path]:
    out = args.out_dir / f"{args.name}.parquet"
    writer, schema, manifest, last_wrist = None, None, [], 1
    for take in takes:
        path = parts.get(take["name"])
        if path is None:
            continue
        t = pq.read_table(path)
        images = t.column("images").to_pylist()
        wi = wrist_index(images)
        if wi is None:
            wi = last_wrist  # ambiguous (e.g. overexposed): order rarely changes between takes
            print(f"  {take['name']}: camera order ambiguous, assuming same as previous take")
        last_wrist = wi
        swapped = wi == 0
        if swapped:
            images = [[fr[1], fr[0]] for fr in images]
            idx = t.schema.get_field_index("images")
            t = t.set_column(idx, "images", pa.array(images, type=t.schema.field("images").type))
        schema = schema or t.schema
        writer = writer or pq.ParquetWriter(out, schema)
        writer.write_table(t.cast(schema))
        ts = t.column("timestamp_ns_since_episode_start")
        manifest.append({**take, "frames": t.num_rows, "seconds": round(ts[-1].as_py() / 1e9, 1),
                         "episode_start_ns": t.column("episode_start_ns")[0].as_py(),
                         "cameras_swapped": swapped})
    if writer is None:
        raise SystemExit("no takes exported: the station keeps only ~2 GB of inference/normvla frames, so takes "
                         "older than ~2 h are purged. Export right after recording, or use --existing.")
    writer.close()
    man = out.with_suffix(".manifest.json")
    man.write_text(json.dumps({
        "task": args.task, "tags": f"{args.tag_task} | {args.session}", "fps": args.fps,
        "cameras": ["cam0=scene", "cam1=wrist"], "episodes": manifest,
    }, indent=1))
    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in (out, man))
    out.with_suffix(".sha256").write_text(sums)
    print(f"merged {len(manifest)} episodes, {sum(m['frames'] for m in manifest)} frames -> {out}")
    return out, man


# ---------------------------------------------------------------------------
# 5-7. Convert, publish, copy

def convert(parquet: Path, repo_id: str, args) -> Path:
    root = args.out_dir / "lerobot" / args.name
    subprocess.run([sys.executable, str(_HERE.parent / "convert_to_lerobot.py"), "--parquets", str(parquet),
                    "--root", str(root), "--repo-id", repo_id, "--fps", str(args.fps)],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"converted -> {root}")
    return root


def push_to_hub(root: Path, repo_id: str, raw_files: list[Path], args) -> None:
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    ds.push_to_hub(private=not args.public, tags=["norma-core", args.tag_task], license="apache-2.0")
    api = HfApi()
    for f in raw_files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"raw/{f.name}", repo_id=repo_id, repo_type="dataset",
                        commit_message=f"Add raw station export {f.name}")
    print(f"pushed -> https://huggingface.co/datasets/{repo_id} ({'public' if args.public else 'private'})")


def copy_to_vm(host: str, files: list[Path], lerobot_root: Path) -> None:
    """Stream into the Jupyter container as jovyan; nothing is stored on the VM host."""
    find = 'C=$(sudo docker ps --format "{{.ID}} {{.Image}}" | grep -i jupyter | cut -d" " -f1)'
    target = "/home/jovyan/datasets"
    tar = subprocess.Popen(["tar", "-C", str(files[0].parent), "-cf", "-", *[f.name for f in files],
                            "-C", str(lerobot_root.parent.parent), f"lerobot/{lerobot_root.name}"],
                           stdout=subprocess.PIPE)
    subprocess.run(["ssh", "-o", "BatchMode=yes", f"nebius@{host}",
                    f"{find}; sudo docker exec -i -u jovyan $C sh -c "
                    f"'mkdir -p {target} && tar -C {target} -xf - && cd {target} && sha256sum -c {files[-1].name}'"],
                   stdin=tar.stdout, check=True)
    tar.wait()
    print(f"copied -> {host}:{target}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo_id = f"{args.hf_user}/{args.name}"
    if args.existing:
        parquet = args.out_dir / f"{args.name}.parquet"
        manifest, sha = parquet.with_suffix(".manifest.json"), parquet.with_suffix(".sha256")
        subprocess.run(["sha256sum", "-c", sha.name], cwd=args.out_dir, check=True)
        publish(parquet, manifest, sha, repo_id, args)
        return

    tags = asyncio.run(read_tags(args.server))
    takes = select_takes(tags, args)
    print(f"{len(takes)} takes for '{args.tag_task} | {args.session}' (rounds >= {args.min_round})")
    if args.dry_run:
        for t in takes:
            print(f"  {t['name']}  {t['op']}  {t.get('quadrant', '')} {t.get('pos', '')} "
                  f"{t.get('boxes', '')}  {t['from']} -> {t['to']}  \"{task_text(t, args)}\"")
        return

    with tempfile.TemporaryDirectory(prefix="export-") as tmp:
        parts = {}
        for take in takes:
            path = export_take(take, args, Path(tmp))
            if path:
                parts[take["name"]] = path
        parquet, manifest = merge(takes, parts, args)

    publish(parquet, manifest, parquet.with_suffix(".sha256"), repo_id, args)


def publish(parquet: Path, manifest: Path, sha: Path, repo_id: str, args) -> None:
    root = convert(parquet, repo_id, args)
    if not args.no_hub:
        push_to_hub(root, repo_id, [parquet, manifest, sha], args)
    for host in args.vm:
        copy_to_vm(host, [parquet, manifest, sha], root)


if __name__ == "__main__":
    main()
