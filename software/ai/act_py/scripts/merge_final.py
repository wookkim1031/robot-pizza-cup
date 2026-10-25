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

"""Merge the per-block station exports into the final hackathon datasets.

Each final dataset gets, in <out-dir>:
  <name>.parquet        all episodes, sorted by round then episode
  <name>.manifest.json  one clean record per episode + dataset notes
  <name>.README.md      the same notes, human-readable
  <name>.sha256
and a LeRobot copy at <out-dir>/lerobot/<name>. Session, operator and the
fixed layout fields are dropped from the per-episode records; the layout is
stated once at the top.

    uv run python scripts/merge_final.py            # build all three
    uv run python scripts/merge_final.py --vm 81.85.4.50 --vm 81.85.0.200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from export_dataset import copy_to_vm  # noqa: E402

DATE = "2026-10-24"
QUADRANTS = {"TL": "upper-left", "TR": "upper-right", "BR": "lower-right", "BL": "lower-left",
             "CENTER": "center", "GAP": "largest empty area"}
# Final task text per target. The run-time sequencer must send exactly these strings.
_WHERE = {"TL": "on the upper-left quadrant of the pizza", "TR": "on the upper-right quadrant of the pizza",
          "BR": "on the lower-right quadrant of the pizza", "BL": "on the lower-left quadrant of the pizza",
          "CENTER": "in the center of the pizza", "GAP": "on the largest empty area of the pizza"}
_ACTION = {"cheese": "Spread one portion of cheese", "pepperoni": "Place one pepperoni slice"}


def task_for(ingredient: str, target: str) -> str:
    return f"{_ACTION[ingredient]} {_WHERE[target]} and return to the start position."
PEPPERONI_ROUND = "6 episodes: TL, TR, BR, BL, CENTER, GAP (GAP = wherever the pizza has the most empty space)"
COMMON = {
    "fps": 10,
    "cameras": {"cam0": "scene (top view)", "cam1": "wrist"},
    "layout": "fixed for every episode: pizza at P0 (challenge 1-2 spot), cheese and pepperoni boxes in the normal order",
    "episode": "starts and ends with the arm at the home pose; one ingredient placement per episode",
    "tasks_note": "one task text per target; the run-time sequencer must send exactly these strings",
}

DATASETS = {
    f"cheese-r1r10-{DATE}": {
        "ingredient": "cheese",
        "pizza": "clean pizza (dough + sauce)",
        "parts": [f"cheese-new-s2-{DATE}", f"cheese-new-s2-r8r9-{DATE}", f"cheese-new-s2-r10-{DATE}"],
        "rounds": 10,
        "per_round": lambda r: ["TL", "TR", "BR", "BL"] * (1 if r <= 2 else 2),
        "round_plan": "R1-R2: 4 episodes (TL, TR, BR, BL); R3-R10: 8 episodes (TL, TR, BR, BL, twice)",
        "missing_reason": "dropped at export: follower servo overload (m2/m3/m8) froze the stream for >500 ms "
                          "mid-take. Fixed before R8 by raising the follower OverloadTorque from 25% to 50%.",
        "notes": [
            "R1 and R2 have only 4 episodes each (one pass over the quadrants); R3-R10 have 8 (two passes).",
            "9 episodes in R2-R6 are missing (see missing_episodes): they had frame gaps and were dropped.",
            "R8-R10 were recorded after the servo fix and have no gaps.",
        ],
    },
    f"pepperoni-r1r10-{DATE}": {
        "ingredient": "pepperoni",
        "pizza": "clean pizza (dough + sauce), no cheese",
        "parts": [f"pepperoni-new-s1-r1r3-{DATE}", f"pepperoni-new-s1-r4r6-{DATE}",
                  f"pepperoni-new-s1-r7r9-{DATE}", f"pepperoni-new-s1-r10-{DATE}"],
        "rounds": 10,
        "per_round": lambda r: ["TL", "TR", "BR", "BL", "CENTER", "GAP"],
        "round_plan": PEPPERONI_ROUND,
        "notes": [
            "One pepperoni slice per episode; a round fills one pizza with 6 slices.",
            "R1 E1: the operator forgot to restart recording after a failed try; the episode start was "
            "set afterwards from the arm's motion onset (~1 s before it left home). Checked visually.",
            "R1 E2 is long (75 s): the arm spends ~45 s re-trying the grab at the bowl before placing.",
        ],
    },
    f"pepperoni-on-cheese-r1r5-{DATE}": {
        "ingredient": "pepperoni",
        "pizza": "pizza with cheese already spread on it by hand (not recorded), clumpy like the robot leaves it",
        "parts": [f"pepperoni-oncheese-r1r3-{DATE}", f"pepperoni-oncheese-r4r5-{DATE}"],
        "rounds": 5,
        "per_round": lambda r: ["TL", "TR", "BR", "BL", "CENTER", "GAP"],
        "round_plan": PEPPERONI_ROUND,
        "notes": [
            "Challenge 2 stage 2: pepperoni placed on top of cheese. Same skill and task texts as "
            f"pepperoni-r1r10-{DATE}; only the pizza underneath differs.",
            "A fresh pizza with hand-spread cheese was used for every round.",
        ],
    },
    "pepperoni-grasp-r1r3-2026-10-25": {
        "ingredient": "pepperoni",
        "pizza": "clean pizza (dough + sauce), no cheese",
        "parts": ["pepperoni-grasp-ep-2026-10-25"],
        "rounds": 3,
        "per_round": lambda r: ["TL", "TR", "BR", "BL", "CENTER", "GAP"],
        "round_plan": PEPPERONI_ROUND,
        "notes": [
            "Grasp-focused demos recorded 2026-10-25 to fix weak grabbing: slow straight descent, a short "
            "stop before and after closing, straight lift. Same task texts as pepperoni-r1r10-2026-10-24.",
            "Missed grabs and double grabs were tagged FAIL and re-recorded, so every episode is one clean slice.",
        ],
    },
    "pepperoni-grasp-on-cheese-r1r2-2026-10-25": {
        "ingredient": "pepperoni",
        "pizza": "pizza with cheese already spread on it by hand (not recorded)",
        "parts": ["pepperoni-grasp-oncheese-2026-10-25"],
        "rounds": 2,
        "per_round": lambda r: ["TL", "TR", "BR", "BL", "CENTER", "GAP"],
        "round_plan": PEPPERONI_ROUND,
        "notes": [
            "Grasp-focused demos (see pepperoni-grasp-r1r3-2026-10-25), placed on top of cheese. "
            "Same task texts as pepperoni-on-cheese-r1r5-2026-10-24.",
            "Missed grabs and double grabs were tagged FAIL and re-recorded, so every episode is one clean slice.",
        ],
    },
    "cheese-grasp-r1r3-2026-10-25": {
        "ingredient": "cheese",
        "pizza": "clean pizza (dough + sauce)",
        "parts": ["cheese-grasp-ep-2026-10-25"],
        "rounds": 3,
        "per_round": lambda r: ["TL", "TR", "BR", "BL"],
        "round_plan": "4 episodes: TL, TR, BR, BL",
        "missing_reason": "dropped at export: 535 ms frame gap in the stream (limit 500 ms).",
        "notes": [
            "Grasp-focused demos recorded 2026-10-25 to fix weak grabbing: slow straight descent, a short "
            "stop before and after closing, straight lift. Same task texts as cheese-r1r10-2026-10-24.",
            "Failed tries were tagged FAIL and re-recorded, so every episode is one clean portion.",
        ],
    },
}


def build(name: str, spec: dict, out: Path) -> list[Path]:
    rows = []
    for part in spec["parts"]:
        m = json.loads((out / f"{part}.manifest.json").read_text())
        rows += [(e, out / f"{part}.parquet") for e in m["episodes"]]
    rows.sort(key=lambda x: (x[0]["round"], x[0]["ep"]))

    parquet = out / f"{name}.parquet"
    writer, schema, episodes = None, None, []
    for i, (e, path) in enumerate(rows):
        t = pq.read_table(path, filters=[("episode_start_ns", "=", e["episode_start_ns"])])
        assert t.num_rows == e["frames"], (name, e["name"], t.num_rows, e["frames"])
        assert len(set(t.column("task").to_pylist())) == 1, (name, e["name"])
        task = task_for(spec["ingredient"], e["quadrant"])
        t = t.set_column(t.schema.get_field_index("task"), "task", pa.array([task] * t.num_rows, pa.string()))
        schema = schema or t.schema
        writer = writer or pq.ParquetWriter(parquet, schema)
        writer.write_table(t.cast(schema))
        episodes.append({
            "index": i,
            "id": f"{spec['ingredient']}_R{e['round']:02d}_E{e['ep']}",
            "round": e["round"],
            "episode": e["ep"],
            "target": e["quadrant"],
            "task": task,
            "frames": e["frames"],
            "seconds": e["seconds"],
            "episode_start_ns": e["episode_start_ns"],
        })
    writer.close()

    have = {(e["round"], e["episode"]) for e in episodes}
    missing = [{"round": r, "episode": ep, "target": q, "reason": spec["missing_reason"]}
               for r in range(1, spec["rounds"] + 1)
               for ep, q in enumerate(spec["per_round"](r), start=1) if (r, ep) not in have]
    per_round = {f"R{r}": sum(e["round"] == r for e in episodes) for r in range(1, spec["rounds"] + 1)}
    targets = {q: sum(e["target"] == q for e in episodes) for q in dict.fromkeys(spec["per_round"](3))}

    manifest = {
        "name": name,
        "ingredient": spec["ingredient"],
        "pizza": spec["pizza"],
        **COMMON,
        "rounds": spec["rounds"],
        "round_plan": spec["round_plan"],
        "episodes_total": len(episodes),
        "frames_total": sum(e["frames"] for e in episodes),
        "episodes_per_round": per_round,
        "episodes_per_target": targets,
        "tasks": sorted({e["task"] for e in episodes}),
        "notes": spec["notes"],
        "missing_episodes": missing,
        "episodes": episodes,
    }
    man = out / f"{name}.manifest.json"
    man.write_text(json.dumps(manifest, indent=1))
    readme = out / f"{name}.README.md"
    readme.write_text(render_readme(manifest))
    sha = out / f"{name}.sha256"
    sha.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                           for p in (parquet, man, readme)))
    print(f"{name}: {len(episodes)} episodes, {manifest['frames_total']} frames, {len(missing)} missing")
    return [parquet, man, readme, sha]


def render_readme(m: dict) -> str:
    lines = [f"# {m['name']}", "",
             f"- **Ingredient:** {m['ingredient']}",
             f"- **Pizza:** {m['pizza']}",
             f"- **Layout:** {m['layout']}",
             f"- **Episode:** {m['episode']}",
             f"- **Rounds:** {m['rounds']} ({m['round_plan']})",
             f"- **Episodes:** {m['episodes_total']} ({m['frames_total']} frames at {m['fps']} fps)",
             f"- **Cameras:** cam0 = {m['cameras']['cam0']}, cam1 = {m['cameras']['cam1']}", "",
             "## Notes", ""] + [f"- {n}" for n in m["notes"]] + ["",
             "## Episodes per round", "",
             "| " + " | ".join(m["episodes_per_round"]) + " |",
             "|" + "---|" * len(m["episodes_per_round"]),
             "| " + " | ".join(str(v) for v in m["episodes_per_round"].values()) + " |", "",
             "## Episodes per target", "",
             "| Target | Episodes |", "|---|---|"] + [
             f"| {q} ({QUADRANTS[q]}) | {n} |" for q, n in m["episodes_per_target"].items()] + ["",
             "## Tasks", "", f"{m['tasks_note'].capitalize()}.", ""] + [f"- `{t}`" for t in m["tasks"]]
    if m["missing_episodes"]:
        lines += ["", "## Missing episodes", "", f"Reason: {m['missing_episodes'][0]['reason']}", "",
                  "| Round | Episode | Target |", "|---|---|---|"]
        lines += [f"| R{x['round']} | E{x['episode']} | {x['target']} |" for x in m["missing_episodes"]]
    return "\n".join(lines) + "\n"


def push_files(host: str, files: list[Path]) -> None:
    """Copy files into the VM's Jupyter container datasets dir; the last one is the sha256 list to check."""
    find = 'C=$(sudo docker ps --format "{{.ID}} {{.Image}}" | grep -i jupyter | cut -d" " -f1)'
    tar = subprocess.Popen(["tar", "-C", str(files[0].parent), "-cf", "-", *[f.name for f in files]],
                           stdout=subprocess.PIPE)
    subprocess.run(["ssh", "-o", "BatchMode=yes", f"nebius@{host}",
                    f"{find}; sudo docker exec -i -u jovyan $C sh -c "
                    f"'cd /home/jovyan/datasets && tar -xf - && sha256sum -c {files[-1].name}'"],
                   stdin=tar.stdout, check=True)
    tar.wait()
    print(f"  copied -> {host}")


def refresh_readme(name: str, out: Path, vms: list[str]) -> None:
    parquet, man = out / f"{name}.parquet", out / f"{name}.manifest.json"
    readme, sha = out / f"{name}.README.md", out / f"{name}.sha256"
    readme.write_text(render_readme(json.loads(man.read_text())))
    (out / "lerobot" / name / "README.md").write_text(readme.read_text())
    sha.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                           for p in (parquet, man, readme)))
    find = 'C=$(sudo docker ps --format "{{.ID}} {{.Image}}" | grep -i jupyter | cut -d" " -f1)'
    for host in vms:
        tar = subprocess.Popen(["tar", "-C", str(out), "-cf", "-", readme.name, sha.name,
                                f"lerobot/{name}/README.md"], stdout=subprocess.PIPE)
        subprocess.run(["ssh", "-o", "BatchMode=yes", f"nebius@{host}",
                        f"{find}; sudo docker exec -i -u jovyan $C sh -c "
                        f"'cd /home/jovyan/datasets && tar -xf - && sha256sum -c {sha.name}'"],
                       stdin=tar.stdout, check=True)
        tar.wait()
    print(f"{name}: README refreshed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path.home() / "datasets")
    p.add_argument("--only", choices=list(DATASETS), help="build just this dataset")
    p.add_argument("--vm", action="append", default=[], help="GPU VM to copy into; repeatable")
    p.add_argument("--no-lerobot", action="store_true",
                   help="skip the LeRobot conversion; copy only the parquet + manifest/README/sha256 "
                        "(smolvla_py train.py reads the parquet directly)")
    p.add_argument("--readme-only", action="store_true",
                   help="only re-render README + sha256 from the existing manifest and push those small files")
    args = p.parse_args()

    for name, spec in DATASETS.items():
        if args.only and name != args.only:
            continue
        if args.readme_only:
            refresh_readme(name, args.out_dir, args.vm)
            continue
        files = build(name, spec, args.out_dir)
        if args.no_lerobot:
            for host in args.vm:
                push_files(host, files)
            continue
        root = args.out_dir / "lerobot" / name
        subprocess.run([sys.executable, str(_HERE.parent / "convert_to_lerobot.py"), "--parquets", str(files[0]),
                        "--root", str(root), "--repo-id", f"local/{name}", "--fps", "10"],
                       check=True, stdout=subprocess.DEVNULL)
        (root / "README.md").write_text(files[2].read_text())
        print(f"  converted -> {root}")
        for host in args.vm:
            copy_to_vm(host, files, root)


if __name__ == "__main__":
    main()
