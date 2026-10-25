"""Train the quarter checker (smolvla/quarter_checker.py) from the recorded episodes.

Labels come from the episode structure, no hand labelling:
  - start vs end of one episode            -> its quarter got a piece
  - start of Ek vs end of Ek+1 (back to back) -> both quarters
  - two frames of the same parked moment, end of Ek vs start of Ek+1,
    or a reversed (after, before) pair      -> nothing added
Only the scene camera (cam0) is used. Rounds are split train/val.

Run:
    uv run python scripts/train_checker.py --parquets ~/datasets/cheese-fixed-s1-2026-10-24.parquet ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from smolvla.quarter_checker import QUARTERS, QuarterChecker, jpeg_to_tensor

ROOT = Path(__file__).resolve().parent.parent
PARKED = 6          # frames at each end of an episode to consider "arm parked"
STILL = 0.004       # max per-frame joint change (norm) to count as still
CHAIN_GAP_S = 15.0  # episodes closer than this are back to back (no failed take between)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquets", type=Path, nargs="+", required=True,
                   help="Raw exports with a .manifest.json next to each")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", type=int, nargs=4, default=[64, 80, 160, 176], metavar=("X0", "Y0", "X1", "Y1"),
                   help="Pizza box in the 224x224 scene frame (same spot in every session so far)")
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "quarter-checker")
    return p.parse_args()


def quarter_of(task: str, ep: int) -> str | None:
    for q in QUARTERS:
        if q in task:
            return q
    if re.match(r"^put (?:the )?\w+ on the pizza$", task):
        return QUARTERS[(ep - 1) % 4]  # recorded clockwise: E1 TL, E2 TR, E3 BR, E4 BL
    return None  # center / most empty space


def parked_rows(pos: np.ndarray, head: bool) -> list[int]:
    n = len(pos)
    rows = list(range(min(PARKED, n))) if head else list(range(max(n - PARKED, 0), n))
    still = [r for r in rows if np.abs(pos[min(r + 1, n - 1)] - pos[max(r - 1, 0)]).max() < 2 * STILL]
    return still if len(still) >= 2 else (rows[:3] if head else rows[-3:])


def load_episodes(paths: list[Path]) -> list[dict]:
    """One dict per episode: quarter, round key, start/end frames (uint8 tensors), chain link."""
    episodes = []
    for path in paths:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())["episodes"]
        meta = {int(e["episode_start_ns"]): e for e in manifest}
        table = pq.read_table(path, columns=["episode_start_ns", "joints", "task", "images"])
        ep_col = table.column("episode_start_ns").to_numpy()
        tasks = table.column("task")
        joints = table.column("joints")
        images = table.column("images")
        prev = None
        for start_ns in [int(e["episode_start_ns"]) for e in manifest]:
            idx = np.flatnonzero(ep_col == start_ns)
            if len(idx) == 0:
                continue
            m = meta[start_ns]
            pos = np.array([[j["position_norm"] for j in joints[int(i)].as_py()] for i in idx], np.float32)

            def frames(rows):
                return [(jpeg_to_tensor(images[int(idx[r])].as_py()[0]["jpeg"]) * 255).to(torch.uint8) for r in rows]

            ep = {
                "name": f"{path.stem}:{m['name']}",
                "round": f"{path.stem}:R{m['round']}",
                "quarter": quarter_of(tasks[int(idx[0])].as_py(), int(m["ep"])),
                "start": frames(parked_rows(pos, head=True)),
                "end": frames(parked_rows(pos, head=False)),
                "next": None,
            }
            if prev is not None and prev["round"] == ep["round"]:
                per_s = (prev["_to"] - prev["_from"]) / max(prev["_seconds"], 1e-6)
                if (m["from"] - prev["_to"]) / per_s < CHAIN_GAP_S:
                    prev["next"] = ep
            ep.update(_from=m["from"], _to=m["to"], _seconds=m["seconds"])
            episodes.append(ep)
            prev = ep
        print(f"{path.name}: {sum(1 for e in episodes if e['round'].startswith(path.stem + ':'))} episodes")
    return episodes


def label(*quarters: str | None) -> torch.Tensor:
    y = torch.zeros(len(QUARTERS))
    for q in quarters:
        if q is not None:
            y[QUARTERS.index(q)] = 1.0
    return y


def sample_pair(episodes: list[dict], rng: random.Random):
    """Draw one (before, after, label) pair. Returns uint8 tensors."""
    while True:
        ep = rng.choice(episodes)
        kind = rng.random()
        if kind < 0.40 and ep["quarter"]:
            return rng.choice(ep["start"]), rng.choice(ep["end"]), label(ep["quarter"])
        if kind < 0.50 and ep["quarter"] and ep["next"] and ep["next"]["quarter"]:
            return rng.choice(ep["start"]), rng.choice(ep["next"]["end"]), label(ep["quarter"], ep["next"]["quarter"])
        if kind < 0.70:
            group = ep["start"] if rng.random() < 0.5 else ep["end"]
            return rng.choice(group), rng.choice(group), label()
        if kind < 0.85:
            return rng.choice(ep["end"]), rng.choice(ep["start"]), label()
        if ep["next"]:
            return rng.choice(ep["end"]), rng.choice(ep["next"]["start"]), label()


def augment(before: torch.Tensor, after: torch.Tensor, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared small shift + lighting (same camera), plus a little independent exposure noise."""
    s = rng.uniform(0.95, 1.0)  # the model crops the pizza itself; keep this jitter small
    h = w = int(224 * s)
    top, left = rng.randint(0, 224 - h), rng.randint(0, 224 - w)
    bright, contrast, sat = rng.uniform(0.7, 1.3), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    out = []
    for img in (before, after):
        img = TF.resized_crop(img, top, left, h, w, [224, 224], antialias=True)
        img = img.float() / 255.0
        img = TF.adjust_brightness(img, bright * rng.uniform(0.97, 1.03))
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, sat)
        out.append(img.clamp(0, 1))
    return out[0], out[1]


def make_batch(episodes, rng, n, train: bool, device):
    bs, as_, ys = [], [], []
    for _ in range(n):
        b, a, y = sample_pair(episodes, rng)
        if train:
            b, a = augment(b, a, rng)
        else:
            b, a = b.float() / 255.0, a.float() / 255.0
        bs.append(b); as_.append(a); ys.append(y)
    return torch.stack(bs).to(device), torch.stack(as_).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate(model, episodes, device) -> dict:
    """Fixed val pairs: every episode's start->end (positive) and end->start + start->start (negative)."""
    model.eval()
    pos_hit = pos_n = neg_ok = neg_n = exact = total = 0
    for ep in episodes:
        pairs = []
        if ep["quarter"]:
            pairs.append((ep["start"][0], ep["end"][-1], label(ep["quarter"])))
        pairs.append((ep["end"][-1], ep["start"][0], label()))
        pairs.append((ep["start"][0], ep["start"][-1], label()))
        for b, a, y in pairs:
            p = torch.sigmoid(model(b[None].float().to(device) / 255, a[None].float().to(device) / 255))[0].cpu()
            pred = p > 0.5
            exact += int(torch.equal(pred, y.bool())); total += 1
            if y.any():
                pos_n += 1; pos_hit += int(p.argmax() == y.argmax() and pred[y.argmax()])
            else:
                neg_n += 1; neg_ok += int(not pred.any())
    model.train()
    return {"exact": exact / max(total, 1), "pos_quarter_acc": pos_hit / max(pos_n, 1),
            "neg_clean": neg_ok / max(neg_n, 1), "n_pos": pos_n, "n_neg": neg_n}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = load_episodes(args.parquets)
    rounds = sorted({e["round"] for e in episodes})
    rng.shuffle(rounds)
    val_rounds = set(rounds[: max(1, round(len(rounds) * args.val_frac))])
    train_eps = [e for e in episodes if e["round"] not in val_rounds]
    val_eps = [e for e in episodes if e["round"] in val_rounds]
    print(f"episodes: train {len(train_eps)}  val {len(val_eps)} ({len(val_rounds)} rounds held out)")

    model = QuarterChecker(crop=args.crop).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    t0, running = time.time(), 0.0
    best = None
    for step in range(1, args.steps + 1):
        b, a, y = make_batch(train_eps, rng, args.batch_size, train=True, device=device)
        loss = F.binary_cross_entropy_with_logits(model(b, a), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        running += loss.item()
        if step % 250 == 0 or step == args.steps:
            metrics = evaluate(model, val_eps, device)
            print(f"step {step:>5}  loss {running / 250:.4f}  val {metrics}  ({time.time() - t0:.0f}s)")
            running = 0.0
            if best is None or metrics["exact"] >= best["exact"]:
                best = {**metrics, "step": step}
                model.save(args.output, {"val": best, "val_rounds": sorted(val_rounds),
                                         "parquets": [str(p) for p in args.parquets]})
    print(f"best {best} -> {args.output}")


if __name__ == "__main__":
    main()
