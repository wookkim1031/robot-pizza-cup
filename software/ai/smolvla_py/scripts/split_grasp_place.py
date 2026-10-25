"""Split pepperoni and cheese episodes into a shared grasp skill and a per-target place skill.

Each recorded episode (home -> grasp at the bowl -> place on a target -> home) becomes two:
    grasp  start .. lift-off after the gripper closed   "Pick up one pepperoni slice from the bowl."
    place  lift-off .. end                               "Place the pepperoni slice you are holding <where> ..."
(cheese: "Pick up one portion of cheese from the bowl." / "Spread the cheese you are holding <where> ...")
The place half gets its own episode_start_ns (start + 1), so training chunks never cross the
split: the grasp half ends padded with "hold closed", the place half starts in the air.

The split comes from the follower gripper: first close (norm < --close), then lift-off = first
frame after it where an arm joint moved more than --lift-tol from the pose at the close. Episodes
without a clean close -> lift -> release are dropped.

Run (works on the raw station parquets, here or on the GPU box):
    uv run python scripts/split_grasp_place.py ~/datasets/pepperoni-r1r10-2026-10-24.parquet ...
Writes <name>-split.parquet next to each input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRASP_TASKS = {"pepperoni": "Pick up one pepperoni slice from the bowl.",
               "cheese": "Pick up one portion of cheese from the bowl."}
WHERE = {"TL": "on the upper-left quadrant of the pizza", "TR": "on the upper-right quadrant of the pizza",
         "BR": "on the lower-right quadrant of the pizza", "BL": "on the lower-left quadrant of the pizza",
         "CENTER": "in the center of the pizza", "GAP": "on the largest empty area of the pizza"}
_PLACE = {"pepperoni": "Place the pepperoni slice you are holding", "cheese": "Spread the cheese you are holding"}
PLACE_TASKS = {(ing, t): f"{verb} {w} and return to the start position."
               for ing, verb in _PLACE.items() for t, w in WHERE.items()}


def ingredient_and_target(task: str) -> tuple[str, str]:
    ings = [i for i in GRASP_TASKS if i in task]
    hits = [t for t, w in WHERE.items() if w in task]
    if len(ings) != 1 or len(hits) != 1:
        raise ValueError(f"not a single-ingredient, single-target task: {task!r}")
    return ings[0], hits[0]


def split(path: Path, close: float, release: float, lift_tol: float) -> Path:
    table = pq.read_table(path)
    starts = table.column("episode_start_ns").to_numpy()
    tasks = table.column("task").to_pylist()
    joints = table.column("joints").to_pylist()
    pos = np.array([[j["position_norm"] for j in row] for row in joints], dtype=np.float32)

    keep, new_start, new_task = [], [], []
    stats, dropped, lifts, holds = Counter(), [], [], []
    bounds = np.r_[0, np.flatnonzero(np.diff(starts) != 0) + 1, len(starts)]
    for e, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        g, arm = pos[a:b, -1], pos[a:b, :6]
        closed = np.flatnonzero(g < close)
        if not closed.size:
            dropped.append((e, "never closes")); continue
        c = int(closed[0])
        opened = c + np.flatnonzero(g[c:] > release)
        moved = c + np.flatnonzero(np.abs(arm[c:] - arm[c]).max(axis=1) > lift_tol)
        if not opened.size or not moved.size or moved[0] >= opened[0]:
            dropped.append((e, "no clean lift before release")); continue
        lift, rel = int(moved[0]), int(opened[0])
        ing, target = ingredient_and_target(tasks[a])
        lifts.append(lift - c)
        holds.append(float(np.median(g[lift:rel])))
        keep.extend(range(a, b))
        new_start += [int(starts[a])] * lift + [int(starts[a]) + 1] * (b - a - lift)
        new_task += [GRASP_TASKS[ing]] * lift + [PLACE_TASKS[(ing, target)]] * (b - a - lift)
        stats[f"{ing} {target}"] += 1

    out_table = table.take(pa.array(keep, type=pa.int64()))
    out_table = out_table.set_column(out_table.schema.get_field_index("episode_start_ns"), "episode_start_ns",
                                     pa.array(new_start, type=pa.uint64()))
    out_table = out_table.set_column(out_table.schema.get_field_index("task"), "task",
                                     pa.array(new_task, type=pa.string()))
    out = path.with_name(path.stem + "-split.parquet")
    pq.write_table(out_table, out)

    print(f"{path.name} -> {out.name}: {sum(stats.values())} episodes split, {len(dropped)} dropped")
    print(f"  targets {dict(stats)}")
    print(f"  close -> lift-off: median {np.median(lifts):.0f} frames (min {min(lifts)}, max {max(lifts)}); "
          f"gripper while holding: median {np.median(holds):.2f} (min {min(holds):.2f}, max {max(holds):.2f})")
    for e, why in dropped:
        print(f"  dropped episode {e}: {why}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parquets", type=Path, nargs="+")
    p.add_argument("--close", type=float, default=0.3, help="gripper norm below this = closed")
    p.add_argument("--release", type=float, default=0.6, help="gripper norm above this = released")
    p.add_argument("--lift-tol", type=float, default=0.03, help="arm joint motion (norm) that marks lift-off")
    args = p.parse_args()
    for path in args.parquets:
        split(path, args.close, args.release, args.lift_tol)


if __name__ == "__main__":
    main()
