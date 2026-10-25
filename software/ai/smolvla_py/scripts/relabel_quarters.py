"""Rewrite the task column so every placement episode names its pizza quarter.

Rounds were recorded clockwise, one piece per episode:
    E1 top-left, E2 top-right, E3 bottom-right, E4 bottom-left (E5-E8 = second lap).

Sessions tagged with a generic prompt ("put the cheese on the pizza") get the
quarter from the manifest's episode number. Sessions that already carry a
quarter prompt keep it. "salami" is renamed to "pepperoni" so both sessions
share one prompt vocabulary. Other prompts (center, most empty space) are kept.

Run:
    uv run python scripts/relabel_quarters.py ~/datasets/cheese-fixed-s1-2026-10-24.parquet ...
Writes <name>-quarters.parquet next to each input.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

QUARTERS = ("top-left", "top-right", "bottom-right", "bottom-left")
GENERIC_RE = re.compile(r"^put (?:the )?(\w+) on the pizza$")


def quarter_task(ingredient: str, quarter: str) -> str:
    return f"put the {ingredient} on the {quarter} quarter of the pizza"


def relabel(path: Path) -> Path:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    ep_number = {int(e["episode_start_ns"]): int(e["ep"]) for e in manifest["episodes"]}

    table = pq.read_table(path)
    episodes = table.column("episode_start_ns").to_numpy()
    tasks = table.column("task").to_pylist()

    new_tasks = []
    for ep, task in zip(episodes, tasks):
        task = task.replace("salami", "pepperoni")
        m = GENERIC_RE.match(task)
        if m:
            task = quarter_task(m.group(1), QUARTERS[(ep_number[int(ep)] - 1) % 4])
        new_tasks.append(task)

    idx = table.schema.get_field_index("task")
    table = table.set_column(idx, "task", pa.array(new_tasks, type=pa.string()))
    out = path.with_name(path.stem + "-quarters.parquet")
    pq.write_table(table, out)

    per_episode = Counter(t for t, first in zip(new_tasks, np.r_[True, np.diff(episodes) != 0]) if first)
    print(f"{path.name} -> {out.name}")
    for t, n in sorted(per_episode.items()):
        print(f"  {n:3d} eps  {t}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parquets", type=Path, nargs="+")
    for path in p.parse_args().parquets:
        relabel(path)


if __name__ == "__main__":
    main()
