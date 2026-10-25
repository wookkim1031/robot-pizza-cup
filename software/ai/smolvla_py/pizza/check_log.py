"""Failure-checker events for the web UI: one events.jsonl plus overlay PNGs in a directory.

    start      v2 localization at the run start (status dict of FailureCheckerV2.start + overlay)
    check      one checked step (v2 result dict + overlay, or the v1 verdict without an image)
    decision   what the step loop did with a check: attempt number, retry or not (refers to a check by seq)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


class CheckLog:
    def __init__(self, path: Path | str):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seq = 0

    def write(self, kind: str, data: dict, image: np.ndarray | None = None) -> int:
        self.seq += 1
        rec = {"seq": self.seq, "kind": kind, "t": round(time.time(), 2), **data}
        if image is not None:
            rec["image"] = f"{self.seq:03d}-{kind}.png"
            Image.fromarray(np.asarray(image, np.uint8)).save(self.dir / rec["image"])
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return self.seq
