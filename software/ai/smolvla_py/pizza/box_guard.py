"""Detect if a side container (pepperoni tray, cheese box) is moved by the arm. Top camera, no training.

    g = BoxGuard.from_frame(top_rgb)     # arm at home: detect the containers and take the reference crops
    g = BoxGuard("boxes.json")           # or fixed rectangles: path or {"<name>": [x0, y0, x1, y1]} ...
    g.calibrate(top_rgb)                 #   ... then calibrate at home
    st = g.update(top_rgb)               # every tick -> {"pepperoni": {...}, "cheese": {...}}
    if g.moved: abort()                  # g.moved = name of the moved container or None

Logic per container: template match of the reference crop in a window around its place.
  - match found but shifted > SHIFT_PX for K frames in a row          -> moved (pushed/dragged)
  - arm gone from the box area but box no longer matches at its place -> moved (lifted/taken)
  - arm covering the box with no shift                                -> fine (occlusion)

numpy port of the cv2 BoxGuard on DevLab (~/v2/scripts/box_guard.py), same thresholds; boxes.json format
{"image_size": [w, h], "boxes": {"<name>": [x0, y0, x1, y1], ...}}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi

from pizza.box_detector import detect_boxes


def _ncc(window: np.ndarray, t: np.ndarray) -> np.ndarray:
    """cv2.TM_CCOEFF_NORMED of template t over every position in window."""
    w = sliding_window_view(window, t.shape)
    t = t - t.mean()
    wm = w - w.mean(axis=(-1, -2), keepdims=True)
    den = np.sqrt((wm ** 2).sum(axis=(-1, -2)) * (t ** 2).sum())
    return np.where(den > 1e-6, (wm * t).sum(axis=(-1, -2)) / np.maximum(den, 1e-6), 0.0)


class BoxGuard:
    SEARCH = 12          # px search margin around the box
    SHIFT_PX = 4.0       # sustained shift above this = moved
    MATCH_OK = 0.6       # NCC score for a reliable match
    MATCH_LOST = 0.45    # below this with no occlusion = box gone/lifted
    OCC_PIX = 30         # gray level change counted as 'covered'
    OCC_FREE = 0.10      # covered fraction below this = arm not over the box
    K = 3                # consecutive frames needed

    def __init__(self, boxes: str | Path | dict):
        if not isinstance(boxes, dict):
            boxes = json.load(open(boxes))
        self.boxes = {n: [int(v) for v in b] for n, b in boxes.get("boxes", boxes).items()}
        self.ref, self.count, self.moved = {}, {}, None
        self.image_size = None

    @classmethod
    def from_frame(cls, top_rgb) -> "BoxGuard":
        """Detect the containers in this frame (arm at home) and calibrate on it."""
        g = cls(detect_boxes(top_rgb))
        g.calibrate(top_rgb)
        return g

    def save(self, path: str | Path) -> None:
        json.dump({"image_size": self.image_size, "boxes": self.boxes}, open(path, "w"), indent=1)

    @staticmethod
    def _gray(img) -> np.ndarray:
        rgb = np.asarray(img, np.uint8).astype(np.float32)
        g = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        k = np.array([0.25, 0.5, 0.25], np.float32)        # cv2.GaussianBlur((3, 3), 0)
        return ndi.convolve1d(ndi.convolve1d(g, k, axis=0, mode="reflect"), k, axis=1, mode="reflect")

    def calibrate(self, img) -> None:
        g = self._gray(img)
        self.image_size = [g.shape[1], g.shape[0]]
        for n, (x0, y0, x1, y1) in self.boxes.items():
            self.ref[n] = g[y0:y1, x0:x1].copy()
            self.count[n] = 0
        self.moved = None

    def update(self, img) -> dict:
        g = self._gray(img)
        h, w = g.shape
        out = {}
        for n, (x0, y0, x1, y1) in self.boxes.items():
            t = self.ref[n]
            m = self.SEARCH
            wx0, wy0, wx1, wy1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
            res = _ncc(g[wy0:wy1, wx0:wx1], t)
            ly, lx = np.unravel_index(int(np.argmax(res)), res.shape)
            score = float(res[ly, lx])
            dx, dy = int(lx) + wx0 - x0, int(ly) + wy0 - y0
            shift = float(np.hypot(dx, dy))
            occ = float((np.abs(g[y0:y1, x0:x1] - t) > self.OCC_PIX).mean())
            pushed = score >= self.MATCH_OK and shift > self.SHIFT_PX
            gone = occ < self.OCC_FREE and score < self.MATCH_LOST
            suspicious = pushed or (gone and not (score >= self.MATCH_OK and shift <= self.SHIFT_PX))
            self.count[n] = self.count[n] + 1 if suspicious else 0
            if self.count[n] >= self.K and self.moved is None:
                self.moved = n
            out[n] = dict(dx=dx, dy=dy, shift=round(shift, 1), score=round(score, 3),
                          covered=round(occ, 3), suspicious=suspicious)
        return out
