"""Locate the pizza in a top-camera frame and build sector boxes for the checkers.
The sector circle (sauce area, from roi.json: center 118,131 r=28) has no clear edge, so we find the
OUTER pizza edge (white crust on brown board) and map it to the sector circle with a scale/offset
calibrated once on a reference frame.
    loc = PizzaLocator(ref_image=top_rgb_reference, method="hough")   # or "sam"
    cx, cy, r = loc.locate(top_rgb)                                  # sector circle on this frame
    boxes = loc.sector_boxes(cx, cy, r, top_rgb.shape)
    change_checker.roi = boxes
Copy of the DevLab locator (gpu2 ~/v2/scripts/pizza_locator.py); only ROI_JSON points at the local checkpoint dir.
Used by smolvla.failure_checker_v2 with method "red"."""
import json
from pathlib import Path

import numpy as np, cv2

ROI_JSON = Path(__file__).resolve().parents[1] / "checkpoints/change-checker-2026-10-25/roi.json"


class PizzaLocator:
    def __init__(self, ref_image, method="hough", roi_json=ROI_JSON):
        cfg = json.load(open(roi_json))
        self.cx0, self.cy0, self.r0 = cfg["pizza"]
        self.mapping = cfg["mapping"]
        self.method = method
        self._sam = None
        self.last_outer = None
        self.R_ref = None
        out = self._outer(np.asarray(ref_image, np.uint8), (self.cx0, self.cy0))
        if out is None:
            raise RuntimeError("could not find the pizza on the reference frame")
        ox, oy, R = out
        self.R_ref = R
        self.k = self.r0 / R                               # sector radius / outer radius
        self.off = ((self.cx0 - ox) / R, (self.cy0 - oy) / R)
        print(f"calibrated ({method}): outer circle {ox:.1f},{oy:.1f} R={R:.1f} -> k={self.k:.3f}, "
              f"offset=({self.off[0]:+.3f},{self.off[1]:+.3f})R")

    # ---- outer pizza edge ----
    def _hough_outer(self, img, guess):
        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
        rmin, rmax = int(1.2 * self.r0), int(2.6 * self.r0)
        c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=5, param1=80, param2=15,
                             minRadius=rmin, maxRadius=rmax)
        if c is None:
            return None
        gx, gy = guess
        best, best_s = None, np.inf
        for x, y, r in c[0][:40]:
            s = np.hypot(x - gx, y - gy) / self.r0
            if self.R_ref is not None:
                s += 2.0 * abs(r - self.R_ref) / self.R_ref
            if s < best_s:
                best, best_s = (float(x), float(y), float(r)), s
        return best

    def _sam_outer(self, img, guess, weights="mobile_sam.pt"):
        from ultralytics import SAM
        if self._sam is None:
            self._sam = SAM(weights)
        h, w = img.shape[:2]
        gx, gy = guess
        R = self.R_ref or 1.7 * self.r0
        box = [max(0, gx - 1.35 * R), max(0, gy - 1.35 * R), min(w, gx + 1.35 * R), min(h, gy + 1.35 * R)]
        res = self._sam(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), bboxes=[box], verbose=False)
        if res[0].masks is None:
            return None
        m = cv2.resize(res[0].masks.data[0].cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        mo = cv2.moments(cnt)
        if mo["m00"] == 0:
            return None
        x, y = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        r = float(np.sqrt(cv2.contourArea(cnt) / np.pi))
        if not (1.1 * self.r0 <= r <= 2.8 * self.r0):
            return None
        return x, y, r

    def _red(self, img, guess):
        """Sauce region by redness (Lab a*), Otsu threshold inside a window around the guess."""
        h, w = img.shape[:2]
        gx, gy = guess
        half = int(2.2 * self.r0)
        x0, y0 = int(max(0, gx - half)), int(max(0, gy - half))
        x1, y1 = int(min(w, gx + half)), int(min(h, gy + half))
        a = cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2LAB)[..., 1], (5, 5), 0)
        _, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_mask = (m, (x0, y0))
        best, best_s = None, np.inf
        for c in cnts:
            area = cv2.contourArea(c)
            r = float(np.sqrt(area / np.pi))
            if not (0.6 * self.r0 <= r <= 1.8 * self.r0):
                continue
            mo = cv2.moments(c)
            cx, cy = mo["m10"] / mo["m00"] + x0, mo["m01"] / mo["m00"] + y0
            sc = np.hypot(cx - gx, cy - gy) / self.r0 + abs(r - self.r0) / self.r0
            if sc < best_s:
                best, best_s = (cx, cy, r), sc
        return best

    def _outer(self, img, guess):
        if self.method == "red":
            return self._red(img, guess)
        if self.method == "sam":
            res = self._sam_outer(img, guess)
            if res is not None:
                return res
        return self._hough_outer(img, guess)

    # ---- public ----
    def locate(self, img, guess=None):
        img = np.asarray(img, np.uint8)
        if guess is None and self.R_ref is not None:
            guess = (self.cx0 - self.off[0] * self.R_ref, self.cy0 - self.off[1] * self.R_ref)
        out = self._outer(img, guess or (self.cx0, self.cy0))
        self.last_outer = out
        if out is None:
            return None
        ox, oy, R = out
        return ox + self.off[0] * R, oy + self.off[1] * R, self.k * R

    def sector_boxes(self, cx, cy, r, img_shape):
        h, w = img_shape[:2]
        cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
        pad = max(2, int(0.08 * r))
        geo = {"img_UL": (cx - r, cy - r, cx, cy), "img_UR": (cx, cy - r, cx + r, cy),
               "img_LR": (cx, cy, cx + r, cy + r), "img_LL": (cx - r, cy, cx, cy + r)}
        clip = lambda b: [int(max(0, b[0] - pad)), int(max(0, b[1] - pad)), int(min(w, b[2] + pad)), int(min(h, b[3] + pad))]
        boxes = {tg: clip(geo[q]) for tg, q in self.mapping.items()}
        c = 0.45 * r
        boxes["CENTER"] = clip((cx - c, cy - c, cx + c, cy + c))
        return boxes
