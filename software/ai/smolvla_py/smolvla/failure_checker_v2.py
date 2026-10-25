"""Failure checker v2: the v1 change checker with sector boxes placed around the pizza found in the frame.
At the start of a run (arm at home, clean or cheese-only pizza) the sauce region is located once by redness
(PizzaLocator method "red") and compared with where the locator puts the pizza at the start of the training
rounds (ref_located). The fixed roi.json boxes are moved by that difference, i.e. the five sector boxes are
rebuilt around roi.json center + offset with a FIXED radius (28 px, the scale the change model was trained at).
Offsets up to dead_zone_px are locator noise (round starts jitter ~2-3 px) and keep the fixed boxes, which the
model was trained on. If the pizza is not found, or the offset is larger than max_offset_px, the fixed roi.json
boxes are used (= v1).
    fc = FailureCheckerV2()
    status = fc.start(top)                                   # once, arm at home
    res = fc.check_step(before, after, "TL", "pepperoni")    # after every pepperoni step
    img = fc.render(after, res)                              # RGB overlay for the UI
Only pepperoni and the sectors TL TR BR BL CENTER are supported (cheese / GAP -> supported False).
Copy of the DevLab module (gpu2 ~/v2/scripts/failure_checker_v2.py, docs FAILURE_CHECKER_V2.md there): same
logic, local paths, the reference frame from a PNG instead of the LeRobot dataset."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from smolvla.change_checker import ChangeChecker
from smolvla.pizza_locator import PizzaLocator

CKPT = Path(__file__).resolve().parents[1] / "checkpoints/change-checker-2026-10-25"
MODEL, ROI_JSON = CKPT / "change.pt", CKPT / "roi.json"
REF_IMAGE = CKPT / "ref_top.png"   # frame 0 of cheese-r1r10-2026-10-24 episode 0 (top camera), lossless
TOPPINGS = {"pepperoni"}                                     # cheese is not reliable (51% detected)
SECTORS = ("TL", "TR", "BR", "BL", "CENTER")
# Median locator center (method red, calibrated on REF_SRC ep 0) over frame 0 of the first episode of the 20
# training rounds (cheese R1-R10 w/o R5/R10, pepperoni the same, pepperoni-on-cheese R1-R4). The pizza that the
# fixed boxes (center 118,131) were trained on is located here, so this point maps to the fixed boxes.
REF_LOCATED = (118.0, 133.8)


def reference_frame(path=REF_IMAGE):
    """The frame the locator is calibrated on (RGB)."""
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class FailureCheckerV2:
    def __init__(self, model=MODEL, roi_json=ROI_JSON, ref_image=None, radius=28, max_offset_px=15,
                 thr=0.65, device=None, ref_located=REF_LOCATED, dead_zone_px=4.0):
        cfg = json.load(open(roi_json))
        self.ref_center = [float(cfg["pizza"][0]), float(cfg["pizza"][1])]
        self.fixed_boxes = {s: [int(v) for v in b] for s, b in cfg["roi"].items()}
        self.radius, self.max_offset_px, self.thr = int(radius), float(max_offset_px), float(thr)
        self.ref_located = [float(v) for v in (ref_located or self.ref_center)]
        self.dead_zone_px = float(dead_zone_px)
        self.checker = ChangeChecker(model, device=device)
        if {s: list(b) for s, b in self.checker.roi.items()} != self.fixed_boxes:
            print(f"WARNING: {roi_json} boxes differ from the boxes {model} was trained with "
                  f"(roi.json pizza {cfg['pizza']}); restore roi_r28.json")
        if ref_image is None:
            ref_image = reference_frame()
        elif isinstance(ref_image, (str, Path)):
            ref_image = reference_frame(ref_image)
        self.locator = PizzaLocator(ref_image, method="red", roi_json=roi_json)
        self.status = self.use_fixed("start() not called yet")

    def _set(self, boxes, source, **status):
        self.boxes, self.boxes_source = boxes, source
        self.checker.roi = boxes
        self.status = {"version": "v2", "method": "red", "radius": self.radius, "boxes": boxes,
                       "boxes_source": source, **status}
        return self.status

    def use_fixed(self, reason="fixed boxes requested"):
        """The roi.json boxes, as v1 uses them."""
        return self._set(dict(self.fixed_boxes), "fixed", found=False, fallback=True, reason=reason,
                         center=list(self.ref_center), offset_px=None, shift_px=None, ms=0.0)

    def start(self, top_rgb):
        """Once per run, arm at home: locate the pizza, build the sector boxes, return the status dict."""
        t0 = time.perf_counter()
        top = np.ascontiguousarray(np.asarray(top_rgb, np.uint8))
        try:
            res = self.locator.locate(top)
        except Exception as e:                                # a bad frame must never stop the run
            res, err = None, f"{type(e).__name__}: {e}"
        else:
            err = "pizza not found"
        ms = round(1000 * (time.perf_counter() - t0), 1)
        if res is None:
            st = self.use_fixed(err)
            st["ms"] = ms
            return st
        dx, dy = float(res[0]) - self.ref_located[0], float(res[1]) - self.ref_located[1]
        off = float(np.hypot(dx, dy))
        cx, cy = self.ref_center[0] + dx, self.ref_center[1] + dy      # pizza center in roi.json terms
        found = dict(found=True, center=[round(cx, 1), round(cy, 1)], offset_px=round(off, 1),
                     shift_px=[round(dx, 1), round(dy, 1)], ms=ms)
        if off > self.max_offset_px:
            st = self.use_fixed(f"located center is {off:.1f}px from the reference (> {self.max_offset_px:g})")
            st.update(found)
            return st
        if off <= self.dead_zone_px:
            st = self._set(dict(self.fixed_boxes), "fixed", fallback=False,
                           reason=f"offset {off:.1f}px <= dead zone {self.dead_zone_px:g}px", **found)
            return st
        boxes = self.locator.sector_boxes(cx, cy, self.radius, top.shape)
        return self._set(boxes, "dynamic", fallback=False, reason="", **found)

    def check_step(self, before_rgb, after_rgb, target, topping="pepperoni"):
        """Was `topping` added to `target` between the two at-home frames? Same decision as v1 check_step."""
        out = {"version": "v2", "target": target, "topping": topping, "threshold": self.thr,
               "boxes_source": self.boxes_source}
        if topping not in TOPPINGS or target not in self.boxes:
            why = f"{topping} is not checked" if topping not in TOPPINGS else f"no box for {target}"
            return {**out, "supported": False, "reason": why, "p": None, "ok": None, "per_sector": {}, "elsewhere": {}}
        t0 = time.perf_counter()
        res = self.checker.scan(np.asarray(before_rgb, np.uint8), np.asarray(after_rgb, np.uint8))
        per = {s: round(float(v[topping]), 3) for s, v in res.items()}
        p = per[target]
        return {**out, "supported": True, "p": p, "ok": bool(p > self.thr), "per_sector": per,
                "elsewhere": {s: v for s, v in per.items() if s != target and v > self.thr},
                "ms": round(1000 * (time.perf_counter() - t0), 1)}

    def render(self, top_rgb, result=None, scale=3):
        """RGB overlay: located circle (red), sector boxes (yellow), target box (green = ok, red = missing,
        blue = not checked), per-sector probabilities. Upscaled `scale` times for legibility."""
        img = cv2.resize(np.asarray(top_rgb, np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        S = lambda v: int(round(v * scale))
        st = self.status
        cx, cy = st["center"]
        if st.get("found"):
            cv2.circle(img, (S(cx), S(cy)), S(self.radius), (255, 60, 60), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (S(self.ref_center[0]), S(self.ref_center[1])), (80, 160, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.drawMarker(img, (S(cx), S(cy)), (255, 60, 60), cv2.MARKER_CROSS, 14, 2)
        target = (result or {}).get("target")
        per = (result or {}).get("per_sector", {})
        for s, (x0, y0, x1, y1) in self.boxes.items():
            col, w = (255, 220, 0), 1
            if s == target:
                ok = result.get("ok")
                col, w = ((0, 220, 0) if ok else (255, 40, 40)) if ok is not None else (80, 160, 255), 3
            elif per.get(s, 0) > self.thr:
                col, w = (255, 140, 0), 2
            cv2.rectangle(img, (S(x0), S(y0)), (S(x1) - 1, S(y1) - 1), col, w)
            label = s if s not in per else f"{s} {per[s]:.2f}"
            org = (S(x0) + 4, S(y0) + 14) if s != "CENTER" else (S(x0) + 4, S(y1) - 6)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        head = (f"v2 {st['boxes_source']} boxes" + (f", pizza off {st['offset_px']}px" if st.get("offset_px") is not None else "")
                + (", FALLBACK" if st["fallback"] else ""))
        if result and result.get("supported"):
            head += f"  |  {result['topping']} {result['target']} p={result['p']:.2f} {'OK' if result['ok'] else 'MISSING'}"
        elif result:
            head += f"  |  not checked: {result.get('reason')}"
        cv2.putText(img, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

