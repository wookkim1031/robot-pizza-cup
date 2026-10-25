"""Find the two topping containers next to the pizza in a top-camera frame (no training, numpy + scipy).

    boxes = detect_boxes(top_rgb)   # {"pepperoni": [x0, y0, x1, y1], "cheese": [x0, y0, x1, y1]}

Run once per run with the arm at home and the containers in view. Layout (from the top camera):
pepperoni tray LEFT of the pizza, cheese box RIGHT of it, both small brown cardboard trays on white paper.

    1. pizza: the largest sauce-red blob (a* high) near the middle, plus a crust ring
    2. topping pixels: pepperoni = strongly saturated red (the sauce is less saturated and masked out with
       the pizza), cheese = bright and yellowish (paper is neutral, the board and crust are darker / less yellow)
    3. a container-sized window slides over the band left (pepperoni) / right (cheese) of the pizza and
       keeps the position that covers the most topping while avoiding paper, cables and the pizza.
       A window, not the topping's bounding box: the tray is the same size whether it holds 2 slices or 8.
    4. cheese only: a watershed on the colour gradient around the window, seeded with cheese + tray-brown
       pixels inside it and paper/pizza/cables/crop border outside, snaps the box to the tray rim. (For the
       pepperoni tray it shrinks onto the slices: their edges are stronger than the tray's.)

Parameters are in pixels of the 224x224 training frames and scale with the image width.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

REF_WIDTH = 224
MARGIN = 3                     # px added around each container
SAUCE_A = 7.0                  # a* above this = sauce/pepperoni red
CRUST = 9                      # px from the sauce edge to the outer crust edge
PEP_A, PEP_S = 18.0, 0.45      # pepperoni slices: a*, saturation (sauce: a* 10-14, s 0.4-0.55)
CHEESE_L, CHEESE_B = 72.0, 6.5  # grated cheese: bright and yellowish (paper b* 0-3, board L* ~60, crust b* 2-7)
CHEESE_TEX = 2.0               # and grainy: 3x3 std of L* (a yellow sticky note on the paper is flat)
TRAY_W, TRAY_H = (30, 36), (34, 42)   # container size range (px, without margin), both trays
MIN_TOPPING = 15               # topping px inside the window below this = container not found
# Window score per pixel: topping, paper, dark (cables/arm), pizza. Rewarding cardboard pulled the
# pepperoni window onto the board under the pizza, which is the same cardboard.
W_TOP, W_PAPER, W_DARK, W_PIZZA = 8.0, -1.0, -1.0, -3.0
REFINE = {"cheese"}            # containers whose window is snapped to the tray rim by a watershed
REFINE_PAD = 10                # px around the window the watershed may use
TRAY_L, TRAY_B = 56.0, 9.0     # tray cardboard seen from above: darker and yellower than the board (L* ~60)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIE Lab (D65), float32."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]],
                       np.float32)
    xyz /= np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    mx, mn = rgb.max(-1).astype(np.float32), rgb.min(-1).astype(np.float32)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)


def _texture(L: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(ndi.uniform_filter(L * L, 3) - ndi.uniform_filter(L, 3) ** 2, 0))


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def find_pizza(top_rgb: np.ndarray, lab: np.ndarray | None = None) -> tuple[float, float, float] | None:
    """(cx, cy, r) of the pizza with its crust: the largest sauce blob near the middle of the frame."""
    lab = _lab(top_rgb) if lab is None else lab
    h, w = lab.shape[:2]
    s = w / REF_WIDTH
    red = ndi.binary_closing(lab[..., 1] > SAUCE_A, _disk(max(1, round(3 * s))))
    red = ndi.binary_fill_holes(ndi.binary_opening(red, _disk(max(1, round(2 * s)))))
    lbl, n = ndi.label(red)
    if not n:
        return None
    area = ndi.sum(red, lbl, range(1, n + 1))
    cy, cx = np.array(ndi.center_of_mass(red, lbl, range(1, n + 1))).T
    central = (cx > 0.2 * w) & (cx < 0.8 * w) & (cy > 0.2 * h) & (cy < 0.9 * h)
    if not central.any():
        return None
    k = int(np.argmax(np.where(central, area, 0)))
    ys, xs = np.nonzero(lbl == k + 1)
    r_sauce = float(np.percentile(np.hypot(xs - cx[k], ys - cy[k]), 95))
    return float(cx[k]), float(cy[k]), r_sauce + CRUST * s


def _best_window(score: np.ndarray, allowed: np.ndarray, topping: np.ndarray, s: float):
    """Container-sized window with the highest score sum whose centre is allowed -> (box, topping px)."""
    h, w = score.shape
    S = np.pad(score.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    T = np.pad(topping.astype(np.float32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = (-np.inf, None, 0.0)
    for bw in range(round(TRAY_W[0] * s), round(TRAY_W[1] * s) + 1, max(1, round(2 * s))):
        for bh in range(round(TRAY_H[0] * s), round(TRAY_H[1] * s) + 1, max(1, round(2 * s))):
            sums = S[bh:, bw:] - S[:-bh, bw:] - S[bh:, :-bw] + S[:-bh, :-bw]
            ok = allowed[bh // 2:bh // 2 + h - bh + 1, bw // 2:bw // 2 + w - bw + 1]
            sums = np.where(ok, sums, -np.inf)
            y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
            if sums[y, x] > best[0]:
                tops = T[y + bh, x + bw] - T[y, x + bw] - T[y + bh, x] + T[y, x]
                best = (sums[y, x], (x, y, x + bw, y + bh), tops)
    return best[1], best[2]


def _refine(lab: np.ndarray, top: np.ndarray, box, pizza, s: float):
    """Snap a window to the container rim: watershed of the colour gradient between the topping/tray
    inside the window and paper, pizza, cables and the border of a crop around it."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    h, w = L.shape
    x0, y0, x1, y1 = box
    pad = round(REFINE_PAD * s)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    sl = np.s_[Y0:Y1, X0:X1]
    grad = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    for ch, wt in ((L, 1.0), (b, 2.0), (a, 1.0)):
        c = ndi.gaussian_filter(ch[sl], 0.7 * s)
        grad += wt * np.hypot(ndi.sobel(c, 0), ndi.sobel(c, 1))
    grad = np.clip(grad / max(float(np.percentile(grad, 99)), 1e-6) * 255, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    tray = (L[sl] < TRAY_L) & (L[sl] > 30) & (b[sl] > TRAY_B) & (a[sl] < 15)
    fg = ndi.binary_erosion((top[sl] | tray) & inside)
    if fg.sum() < 5:
        return box
    paper = (L[sl] > 68) & (b[sl] < 3.5) & (_texture(L)[sl] < 2.5)
    border = np.zeros_like(fg)
    border[[0, -1], :] = border[:, [0, -1]] = True
    bg = (paper | (np.hypot(xx - pizza[0], yy - pizza[1]) < pizza[2]) | (L[sl] < 30) | border) & ~fg
    markers = np.zeros(fg.shape, np.int16)
    markers[bg], markers[fg] = 1, 2
    reg = ndi.binary_opening(ndi.watershed_ift(grad, markers) == 2, _disk(1))
    lbl, n = ndi.label(reg)
    if not n:
        return box
    sizes = ndi.sum(reg, lbl, range(1, n + 1))
    ys, xs = np.nonzero(np.isin(lbl, 1 + np.flatnonzero(sizes >= 0.15 * sizes.max())))
    return X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1


def detect_boxes(top_rgb, pizza=None) -> dict:
    """Return {"pepperoni": [x0, y0, x1, y1], "cheese": [x0, y0, x1, y1]} in pixel coords of top_rgb,
    each box tight around the container plus a small margin (about 3 px). Missing container -> key absent.
    pizza: optional (cx, cy, r) of the pizza to restrict the search (left / right of it)."""
    rgb = np.asarray(top_rgb, np.uint8)
    h, w = rgb.shape[:2]
    s = w / REF_WIDTH
    lab = _lab(rgb)
    pizza = pizza or find_pizza(rgb, lab)
    if pizza is None:
        return {}
    cx, cy, r = pizza
    yy, xx = np.mgrid[0:h, 0:w]
    on_pizza = np.hypot(xx - cx, yy - cy) < r
    band = np.abs(yy - cy) < 0.9 * r                         # window centres: at the pizza's height
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    toppings = {
        "pepperoni": ((a > PEP_A) & (_saturation(rgb) > PEP_S), (xx < cx - 0.6 * r) & (xx > cx - 2.5 * r)),
        "cheese": ((L > CHEESE_L) & (b > CHEESE_B) & (a < 3) & (_texture(L) > CHEESE_TEX),
                   (xx > cx + 0.6 * r) & (xx < cx + 2.5 * r)),
    }
    out = {}
    for name, (top, side) in toppings.items():
        top = ndi.binary_opening(top & ~on_pizza, _disk(1))
        score = np.where(top, W_TOP, 0.0)
        score += np.where(~top & (L > 68) & (b < 4.5), W_PAPER, 0.0)
        score += np.where(L < 30, W_DARK, 0.0)
        score += np.where(on_pizza, W_PIZZA, 0.0)
        box, n_top = _best_window(score.astype(np.float32), band & side, top, s)
        if box is None or n_top < MIN_TOPPING * s * s:
            continue
        if name in REFINE:
            box = _refine(lab, top, box, pizza, s)
        m = round(MARGIN * s)
        x0, y0, x1, y1 = box
        out[name] = [int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(w, x1 + m)), int(min(h, y1 + m))]
    return out
