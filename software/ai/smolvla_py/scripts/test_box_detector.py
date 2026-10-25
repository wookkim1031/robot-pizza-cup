"""Check pizza.box_detector on frame 0 (arm at home) of every episode, and the box guard with detected boxes.

    python scripts/test_box_detector.py ~/v2/lerobot/cheese-r1r10-2026-10-24 ... --out ~/v2/boxguard
    python scripts/test_box_detector.py ... --scan     # + BoxGuard.from_frame(frame 0) over every episode

Reads LeRobot datasets with the top camera as video (v2: info["video_path"], needs av) or as images in
the parquet files (v3). Reports per container: detection rate, IoU against the hand-measured rectangles
(fixed, and moved by the shift a template match finds in each episode: the cheese box sits ~6 px left
of the hand rectangle in most episodes), runtime; saves detect_test.jpg (worst cases: hand rectangle
green, detection magenta). --scan reports the episodes where the guard fires (scan_detected.csv): the
containers stay put in nearly all demos, so every trigger is a false alarm or a real push to look at.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pizza.box_detector import detect_boxes  # noqa: E402
from pizza.box_guard import BoxGuard, _ncc  # noqa: E402

HAND = {"pepperoni": [41, 105, 79, 148], "cheese": [159, 105, 197, 144]}  # measured on cheese-r1r10 ep 0
TOP_KEYS = ("observation.images.top", "observation.images.cam0", "observation.images.cam1")


def episodes(root: Path, first_only: bool):
    """Yield (episode index, list of top-camera RGB frames)."""
    info = json.load(open(root / "meta/info.json"))
    if "video_path" in info and (root / "videos").exists():
        import av
        for e in map(json.loads, open(root / "meta/episodes.jsonl")):
            i = e["episode_index"]
            vp = root / info["video_path"].format(episode_chunk=i // info["chunks_size"],
                                                  video_key="observation.images.top", episode_index=i)
            with av.open(str(vp)) as c:
                frames = []
                for f in c.decode(video=0):
                    frames.append(f.to_ndarray(format="rgb24"))
                    if first_only:
                        break
            yield i, frames
        return
    import pyarrow.parquet as pq
    cams = [k for k in TOP_KEYS if k in info["features"]]
    eps: dict[int, list] = {}
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", *cams])
        for r in range(t.num_rows):
            if first_only and t.column("frame_index")[r].as_py() != 0:
                continue
            ims = [np.asarray(Image.open(io.BytesIO(t.column(c)[r].as_py()["bytes"])).convert("RGB")) for c in cams]
            top = next((im for im in ims if im[-40:].mean() >= 70), ims[0])  # wrist view: dark gripper band
            eps.setdefault(t.column("episode_index")[r].as_py(), []).append(top)
    yield from sorted(eps.items())


def iou(p, q) -> float:
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    return ix * iy / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy)


def tracked(gray: np.ndarray, ref_gray: np.ndarray, box, m: int = 10):
    """Hand rectangle moved to where the reference episode's crop matches best in this frame."""
    x0, y0, x1, y1 = box
    res = _ncc(gray[y0 - m:y1 + m, x0 - m:x1 + m], ref_gray[y0:y1, x0:x1])
    dy, dx = (int(v) - m for v in np.unravel_index(int(np.argmax(res)), res.shape))
    return [x0 + dx, y0 + dy, x1 + dx, y1 + dy]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--scan", action="store_true", help="also run the guard over every frame of every episode")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, ref_gray = [], None
    for root in args.datasets:
        for ep, frames in episodes(root, first_only=not args.scan):
            img = frames[0]
            gray = BoxGuard._gray(img)
            ref_gray = gray if ref_gray is None else ref_gray
            t0 = time.perf_counter()
            det = detect_boxes(img)
            ms = (time.perf_counter() - t0) * 1000
            row = dict(src=root.name, ep=ep, ms=ms, img=img, det=det)
            for n, hand in HAND.items():
                trk = tracked(gray, ref_gray, hand)
                row[n] = (iou(det[n], hand) if n in det else 0.0, iou(det[n], trk) if n in det else 0.0)
            if args.scan and det:
                g = BoxGuard(det)
                g.calibrate(img)
                for k, f in enumerate(frames[1:], 1):
                    g.update(f)
                    if g.moved:
                        row["trigger"] = (k, g.moved)
                        row["trigger_img"] = frames[min(k + 5, len(frames) - 1)]
                        row["end_img"] = frames[-1]
                        break
            rows.append(row)
            print(f"{root.name} ep{ep}: {det} {ms:.0f} ms" + (f"  TRIGGER {row['trigger']}" if "trigger" in row else ""),
                  flush=True)

    print(f"\n{len(rows)} episodes, runtime median {np.median([r['ms'] for r in rows]):.1f} ms, "
          f"max {max(r['ms'] for r in rows):.1f} ms")
    for n in HAND:
        v = np.array([r[n] for r in rows])
        print(f"{n:10s} detected {np.mean([n in r['det'] for r in rows]):6.1%}   IoU>=0.7 vs hand rect "
              f"{np.mean(v[:, 0] >= 0.7):6.1%} (median {np.median(v[:, 0]):.2f})   vs tracked hand rect "
              f"{np.mean(v[:, 1] >= 0.7):6.1%} (median {np.median(v[:, 1]):.2f}, min {v[:, 1].min():.2f})")

    worst = sorted(rows, key=lambda r: min(r[n][1] for n in HAND))[:12]
    sheet = Image.new("RGB", (224 * 4, 224 * 3))
    for k, r in enumerate(worst):
        im = Image.fromarray(r["img"]).resize((224, 224))
        sx, sy = 224 / r["img"].shape[1], 224 / r["img"].shape[0]
        d = ImageDraw.Draw(im)
        for n, hand in HAND.items():
            d.rectangle(hand, outline=(0, 255, 0))
            if n in r["det"]:
                b = r["det"][n]
                d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=(255, 0, 255))
        d.text((2, 2), f"{r['src'][:12]} ep{r['ep']} {min(r[n][1] for n in HAND):.2f}", fill=(255, 255, 0))
        sheet.paste(im, ((k % 4) * 224, (k // 4) * 224))
    sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST).save(args.out / "detect_test.jpg")
    print(f"saved {args.out / 'detect_test.jpg'} (worst 12 by IoU vs tracked hand rect)")

    if args.scan:
        hit = [r for r in rows if "trigger" in r]
        print(f"\nguard with detected boxes: {len(hit)}/{len(rows)} episodes trigger")
        for r in hit:
            print(f"  {r['src']} ep{r['ep']}: {r['trigger'][1]} at frame {r['trigger'][0]}")
        with open(args.out / "scan_detected.csv", "w") as f:
            f.write("src,ep,triggered,trigger_frame,box\n")
            for r in rows:
                t = r.get("trigger")
                f.write(f"{r['src']},{r['ep']},{t is not None},{t[0] if t else ''},{t[1] if t else ''}\n")
        if hit:
            tiles = []
            for r in hit[:12]:
                tile = Image.new("RGB", (r["img"].shape[1] * 3, r["img"].shape[0]))
                for k, im in enumerate((r["img"], r["trigger_img"], r["end_img"])):
                    pi = Image.fromarray(im)
                    ImageDraw.Draw(pi).rectangle(r["det"][r["trigger"][1]], outline=(255, 0, 0))
                    ImageDraw.Draw(pi).text((2, 2), f"{r['src'][:10]} ep{r['ep']} {['start', 'trigger', 'end'][k]}",
                                            fill=(255, 255, 0))
                    tile.paste(pi, (k * im.shape[1], 0))
                tiles.append(tile)
            out = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
            for k, t in enumerate(tiles):
                out.paste(t, (0, k * t.height))
            out.save(args.out / "scan_detected.jpg")
            print(f"saved {args.out / 'scan_detected.jpg'} (start / trigger / end)")


if __name__ == "__main__":
    main()
