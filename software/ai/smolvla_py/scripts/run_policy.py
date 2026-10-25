"""Interactive policy-driven hardware loop.

Per SPACE-press tick:
  1. Fetch one Frame from `inference/normvla`.
  2. Build obs (state from position_norm, both camera JPEGs → tensors).
  3. Tokenize the task prompt, run policy.predict_action_chunk.
  4. Unnormalize chunk[0] → 6 goal_norm in [0, 1] (clipped).
  5. Map each joint to a raw servo tick via the frame's inline
     (range_min, range_max), clipped to that range.
  6. Print preview, then send immediately (one SPACE = one send).
  7. Auto-abort if any joint delta exceeds --max-delta-ticks.
  8. After send, fetch again and print motion (position delta).

Keys:  SPACE = predict+send | q / Ctrl-C = quit

Run:
    uv run python scripts/run_policy.py \\
        --checkpoint checkpoints/run/final \\
        --task       "place the yellow cube on top of the other" \\
        --bus-serial 5AB9068807

`--server` defaults to localhost; pass it if the station daemon is on a
different host (e.g. --server ab-rpi5.server). `--bus-serial` is the ST3215
bus id from the station web interface.

Recipe mode — one prompt, planned into steps (smolvla/planner.py):
    uv run python scripts/run_policy.py \\
        --checkpoint checkpoints/run/final --bus-serial 5B3D050088 \\
        --plan "make a pizza: cheese then pepperoni"
Each step runs the policy on its quarter prompt until the arm is back home
(or --step-timeout). The quarter checker then compares the scene before and
after: a missed quarter is retried (--max-tries), and a quarter that already
got a stray piece is skipped. `--plan-only` prints the plan and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]
sys.path.insert(0, str(_REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(_REPO))

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.inferences import normvla  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

from smolvla import SmolVLAPolicy  # noqa: E402
from smolvla.normalize import normalize_state, unnormalize_action  # noqa: E402
from smolvla.planner import CENTER, GAP, Step, describe, plan_llm, plan_rules  # noqa: E402
from smolvla.quarter_checker import QUARTERS, QuarterChecker  # noqa: E402
from smolvla.stats import load_stats  # noqa: E402


QUEUE_ID = "inference/normvla"
ST3215_TARGET_POS_REGISTER = 0x2A
IMAGE_KEYS = ("observation.images.cam0", "observation.images.cam1")
CAMERA_ORDER: tuple[int, ...] = (0, 1)  # set from --camera-order in main_async


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Saved checkpoint dir with config.json, model.safetensors, stats.safetensors.")
    p.add_argument("--task",
                   help='Natural-language prompt the policy is conditioned on. '
                        'Example: --task "place the yellow cube on top of the other"')
    p.add_argument("--plan",
                   help='Recipe prompt for recipe mode, e.g. "make a pizza: cheese then pepperoni". '
                        "Replaces --task.")
    p.add_argument("--planner", choices=["rules", "llm"], default="rules")
    p.add_argument("--planner-llm", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Open-source instruct LLM for --planner llm (falls back to rules on failure).")
    p.add_argument("--plan-only", action="store_true", help="Print the plan and exit (no robot).")
    p.add_argument("--prompt-style", choices=["quarter", "generic"], default="quarter",
                   help="quarter: one prompt per quarter; generic: 'cover the pizza with X' and the "
                        "policy picks the quarter itself.")
    p.add_argument("--checker", type=Path,
                   help="Quarter checker dir (scripts/train_checker.py). Without it each step runs once, "
                        "unchecked. Experimental: ~45%% right quarter on held-out rounds so far.")
    p.add_argument("--checker-threshold", type=float, default=0.5)
    p.add_argument("--max-tries", type=int, default=2, help="Attempts per step when the checker says missed.")
    p.add_argument("--step-timeout", type=float, default=45.0, help="Seconds before a step is cut off.")
    p.add_argument("--settle-ticks", type=int, default=8,
                   help="Ticks back at the home pose that end a step.")
    p.add_argument("--step-confirm", action="store_true", help="Wait for ENTER before every step.")
    p.add_argument("--server", default="localhost",
                   help="Hostname of the machine running the station daemon.")
    p.add_argument("--bus-serial", required=True,
                   help='ST3215 bus id from the station web interface (open the station '
                        'viewer, look at the bus card). Example: --bus-serial 5AB9068807')
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8",
                   help="Comma-separated motor IDs, same order as joints in the Frame.")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--max-delta-ticks", type=int, default=1000,
                   help="Abort a tick if the predicted goal is more than this many ticks "
                        "from the current position on any joint. 0 to disable.")
    p.add_argument("--auto", action="store_true",
                   help="Run continuously — predict + send each tick, no SPACE. Ctrl-C to stop.")
    p.add_argument("--max-ticks", type=int, default=0,
                   help="Stop after N ticks in --auto mode. 0 = unlimited.")
    p.add_argument("--camera-order", default="auto",
                   help="Live image index fed to cam0,cam1, e.g. 1,0. 'auto' puts the scene camera "
                        "in cam0 and the wrist camera (dark gripper band at the bottom) in cam1, "
                        "decided once from the first frame. The station's order can swap between restarts.")
    return p.parse_args()


def detect_camera_order(frame: normvla.FrameReader) -> tuple[int, ...]:
    """Scene first, wrist second, judged by the wrist view's dark gripper band."""
    images = frame.get_images() or []
    is_wrist = []
    for img in images[: len(IMAGE_KEYS)]:
        with Image.open(io.BytesIO(bytes(img.get_jpeg()))) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32)
        is_wrist.append(gray[-40:].mean() < 70)
    if is_wrist.count(True) != 1:
        raise SystemExit(f"--camera-order auto: can't tell the cameras apart (wrist flags {is_wrist}); "
                         "pass --camera-order explicitly")
    wrist = is_wrist.index(True)
    return (1 - wrist, wrist)


# ---------------------------------------------------------------------------
# Queue helpers

async def fetch_frame(client, timeout: float) -> normvla.FrameReader:
    qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
    if entry is None:
        raise RuntimeError(f"{QUEUE_ID} closed without delivering a frame ({qr.err})")
    return normvla.FrameReader(memoryview(bytes(entry.Data)))


async def fetch_fresh_frame(
    client, last_id: bytes, timeout: float, retry_ms: int = 20,
) -> tuple[normvla.FrameReader, bytes]:
    """Fetch a frame whose global_frame_id differs from last_id.

    Inference/normvla updates at ~10 Hz; if we fetch faster than that we get
    the same frame back. Polls with a short sleep until a new id appears.
    """
    while True:
        frame = await fetch_frame(client, timeout)
        fid = bytes(frame.get_global_frame_id())
        if fid != last_id:
            return frame, fid
        await asyncio.sleep(retry_ms / 1000.0)


# ---------------------------------------------------------------------------
# Frame → model batch

def frame_to_batch(
    frame: normvla.FrameReader,
    stats: dict[str, torch.Tensor],
    image_keys: tuple[str, ...],
    device: torch.device,
) -> tuple[dict, list[tuple[int, int]]]:
    """Return (batch, per-joint (range_min, range_max))."""
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    if len(images) < len(image_keys):
        raise RuntimeError(f"Frame has {len(images)} images but policy expects {len(image_keys)}")

    state = torch.tensor(
        [j.get_position_norm() for j in joints], dtype=torch.float32, device=device
    ).unsqueeze(0)

    batch = {"observation.state": normalize_state(state, stats)}
    for i, key in zip(CAMERA_ORDER, image_keys):
        jpeg = bytes(images[i].get_jpeg())
        with Image.open(io.BytesIO(jpeg)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        batch[key] = (
            torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        )

    ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]
    return batch, ranges


def build_sync_write_command(
    bus_serial: str, motor_ids: list[int], raw_goals: list[int]
) -> commands.DriverCommand:
    """One sync-write: all joints update their target position atomically."""
    motors = [
        st3215.ST3215SyncWriteCommand_MotorWrite(
            motor_id=mid,
            value=raw.to_bytes(2, byteorder="little"),
        )
        for mid, raw in zip(motor_ids, raw_goals)
    ]
    sync = st3215.ST3215SyncWriteCommand(
        address=ST3215_TARGET_POS_REGISTER,
        motors=motors,
    )
    cmd = st3215.Command(target_bus_serial=bus_serial, sync_write=sync)
    return commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND,
        body=cmd.encode(),
    )


# ---------------------------------------------------------------------------
# One-tick inference + command build

@torch.no_grad()
def predict_commands(
    frame: normvla.FrameReader,
    policy: SmolVLAPolicy,
    stats: dict[str, torch.Tensor],
    task: str,
    bus_serial: str,
    motor_ids: list[int],
    device: torch.device,
) -> tuple[commands.DriverCommand, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Returns (single sync-write command, predicted_goal_norm[6], raw_ticks[6], ranges)."""
    batch, ranges = frame_to_batch(frame, stats, IMAGE_KEYS, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask

    # Same bf16 autocast as training: the SDPA attention path needs matching query/key dtypes.
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        pred_norm = policy.predict_action_chunk(batch)[0]    # (chunk_size, 6)
    pred_norm = pred_norm.float()
    pred_goal = unnormalize_action(pred_norm, stats)          # in [0, 1]-ish
    next_goal = pred_goal[0].cpu().clamp(0.0, 1.0).numpy()    # (6,)

    raws: list[int] = []
    for g_norm, (rmin, rmax) in zip(next_goal, ranges):
        raw = int(round(rmin + float(g_norm) * (rmax - rmin)))
        raws.append(max(rmin, min(rmax, raw)))
    cmd = build_sync_write_command(bus_serial, motor_ids, raws)
    return cmd, next_goal, np.asarray(raws, dtype=np.int64), ranges


def print_preview(
    motor_ids: list[int],
    frame: normvla.FrameReader,
    pred_goal_norm: np.ndarray,
    pred_raw: np.ndarray,
    ranges: list[tuple[int, int]],
    header: str = "",
) -> int:
    """Print a compact comparison table and return max |delta_ticks| across joints."""
    joints = frame.get_joints() or []
    deltas = [int(r) - int(j.get_position()) for r, j in zip(pred_raw, joints)]
    max_delta = max(abs(d) for d in deltas) if deltas else 0

    if header:
        print(f"[{header}]  max|Δ|={max_delta}")
    print(f"  {'motor':>5}  {'pos':>6}  {'pred':>6}  {'Δ':>6}  {'pred_norm':>9}")
    for mid, j, gn, raw, d in zip(motor_ids, joints, pred_goal_norm, pred_raw, deltas):
        print(f"  {mid:>5}  {int(j.get_position()):>6}  {int(raw):>6}  "
              f"{d:>+6}  {gn:>9.4f}")
    return max_delta


async def read_key() -> str:
    return await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)


async def run_one_tick(
    args, client, policy, stats, motor_ids, device, tick: int, last_id: bytes,
) -> bytes:
    t_start = time.perf_counter()
    frame, frame_id = await fetch_fresh_frame(client, last_id, args.timeout)
    t_fetched = time.perf_counter()
    cmd, pred_norm, pred_raw, ranges = predict_commands(
        frame, policy, stats, args.task, args.bus_serial, motor_ids, device
    )
    t_predicted = time.perf_counter()
    max_delta = print_preview(
        motor_ids, frame, pred_norm, pred_raw, ranges, header=f"tick {tick}"
    )

    fetch_ms = (t_fetched - t_start) * 1000
    predict_ms = (t_predicted - t_fetched) * 1000

    if args.max_delta_ticks and max_delta > args.max_delta_ticks:
        print(f"  aborted: max|Δ|={max_delta} > {args.max_delta_ticks} ticks  "
              f"(fetch {fetch_ms:.0f}ms, predict {predict_ms:.0f}ms)\n")
    else:
        await send_commands(client, [cmd])
        send_ms = (time.perf_counter() - t_predicted) * 1000
        print(f"  sent sync-write  (fetch {fetch_ms:.0f}ms, "
              f"predict {predict_ms:.0f}ms, send {send_ms:.0f}ms, "
              f"total {(time.perf_counter() - t_start) * 1000:.0f}ms)\n")
    return frame_id


async def auto_loop(args, client, policy, stats, motor_ids, device) -> None:
    print("\nauto: predict + send continuously.  Ctrl-C to stop.\n")
    tick = 0
    last_id = b""
    try:
        while True:
            if args.max_ticks and tick >= args.max_ticks:
                print(f"reached --max-ticks={args.max_ticks}, stopping.")
                return
            tick += 1
            last_id = await run_one_tick(
                args, client, policy, stats, motor_ids, device, tick, last_id,
            )
    except KeyboardInterrupt:
        print("\ninterrupted — stopping.")


async def interactive_loop(args, client, policy, stats, motor_ids, device) -> None:
    print("\nSPACE = predict + send   |   q / Ctrl-C = quit\n")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        tick = 0
        last_id = b""
        while True:
            ch = await read_key()
            if ch in ("q", "Q", "\x03", "\x04", ""):
                print("quitting.")
                return
            if ch != " ":
                continue

            tick += 1
            last_id = await run_one_tick(
                args, client, policy, stats, motor_ids, device, tick, last_id,
            )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Recipe mode: planner -> policy per step -> quarter checker

HOME_JOINTS = slice(0, 6)  # arm joints that define "home"; wrist roll and gripper vary
HOME_TOL = 0.08            # max |Δ| (norm) to count as back home
LEAVE_TOL = 0.12           # the arm has left home once any joint is this far away
STILL_TOL = 0.004


def joint_positions(frame: normvla.FrameReader) -> np.ndarray:
    return np.asarray([j.get_position_norm() for j in frame.get_joints() or []], dtype=np.float32)


def scene_jpeg(frame: normvla.FrameReader) -> bytes:
    return bytes(frame.get_images()[CAMERA_ORDER[0]].get_jpeg())


async def wait_parked(client, timeout: float, ticks: int = 5, max_wait: float = 10.0) -> normvla.FrameReader:
    """Return a frame once the arm has been still for `ticks` frames (or after max_wait)."""
    t0, still, last_id = time.perf_counter(), 0, b""
    frame, last_id = await fetch_fresh_frame(client, last_id, timeout)
    prev = joint_positions(frame)
    while still < ticks and time.perf_counter() - t0 < max_wait:
        frame, last_id = await fetch_fresh_frame(client, last_id, timeout)
        pos = joint_positions(frame)
        still = still + 1 if np.abs(pos - prev).max() < STILL_TOL else 0
        prev = pos
    return frame


async def run_step(args, client, policy, stats, motor_ids, device, prompt: str, home: np.ndarray) -> str:
    """Drive the policy on one prompt until it has left home and come back, or times out."""
    t0, tick, settled, left, last_id = time.perf_counter(), 0, 0, False, b""
    while True:
        frame, last_id = await fetch_fresh_frame(client, last_id, args.timeout)
        dist = float(np.abs(joint_positions(frame)[HOME_JOINTS] - home[HOME_JOINTS]).max())
        left = left or dist > LEAVE_TOL
        settled = settled + 1 if left and dist < HOME_TOL else 0
        if settled >= args.settle_ticks:
            return f"back home after {tick} ticks"
        if time.perf_counter() - t0 > args.step_timeout:
            return f"timeout after {tick} ticks ({'left' if left else 'never left'} home)"

        cmd, _, pred_raw, _ = predict_commands(
            frame, policy, stats, prompt, args.bus_serial, motor_ids, device
        )
        deltas = [abs(int(r) - int(j.get_position())) for r, j in zip(pred_raw, frame.get_joints() or [])]
        tick += 1
        if args.max_delta_ticks and max(deltas) > args.max_delta_ticks:
            print(f"    tick {tick}: skipped, max|Δ|={max(deltas)} > {args.max_delta_ticks}")
            continue
        await send_commands(client, [cmd])
        if tick % 20 == 0:
            print(f"    tick {tick}  {time.perf_counter() - t0:4.0f}s  dist-from-home {dist:.2f}")


async def recipe_loop(args, client, policy, stats, motor_ids, device, steps: list[Step], checker) -> None:
    print(f"\nrecipe: {args.plan!r} -> {len(steps)} steps\n{describe(steps)}\n")
    home = joint_positions(await wait_parked(client, args.timeout))
    print(f"home pose: {np.round(home, 2).tolist()}")
    placed = {s.ingredient: dict.fromkeys(QUARTERS, 0) for s in steps}
    planned = {s.ingredient: dict.fromkeys(QUARTERS, 0) for s in steps}
    log = []
    try:
        for i, step in enumerate(steps, 1):
            if step.target in QUARTERS:
                planned[step.ingredient][step.target] += 1
                if checker and placed[step.ingredient][step.target] >= planned[step.ingredient][step.target]:
                    print(f"[{i}/{len(steps)}] skip: {step.target} already has its {step.ingredient}")
                    log.append((step, "skipped (already covered)"))
                    continue
            prompt = step.prompt if args.prompt_style == "quarter" else step.generic_prompt
            for attempt in range(1, args.max_tries + 1):
                if args.step_confirm:
                    await asyncio.get_event_loop().run_in_executor(
                        None, input, f"[{i}/{len(steps)}] ENTER to run: {prompt!r} ")
                print(f"[{i}/{len(steps)}] attempt {attempt}: {prompt!r}")
                before = scene_jpeg(await wait_parked(client, args.timeout))
                ended = await run_step(args, client, policy, stats, motor_ids, device, prompt, home)
                after = scene_jpeg(await wait_parked(client, args.timeout))
                if checker is None:
                    log.append((step, ended)); break
                probs = checker.added(before, after)
                hits = [q for q, p in probs.items() if p >= args.checker_threshold]
                for q in hits:
                    placed[step.ingredient][q] += 1
                print(f"    {ended}; checker: " + "  ".join(f"{q} {p:.2f}" for q, p in probs.items()))
                ok = step.target in hits if step.target in QUARTERS else bool(hits)
                if ok or step.target in (CENTER, GAP) or args.prompt_style == "generic":
                    log.append((step, f"placed in {hits or 'nothing detected'}")); break
                print(f"    missed {step.target} (landed: {hits or 'nothing'})")
            else:
                log.append((step, f"missed after {args.max_tries} tries"))
    finally:
        print("\nsummary:")
        for step, result in log:
            print(f"  {step.ingredient:9s} {step.target:12s} {result}")
        for ing, counts in placed.items():
            print(f"  {ing} pieces per quarter (checker): {counts}")


# ---------------------------------------------------------------------------

async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_policy")
    motor_ids = [int(x) for x in args.motor_ids.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps = None
    if args.plan:
        steps = (plan_llm(args.plan, args.planner_llm, str(device)) if args.planner == "llm"
                 else plan_rules(args.plan))
        if args.plan_only:
            print(f"{args.plan!r} -> {len(steps)} steps\n{describe(steps)}")
            return
    elif not args.task:
        raise SystemExit("pass --task (single prompt) or --plan (recipe mode)")

    stats_path = args.checkpoint / "stats.safetensors"
    if not stats_path.exists():
        raise SystemExit(f"No stats.safetensors in {args.checkpoint}.")
    stats = {k: v.to(device) for k, v in load_stats(stats_path).items()}

    print(f"loading {args.checkpoint} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config_overrides={"load_vlm_weights": False},
        strict=False,
    ).to(device)
    policy.eval()

    checker = None
    if steps is not None and args.checker:
        checker = QuarterChecker.load(args.checker, device)
        print(f"checker:   {args.checker}")
    elif steps is not None:
        print("checker:   none; every step runs once, unchecked")

    print(f"task:      {args.plan if steps is not None else args.task!r}")
    print(f"server:    {args.server}")
    print(f"bus:       {args.bus_serial}")
    print(f"motors:    {motor_ids}")
    print(f"safety:    max_delta_ticks={args.max_delta_ticks}")

    client = await new_station_client(args.server, logger)
    global CAMERA_ORDER
    if args.camera_order == "auto":
        CAMERA_ORDER = detect_camera_order(await fetch_frame(client, args.timeout))
    else:
        CAMERA_ORDER = tuple(int(i) for i in args.camera_order.split(","))
    print(f"cameras:   live image {CAMERA_ORDER[0]} -> cam0 (scene), {CAMERA_ORDER[1]} -> cam1 (wrist)")
    if steps is not None:
        await recipe_loop(args, client, policy, stats, motor_ids, device, steps, checker)
    elif args.auto:
        await auto_loop(args, client, policy, stats, motor_ids, device)
    else:
        await interactive_loop(args, client, policy, stats, motor_ids, device)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
