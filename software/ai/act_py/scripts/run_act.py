# Copyright 2026 Norma Core contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a lerobot ACT checkpoint on the station hardware.

Port of smolvla_py/scripts/run_policy.py. Per tick:
  1. Fetch a fresh Frame from `inference/normvla` (~10 Hz, the training fps).
  2. Build obs (state from position_norm, both camera JPEGs → tensors).
  3. policy.select_action: ACT predicts a chunk and replays it one action
     per call, re-predicting when the queue runs dry.
  4. Map each goal_norm to a raw servo tick via the frame's inline
     (range_min, range_max), clipped to that range.
  5. Skip the send if any joint delta exceeds --max-delta-ticks.

Several --checkpoint dirs run one after another (e.g. one ACT model per pizza
quadrant). Each gets a fresh action queue; --auto runs each for --max-ticks,
interactive mode switches with `n`. All are loaded up front.

lerobot 0.3.x checkpoints (normalization baked into model.safetensors, no
policy_pre/postprocessor files) must be migrated first:
    uv run python -m lerobot.processor.migrate_policy_normalization \\
        --pretrained-path <old pretrained_model> --output-dir <new dir>

On start the follower holds its present position, then torque is enabled.
Stop teleop first: two writers on the same bus fight each other.

Keys:  SPACE = predict+send | n = next checkpoint | q / Ctrl-C = quit

Run:
    uv run python scripts/run_act.py \\
        --checkpoint outputs/act-cheese-pizza-2026-10-24/checkpoints/last/pretrained_model \\
        --bus-serial 5B3D050088

    # continuous
    uv run python scripts/run_act.py --checkpoint ... --bus-serial 5B3D050088 --auto

    # predictions only, nothing written to the bus (safe while teleop runs)
    uv run python scripts/run_act.py --checkpoint ... --bus-serial 5B3D050088 --auto --dry-run

    # cheese quadrants in order, 250 ticks (~25 s) each
    Q=outputs/act-quadrants-2026-10-24
    uv run python scripts/run_act.py --bus-serial 5B3D050088 --auto --max-ticks 250 \\
        --checkpoint $Q/top-left $Q/top-right $Q/bottom-right $Q/bottom-left
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

from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402


QUEUE_ID = "inference/normvla"
RAM_TORQUE_ENABLE = 0x28
RAM_ACC = 0x29
RAM_GOAL_POSITION = 0x2A
RAM_GOAL_SPEED = 0x2E
IMAGE_KEYS = ("observation.images.cam0", "observation.images.cam1")  # set from the checkpoint
CAMERA_ORDER = (0, 1)  # set from --camera-order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                   help="lerobot pretrained_model dir(s) (config.json, model.safetensors, "
                        "policy_pre/postprocessor files). Several run in the given order.")
    p.add_argument("--n-action-steps", type=int, default=0,
                   help="Actions executed per predicted chunk before re-predicting. "
                        "0 = checkpoint value. Lower is more reactive.")
    p.add_argument("--server", default="localhost",
                   help="Hostname of the machine running the station daemon.")
    p.add_argument("--bus-serial", required=True,
                   help="Follower ST3215 bus id, the one recorded in the dataset. "
                        "Example: --bus-serial 5B3D050088")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8",
                   help="Comma-separated motor IDs, same order as joints in the Frame.")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--max-delta-ticks", type=int, default=200,
                   help="Skip a tick if the predicted goal is more than this many ticks "
                        "from the current position on any joint. 0 to disable.")
    p.add_argument("--speed", type=int, default=2000,
                   help="Goal speed written once at start (teleop uses 300-3300).")
    p.add_argument("--accel", type=int, default=50,
                   help="Goal acceleration written once at start (teleop uses 5-100).")
    p.add_argument("--auto", action="store_true",
                   help="Run continuously — predict + send each tick, no SPACE. Ctrl-C to stop.")
    p.add_argument("--max-ticks", type=int, default=0,
                   help="In --auto mode, run each checkpoint for N ticks. 0 = unlimited "
                        "(only the first checkpoint ever runs).")
    p.add_argument("--release-on-exit", action="store_true",
                   help="Disable torque on exit (arm goes limp). Default keeps holding.")
    p.add_argument("--dry-run", action="store_true",
                   help="Predict from live frames but never write to the bus (torque untouched).")
    p.add_argument("--camera-order", default="0,1",
                   help="Live image index fed to cam0,cam1. The station's camera order can swap "
                        "between restarts; use 1,0 when it is the reverse of the training data.")
    return p.parse_args()


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
    """Fetch a frame whose global_frame_id differs from last_id."""
    while True:
        frame = await fetch_frame(client, timeout)
        fid = bytes(frame.get_global_frame_id())
        if fid != last_id:
            return frame, fid
        await asyncio.sleep(retry_ms / 1000.0)


# ---------------------------------------------------------------------------
# Commands

def sync_write(bus_serial: str, address: int, motors: list[tuple[int, bytes]]) -> commands.DriverCommand:
    cmd = st3215.Command(
        target_bus_serial=bus_serial,
        sync_write=st3215.ST3215SyncWriteCommand(
            address=address,
            motors=[
                st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=val)
                for mid, val in motors
            ],
        ),
    )
    return commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND,
        body=cmd.encode(),
    )


def goal_command(bus_serial: str, motor_ids: list[int], raw_goals: list[int]) -> commands.DriverCommand:
    return sync_write(bus_serial, RAM_GOAL_POSITION,
                      [(mid, raw.to_bytes(2, "little")) for mid, raw in zip(motor_ids, raw_goals)])


async def engage(client, args, motor_ids: list[int]) -> None:
    """Hold the present position, set the speed envelope, then enable torque."""
    frame = await fetch_frame(client, args.timeout)
    present = [int(j.get_position()) for j in frame.get_joints() or []]
    await send_commands(client, [
        goal_command(args.bus_serial, motor_ids, present),
        sync_write(args.bus_serial, RAM_GOAL_SPEED,
                   [(mid, args.speed.to_bytes(2, "little")) for mid in motor_ids]),
        sync_write(args.bus_serial, RAM_ACC, [(mid, bytes([args.accel])) for mid in motor_ids]),
        sync_write(args.bus_serial, RAM_TORQUE_ENABLE, [(mid, b"\x01") for mid in motor_ids]),
    ])


async def release(client, args, motor_ids: list[int]) -> None:
    await send_commands(client, [
        sync_write(args.bus_serial, RAM_TORQUE_ENABLE, [(mid, b"\x00") for mid in motor_ids]),
    ])


# ---------------------------------------------------------------------------
# Frame → policy obs → goal ticks

def frame_to_obs(frame: normvla.FrameReader) -> tuple[dict, list[tuple[int, int]]]:
    """Unbatched obs in dataset layout; the preprocessor adds batch dim + device."""
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    if len(images) < len(IMAGE_KEYS):
        raise RuntimeError(f"Frame has {len(images)} images but policy expects {len(IMAGE_KEYS)}")

    obs = {
        "observation.state": torch.tensor([j.get_position_norm() for j in joints], dtype=torch.float32),
    }
    for i, key in zip(CAMERA_ORDER, IMAGE_KEYS):
        with Image.open(io.BytesIO(bytes(images[i].get_jpeg()))) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        obs[key] = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0

    ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]
    return obs, ranges


@torch.no_grad()
def predict_goal(frame, policy, preprocessor, postprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Returns (goal_norm, raw_ticks), one entry per joint."""
    obs, ranges = frame_to_obs(frame)
    action = postprocessor(policy.select_action(preprocessor(obs)))
    goal = action.squeeze(0).float().cpu().clamp(0.0, 1.0).numpy()

    raws = []
    for g, (rmin, rmax) in zip(goal, ranges):
        raw = int(round(rmin + float(g) * (rmax - rmin)))
        raws.append(max(rmin, min(rmax, raw)))
    return goal, np.asarray(raws, dtype=np.int64)


def print_preview(motor_ids, frame, goal_norm, raws, header: str) -> int:
    """Print a compact comparison table and return max |delta_ticks| across joints."""
    joints = frame.get_joints() or []
    deltas = [int(r) - int(j.get_position()) for r, j in zip(raws, joints)]
    max_delta = max(abs(d) for d in deltas) if deltas else 0

    print(f"[{header}]  max|Δ|={max_delta}")
    print(f"  {'motor':>5}  {'pos':>6}  {'pred':>6}  {'Δ':>6}  {'pred_norm':>9}")
    for mid, j, gn, raw, d in zip(motor_ids, joints, goal_norm, raws, deltas):
        print(f"  {mid:>5}  {int(j.get_position()):>6}  {int(raw):>6}  {d:>+6}  {gn:>9.4f}")
    return max_delta


async def run_one_tick(args, client, loaded, motor_ids, tick: int, last_id: bytes) -> bytes:
    name, policy, pre, post = loaded
    t_start = time.perf_counter()
    frame, frame_id = await fetch_fresh_frame(client, last_id, args.timeout)
    t_fetched = time.perf_counter()
    goal_norm, raws = predict_goal(frame, policy, pre, post)
    t_predicted = time.perf_counter()
    max_delta = print_preview(motor_ids, frame, goal_norm, raws, header=f"{name} tick {tick}")

    fetch_ms = (t_fetched - t_start) * 1000
    predict_ms = (t_predicted - t_fetched) * 1000
    if args.dry_run:
        print(f"  dry-run, not sent  (fetch {fetch_ms:.0f}ms, predict {predict_ms:.0f}ms)\n")
    elif args.max_delta_ticks and max_delta > args.max_delta_ticks:
        print(f"  skipped: max|Δ|={max_delta} > {args.max_delta_ticks} ticks  "
              f"(fetch {fetch_ms:.0f}ms, predict {predict_ms:.0f}ms)\n")
    else:
        await send_commands(client, [goal_command(args.bus_serial, motor_ids, raws.tolist())])
        print(f"  sent  (fetch {fetch_ms:.0f}ms, predict {predict_ms:.0f}ms, "
              f"total {(time.perf_counter() - t_start) * 1000:.0f}ms)\n")
    return frame_id


async def auto_loop(args, client, policies, motor_ids) -> None:
    print("\nauto: predict + send continuously.  Ctrl-C to stop.\n")
    last_id = b""
    for loaded in policies:
        print(f"=== {loaded[0]} ===\n")
        loaded[1].reset()
        tick = 0
        while not args.max_ticks or tick < args.max_ticks:
            tick += 1
            last_id = await run_one_tick(args, client, loaded, motor_ids, tick, last_id)
        print(f"{loaded[0]}: reached --max-ticks={args.max_ticks}.\n")
    print("all checkpoints done, stopping.")


async def read_key() -> str:
    return await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)


async def interactive_loop(args, client, policies, motor_ids) -> None:
    print("\nSPACE = predict + send   |   n = next checkpoint   |   q / Ctrl-C = quit\n")
    print(f"=== {policies[0][0]} ===\n")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        current = 0
        tick = 0
        last_id = b""
        while True:
            ch = await read_key()
            if ch in ("q", "Q", "\x03", "\x04", ""):
                print("quitting.")
                return
            if ch in ("n", "N"):
                if current + 1 == len(policies):
                    print("already on the last checkpoint.")
                    continue
                current += 1
                tick = 0
                policies[current][1].reset()
                print(f"=== {policies[current][0]} ===\n")
                continue
            if ch != " ":
                continue
            tick += 1
            last_id = await run_one_tick(args, client, policies[current], motor_ids, tick, last_id)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------

def load_policy(checkpoint: Path, device: str, n_action_steps: int) -> tuple:
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.config.device = device
    if n_action_steps:
        policy.config.n_action_steps = n_action_steps
    policy.to(device).eval()
    policy.reset()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return checkpoint.name, policy, pre, post


async def main_async() -> None:
    global CAMERA_ORDER, IMAGE_KEYS
    args = parse_args()
    CAMERA_ORDER = tuple(int(i) for i in args.camera_order.split(","))
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_act")
    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policies = []
    for checkpoint in args.checkpoint:
        print(f"loading {checkpoint} on {device} ...")
        policies.append(load_policy(checkpoint, device, args.n_action_steps))
    image_keys = {tuple(p.config.image_features) for _, p, _, _ in policies}
    if len(image_keys) != 1:
        raise SystemExit(f"checkpoints disagree on camera keys: {image_keys}")
    IMAGE_KEYS = image_keys.pop()
    policy = policies[0][1]

    print(f"server:    {args.server}")
    print(f"bus:       {args.bus_serial}")
    print(f"motors:    {motor_ids}")
    print(f"order:     {' -> '.join(name for name, *_ in policies)}")
    print(f"cameras:   {', '.join(f'images[{i}] -> {k}' for i, k in zip(CAMERA_ORDER, IMAGE_KEYS))}")
    print(f"chunk:     {policy.config.chunk_size} predicted, {policy.config.n_action_steps} executed")
    print(f"safety:    max_delta_ticks={args.max_delta_ticks} speed={args.speed} accel={args.accel}")

    client = await new_station_client(args.server, logger)
    if not args.dry_run:
        await engage(client, args, motor_ids)
        print("torque on, holding present position.")
    try:
        if args.auto:
            await auto_loop(args, client, policies, motor_ids)
        else:
            await interactive_loop(args, client, policies, motor_ids)
    finally:
        if args.dry_run:
            pass
        elif args.release_on_exit:
            await release(client, args, motor_ids)
            print("torque off.")
        else:
            print("torque left on (holding). Pass --release-on-exit to go limp.")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\ninterrupted.")


if __name__ == "__main__":
    main()
