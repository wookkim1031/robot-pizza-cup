"""Run a LeRobot 0.3.x policy (ACT or SmolVLA) on ElRobot via NormaCore Station.

Adapted from norma-core smolvla_py/scripts/run_policy.py. The Station side
(frame fetch, tick mapping, sync-write, safety abort) is unchanged; only the
model side differs:
  * the checkpoint is a LeRobot `pretrained_model` folder (config.json +
    model.safetensors); normalization stats live inside the model,
  * policy.select_action() returns one action per tick and re-plans a new
    chunk every --n-action-steps ticks,
  * all 8 ElRobot joints are used by default.

Put this file next to run_policy.py (norma-core/software/ai/smolvla_py/scripts/)
or pass --norma-root /path/to/norma-core. Needs an env with lerobot==0.3.3
(+ "[smolvla]" extra for SmolVLA) and whatever station_py imports.

Examples:
  python run_lerobot_policy.py \
      --checkpoint ~/outputs/smolvla_quadrants/checkpoints/020000/pretrained_model \
      --task "put the cheese on the top-left quadrant of the pizza" \
      --bus-serial <FOLLOWER_SERIAL> --auto --max-ticks 300

  python run_lerobot_policy.py \
      --checkpoint ~/outputs/act_top-left/checkpoints/030000/pretrained_model \
      --bus-serial <FOLLOWER_SERIAL> --n-action-steps 10 --auto --max-ticks 300

  python run_lerobot_policy.py --checkpoint ... --task "..." --bus-serial ... --dry-run
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

QUEUE_ID = "inference/normvla"
ST3215_TORQUE_ENABLE_REGISTER = 0x28
ST3215_ACC_REGISTER = 0x29
ST3215_TARGET_POS_REGISTER = 0x2A
ST3215_GOAL_SPEED_REGISTER = 0x2E


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="LeRobot pretrained_model dir (config.json + model.safetensors).")
    p.add_argument("--task", default="",
                   help="Prompt for SmolVLA; must match the training string exactly. Ignored by ACT.")
    p.add_argument("--server", default="localhost")
    p.add_argument("--bus-serial", required=True, help="Follower ST3215 bus id from the Station UI.")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8",
                   help="Motor IDs in the same order as joints in the Frame.")
    p.add_argument("--image-keys", default="observation.images.top,observation.images.front",
                   help="Policy image keys in Frame camera order (cam0=scene/top, cam1=wrist/front).")
    p.add_argument("--n-action-steps", type=int, default=0,
                   help="Ticks executed per predicted chunk before re-planning. 0 = keep checkpoint value.")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--delta-ignore-motors", default="8",
                   help="Motor IDs left out of the --max-delta-ticks check (gripper stalls on the topping when grasping).")
    p.add_argument("--max-delta-ticks", type=int, default=200,
                   help="Skip sending if any joint target is further than this from its position. 0 = off.")
    p.add_argument("--speed", type=int, default=2000,
                   help="Goal speed written once at start (teleop uses 300-3300).")
    p.add_argument("--accel", type=int, default=50,
                   help="Goal acceleration written once at start (teleop uses 5-100).")
    p.add_argument("--release-on-exit", action="store_true",
                   help="Disable torque on exit (arm goes limp). Default keeps holding.")
    p.add_argument("--auto", action="store_true", help="Run continuously instead of SPACE per tick.")
    p.add_argument("--max-ticks", type=int, default=0, help="Stop after N ticks in --auto mode.")
    p.add_argument("--dry-run", action="store_true", help="Predict and print, never send.")
    p.add_argument("--norma-root", type=Path, default=Path(__file__).resolve().parents[4],
                   help="norma-core repo root (default assumes this file sits in smolvla_py/scripts).")
    return p.parse_args()


def import_station(root: Path):
    sys.path.insert(0, str(root / "software" / "station" / "shared"))
    sys.path.insert(0, str(root))
    from station_py import new_station_client, send_commands
    from target.gen_python.protobuf.drivers.inferences import normvla
    from target.gen_python.protobuf.drivers.st3215 import st3215
    from target.gen_python.protobuf.station import commands, drivers
    return new_station_client, send_commands, normvla, st3215, commands, drivers


def load_policy(ckpt: Path, n_action_steps: int, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt)
    if n_action_steps:
        if n_action_steps > policy.config.chunk_size:
            raise SystemExit(f"--n-action-steps {n_action_steps} > chunk_size {policy.config.chunk_size}")
        policy.config.n_action_steps = n_action_steps
    policy.to(device).eval()
    policy.reset()
    return policy


def frame_to_batch(frame, image_keys, task: str, policy_type: str, device):
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    if len(images) < len(image_keys):
        raise RuntimeError(f"Frame has {len(images)} images, policy expects {len(image_keys)}")

    batch = {
        "observation.state": torch.tensor(
            [j.get_position_norm() for j in joints], dtype=torch.float32, device=device
        ).unsqueeze(0)
    }
    for i, key in enumerate(image_keys):
        with Image.open(io.BytesIO(bytes(images[i].get_jpeg()))) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        batch[key] = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    if policy_type == "smolvla":
        batch["task"] = [task]

    ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]
    return batch, ranges


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_lerobot_policy")
    new_station_client, send_commands, normvla, st3215, commands, drivers = import_station(args.norma_root)

    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    image_keys = tuple(args.image_keys.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.checkpoint} on {device} ...")
    policy = load_policy(args.checkpoint, args.n_action_steps, device)
    ptype = policy.config.type
    state_dim = policy.config.input_features["observation.state"].shape[0]
    if state_dim != len(motor_ids):
        raise SystemExit(f"Policy state has {state_dim} joints but --motor-ids lists {len(motor_ids)}")
    missing = [k for k in image_keys if k not in policy.config.input_features]
    if missing:
        raise SystemExit(f"Image keys {missing} not in policy inputs: {list(policy.config.input_features)}")
    if ptype == "smolvla" and not args.task:
        raise SystemExit("SmolVLA needs --task (the exact training prompt).")

    print(f"policy:    {ptype}  (chunk {policy.config.chunk_size}, "
          f"re-plan every {policy.config.n_action_steps} ticks)")
    print(f"task:      {args.task!r}" + ("  (ignored by ACT)" if ptype != "smolvla" else ""))
    print(f"motors:    {motor_ids}   bus: {args.bus_serial}   server: {args.server}")
    print(f"safety:    max_delta_ticks={args.max_delta_ticks}" + ("   DRY RUN" if args.dry_run else ""))

    async def fetch_fresh(client, last_id: bytes):
        while True:
            qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
            entry = await asyncio.wait_for(qr.data.get(), timeout=args.timeout)
            if entry is None:
                raise RuntimeError(f"{QUEUE_ID} closed without a frame ({qr.err})")
            frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
            fid = bytes(frame.get_global_frame_id())
            if fid != last_id:
                return frame, fid
            await asyncio.sleep(0.02)

    def write_all(address: int, values: list[bytes]):
        motors = [st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=v)
                  for m, v in zip(motor_ids, values)]
        sync = st3215.ST3215SyncWriteCommand(address=address, motors=motors)
        cmd = st3215.Command(target_bus_serial=args.bus_serial, sync_write=sync)
        return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())

    def sync_write(raw_goals: list[int]):
        return write_all(ST3215_TARGET_POS_REGISTER, [r.to_bytes(2, "little") for r in raw_goals])

    async def engage(client):
        """Hold the present position, set the speed envelope, then enable torque."""
        frame, _ = await fetch_fresh(client, b"")
        present = [int(j.get_position()) for j in frame.get_joints() or []]
        n = len(motor_ids)
        await send_commands(client, [
            sync_write(present),
            write_all(ST3215_GOAL_SPEED_REGISTER, [args.speed.to_bytes(2, "little")] * n),
            write_all(ST3215_ACC_REGISTER, [bytes([args.accel])] * n),
            write_all(ST3215_TORQUE_ENABLE_REGISTER, [b"\x01"] * n),
        ])
        print(f"torque on, holding present position {present}")

    async def finish(client):
        if args.dry_run:
            return
        if args.release_on_exit:
            await send_commands(client, [write_all(ST3215_TORQUE_ENABLE_REGISTER, [b"\x00"] * len(motor_ids))])
            print("torque off.")
        else:
            print("torque left on (holding). Pass --release-on-exit to go limp.")

    @torch.no_grad()
    async def tick(client, n: int, last_id: bytes) -> bytes:
        t0 = time.perf_counter()
        frame, fid = await fetch_fresh(client, last_id)
        batch, ranges = frame_to_batch(frame, image_keys, args.task, ptype, device)
        t1 = time.perf_counter()
        goal = policy.select_action(batch)[0].float().cpu().clamp(0.0, 1.0).numpy()
        t2 = time.perf_counter()

        raws = [max(lo, min(hi, int(round(lo + float(g) * (hi - lo))))) for g, (lo, hi) in zip(goal, ranges)]
        joints = frame.get_joints() or []
        deltas = [r - int(j.get_position()) for r, j in zip(raws, joints)]
        ignored = {int(x) for x in args.delta_ignore_motors.split(",") if x}
        max_d = max(abs(d) for m, d in zip(motor_ids, deltas) if m not in ignored)

        print(f"[tick {n}] max|Δ|={max_d}  (fetch {(t1 - t0) * 1e3:.0f}ms, predict {(t2 - t1) * 1e3:.0f}ms)")
        print("  " + "  ".join(f"m{m}:{d:+d}" for m, d in zip(motor_ids, deltas)))
        if args.dry_run:
            print("  dry run, not sent")
        elif args.max_delta_ticks and max_d > args.max_delta_ticks:
            print(f"  aborted: max|Δ| {max_d} > {args.max_delta_ticks}")
        else:
            await send_commands(client, [sync_write(raws)])
        return fid

    client = await new_station_client(args.server, logger)
    if not args.dry_run:
        await engage(client)
    n, last_id = 0, b""
    if args.auto:
        print("\nauto mode, Ctrl-C to stop\n")
        try:
            while not (args.max_ticks and n >= args.max_ticks):
                n += 1
                last_id = await tick(client, n, last_id)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\ninterrupted")
        finally:
            await finish(client)
        return

    print("\nSPACE = one tick | r = reset policy | q = quit\n")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read, 1)
            if ch in ("q", "Q", "\x03", "\x04", ""):
                return
            if ch in ("r", "R"):
                policy.reset()
                print("policy reset (action queue cleared)")
            elif ch == " ":
                n += 1
                last_id = await tick(client, n, last_id)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        await finish(client)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
