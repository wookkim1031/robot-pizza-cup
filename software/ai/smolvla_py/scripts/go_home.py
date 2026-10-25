"""Reset only: move the follower slowly to the home pose (pizza/station.py HOME_POSE).

Linear in joint space over --seconds with the gripper held as it is, then the gripper opens
(the policies only start with the gripper open). Not part of a scored run: it is a scripted
motion, for resetting between runs. Teleop is stopped with SIGKILL first so the follower holds.

    uv run python scripts/go_home.py
    uv run python scripts/go_home.py --seconds 10 --keep-gripper
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pizza.station import HOME_POSE, Station, preflight, rp  # noqa: E402


async def main_async(args) -> None:
    station = Station(args.server)
    station.client = await rp.new_station_client(args.server, logging.getLogger("home"))
    frame = await station.parked()
    start = rp.joint_positions(frame)
    print(f"from {np.round(start, 2).tolist()}")
    arm_target = HOME_POSE.copy()
    arm_target[-1] = start[-1]  # gripper stays as it is during the move
    await station.glide(frame, arm_target, args.seconds, 10.0)
    if not args.keep_gripper:
        frame = await station.parked()
        await station.glide(frame, HOME_POSE, 1.0, 10.0)
    end = rp.joint_positions(await station.parked())
    print(f"now  {np.round(end, 2).tolist()}")
    off = np.abs(end - HOME_POSE)[:7].max()
    print(f"home: max joint offset {off:.3f}" + ("  (more than 0.05: check for a blocked joint)" if off > 0.05 else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=6.0, help="duration of the move")
    p.add_argument("--keep-gripper", action="store_true", help="do not open the gripper at the end")
    p.add_argument("--server", default="localhost")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    preflight(args.server)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
