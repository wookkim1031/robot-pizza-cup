"""Record one joint's calibrated arc by hand, without touching the motors.

Move only the chosen joint slowly from one hard stop to the other (a few times
is fine), then press Ctrl+C. The lowest/highest positions seen become that
joint's range override in range_overrides.json, which main.py applies in place
of the station's range_min/range_max. Nothing is written to the motors.

Usage:
    uv run python record_range.py --server localhost --bus 5B61034559 --motor 4
    uv run python record_range.py --server localhost --bus 5B61034559 --motor 4 --clear

Re-run after any station calibration of that bus: a station Save rewrites the
motor's zero offset, so old overrides no longer line up.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add repo root so the generated protobuf and station_py imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from software.station.shared.station_py import new_station_client

from config import (
    DEFAULT_RANGE_OVERRIDES_PATH,
    MotorConfig,
    load_range_overrides,
    save_range_overrides,
)
from main import BusReader, _wait_for_first_frame
from mirror import get_steps_range
from state import find_bus, parse_motor_state

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Sample fast enough that a joint can't move half a turn (2048 steps) between
# reads, or the unwrapping below would mistake it for a zero crossing.
SAMPLE_INTERVAL_S = 0.02

# Refuse to save an arc narrower than this; it means the joint wasn't moved.
MIN_RECORDED_RANGE = 200


class ArcRecorder:
    """Tracks min/max of an encoder that may cross the 4095 -> 0 boundary.

    Positions are unwrapped (each new reading is added as the shortest step
    from the previous one), so an arc spanning the zero crossing still comes
    out as one contiguous range.
    """

    def __init__(self, config: MotorConfig):
        self.config = config
        self.max_steps = config.max_steps
        self.last_raw: int | None = None
        self.unwrapped = 0
        self.lo = 0
        self.hi = 0

    def add(self, raw: int) -> None:
        if self.last_raw is None:
            self.unwrapped = raw
            self.lo = self.hi = raw
        else:
            delta = (raw - self.last_raw) % self.max_steps
            if delta > self.max_steps // 2:
                delta -= self.max_steps
            self.unwrapped += delta
            self.lo = min(self.lo, self.unwrapped)
            self.hi = max(self.hi, self.unwrapped)
        self.last_raw = raw

    @property
    def span(self) -> int:
        return self.hi - self.lo

    def arc(self) -> tuple[int, int]:
        return self.lo % self.max_steps, self.hi % self.max_steps


async def record(args, recorder: ArcRecorder) -> None:
    client = await new_station_client(args.server, logger)
    reader = BusReader(client, label=args.bus)
    task = asyncio.create_task(reader.run())
    await _wait_for_first_frame(reader)

    station_range = None
    print(f"Move M{args.motor} on bus {args.bus} between both hard stops, then press Ctrl+C.")
    while True:
        if task.done():
            task.result()
        bus = find_bus(reader.latest, args.bus)
        motor = next(
            (m for m in (bus.get_motors() or []) if m.get_id() == args.motor), None,
        ) if bus else None
        if motor is None:
            print(f"\rM{args.motor} not found on bus {args.bus}        ", end="", flush=True)
        else:
            state = parse_motor_state(motor)
            station_range = (state.range_min, state.range_max)
            recorder.add(state.present_position)
            lo, hi = recorder.arc()
            print(
                f"\rpos={state.present_position:4d}  recorded={lo:4d}-{hi:4d} "
                f"(span {recorder.span:4d})  station={station_range[0]}-{station_range[1]} "
                f"(span {get_steps_range(*station_range, recorder.config)})   ",
                end="", flush=True,
            )
        await asyncio.sleep(SAMPLE_INTERVAL_S)


def main():
    parser = argparse.ArgumentParser(description="Record one joint's range override")
    parser.add_argument("--server", required=True, help="Station address")
    parser.add_argument("--bus", required=True, help="Bus serial of the arm the joint is on")
    parser.add_argument("--motor", required=True, type=int, help="Motor id, e.g. 4 for M4")
    parser.add_argument(
        "--overrides", default=str(DEFAULT_RANGE_OVERRIDES_PATH),
        help="Overrides JSON file to update",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Remove this joint's override (fall back to the station range) and exit",
    )
    args = parser.parse_args()

    path = Path(args.overrides)
    overrides = load_range_overrides(path)

    if args.clear:
        removed = overrides.get(args.bus, {}).pop(args.motor, None)
        save_range_overrides(path, overrides)
        print(f"Removed override for M{args.motor} on {args.bus}: {removed}")
        return

    recorder = ArcRecorder(MotorConfig())
    try:
        asyncio.run(record(args, recorder))
    except KeyboardInterrupt:
        pass
    print()

    if recorder.span < MIN_RECORDED_RANGE:
        print(f"Recorded span {recorder.span} < {MIN_RECORDED_RANGE} steps; not saving.")
        return

    arc = recorder.arc()
    overrides.setdefault(args.bus, {})[args.motor] = arc
    save_range_overrides(path, overrides)
    print(f"Saved M{args.motor} on {args.bus}: {arc[0]}-{arc[1]} (span {recorder.span}) -> {path}")
    print("Restart teleop to use it.")


if __name__ == "__main__":
    main()
