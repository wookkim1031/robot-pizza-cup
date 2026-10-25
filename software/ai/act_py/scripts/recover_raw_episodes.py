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

"""Rebuild dataset-generator parquet episodes straight from the raw station queues.

For when `inference/normvla` stopped being written but `inference-states`,
`st3215/inference` and the usbvideo queues kept recording (e.g. cameras
re-enumerated and the stale camera queues made normvla reject every frame).

Mirrors drivers/inferences/src/formats/normvla.rs: one frame per 100 ms of
inference-states, joints from the single torque-enabled bus sorted by motor
id, the latest JPEG of each camera, frames skipped on stale stamps (>100 ms)
or motor errors. Unlike normvla, a stale camera queue is dropped instead of
rejecting the whole frame.

Episodes come from a text file of `<name> <from_ptr> <to_ptr>` lines
(inference-states pointers, e.g. START/END tags). Output has the same schema
as dataset-generator, so convert_to_lerobot.py reads it unchanged.

    python scripts/recover_raw_episodes.py --episodes eps.txt \\
        --output ~/datasets/cheese-recovered.parquet --task "put the cheese on the pizza"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

from software.station.shared.station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo.usbvideo import RxEnvelopeReader  # noqa: E402
from target.gen_python.protobuf.station.inference import InferenceRxReader  # noqa: E402

INTERVAL_NS = 100_000_000
MAX_STAMP_DIFF_NS = 100_000_000

# ST3215 RAM addresses (drivers/st3215/src/protocol/memory.rs)
TORQUE_ENABLE = 0x28
GOAL_POSITION = 0x2A
PRESENT_POSITION = 0x38
PRESENT_SPEED = 0x3A
STATUS = 0x41
PRESENT_CURRENT = 0x45

JOINT = pa.struct([
    pa.field("range_min", pa.uint32(), False),
    pa.field("range_max", pa.uint32(), False),
    pa.field("position", pa.uint32(), False),
    pa.field("position_norm", pa.float32(), False),
    pa.field("goal", pa.uint32(), False),
    pa.field("goal_norm", pa.float32(), False),
    pa.field("current_ma", pa.uint32(), False),
    pa.field("velocity", pa.uint32(), False),
])
IMAGE = pa.struct([pa.field("jpeg", pa.binary(), False)])
SCHEMA = pa.schema([
    pa.field("episode_start_ns", pa.uint64(), False),
    pa.field("global_frame_id", pa.binary(), False),
    pa.field("timestamp_ns_since_episode_start", pa.uint64(), False),
    pa.field("joints", pa.list_(pa.field("element", JOINT, False)), False),
    pa.field("images", pa.list_(pa.field("element", IMAGE, False)), False),
    pa.field("task", pa.string(), False),
])


def u16(b: bytes, addr: int) -> int:
    return int.from_bytes(b[addr:addr + 2], "little") if len(b) >= addr + 2 else 0


def normal_position(raw: int) -> int:
    if raw & 0x8000:
        return (4096 - (raw & 4095)) & 4095
    return raw & 4095


def normalize(value: int, lo: int, hi: int) -> float:
    """Port of normalize_motor_position (drivers/st3215/src/protocol/units.rs)."""
    if lo == hi:
        return 0.0
    rng = (hi - lo) if hi > lo else (4096 + hi - lo)
    threshold = rng * 0.04
    if hi > lo:
        if value < lo and (lo - value) < threshold:
            return 0.0
        if value > hi and (value - hi) < threshold:
            return 1.0
    elif hi < value < lo:
        to_max, to_min = value - hi, lo - value
        if to_min <= to_max and to_min < threshold:
            return 0.0
        if to_max < to_min and to_max < threshold:
            return 1.0
    pos = (value - lo) if (hi > lo or value >= lo) else (4096 + value - lo)
    return pos / rng


def parse_joints(state: st3215.InferenceStateReader, stamp_ns: int) -> list[dict] | None:
    buses = [b for b in state.get_buses() or []
             if any(bytes(m.get_state())[TORQUE_ENABLE:TORQUE_ENABLE + 1] not in (b"", b"\x00")
                    for m in b.get_motors() or [])]
    if len(buses) != 1:
        return None
    bus = buses[0]
    bus_stamp = bus.get_monotonic_stamp_ns()
    if bus_stamp > stamp_ns or stamp_ns - bus_stamp > MAX_STAMP_DIFF_NS:
        return None
    joints = []
    for m in sorted(bus.get_motors() or [], key=lambda m: m.get_id()):
        s = bytes(m.get_state())
        lo, hi = m.get_range_min(), m.get_range_max()
        if lo == hi or len(s) <= STATUS or s[STATUS] != 0:
            return None
        pos = normal_position(u16(s, PRESENT_POSITION))
        goal = normal_position(u16(s, GOAL_POSITION)) if s[TORQUE_ENABLE] else pos
        joints.append({
            "range_min": lo, "range_max": hi,
            "position": pos, "position_norm": normalize(pos, lo, hi),
            "goal": goal, "goal_norm": normalize(goal, lo, hi),
            "current_ma": u16(s, PRESENT_CURRENT), "velocity": u16(s, PRESENT_SPEED),
        })
    return joints or None


def latest_jpeg(env: RxEnvelopeReader, stamp_ns: int) -> bytes | None:
    pack = env.get_frames()
    if pack is None:
        return None
    data, stamps = pack.get_frames_data(), pack.get_stamps()
    if not data or not stamps:
        return None
    frame_stamp = stamps[-1].get_monotonic_stamp_ns()
    if frame_stamp > stamp_ns or stamp_ns - frame_stamp > MAX_STAMP_DIFF_NS:
        return None
    return bytes(data[-1]) or None


class Station:
    def __init__(self, client):
        self.client = client

    async def read(self, queue: str, ptr: int, limit: int = 1):
        qr = self.client.read_from_offset(queue, ptr.to_bytes(8, "little"), limit, 1, 256)
        while (entry := await qr.data.get()) is not None:
            yield entry
        if qr.err:
            raise RuntimeError(f"read {queue}@{ptr}: {qr.err}")

    async def one(self, queue: str, ptr: int) -> bytes | None:
        async for entry in self.read(queue, ptr):
            return bytes(entry.Data)
        return None


def short_queue(q: str) -> str:
    # "/<station-id>/usbvideo/<id>" -> "usbvideo/<id>"
    return q.split("/", 2)[2] if q.startswith("/") else q


async def recover_episode(st: Station, a: int, b: int, task: str) -> tuple[list[dict], dict]:
    rows, stats = [], {"sampled": 0, "no_joints": 0, "no_images": 0, "camera_count": set()}
    # Pick the 100 ms samples first: issuing other reads while this stream is
    # still open stalls the client.
    sampled, last_ts = [], 0
    async for entry in st.read("inference-states", a, limit=b - a + 1):
        data = bytes(entry.Data)
        ts = InferenceRxReader(data).get_monotonic_stamp_ns()
        if ts - last_ts >= INTERVAL_NS:
            last_ts = ts
            sampled.append((bytes(entry.ID.ID), data))

    t0_mono = t0_local = None
    for frame_id, data in sampled:
        rx = InferenceRxReader(data)
        ts = rx.get_monotonic_stamp_ns()
        stats["sampled"] += 1

        joints, images = None, []
        for e in rx.get_entries():
            kind = e.get_type().name
            ptr = int.from_bytes(bytes(e.get_ptr()), "little")
            if kind == "QDT_ST3215_INFERENCE":
                data = await st.one(short_queue(e.get_queue()), ptr)
                if data:
                    joints = parse_joints(st3215.InferenceStateReader(data), ts)
            elif kind == "QDT_USB_VIDEO_FRAMES":
                data = await st.one(short_queue(e.get_queue()), ptr)
                jpeg = latest_jpeg(RxEnvelopeReader(data), ts) if data else None
                if jpeg:
                    images.append({"jpeg": jpeg})
        if not joints:
            stats["no_joints"] += 1
            continue
        if not images:
            stats["no_images"] += 1
            continue
        stats["camera_count"].add(len(images))
        if t0_mono is None:
            t0_mono, t0_local = ts, rx.get_local_stamp_ns()
        rows.append({
            "episode_start_ns": t0_local,
            "global_frame_id": frame_id,
            "timestamp_ns_since_episode_start": ts - t0_mono,
            "joints": joints,
            "images": images,
            "task": task,
        })
    return rows, stats


async def main_async(args):
    st = Station(await new_station_client(args.server, logging.getLogger("recover")))
    writer = pq.ParquetWriter(args.output, SCHEMA)
    total = 0
    try:
        for line in Path(args.episodes).read_text().splitlines():
            if not line.strip():
                continue
            name, a, b = line.split()
            rows, stats = await recover_episode(st, int(a), int(b), args.task)
            cams = sorted(stats["camera_count"])
            if not rows or cams != [args.cameras]:
                print(f"{name}: SKIPPED ({len(rows)} frames, cameras per frame {cams}, {stats})")
                continue
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            total += len(rows)
            dur = rows[-1]["timestamp_ns_since_episode_start"] / 1e9
            print(f"{name}: {len(rows)} frames, {dur:.1f}s, skipped no_joints={stats['no_joints']} "
                  f"no_images={stats['no_images']}")
    finally:
        writer.close()
    print(f"Wrote {total} frames to {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", required=True, type=Path, help="Lines of '<name> <from_ptr> <to_ptr>'")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--server", default="localhost")
    p.add_argument("--cameras", type=int, default=2, help="Expected images per frame; other episodes are skipped")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
