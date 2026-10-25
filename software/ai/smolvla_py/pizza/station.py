"""Station connection for the follower arm: frames in, sync-write joint targets out.

Thin wrapper over the queue/command helpers in scripts/run_policy.py, so that frame_to_batch
(which reads run_policy.CAMERA_ORDER) sees the camera order picked here.
"""

from __future__ import annotations

import asyncio
import io
import logging
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_policy as rp  # noqa: E402

FOLLOWER_BUS = "5B3D050088"
# Start pose of the runs on 2026-10-25 (position_norm M1..M8); matches the training episode starts.
HOME_POSE = np.array([0.48, 0.01, 0.47, 0.96, 0.99, 0.45, 0.28, 0.93])
HOME_GLIDE_TOL = 0.10  # connect() glides to HOME_POSE first when an arm joint is further away than this (norm)
TELEOP_PATTERN = "main.py --leader-server"


def preflight(server: str = "localhost", port: int = 8888, stop_teleop: bool | None = True) -> None:
    """Station must be up; teleop must be gone (SIGKILL, so the follower keeps torque and holds).
    stop_teleop=None: only check the station (read-only use, teleop may keep running)."""
    try:
        socket.create_connection((server, port), timeout=2).close()
    except OSError:
        raise SystemExit(f"station is not reachable on {server}:{port}; start it first "
                         "(cd ~/Downloads && ./station --web --tcp 127.0.0.1:8888)")
    if stop_teleop is None:
        return
    if subprocess.run(["pgrep", "-f", TELEOP_PATTERN], capture_output=True).returncode == 0:
        if not stop_teleop:
            raise SystemExit("teleop is running; stop it with SIGKILL (pkill -KILL -f 'main.py --leader-server')")
        print("stopping teleop with SIGKILL (the follower keeps holding)")
        subprocess.run(["pkill", "-KILL", "-f", TELEOP_PATTERN])


def to_rgb(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.asarray(im.convert("RGB"))


class Station:
    def __init__(self, server: str = "localhost", bus_serial: str = FOLLOWER_BUS,
                 motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8), timeout: float = 5.0,
                 camera_order: str = "auto"):
        self.server, self.bus_serial, self.motor_ids = server, bus_serial, list(motor_ids)
        self.timeout, self.camera_order = timeout, camera_order
        self.client = None
        self.home: np.ndarray | None = None
        self.box_guard = None                 # pizza.box_guard.BoxGuard while drive(box_guard=...) watches
        self.guard_frame_id = b""             # last frame the guard has seen
        self.trail: list[np.ndarray] = []     # commands sent since the last skill start at home (for retrace)

    async def connect(self) -> None:
        """Connect, pick the camera order, and take the current (parked) pose as home."""
        self.client = await rp.new_station_client(self.server, logging.getLogger("pizza"))
        if self.camera_order == "auto":
            rp.CAMERA_ORDER = rp.detect_camera_order(await rp.fetch_frame(self.client, self.timeout))
        else:
            rp.CAMERA_ORDER = tuple(int(i) for i in self.camera_order.split(","))
        print(f"cameras: live image {rp.CAMERA_ORDER[0]} -> scene, {rp.CAMERA_ORDER[1]} -> wrist")
        frame = await self.parked()
        off = float(np.abs(rp.joint_positions(frame) - HOME_POSE)[:6].max())
        if off > HOME_GLIDE_TOL:
            # A run started with the arm at the bowl took that as home, so the real home looked 0.50
            # away and every step after the first sat there until the timeout (grasp5beta, 09:55).
            print(f"arm is {off:.2f} from the home pose: gliding home first")
            target = HOME_POSE.copy()
            target[-1] = rp.joint_positions(frame)[-1]
            await self.glide(frame, target, 6.0, 10.0)
            await self.glide(await self.parked(), HOME_POSE, 1.0, 10.0)
        self.home = rp.joint_positions(await self.parked())
        print(f"home pose: {np.round(self.home, 2).tolist()}\n")

    async def connect_readonly(self) -> None:
        """Connect and pick the camera order only: no glide home, nothing is sent to the arm."""
        self.client = await rp.new_station_client(self.server, logging.getLogger("pizza"))
        if self.camera_order == "auto":
            rp.CAMERA_ORDER = rp.detect_camera_order(await rp.fetch_frame(self.client, self.timeout))
        else:
            rp.CAMERA_ORDER = tuple(int(i) for i in self.camera_order.split(","))

    async def fresh_frame(self, last_id: bytes = b""):
        return await rp.fetch_fresh_frame(self.client, last_id, self.timeout)

    async def latest_frame(self):
        """The newest frame without waiting for a new one."""
        return await rp.fetch_frame(self.client, self.timeout)

    async def send(self, raw_ticks) -> None:
        cmd = rp.build_sync_write_command(self.bus_serial, self.motor_ids, [int(x) for x in raw_ticks])
        await rp.send_commands(self.client, [cmd])

    async def parked(self):
        """A frame once the arm has stopped moving (or after 10 s)."""
        return await rp.wait_parked(self.client, self.timeout)

    async def scene(self, settle: float = 0.5) -> np.ndarray:
        """Scene camera RGB once the arm is parked and has settled (the checker's before/after frames)."""
        await self.parked()
        await asyncio.sleep(settle)
        frame, _ = await self.fresh_frame()
        return self.scene_rgb(frame)

    @staticmethod
    def scene_rgb(frame) -> np.ndarray:
        return to_rgb(bytes(frame.get_images()[rp.CAMERA_ORDER[0]].get_jpeg()))

    @staticmethod
    def wrist_rgb(frame) -> np.ndarray:
        return to_rgb(bytes(frame.get_images()[rp.CAMERA_ORDER[1]].get_jpeg()))

    async def glide(self, frame, target: np.ndarray, seconds: float, hz: float) -> None:
        """Move linearly in joint space from the frame's pose to target (norm) over `seconds`."""
        joints = frame.get_joints()
        start = rp.joint_positions(frame)
        lo = np.array([j.get_range_min() for j in joints]); hi = np.array([j.get_range_max() for j in joints])
        n = max(1, int(seconds * hz))
        for k in range(1, n + 1):
            g = start + (target - start) * k / n
            await self.send(np.clip(np.round(lo + g * (hi - lo)), lo, hi).astype(np.int64))
            await asyncio.sleep(1.0 / hz)
