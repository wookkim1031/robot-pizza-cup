"""Run the SmolVLA policy on one prompt until the skill is done: the low-level step executor.

The policy produces every joint command: every target sent here is one the policy predicted. Code
above it only picks the prompt and decides when a skill is over (allowed by the hackathon rules:
sequencing, prompt choice, done flag). The kicks and commits below choose which predicted targets
are played and when (and transit_speed plays some faster); they never make up a target.

Each cycle predicts a 50-step chunk from the newest frame and plays part of it, then re-plans.
Re-planning every few steps works badly with these demos, because every pause in the demos becomes
a pause at the start of each plan: the arm idles at home, at the bowl after closing and over the
pizza after the release. The "kicks" and "commits" below play past those pauses; each was added
for a failure seen on the robot (dates in the comments).

Box guard (--drive box_guard=auto, off by default): at each skill start from home the side containers are
found in the scene image (pizza.box_detector) and watched on every played step (pizza.box_guard). If the
policy grabs or drags one, the skill stops, the gripper opens where the arm is and the commands sent since
the skill start are played backwards (gripper open) to bring the arm home along its own path.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from pizza import tasks
from pizza.box_guard import BoxGuard
from pizza.station import Station, rp
from smolvla import SmolVLAPolicy
from smolvla.normalize import unnormalize_action
from smolvla.stats import load_stats

HOME_JOINTS = slice(0, 6)  # arm joints that define "home"; wrist roll and gripper vary
HOME_TOL = 0.08            # max |delta| (norm) to count as back home
LEAVE_TOL = 0.30           # the arm has left home once any joint is this far away (0.12 let idle drift count)


@dataclass
class DriveConfig:
    """Step execution settings; the defaults are the values tuned on the robot on 2026-10-25."""
    step_timeout: float = 45.0    # seconds before a skill is cut off (was 120; grasps p90 22 s, places p90 13 s)
    hz: float = 10.0              # action rate; the demos were recorded at 10 fps
    exec_steps: int = 5           # actions played from each 50-step plan before re-planning
    settle_ticks: int = 8         # frames back at home that end a step
    max_delta_ticks: int = 1000   # skip a plan whose first command is this far (servo ticks) from now; 0 = off
    grip_commit: int = 10         # >0: play a plan that closes the open gripper to the close + this many steps (off 06:58-07:50: BR retry hovered 68 s, close always at +49)
    release_commit: int = 0       # >0: same for the release; off since forcing it turned early opens at the bowl into drops + re-grabs (06:58)
    close_wait: float = 2.0       # >0: at a planned close, hold the pose (gripper open) up to this long until the arm arrives
    close_tol: float = 0.05       # arm joints (norm) this close to the planned pose count as arrived
    grip_close: float = 0.3       # gripper (norm) below this = closed
    grip_open: float = 0.6        # gripper (norm) above this = open
    kick_tol: float = 0.04        # joint motion (norm) that counts as "the plan moves"
    hold_kick: int = 20           # holding, no release planned: skip the still prefix, play this many steps
    return_kick: int = 20         # after the release: skip the still prefix, play this many steps
    return_tol: float = 0.25      # after grab + release, the step ends this close to home (must stay < 0.30; 0.20 parked at 0.21 for 25 s)
    transit_speed: float = 1.5    # play actions this much faster near home and on the way back; 1 = off
    fast_zone: float = 0.30       # max |joint - home| (norm) that counts as near home
    grasp_frames: int = 1         # grasp skill ends after the gripper is seen closed on this many re-plans
    split_cheese: bool = True     # grasp prompt + place prompt (grasp4 and later); False = one full prompt
    split_pepperoni: bool = True  # same for pepperoni (grasp3 and later)
    box_guard: str = "off"        # "auto": detect the containers at each skill start (arm at home) and stop + retrace if one moves; a boxes.json path = fixed rectangles
    retrace_speed: float = 2.0    # box guard: play the way back this much faster than hz
    verbose: bool = False

    @classmethod
    def with_overrides(cls, pairs: list[str]) -> "DriveConfig":
        """From KEY=VALUE strings (CLI --drive), typed like the defaults."""
        cfg, types = cls(), {f.name: type(f.default) for f in fields(cls)}
        for pair in pairs:
            key, _, value = pair.partition("=")
            key = key.replace("-", "_")
            if key not in types:
                raise SystemExit(f"--drive: unknown setting {key!r}; known: {', '.join(types)}")
            t = types[key]
            setattr(cfg, key, value.lower() in ("1", "true", "yes", "on") if t is bool else t(value))
        return cfg


@dataclass
class SkillResult:
    outcome: str     # "grasped" | "done" | "timeout" | "grasp failed" | "box moved"
    seconds: float
    text: str

    @property
    def ok(self) -> bool:
        return self.outcome in ("grasped", "done")


class BoxMoved(Exception):
    def __init__(self, name: str, elapsed: float):
        super().__init__(name)
        self.name, self.elapsed = name, elapsed


RETRACE_TOL = 8  # ticks: consecutive trail targets closer than this (holds) are played once on the way back


def _start_guard(station: Station, cfg: DriveConfig, frame) -> None:
    """Skill start with the arm at home: (re)find the containers and take their reference crops."""
    top = station.scene_rgb(frame)
    if cfg.box_guard == "auto":
        g = BoxGuard.from_frame(top)
    else:
        g = BoxGuard(cfg.box_guard)
        g.calibrate(top)
    missing = {"pepperoni", "cheese"} - set(g.boxes) if cfg.box_guard == "auto" else set()
    print(f"    box guard: watching {g.boxes}" + (f", NOT found: {sorted(missing)}" if missing else ""))
    station.box_guard, station.guard_frame_id = (g if g.boxes else None), b""


def _check_guard(station: Station, frame, frame_id: bytes, elapsed: float) -> None:
    """Feed a new scene frame to the guard; raise BoxMoved once it has seen a container move."""
    g = station.box_guard
    if g is None or frame_id == station.guard_frame_id:
        return
    station.guard_frame_id = frame_id
    st = g.update(station.scene_rgb(frame))
    if g.moved:
        print(f"    {elapsed:4.0f}s  BOX GUARD: {g.moved} moved {st[g.moved]}")
        raise BoxMoved(g.moved, elapsed)


async def _send(station: Station, raw, elapsed: float) -> None:
    """Send one command; with the box guard on, record it and check the newest frame."""
    await station.send(raw)
    if station.box_guard is not None:
        station.trail.append(np.asarray(raw).copy())
        frame = await station.latest_frame()
        _check_guard(station, frame, bytes(frame.get_global_frame_id()), elapsed)


async def _retrace(station: Station, cfg: DriveConfig) -> int:
    """Open the gripper at the last commanded pose, then play the trail backwards with the gripper open."""
    frame = await station.latest_frame()
    joints = frame.get_joints()
    lo = np.array([j.get_range_min() for j in joints]); hi = np.array([j.get_range_max() for j in joints])
    open_raw = int(round(lo[-1] + station.home[-1] * (hi[-1] - lo[-1])))
    back: list[np.ndarray] = []
    for raw in reversed(station.trail):
        if not back or np.abs(raw[:-1] - back[-1][:-1]).max() > RETRACE_TOL:
            back.append(raw.copy())
    station.trail, station.box_guard = [], None
    period = 1.0 / (cfg.hz * cfg.retrace_speed)
    for k, raw in enumerate(back[:1] * int(0.5 * cfg.hz) + back):
        raw[-1] = open_raw
        await station.send(raw)
        await asyncio.sleep(period if k >= int(0.5 * cfg.hz) else 1.0 / cfg.hz)
    await station.parked()
    return len(back)


class Policy:
    """A SmolVLA checkpoint loaded for inference."""

    def __init__(self, path: Path, device: torch.device):
        self.path, self.device = Path(path), device
        self.stats = {k: v.to(device) for k, v in load_stats(self.path / "stats.safetensors").items()}
        model = SmolVLAPolicy.from_pretrained(self.path, config_overrides={"load_vlm_weights": False}, strict=False)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def predict(self, frame, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted chunk as raw servo ticks (chunk_size, n_joints), plus per-joint range lo, hi."""
        batch, ranges = rp.frame_to_batch(frame, self.stats, rp.IMAGE_KEYS, self.device)
        tokens, mask = self.model.tokenize_task(task, device=self.device)
        batch["observation.language.tokens"] = tokens
        batch["observation.language.attention_mask"] = mask
        bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=bf16):
            chunk = self.model.predict_action_chunk(batch)[0].float()
        goal = unnormalize_action(chunk, self.stats).cpu().clamp(0.0, 1.0).numpy()
        lo = np.array([r[0] for r in ranges]); hi = np.array([r[1] for r in ranges])
        return np.clip(np.round(lo + goal * (hi - lo)), lo, hi).astype(np.int64), lo, hi


def _motion_start(raws, now, lo, hi, tol: float) -> int | None:
    """First plan index where an arm joint is more than tol (norm) from the current pose."""
    d = np.abs((raws[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - (now[:6] - lo[:6]) / (hi[:6] - lo[:6])).max(axis=1)
    moving = np.flatnonzero(d > tol)
    return int(moving[0]) if moving.size else None


async def _arrive_before_close(station: Station, raws, k: int, lo, hi, cfg: DriveConfig,
                               last_id: bytes, elapsed: float) -> bytes:
    """Hold the planned pose at the close step, gripper still open, until the arm gets there.

    Plans are played open loop and the servos lag, so a close played on schedule fired while the
    arm was still on its way down: cheese closed at shoulder 0.6 / wrist 0.5 instead of the demos'
    0.8 / 0.35 and the jaws caught the box rim (grasp5-4, 09:29).
    """
    hold = raws[k].copy()
    hold[-1] = raws[k - 1, -1] if k else hi[-1]
    target = (raws[k, HOME_JOINTS] - lo[HOME_JOINTS]) / (hi[HOME_JOINTS] - lo[HOME_JOINTS])
    t0 = time.perf_counter()
    while True:
        await station.send(hold)
        frame, last_id = await station.fresh_frame(last_id)
        _check_guard(station, frame, last_id, elapsed)
        off = np.abs(rp.joint_positions(frame)[HOME_JOINTS] - target)
        waited = time.perf_counter() - t0
        if off.max() <= cfg.close_tol or waited >= cfg.close_wait:
            print(f"    {elapsed:4.0f}s  close at +{k}: waited {waited:.1f}s, arm off by {off.max():.2f} "
                  f"(joint {int(off.argmax()) + 1}), planned pose {np.round(target, 2)}")
            return last_id
        await asyncio.sleep(1.0 / cfg.hz)


async def drive(station: Station, policy: Policy, task: str, cfg: DriveConfig,
                until_grasped: bool = False, holding: bool = False) -> SkillResult:
    """Drive the policy on one prompt until it has left home and come back, or times out.

    until_grasped: stop once the gripper has been closed for cfg.grasp_frames re-plans (grasp skill).
    holding: start mid-task with the topping in the gripper (place skill after a grasp); keeps the box
    guard and trail of the grasp, since the containers are not in clear view from the bowl.
    """
    if not holding:
        station.trail, station.box_guard = [], None
        if cfg.box_guard != "off":
            _start_guard(station, cfg, (await station.fresh_frame())[0])
    try:
        return await _drive(station, policy, task, cfg, until_grasped, holding)
    except BoxMoved as e:
        n = await _retrace(station, cfg)
        return SkillResult("box moved", e.elapsed, f"BOX MOVED ({e.name}) after {e.elapsed:.0f}s: "
                                                   f"gripper opened, retraced {n} poses home")


async def _drive(station: Station, policy: Policy, task: str, cfg: DriveConfig,
                 until_grasped: bool, holding: bool) -> SkillResult:
    home = station.home
    t0, plans, settled, left, last_id = time.perf_counter(), 0, 0, holding, b""
    grabbed, released, closed_frames = holding, False, 0
    period = 1.0 / cfg.hz
    while True:
        frame, last_id = await station.fresh_frame(last_id)
        elapsed = time.perf_counter() - t0
        _check_guard(station, frame, last_id, elapsed)
        pos = rp.joint_positions(frame)
        dist = float(np.abs(pos[HOME_JOINTS] - home[HOME_JOINTS]).max())
        left = left or dist > LEAVE_TOL
        settled = settled + cfg.exec_steps if left and dist < HOME_TOL else 0
        grip = float(pos[-1])
        grabbed = grabbed or (left and grip < cfg.grip_close)
        released = released or (grabbed and grip > cfg.grip_open)
        closed_frames = closed_frames + 1 if left and grip < cfg.grip_close else 0
        if until_grasped and closed_frames >= cfg.grasp_frames:
            return SkillResult("grasped", elapsed, f"grasped in {elapsed:.0f}s")
        if settled >= cfg.settle_ticks:
            return SkillResult("done", elapsed, f"done in {elapsed:.0f}s")
        # Near home the policy sees a start frame again and repeats the task, so end the step (done
        # flag) on the way back once it has grabbed and let go. The arm stays where the policy put it;
        # return_tol < LEAVE_TOL, so the next step does not count it as having left home already.
        if released and dist < cfg.return_tol:
            return SkillResult("done", elapsed, f"done in {elapsed:.0f}s (released, {dist:.2f} from home)")
        if elapsed > cfg.step_timeout:
            return SkillResult("timeout", elapsed, f"TIMEOUT after {elapsed:.0f}s "
                                                   f"({'left' if left else 'never left'} home)")

        raws, lo, hi = policy.predict(frame, task)
        plans += 1
        now = np.array([int(j.get_position()) for j in frame.get_joints() or []])
        delta = int(np.abs(raws[0] - now).max())
        if cfg.max_delta_ticks and delta > cfg.max_delta_ticks:
            print(f"    plan {plans}: skipped, max|delta|={delta} > {cfg.max_delta_ticks}")
            continue
        grip_plan = (raws[:, -1] - lo[-1]) / (hi[-1] - lo[-1])
        grip_now = (now[-1] - lo[-1]) / (hi[-1] - lo[-1])
        n_exec, start = cfg.exec_steps, 0
        # Start kick: at home the policy idles ~4 s (as in the demos) and only starts the reach at the
        # end of each plan, so short re-plans never move. Not only within HOME_TOL: a kick that just
        # turns the wrist can park the arm at 0.12, where 5-step re-plans play only idle (03:53).
        if cfg.kick_tol and not left:
            m = _motion_start(raws, now, lo, hi, cfg.kick_tol)
            if m is None:
                continue
            start, n_exec = max(0, m - 2), len(raws)
            print(f"    {elapsed:4.0f}s  start kick: motion from +{m}, playing +{start}..+{len(raws) - 1}")
        closes = np.flatnonzero(grip_plan < cfg.grip_close)
        opens = np.flatnonzero(grip_plan > cfg.grip_open)
        if cfg.grip_commit and grip_now > 0.5 and closes.size:
            # The policy keeps putting the grasp a few seconds ahead, so re-planning every few steps
            # never reaches it: once a plan closes the gripper, play it through.
            n_exec = max(n_exec, int(closes[0]) + cfg.grip_commit)
            print(f"    {elapsed:4.0f}s  grip commit: closes at +{closes[0]}, playing {min(n_exec, len(raws))} steps")
        elif cfg.release_commit and grip_now < 0.5 and opens.size and not until_grasped:
            # Same at the release: the drop is always planned a few seconds out.
            n_exec = max(n_exec, int(opens[0]) + cfg.release_commit)
            print(f"    {elapsed:4.0f}s  release commit: opens at +{opens[0]}, playing {min(n_exec, len(raws))} steps")
        elif cfg.hold_kick and left and grip_now < 0.5 and not until_grasped:
            # Hold kick: the grasp demos stay still 1.5-3 s after closing, so 5-step re-plans kept the
            # arm parked at the bowl (03:18, 18->54 s). Skip to where the plan lifts.
            m = _motion_start(raws, now, lo, hi, cfg.kick_tol)
            if m is not None and m >= cfg.exec_steps:
                start = max(0, m - 2)
                n_exec = start + cfg.hold_kick
                print(f"    {elapsed:4.0f}s  hold kick: motion from +{m}, playing +{start}..+{min(n_exec, len(raws)) - 1}")
        elif cfg.return_kick and released and grip_now > 0.5:
            # Return kick: the demos pause ~1 s over the pizza after the release, and the arm hovered
            # there (cheese TL 18->80 s). Skip to where the plan heads home.
            m = _motion_start(raws, now, lo, hi, cfg.kick_tol)
            if m is not None and m >= cfg.exec_steps:
                start = max(0, m - 2)
                n_exec = start + cfg.return_kick
                print(f"    {elapsed:4.0f}s  return kick: motion from +{m}, playing +{start}..+{min(n_exec, len(raws)) - 1}")
        # Leaving home, approaching home and the way back after the release need no precision: play
        # those actions transit_speed times faster (the grasp and the drop stay at hz).
        plan_dist = np.abs((raws[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - home[:6]).max(axis=1)
        fast = released | (plan_dist < cfg.fast_zone)
        t_next = time.perf_counter()
        close_at = int(closes[0]) if cfg.close_wait and grip_now > 0.5 and closes.size else -1
        for k in range(start, min(n_exec, len(raws))):
            if k == close_at:
                last_id = await _arrive_before_close(station, raws, k, lo, hi, cfg, last_id, elapsed)
                t_next = time.perf_counter()
            await _send(station, raws[k], elapsed)
            t_next += period / cfg.transit_speed if fast[k] else period
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
        if cfg.verbose or plans % 5 == 0:
            p_now, plan = (now - lo) / (hi - lo), (raws - lo) / (hi - lo)
            print(f"    {elapsed:4.0f}s  dist-from-home {dist:.2f}  pose {np.round(p_now[:7], 2)}  "
                  f"grip now {p_now[-1]:.2f} plan +{cfg.exec_steps - 1} {plan[cfg.exec_steps - 1, -1]:.2f} "
                  f"+49 {plan[-1, -1]:.2f}")


async def grasp(station: Station, policy: Policy, ingredient: str, cfg: DriveConfig) -> SkillResult:
    """Grasp skill: from home to the topping closed in the gripper (split checkpoints only)."""
    r = await drive(station, policy, tasks.grasp_task(ingredient), cfg, until_grasped=True)
    return r if r.outcome in ("grasped", "box moved") else SkillResult("grasp failed", r.seconds, f"grasp {r.text}")


async def place(station: Station, policy: Policy, step: tasks.Step, cfg: DriveConfig) -> SkillResult:
    """Place skill: from holding the topping to placed on the target and back home."""
    return await drive(station, policy, tasks.place_task(step.ingredient, step.target), cfg, holding=True)


async def add_topping(station: Station, policy: Policy, step: tasks.Step, cfg: DriveConfig) -> SkillResult:
    """One portion/slice from its bowl onto the target, back home: grasp + place, or one full prompt."""
    if not getattr(cfg, f"split_{step.ingredient}"):
        return await drive(station, policy, tasks.full_task(step.ingredient, step.target), cfg)
    g = await grasp(station, policy, step.ingredient, cfg)
    if not g.ok:
        return g
    p = await place(station, policy, step, cfg)
    return SkillResult(p.outcome, g.seconds + p.seconds, f"{g.text}; place {p.text}")
