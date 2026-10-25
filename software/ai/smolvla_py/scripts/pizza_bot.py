"""Pizza bot: our own sequencer on top of the final-dataset SmolVLA checkpoints.

The policy produces every joint command. This script only picks the task text for
each step (allowed by the hackathon rules: stage sequencing + prompt choice + done flag)
and decides when a step is done: the arm has left the home pose and come back.

The task texts are exactly the strings the final datasets were trained on
(cheese-r1r10, pepperoni-r1r10, pepperoni-on-cheese-r1r5); see TASKS below.

Recipes (in the order the demos were recorded):
    cheese      Ch1 cheese: 8 portions, TL TR BR BL twice
    pepperoni   Ch1 pepperoni: 6 slices, TL TR BR BL CENTER GAP
    pizza       Ch2: the cheese recipe, then the pepperoni recipe (14 steps)
    pepperoni-cheese   the pepperoni recipe, then the cheese recipe
    one         a single step: --ingredient and --target

After every pepperoni step on TL/TR/BR/BL/CENTER the change checker compares the scene
frames from before and after the step (arm at home in both) and repeats the step, up to
--retries times, if no slice landed in the target sector. Cheese and GAP steps are not
checked (the checker cannot see cheese reliably; GAP has no fixed sector). --no-check turns it off.

Time: --time-cheese / --time-pepperoni cap each ingredient's share of a run (a step or retry that
would not fit is skipped); --transit-speed plays the moves near home and the way back faster.

Run (station up, teleop stopped with SIGKILL so the follower holds, arm at home):
    uv run python scripts/pizza_bot.py pizza --confirm
    uv run python scripts/pizza_bot.py one --ingredient pepperoni --target TL
    uv run python scripts/pizza_bot.py pizza --dry-run          # print the plan only
    uv run python scripts/pizza_bot.py pizza --checkpoint-cheese <dir>   # specialist for the cheese steps
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_policy as rp  # noqa: E402  (low-level helpers only: frames, prediction, sending)
from split_grasp_place import GRASP_TASKS, PLACE_TASKS  # noqa: E402

from smolvla import SmolVLAPolicy  # noqa: E402
from smolvla.change_checker import ChangeChecker  # noqa: E402
from smolvla.normalize import unnormalize_action  # noqa: E402
from smolvla.stats import load_stats  # noqa: E402

DEFAULT_CHECKPOINT = Path("checkpoints/smolvla-grasp4-2026-10-25-step006000")  # split cheese + pepperoni
DEFAULT_CHECK_MODEL = Path("checkpoints/change-checker-2026-10-25/change.pt")
CHECKED_TOPPINGS = {"pepperoni"}  # cheese: 51% detected on held-out rounds, unusable
FOLLOWER_BUS = "5B3D050088"
DATASETS = Path.home() / "datasets"

_WHERE = {"TL": "on the upper-left quadrant of the pizza", "TR": "on the upper-right quadrant of the pizza",
          "BR": "on the lower-right quadrant of the pizza", "BL": "on the lower-left quadrant of the pizza",
          "CENTER": "in the center of the pizza", "GAP": "on the largest empty area of the pizza"}
_ACTION = {"cheese": "Spread one portion of cheese", "pepperoni": "Place one pepperoni slice"}
TASKS = {(ing, t): f"{_ACTION[ing]} {w} and return to the start position."
         for ing in _ACTION for t, w in _WHERE.items()}

RECIPES = {
    "cheese": [("cheese", t) for t in ["TL", "TR", "BR", "BL"] * 2],
    "pepperoni": [("pepperoni", t) for t in ["TL", "TR", "BR", "BL", "CENTER", "GAP"]],
}
RECIPES["pizza"] = RECIPES["cheese"] + RECIPES["pepperoni"]
RECIPES["pepperoni-cheese"] = RECIPES["pepperoni"] + RECIPES["cheese"]
# --split-pepperoni / --split-cheese: full task -> (shared grasp prompt, place prompt for its target)
SPLIT = {TASKS[(ing, t)]: (ing, GRASP_TASKS[ing], PLACE_TASKS[(ing, t)]) for ing in _ACTION for t in _WHERE}

HOME_JOINTS = slice(0, 6)  # arm joints that define "home"; wrist roll and gripper vary
HOME_TOL = 0.08            # max |delta| (norm) to count as back home
LEAVE_TOL = 0.30           # the arm has left home once any joint is this far away (0.12 let idle drift count)


@dataclass
class Step:
    ingredient: str
    target: str

    @property
    def task(self) -> str:
        return TASKS[(self.ingredient, self.target)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recipe", choices=[*RECIPES, "one"])
    p.add_argument("--ingredient", choices=list(_ACTION), help="with 'one'")
    p.add_argument("--target", choices=list(_WHERE), help="with 'one'")
    p.add_argument("--confirm", action="store_true", help="wait for ENTER before every step")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit (no robot)")
    p.add_argument("--retries", type=int, default=2, help="extra attempts when the checker sees no slice")
    add_run_args(p)
    return p.parse_args()


def add_run_args(p: argparse.ArgumentParser) -> None:
    """Policy, step-execution, checker and station options (shared with pizza_agent.py)."""
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="joint checkpoint, used for every step")
    p.add_argument("--checkpoint-cheese", type=Path, help="optional specialist for cheese steps")
    p.add_argument("--checkpoint-pepperoni", type=Path, help="optional specialist for pepperoni steps")
    p.add_argument("--step-timeout", type=float, default=120.0, help="seconds before a step is cut off")
    p.add_argument("--exec-steps", type=int, default=5,
                   help="actions executed from each predicted 50-step chunk (at --hz) before re-planning; "
                        "1 = re-plan every frame (sluggish: step 0 of a plan is always close to 'stay')")
    p.add_argument("--hz", type=float, default=10.0, help="action rate; the demos were recorded at 10 fps")
    p.add_argument("--settle-ticks", type=int, default=8, help="frames back at home that end a step")
    p.add_argument("--max-delta-ticks", type=int, default=1000,
                   help="skip a command more than this many servo ticks from the current pose (0 = off)")
    p.add_argument("--grip-commit", type=int, default=10,
                   help="when a plan closes the open gripper, play it to the close plus this many steps (0 = off)")
    p.add_argument("--grip-close", type=float, default=0.3, help="planned gripper (norm) below this counts as closing")
    p.add_argument("--grip-open", type=float, default=0.6, help="planned gripper (norm) above this counts as opening")
    p.add_argument("--kick-tol", type=float, default=0.04,
                   help="at home, skip a plan's idle prefix and play from where it moves this far (norm); 0 = off")
    p.add_argument("--return-kick", type=int, default=20,
                   help="after the release: skip the plan's still prefix (the demos pause ~1 s over the pizza) and "
                        "play this many steps from where it moves; 0 = off")
    p.add_argument("--hold-kick", type=int, default=20,
                   help="gripper closed and no release planned: skip the plan's still prefix and play this many "
                        "steps from where it moves (the grasp demos stop 1.5-3 s after closing); 0 = off")
    p.add_argument("--return-tol", type=float, default=0.20,
                   help="after a grab and release, the step ends once the arm is this close to home (norm)")
    p.add_argument("--transit-speed", type=float, default=1.5,
                   help="play the policy's actions this much faster while the planned pose is within --fast-zone "
                        "of home (leaving and approaching home) and after the release (way back); 1 = off")
    p.add_argument("--fast-zone", type=float, default=0.30, help="max |joint - home| (norm) that counts as near home")
    p.add_argument("--time-cheese", type=float, default=0.0,
                   help="seconds for all cheese steps of a run, retries included; a step (or retry) that would not fit "
                        "in what is left, judged by the median attempt time so far, is skipped; 0 = no limit")
    p.add_argument("--time-pepperoni", type=float, default=0.0, help="same for the pepperoni steps")
    p.add_argument("--glide-seconds", type=float, default=1.0,
                   help="after such an early end, glide linearly to home over this many seconds")
    p.add_argument("--split-pepperoni", action=argparse.BooleanOptionalAction, default=True,
                   help="run pepperoni steps as a grasp prompt, then a place prompt (needs a checkpoint trained "
                        "on split_grasp_place.py data: grasp3 or later); --no-split-pepperoni sends the full "
                        "one-step task text (joint 20k, grasp2 and older)")
    p.add_argument("--split-cheese", action=argparse.BooleanOptionalAction, default=True,
                   help="same for cheese steps (split cheese is only in grasp4); --no-split-cheese for older "
                        "checkpoints, and for grasp3")
    p.add_argument("--grasp-frames", type=int, default=1,
                   help="with --split-pepperoni: gripper seen closed on this many re-plans (not frames: a grip "
                        "commit plays ~50 steps per re-plan) ends the grasp skill; the grip commit has already "
                        "played the close plus --grip-commit steps by then")
    p.add_argument("--check-model", type=Path, default=DEFAULT_CHECK_MODEL,
                   help="change checker checkpoint (did the slice land in the target sector?)")
    p.add_argument("--no-check", action="store_true", help="never check or retry steps")
    p.add_argument("--check-thr", type=float, default=0.65,
                   help="p(pepperoni added) above this counts as placed; a false 'placed' skips a needed retry")
    p.add_argument("--check-settle", type=float, default=0.5, help="seconds to wait at home before the after-frame")
    p.add_argument("--verbose", action="store_true", help="print pose and planned gripper on every plan")
    p.add_argument("--server", default="localhost")
    p.add_argument("--bus-serial", default=FOLLOWER_BUS)
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--camera-order", default="auto", help="'auto' or e.g. 1,0 (live image index for cam0,cam1)")


def check_tasks_against_datasets(steps: list[Step]) -> None:
    """Warn if a step's text is not one the final datasets were trained on."""
    known = set()
    for name in ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24", "pepperoni-on-cheese-r1r5-2026-10-24"]:
        man = DATASETS / f"{name}.manifest.json"
        if man.exists():
            known |= set(json.loads(man.read_text())["tasks"])
    if not known:
        print("note: dataset manifests not found, task texts not cross-checked")
        return
    for s in steps:
        if s.task not in known:
            print(f"WARNING: not a training task text: {s.task!r}")


def load_policy(path: Path, device: torch.device):
    stats = {k: v.to(device) for k, v in load_stats(path / "stats.safetensors").items()}
    policy = SmolVLAPolicy.from_pretrained(path, config_overrides={"load_vlm_weights": False}, strict=False)
    return policy.to(device).eval(), stats


def checked(step: Step, checker: ChangeChecker | None) -> bool:
    return checker is not None and step.ingredient in CHECKED_TOPPINGS and step.target in checker.roi


def scene_rgb(frame) -> np.ndarray:
    jpeg = bytes(frame.get_images()[rp.CAMERA_ORDER[0]].get_jpeg())
    with Image.open(io.BytesIO(jpeg)) as im:
        return np.asarray(im.convert("RGB"))


async def scene_at_home(args, client) -> np.ndarray:
    """Scene frame once the arm is parked and has settled (the checker's before/after frames)."""
    await rp.wait_parked(client, args.timeout)
    await asyncio.sleep(args.check_settle)
    frame, _ = await rp.fetch_fresh_frame(client, b"", args.timeout)
    return scene_rgb(frame)


@torch.no_grad()
def predict_chunk(frame, policy, stats, task: str, device) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Full predicted chunk as raw servo ticks, shape (chunk_size, n_joints), plus per-joint ranges."""
    batch, ranges = rp.frame_to_batch(frame, stats, rp.IMAGE_KEYS, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        chunk = policy.predict_action_chunk(batch)[0].float()
    goal = unnormalize_action(chunk, stats).cpu().clamp(0.0, 1.0).numpy()
    lo = np.array([r[0] for r in ranges]); hi = np.array([r[1] for r in ranges])
    return np.clip(np.round(lo + goal * (hi - lo)), lo, hi).astype(np.int64), ranges


async def glide_home(args, client, motor_ids, frame, home: np.ndarray) -> None:
    """Move linearly in joint space from the frame's pose to home over --glide-seconds."""
    joints = frame.get_joints()
    start = rp.joint_positions(frame)
    lo = np.array([j.get_range_min() for j in joints]); hi = np.array([j.get_range_max() for j in joints])
    n = max(1, int(args.glide_seconds * args.hz))
    for k in range(1, n + 1):
        g = start + (home - start) * k / n
        raw = np.clip(np.round(lo + g * (hi - lo)), lo, hi).astype(np.int64)
        await rp.send_commands(client, [rp.build_sync_write_command(args.bus_serial, motor_ids, raw.tolist())])
        await asyncio.sleep(1.0 / args.hz)


async def run_step(args, client, policy, stats, motor_ids, device, task: str, home: np.ndarray) -> str:
    """Run one step. With --split-<ingredient> such a step is two prompts: grasp, then place."""
    ing, grasp_task, place_task = SPLIT.get(task, (None, None, None))
    if not getattr(args, f"split_{ing}", False):
        return await drive(args, client, policy, stats, motor_ids, device, task, home)
    grasp = await drive(args, client, policy, stats, motor_ids, device, grasp_task, home, until_grasped=True)
    if not grasp.startswith("grasped"):
        return f"grasp {grasp}"
    place = await drive(args, client, policy, stats, motor_ids, device, place_task, home, holding=True)
    return f"{grasp}; place {place}"


async def drive(args, client, policy, stats, motor_ids, device, task: str, home: np.ndarray,
                until_grasped: bool = False, holding: bool = False) -> str:
    """Drive the policy on one task until it has left home and come back, or times out.

    Each cycle predicts a 50-step chunk from the newest frame and plays its first
    --exec-steps actions at --hz, then re-plans from a fresh frame.
    until_grasped: stop once the gripper has been closed for --grasp-frames frames (grasp skill).
    holding: start mid-task with the slice in the gripper (place skill after a grasp).
    """
    t0, plans, settled, left, last_id = time.perf_counter(), 0, 0, holding, b""
    grabbed, released, closed_frames = holding, False, 0
    period = 1.0 / args.hz
    while True:
        frame, last_id = await rp.fetch_fresh_frame(client, last_id, args.timeout)
        dist = float(np.abs(rp.joint_positions(frame)[HOME_JOINTS] - home[HOME_JOINTS]).max())
        left = left or dist > LEAVE_TOL
        settled = settled + args.exec_steps if left and dist < HOME_TOL else 0
        elapsed = time.perf_counter() - t0
        grip = float(rp.joint_positions(frame)[-1])
        grabbed = grabbed or (left and grip < args.grip_close)
        released = released or (grabbed and grip > args.grip_open)
        closed_frames = closed_frames + 1 if left and grip < args.grip_close else 0
        if until_grasped and closed_frames >= args.grasp_frames:
            return f"grasped in {elapsed:.0f}s"
        if settled >= args.settle_ticks:
            return f"done in {elapsed:.0f}s"
        # Near home the policy sees a start frame again and repeats the task, so end the
        # step on the way back once it has grabbed and let go.
        # Glide the rest of the way: otherwise the next step starts off home, counts as already
        # left, and ends as soon as the policy drives back (the TR step "done in 5s").
        if released and dist < args.return_tol:
            await glide_home(args, client, motor_ids, frame, home)
            return f"done in {elapsed:.0f}s (released, back within {args.return_tol}, glided home)"
        if elapsed > args.step_timeout:
            return f"TIMEOUT after {elapsed:.0f}s ({'left' if left else 'never left'} home)"

        raws, ranges = predict_chunk(frame, policy, stats, task, device)
        plans += 1
        now = np.array([int(j.get_position()) for j in frame.get_joints() or []])
        delta = int(np.abs(raws[0] - now).max())
        if args.max_delta_ticks and delta > args.max_delta_ticks:
            print(f"    plan {plans}: skipped, max|delta|={delta} > {args.max_delta_ticks}")
            continue
        # The policy keeps putting the grasp (and the release) a few seconds ahead, so re-planning
        # every --exec-steps never reaches it: once a plan flips the gripper, play it through.
        lo, hi = np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges])
        grip_plan = (raws[:, -1] - lo[-1]) / (hi[-1] - lo[-1])
        grip_now = (now[-1] - lo[-1]) / (hi[-1] - lo[-1])
        n_exec, start = args.exec_steps, 0
        # At home the policy idles for ~4 s (as in the demos) and only starts the reach at
        # the end of each 50-step plan, so short re-plans never move. Skip the idle prefix.
        # Not only within HOME_TOL: a kick that just turns the wrist can park the arm at 0.12,
        # where 5-step re-plans play nothing but idle (joint 20k, TL, 2026-10-25 03:53).
        if args.kick_tol and not left:
            d = np.abs((raws[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - (now[:6] - lo[:6]) / (hi[:6] - lo[:6])).max(axis=1)
            moving = np.flatnonzero(d > args.kick_tol)
            if not moving.size:
                continue
            start, n_exec = max(0, int(moving[0]) - 2), len(raws)
            print(f"    {elapsed:4.0f}s  start kick: motion from +{moving[0]}, playing +{start}..+{len(raws) - 1}")
        closes = np.flatnonzero(grip_plan < args.grip_close)
        opens = np.flatnonzero(grip_plan > args.grip_open)
        if args.grip_commit and grip_now > 0.5 and closes.size:
            n_exec = max(n_exec, int(closes[0]) + args.grip_commit)
            print(f"    {elapsed:4.0f}s  grip commit: closes at +{closes[0]}, playing {min(n_exec, len(raws))} steps")
        elif args.grip_commit and grip_now < 0.5 and opens.size and not until_grasped:
            # Same at the release: the drop is always planned a few seconds out.
            n_exec = max(n_exec, int(opens[0]) + args.grip_commit)
            print(f"    {elapsed:4.0f}s  release commit: opens at +{opens[0]}, playing {min(n_exec, len(raws))} steps")
        elif args.hold_kick and left and grip_now < 0.5 and not until_grasped:
            # Holding after the grasp: each plan starts with the demos' post-close pause, so
            # 5-step re-plans keep the arm parked at the bowl. Skip to where the plan lifts.
            d = np.abs((raws[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - (now[:6] - lo[:6]) / (hi[:6] - lo[:6])).max(axis=1)
            moving = np.flatnonzero(d > args.kick_tol)
            if moving.size and moving[0] >= args.exec_steps:
                start = max(0, int(moving[0]) - 2)
                n_exec = start + args.hold_kick
                print(f"    {elapsed:4.0f}s  hold kick: motion from +{moving[0]}, playing +{start}..+{min(n_exec, len(raws)) - 1}")
        elif args.return_kick and released and grip_now > 0.5:
            # Same after the release: the demos pause ~1 s over the pizza before heading home, so
            # every re-plan starts with that pause and the arm hovered there (cheese TL 18->80 s).
            d = np.abs((raws[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - (now[:6] - lo[:6]) / (hi[:6] - lo[:6])).max(axis=1)
            moving = np.flatnonzero(d > args.kick_tol)
            if moving.size and moving[0] >= args.exec_steps:
                start = max(0, int(moving[0]) - 2)
                n_exec = start + args.return_kick
                print(f"    {elapsed:4.0f}s  return kick: motion from +{moving[0]}, playing +{start}..+{min(n_exec, len(raws)) - 1}")
        # Leaving home, coming back and the way back after the release need no precision:
        # play those actions --transit-speed times faster (the grasp and the drop stay at --hz).
        plan_dist = np.abs((raws[:, :6] - lo[:6]) / (hi[:6] - lo[:6]) - home[:6]).max(axis=1)
        fast = released | (plan_dist < args.fast_zone)
        t_next = time.perf_counter()
        for k in range(start, min(n_exec, len(raws))):
            await rp.send_commands(client, [rp.build_sync_write_command(args.bus_serial, motor_ids, raws[k].tolist())])
            t_next += period / args.transit_speed if fast[k] else period
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
        if args.verbose or plans % 5 == 0:
            pos = (now - lo) / (hi - lo)
            plan = (raws - lo) / (hi - lo)
            print(f"    {elapsed:4.0f}s  dist-from-home {dist:.2f}  pose {np.round(pos[:7], 2)}  "
                  f"grip now {pos[-1]:.2f} plan +{args.exec_steps - 1} {plan[args.exec_steps - 1, -1]:.2f} "
                  f"+49 {plan[-1, -1]:.2f}")


def load_policies(args, ingredients, device) -> tuple[dict[str, Path], dict[Path, tuple]]:
    """Checkpoint path per ingredient, and each distinct checkpoint loaded once."""
    paths = {ing: getattr(args, f"checkpoint_{ing}") or args.checkpoint for ing in _ACTION}
    loaded: dict[Path, tuple] = {}
    for ing in ingredients:
        if paths[ing] not in loaded:
            print(f"loading {paths[ing]} on {device} ...")
            loaded[paths[ing]] = load_policy(paths[ing], device)
        print(f"  {ing}: {paths[ing]}")
    return paths, loaded


class Budget:
    """Per-ingredient time limits (--time-cheese, --time-pepperoni) for a combined recipe."""

    DEFAULT_ATTEMPT = 30.0  # seconds, until an attempt of that ingredient has been timed

    def __init__(self, args):
        self.limit = {ing: getattr(args, f"time_{ing}", 0.0) for ing in _ACTION}
        self.spent = {ing: 0.0 for ing in _ACTION}
        self.times: dict[str, list[float]] = {ing: [] for ing in _ACTION}

    def estimate(self, ing: str) -> float:
        return float(np.median(self.times[ing])) if self.times[ing] else self.DEFAULT_ATTEMPT

    def fits(self, ing: str) -> bool:
        return not self.limit[ing] or self.limit[ing] - self.spent[ing] >= self.estimate(ing)

    def status(self, ing: str) -> str:
        return (f"{ing} budget {self.limit[ing]:.0f}s, {self.limit[ing] - self.spent[ing]:.0f}s left "
                f"< ~{self.estimate(ing):.0f}s per attempt")

    @contextlib.contextmanager
    def attempt(self, ing: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.spent[ing] += dt
            self.times[ing].append(dt)


async def connect_station(args):
    """Station client (camera order set) and the home pose the arm is parked in now."""
    client = await rp.new_station_client(args.server, logging.getLogger("pizza_bot"))
    if args.camera_order == "auto":
        rp.CAMERA_ORDER = rp.detect_camera_order(await rp.fetch_frame(client, args.timeout))
    else:
        rp.CAMERA_ORDER = tuple(int(i) for i in args.camera_order.split(","))
    print(f"cameras: live image {rp.CAMERA_ORDER[0]} -> scene, {rp.CAMERA_ORDER[1]} -> wrist")
    home = rp.joint_positions(await rp.wait_parked(client, args.timeout))
    print(f"home pose: {np.round(home, 2).tolist()}\n")
    return client, home


async def main_async() -> None:
    args = parse_args()
    if args.recipe == "one":
        if not (args.ingredient and args.target):
            raise SystemExit("'one' needs --ingredient and --target")
        steps = [Step(args.ingredient, args.target)]
    else:
        steps = [Step(i, t) for i, t in RECIPES[args.recipe]]

    print(f"recipe {args.recipe}: {len(steps)} steps")
    for n, s in enumerate(steps, 1):
        print(f"  {n:>2}. {s.task}")
    check_tasks_against_datasets(steps)
    if args.dry_run:
        return

    logging.basicConfig(level=logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, loaded = load_policies(args, {s.ingredient for s in steps}, device)
    checker = None
    if not args.no_check and any(s.ingredient in CHECKED_TOPPINGS for s in steps):
        checker = ChangeChecker(args.check_model, device)
        print(f"change checker: {args.check_model} (thr {args.check_thr}, {args.retries} retries) "
              f"checks {[f'{s.ingredient} {s.target}' for s in steps if checked(s, checker)]}")

    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    client, home = await connect_station(args)

    log = []
    budget = Budget(args)
    scene = None  # scene frame at home after the last checked attempt = the next one's before-frame
    t_start = time.perf_counter()
    try:
        for n, s in enumerate(steps, 1):
            if not budget.fits(s.ingredient):
                print(f"[{n}/{len(steps)}] SKIPPED {s.ingredient} {s.target}: {budget.status(s.ingredient)}")
                log.append((s, f"skipped, {budget.status(s.ingredient)}"))
                continue
            if args.confirm:
                await asyncio.get_event_loop().run_in_executor(
                    None, input, f"[{n}/{len(steps)}] ENTER to run {s.ingredient} {s.target} ")
            print(f"[{n}/{len(steps)}] {s.task}")
            policy, stats = loaded[paths[s.ingredient]]
            if not checked(s, checker):
                with budget.attempt(s.ingredient):
                    result = await run_step(args, client, policy, stats, motor_ids, device, s.task, home)
                    await rp.wait_parked(client, args.timeout)
                print(f"    {result}")
                log.append((s, result))
                scene = None
                continue
            for attempt in range(1, args.retries + 2):
                if attempt > 1 and not budget.fits(s.ingredient):
                    print(f"    no retry: {budget.status(s.ingredient)}")
                    attempt -= 1
                    break
                with budget.attempt(s.ingredient):
                    before = scene if scene is not None else await scene_at_home(args, client)
                    result = await run_step(args, client, policy, stats, motor_ids, device, s.task, home)
                    after = scene = await scene_at_home(args, client)
                ok, p, elsewhere = checker.check_step(before, after, s.target, s.ingredient, args.check_thr)
                result += f"; check {'OK' if ok else 'MISSING'} p={p:.2f}"
                if elsewhere:
                    result += f", also in {elsewhere}"
                print(f"    attempt {attempt}: {result}")
                if ok:
                    break
                if attempt <= args.retries:
                    print(f"    no {s.ingredient} in {s.target}, retrying ({attempt}/{args.retries})")
            else:
                result += f" -> gave up after {args.retries + 1} attempts"
            log.append((s, f"[{attempt} attempt{'s' * (attempt > 1)}] {result}"))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print(f"\nsummary ({time.perf_counter() - t_start:.0f}s total; "
              f"{', '.join(f'{i} {budget.spent[i]:.0f}s' for i in budget.spent)}):")
        for s, result in log:
            print(f"  {s.ingredient:9s} {s.target:6s} {result}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
