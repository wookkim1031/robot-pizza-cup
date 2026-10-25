"""Run one hackathon challenge as a fixed step sequence on the learned SmolVLA skills.

The policy produces every joint command; this script only chooses the prompt for each step,
decides when a step is done, and repeats pepperoni steps the change checker says missed.

    ch1-cheese      8 cheese portions, TL TR BR BL twice                 budget 170 s
    ch1-pepperoni   6 slices: TL TR BR BL CENTER GAP                      budget 170 s
    ch2             ch1-cheese, then ch1-pepperoni                        cheese 200 s + pepperoni 270 s
    ch2a            ch1-pepperoni, then ch1-cheese                        pepperoni 270 s + cheese 200 s
    ch3a            as ch2, checker off (the pizza moves)                 cheese 200 s + pepperoni 270 s
    ch3b "<text>"   steps parsed from the recipe text, checker off        470 s split by step count
    one <ingredient> <target>     a single step, for testing

Before a run: station up, arm parked at home (gripper open). Teleop is stopped with SIGKILL
automatically so the follower keeps holding.

    uv run python scripts/run_challenge.py ch2
    uv run python scripts/run_challenge.py ch3b "4 pepperoni on the right half, then cheese over the whole pizza"
    uv run python scripts/run_challenge.py ch1-pepperoni --dry-run           # plan only, no robot
    uv run python scripts/run_challenge.py one pepperoni TL --confirm
    uv run python scripts/run_challenge.py ch2 --time-limit 400 --drive transit_speed=1.0 --drive exec_steps=3
    uv run python scripts/run_challenge.py ch2 --box-guard auto              # stop + retrace if a container moves
    uv run python scripts/run_challenge.py ch2 --failure-checker v2          # checker boxes follow the located pizza
    uv run python scripts/run_challenge.py ch2 --failure-checker v2 --dry-run   # + locate the pizza on the live
                                                                                 #   scene image, arm not moved
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pizza.recipe import CHALLENGES, Budget, RunLog, run_steps  # noqa: E402
from pizza.robot import DEFAULT_CHECKPOINT, PizzaRobot, RobotConfig, SimRobot  # noqa: E402
from pizza.skills import DriveConfig  # noqa: E402
from pizza.station import preflight  # noqa: E402
from pizza.tasks import INGREDIENTS, TARGETS, Step  # noqa: E402


def add_robot_args(p: argparse.ArgumentParser) -> None:
    """Checkpoint, checker, drive and station options (shared with run_agent.py)."""
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--checkpoint-cheese", type=Path, help="specialist for cheese steps")
    p.add_argument("--checkpoint-pepperoni", type=Path, help="specialist for pepperoni steps")
    p.add_argument("--check", action=argparse.BooleanOptionalAction, default=None,
                   help="change checker: after each pepperoni step, compare scene images to see whether a slice "
                        "landed in the target quadrant (and retry if not). Only valid with the pizza in its usual "
                        "spot. Default: on for ch1/ch2/one, off for ch3")
    p.add_argument("--check-thr", type=float, default=0.65)
    p.add_argument("--failure-checker", choices=["v1", "v2"], default="v1",
                   help="v1: fixed sector boxes. v2: the pizza is located once at the start (arm at home) and the "
                        "sector boxes follow it; falls back to the v1 boxes if it is not found or too far off")
    p.add_argument("--checker-max-offset", type=float, default=15.0, metavar="PX",
                   help="v2: fall back to the v1 boxes when the located pizza is further than this from its usual spot")
    p.add_argument("--checker-log", type=Path, metavar="DIR",
                   help="write checker events and overlay images here (the web UI sets this)")
    p.add_argument("--drive", action="append", default=[], metavar="KEY=VALUE",
                   help=f"step execution setting, repeatable; keys: {', '.join(f.name for f in fields(DriveConfig))}")
    p.add_argument("--box-guard", metavar="auto|PATH",
                   help="watch the pepperoni tray and cheese box on the scene camera; if the policy drags or lifts "
                        "one, open the gripper and retrace the arm's path home (the step reports 'box moved'). "
                        "auto = find them at each step start (arm at home); PATH = boxes.json with fixed "
                        "rectangles. Same as --drive box_guard=...")
    p.add_argument("--camera-order", default="auto", help="'auto' or e.g. 1,0 (live image index for scene,wrist)")
    p.add_argument("--server", default="localhost")
    p.add_argument("--sim", action="store_true", help="no robot: simulated steps")


def robot_from_args(args, check_default: bool) -> PizzaRobot | SimRobot:
    cfg = RobotConfig(checkpoint=args.checkpoint, checkpoint_cheese=args.checkpoint_cheese,
                      checkpoint_pepperoni=args.checkpoint_pepperoni,
                      check=check_default if args.check is None else args.check, check_thr=args.check_thr,
                      failure_checker=args.failure_checker, checker_log=args.checker_log,
                      checker_max_offset=args.checker_max_offset,
                      drive=DriveConfig.with_overrides(args.drive + [f"box_guard={args.box_guard}"] * bool(args.box_guard)),
                      server=args.server,
                      camera_order=args.camera_order)
    return SimRobot(cfg) if args.sim else PizzaRobot(cfg)


def scale_budget(limits: dict[str, float], steps: list[Step], total: float) -> dict[str, float]:
    """Split total over the run's ingredients: in the preset's proportions, else by step count."""
    if not total:
        return {}
    ings = {s.ingredient for s in steps}
    weights = {i: limits[i] for i in ings if limits.get(i)} if ings <= set(limits) else \
        {i: sum(s.ingredient == i for s in steps) for i in ings}
    return {i: round(total * w / sum(weights.values())) for i, w in weights.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("challenge", choices=[*CHALLENGES, "one"])
    p.add_argument("text", nargs="*", help="ch3b: the recipe text; one: <ingredient> <target>")
    p.add_argument("--retries", type=int, default=2, help="extra attempts when the checker sees no slice")
    p.add_argument("--time-limit", type=float,
                   help="seconds for the whole run, split over the ingredients like the preset's budgets "
                        "(default: ch1 170, ch2/ch3 470; 0 = no limit)")
    p.add_argument("--time", action="append", default=[], metavar="INGREDIENT=SECONDS",
                   help="override one ingredient's time budget (0 = no limit)")
    p.add_argument("--confirm", action="store_true", help="wait for ENTER before every step")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    add_robot_args(p)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


async def main_async() -> None:
    args = parse_args()
    if args.challenge == "one":
        if len(args.text) != 2 or args.text[0] not in INGREDIENTS or args.text[1].upper() not in TARGETS:
            raise SystemExit(f"one <{'|'.join(INGREDIENTS)}> <{'|'.join(TARGETS)}>")
        title, steps, limits, check = "single step", [Step(args.text[0], args.text[1].upper())], {}, True
    else:
        ch = CHALLENGES[args.challenge]
        title, check = ch.title, ch.check
        steps, limits = ch.plan(" ".join(args.text))
    if args.time_limit is not None:
        limits = scale_budget(limits, steps, args.time_limit)
    for pair in args.time:
        ing, _, sec = pair.partition("=")
        limits[ing] = float(sec)

    robot = robot_from_args(args, check)
    print(f"{title}: {len(steps)} steps, budget {limits or 'none'}, "
          f"checker {robot.cfg.failure_checker if robot.cfg.check else 'off'}")
    for n, s in enumerate(steps, 1):
        print(f"  {n:>2}. {s}")
    if args.dry_run:
        if robot.cfg.check and robot.cfg.failure_checker == "v2" and not args.sim:
            preflight(args.server, stop_teleop=None)
            await robot.localize()
        return

    logging.basicConfig(level=logging.WARNING)
    if not args.sim:
        preflight(args.server)
    await robot.connect({s.ingredient for s in steps})
    budget, log = Budget(limits), RunLog()
    try:
        await run_steps(robot, steps, budget, args.retries, log, confirm=args.confirm)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted (Stop / Ctrl-C)")
    finally:
        log.print(budget)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass  # Stop / Ctrl-C: the summary is already printed


if __name__ == "__main__":
    main()
