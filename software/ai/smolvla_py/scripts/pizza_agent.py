"""Pizza agent: Claude runs the recipe closed loop, one trained policy step at a time.

Same split as pizza_bot.py (allowed by the hackathon rules: stage sequencing + prompt choice +
done flag): the SmolVLA policy produces every joint command. Instead of a fixed recipe, Claude
reads the recipe text and picks one step at a time (an ingredient + target from pizza_bot.TASKS,
i.e. exactly the task texts the policy was trained on). After every step it gets the result:
how long it took, the change checker's verdict for pepperoni, and the scene camera image. It then
decides to retry, move on, or finish (the done flag).

Claude cannot send the policy anything else: the step tool's schema only allows the trained
ingredient/target pairs. The script also enforces --max-steps and --time-limit. If the API fails
mid-run (network, rate limit), the run continues with what is left of the --fallback recipe.

Needs internet and ANTHROPIC_API_KEY on the laptop.

Run (station up, teleop stopped with SIGKILL so the follower holds, arm at home):
    uv run python scripts/pizza_agent.py "Make a pizza: spread the cheese, then place 6 pepperoni slices"
    uv run python scripts/pizza_agent.py "4 pepperoni on the right half, then cheese over the whole pizza" \\
        --fallback none
    uv run python scripts/pizza_agent.py "..." --sim      # no robot: simulated steps, real Claude
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pizza_bot as pb  # noqa: E402  (steps, policy execution, checker, station setup)

from smolvla.change_checker import ChangeChecker  # noqa: E402

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """\
You run a pizza-topping robot arm at a robotics competition. A learned policy moves the arm; you \
choose what it does next, one step at a time, with the run_step tool. One step = one cheese portion \
or one pepperoni slice, taken from its bowl and put on the pizza; the arm then returns home.

Targets: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left quarter), CENTER, and GAP \
(the largest empty area). Quarters are named from the robot operator's side. Every image you get is \
already rotated to match, so the top-left of the image is TL.

After each step you get the time it took and the scene camera image. For pepperoni on TL/TR/BR/BL/ \
CENTER you also get a change checker verdict. It finds 99% of placed slices, but it sometimes reports \
a slice that is not there. Cheese and GAP are never checked: judge them from the image. The image is \
small, so when the image and the checker disagree, weigh both and say why you chose.

Only cheese and pepperoni have trained skills (the pepperoni skill grasps one flat salami slice; the \
cheese skill takes one pinch of shredded cheese). The recipe may name a new ingredient that has no \
skill. Never skip it: run it with the skill closest in form, as its own stage with exactly the \
count, region and place in the order the recipe gives it (so one skill may run in two stages):
- cheese for anything shredded, grated, crumbled, chopped small or sprinkled/spread: other cheeses \
(parmesan, cheddar, feta, ...), sauce, pesto, herbs, basil, oregano, spinach, rocket, corn, diced onion.
- pepperoni for anything placed as separate pieces: ham, bacon, sausage, chorizo, chicken, mushroom, \
olive, tomato slice, pepper, jalapeno, pineapple, anchovy, egg.
- not sure: pepperoni.
With no amount given, use the substitute's default amounts. Say in finish which ingredient ran as \
which skill.

Scoring (do what the recipe says; these decide the trade-offs):
- Pepperoni: 1 point per slice on the pizza (two slices dropped together count as one), plus 1 per \
quarter with at least one slice.
- Cheese: points for coverage (full points above 70% of the pizza) plus 1 per quarter at least ~25% covered.
- When the recipe gives counts, regions or an order, they are scored exactly: extra pieces cost \
points, a piece outside the asked region counts half, the stages must come in the recipe's order.
- Default amounts when the recipe gives none: cheese 8 portions (TL, TR, BR, BL, twice); pepperoni \
6 slices (TL, TR, BR, BL, CENTER, GAP). Spread pieces over the allowed quarters, clockwise.
- A step takes about 20-60 s. Stay inside the time limit. Ties go to the faster team, so call \
finish as soon as the recipe is done.

Retry a step only when the piece clearly did not land (no more than twice for the same piece). A \
step that timed out without leaving home can be retried once. Keep each message to one short line \
of reasoning, then call a tool. Call finish when done or when no useful step fits the time left."""

TOOLS = [
    {
        "name": "run_step",
        "description": "Run one trained policy step: place one portion of the ingredient on the target, "
                       "then return home. Returns the outcome, the time used and the scene image afterwards.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ingredient": {"type": "string", "enum": list(pb._ACTION)},
                "target": {"type": "string", "enum": list(pb._WHERE)},
            },
            "required": ["ingredient", "target"],
            "additionalProperties": False,
        },
    },
    {
        "name": "look",
        "description": "Get a fresh scene camera image without moving the arm.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "finish",
        "description": "End the run. Summarise what is on the pizza and anything skipped.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recipe", help="the recipe text, as given at evaluation")
    p.add_argument("--time-limit", type=float, default=480.0, help="seconds for the whole run (Ch1 180, Ch2 480)")
    p.add_argument("--max-steps", type=int, default=24, help="hard cap on policy steps, retries included")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                   help="Claude's effort per decision; low keeps the gaps between steps short")
    p.add_argument("--api-timeout", type=float, default=60.0, help="seconds per Claude request")
    p.add_argument("--fallback", default="pizza", choices=[*pb.RECIPES, "none"],
                   help="fixed pizza_bot recipe to finish with if the API fails mid-run")
    p.add_argument("--no-images", action="store_true", help="send Claude text results only")
    p.add_argument("--sim", action="store_true", help="no robot: simulated step results (tests the agent loop)")
    pb.add_run_args(p)
    return p.parse_args()


def image_block(rgb: np.ndarray) -> dict:
    """Scene frame rotated 180 degrees (image top-left = TL, as the quarters are named), scaled 2x."""
    im = Image.fromarray(np.ascontiguousarray(rgb[::-1, ::-1]))
    im = im.resize((im.width * 2, im.height * 2), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


class Robot:
    """The policy steps on the real station; keeps the latest at-home scene frame for the checker."""

    def __init__(self, args):
        self.args = args

    async def connect(self) -> None:
        args = self.args
        logging.basicConfig(level=logging.WARNING)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths, self.loaded = pb.load_policies(args, list(pb._ACTION), self.device)
        self.checker = None if args.no_check else ChangeChecker(args.check_model, self.device)
        self.motor_ids = [int(x) for x in args.motor_ids.split(",")]
        self.client, self.home = await pb.connect_station(args)
        self.scene = await pb.scene_at_home(args, self.client)

    async def look(self) -> np.ndarray | None:
        self.scene = await pb.scene_at_home(self.args, self.client)
        return self.scene

    async def step(self, s: pb.Step) -> tuple[str, np.ndarray | None]:
        args = self.args
        before = self.scene
        policy, stats = self.loaded[self.paths[s.ingredient]]
        result = await pb.run_step(args, self.client, policy, stats, self.motor_ids, self.device, s.task, self.home)
        self.scene = await pb.scene_at_home(args, self.client)
        if pb.checked(s, self.checker):
            ok, p, elsewhere = self.checker.check_step(before, self.scene, s.target, s.ingredient, args.check_thr)
            result += f"; checker: {'slice ADDED' if ok else 'NO slice'} in {s.target} (p={p:.2f})"
            if elsewhere:
                result += f", slices appeared elsewhere: {elsewhere}"
        else:
            result += "; not checked"
        return result, self.scene


class SimRobot:
    """--sim: no station, steps succeed 80% of the time, no images."""

    def __init__(self, args):
        self.args = args
        self.checker = None if args.no_check else ChangeChecker(args.check_model, "cpu")

    async def connect(self) -> None:
        print("sim: no robot")

    async def look(self) -> np.ndarray | None:
        return None

    async def step(self, s: pb.Step) -> tuple[str, np.ndarray | None]:
        result = f"done in {random.randint(20, 50)}s (simulated)"
        if pb.checked(s, self.checker):
            ok = random.random() < 0.8
            result += f"; checker: {'slice ADDED' if ok else 'NO slice'} in {s.target} (p={0.9 if ok else 0.1:.2f})"
        else:
            result += "; not checked"
        return result, None


async def run_agent(args, robot, log: list) -> str:
    """Claude's loop. Appends (step, result) to log; returns why it stopped."""
    client = anthropic.AsyncAnthropic(timeout=args.api_timeout)
    t0 = time.perf_counter()

    def status() -> str:
        return (f"[{time.perf_counter() - t0:.0f}s of {args.time_limit:.0f}s used, "
                f"{len(log)} of {args.max_steps} steps used]")

    def with_image(text: str, rgb: np.ndarray | None) -> list[dict]:
        blocks = [{"type": "text", "text": text}]
        if rgb is not None and not args.no_images:
            blocks.append(image_block(rgb))
        return blocks

    messages = [{"role": "user", "content": with_image(
        f"Recipe: {args.recipe}\nTime limit {args.time_limit:.0f}s, at most {args.max_steps} steps. "
        f"Current scene attached.", await robot.look())}]
    while True:
        try:
            resp = await client.beta.messages.create(
                model=args.model,
                max_tokens=8000,
                system=SYSTEM,
                tools=TOOLS,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=messages,
                output_config={"effort": args.effort},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as e:
            return f"API unreachable ({e})"
        except anthropic.RateLimitError as e:
            return f"API rate limited ({e.message})"
        except anthropic.APIStatusError as e:
            return f"API error {e.status_code} ({e.message})"
        if resp.stop_reason == "refusal":
            return "Claude declined"
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                print(f"claude: {b.text.strip()}")
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            return "finished (no tool call)"

        results = []
        for u in uses:
            if u.name == "finish":
                print(f"claude finished: {u.input['summary']}")
                return "finished"
            if u.name == "look":
                content = with_image(f"Current scene. {status()}", await robot.look())
            elif len(log) >= args.max_steps or time.perf_counter() - t0 > args.time_limit:
                content = [{"type": "text", "text": f"Not run: step or time limit reached {status()}. Call finish."}]
            else:
                s = pb.Step(u.input["ingredient"], u.input["target"])
                print(f"[{len(log) + 1}] {s.ingredient} {s.target}: {s.task}")
                result, rgb = await robot.step(s)
                print(f"    {result}")
                log.append((s, result))
                content = with_image(f"{s.ingredient} {s.target}: {result} {status()}", rgb)
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": content})
        messages.append({"role": "user", "content": results})


async def main_async() -> None:
    args = parse_args()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit("set ANTHROPIC_API_KEY (or run the fixed recipes with pizza_bot.py)")
    robot = SimRobot(args) if args.sim else Robot(args)
    await robot.connect()

    log: list = []
    t_start = time.perf_counter()
    try:
        why = await run_agent(args, robot, log)
        print(f"agent stopped: {why}")
        if why.startswith("API") and args.fallback != "none":
            # Finish with the fixed recipe minus what already ran (no retries).
            todo = [pb.Step(i, t) for i, t in pb.RECIPES[args.fallback]]
            done = Counter((s.ingredient, s.target) for s, _ in log)
            print(f"falling back to pizza_bot recipe {args.fallback!r}")
            for s in todo:
                if done[(s.ingredient, s.target)]:
                    done[(s.ingredient, s.target)] -= 1
                    continue
                if time.perf_counter() - t_start > args.time_limit:
                    print("time limit reached")
                    break
                print(f"[{len(log) + 1}] {s.ingredient} {s.target}: {s.task}")
                result, _ = await robot.step(s)
                print(f"    {result}")
                log.append((s, f"{result} (fallback)"))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print(f"\nsummary ({time.perf_counter() - t_start:.0f}s total):")
        for s, result in log:
            print(f"  {s.ingredient:9s} {s.target:6s} {result}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
