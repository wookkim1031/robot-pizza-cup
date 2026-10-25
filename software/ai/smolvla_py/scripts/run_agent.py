"""Pizza agent: Claude reads the recipe and drives the learned skills closed loop, one tool call at a time.

The SmolVLA policy produces every joint command. Claude only chooses which trained skill runs next
(sequencing + prompt choice + done flag, as the rules allow) through these tools:

    add_topping(ingredient, target)   one portion/slice from its bowl onto the target, back home
    pick(ingredient)                  grasp only; returns the wrist image so Claude can see the grip
    place(target)                     put the held topping on the target, back home
    look()                            scene + wrist images, arm not moved
    finish(summary)                   done flag

Ingredients and targets are strict enums, so Claude can only send prompts the policy was trained on.
Every result carries the outcome, time, the change checker's verdict (pepperoni, fixed layout only),
a tally of what is on the pizza, the time left, and the scene image. The script enforces
--time-limit and --max-actions. If the API fails mid-run, the rest of the recipe runs as parsed by
the rule parser (pizza/tasks.py parse_recipe), minus what is already on the pizza.

Claude infers the task from the prompt alone (one topping as in Ch1, or a full recipe); give the
time limit with --time-limit. Turn the change checker off (--no-check) when the pizza is not in its
usual spot (Ch3): its sector boxes are fixed pixels.

Needs internet and an API key: ANTHROPIC_API_KEY, or the key saved from the web UI (pizza/apikey.py).

Or run it on a local model instead: a --model starting with "ollama:" (e.g. ollama:qwen3.8:27b) talks to
Ollama's Anthropic-compatible API at --ollama-url (default $OLLAMA_URL or http://localhost:11434). No key,
no Anthropic API. The local model has no strict tool decoding, so tool inputs are checked here. Ollama
runs on gpu1 (~/ollama-serve.sh in the Jupyter container); if it is not reachable on localhost, the script
opens the tunnel itself (scripts/ollama_tunnel.sh) and loads the model before the run starts.

    uv run python scripts/run_agent.py "Place 6 pepperoni pieces on the pizza and spread them evenly." --time-limit 180
    uv run python scripts/run_agent.py "4 pepperoni on the right half, then cheese over the whole pizza" --no-check
    uv run python scripts/run_agent.py "Spread the cheese evenly over the pizza." --sim       # no robot, real Claude
    uv run python scripts/run_agent.py "Spread the cheese evenly over the pizza." --model ollama:qwen3.8:27b
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import time
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pizza import apikey  # noqa: E402
from pizza.robot import Attempt  # noqa: E402
from pizza.station import preflight  # noqa: E402
from pizza.tasks import INGREDIENTS, TARGETS, Step, parse_recipe  # noqa: E402
from run_challenge import add_robot_args, robot_from_args  # noqa: E402

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Models that take the server-side refusal fallback (others get the request without it).
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

# --model ollama:<name> runs the agent on a local Ollama model (gpu1, through an SSH tunnel).
OLLAMA_PREFIX = "ollama:"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = OLLAMA_PREFIX + "qwen3.8:27b"   # pulled on gpu1
OLLAMA_TUNNEL = Path(__file__).resolve().parent / "ollama_tunnel.sh"

SYSTEM = """\
You run a pizza-topping robot arm at a robotics competition. A learned policy moves the arm; you \
choose which trained skill it runs next, one tool call at a time. The prompt you get is the task as \
the judges gave it: it may ask for one topping only (then use only that topping) or a whole recipe.

Skills:
- add_topping(ingredient, target): one cheese portion or one pepperoni slice from its bowl onto the \
target, then back home. This is the normal step: one call per piece.
- pick(ingredient) then place(target): the same step in two halves. After pick you see the wrist \
camera, so you can tell whether the gripper holds the topping before placing it. Use this only when \
grasps have been failing; it costs an extra decision (~5 s).
- look(): fresh scene and wrist images without moving.
- finish(summary): end the run.

Targets: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left quarter), CENTER, and GAP \
(the largest empty area). Quarters are named from the robot operator's side. The scene image you get \
is already rotated to match: the top-left of the image is TL.

Every result tells you the outcome, the time used and left, a tally of the pieces placed so far, and \
for pepperoni (when the pizza is in its usual spot) a change-checker verdict. The checker finds 99% \
of placed slices but sometimes reports a slice that is not there. Cheese and GAP are never checked: \
judge them from the image. The image is small, so when the image and the checker disagree, weigh both.

An outcome "box moved" means the arm dragged or lifted a topping container: the robot opened the \
gripper and went back home along its own path. Look at the scene before trying that topping again.

Only cheese and pepperoni have trained skills. If the recipe asks for anything else, do what you can \
and say in finish what you skipped.

Scoring (do what the recipe says; these decide the trade-offs):
- Pepperoni: 1 point per slice on the pizza (two slices dropped together count as one), plus 1 per \
quarter with at least one slice.
- Cheese: points for coverage (full points above 70% of the pizza) plus 1 per quarter at least ~25% covered.
- When the recipe gives counts, regions or an order, they are scored exactly: extra pieces cost \
points, a piece outside the asked region counts half, and the stages must come in the recipe's order.
- Default amounts when the recipe gives none: cheese 8 portions (TL, TR, BR, BL, twice); pepperoni \
6 slices (TL, TR, BR, BL, CENTER, GAP). Spread pieces over the allowed quarters, clockwise.
- When the recipe limits a topping to a region but gives no count: cheese 2 portions per quarter \
in the region (e.g. left half: TL, BL, TL, BL); pepperoni 1 slice per quarter in the region (e.g. \
right half: TR, BR). The center counts as one quarter here (cheese 2 portions, pepperoni 1 slice).
- Stage order: the order the toppings are named in the recipe. A recipe that only asks for a pizza \
without naming the toppings means cheese first, then pepperoni.
- A step takes about 20-60 s. Stay inside the time limit; ties go to the faster team, so call \
finish as soon as the recipe is done.

Retry a piece only when it clearly did not land (at most twice for the same piece). A step that timed \
out without leaving home can be retried once. Keep each message to one short line of reasoning, then \
call a tool."""


def _schema(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_ING = {"type": "string", "enum": list(INGREDIENTS)}
_TGT = {"type": "string", "enum": list(TARGETS)}
TOOLS = [
    {"name": "add_topping", "strict": True, "input_schema": _schema({"ingredient": _ING, "target": _TGT}),
     "description": "Run one trained step: one portion of the ingredient from its bowl onto the target, then "
                    "back home. Returns the outcome, the checker verdict if any, the tally and the scene image."},
    {"name": "pick", "strict": True, "input_schema": _schema({"ingredient": _ING}),
     "description": "Grasp skill only: take one portion of the ingredient from its bowl and hold it. Returns "
                    "the wrist image. Must be followed by place."},
    {"name": "place", "strict": True, "input_schema": _schema({"target": _TGT}),
     "description": "Place skill only: put the held topping on the target, then back home."},
    {"name": "look", "strict": True, "input_schema": _schema({}),
     "description": "Fresh scene and wrist camera images without moving the arm."},
    {"name": "finish", "strict": True, "input_schema": _schema({"summary": {"type": "string"}}),
     "description": "End the run. Summarise what is on the pizza and anything skipped."},
]


def image_block(rgb: np.ndarray, rotate: bool) -> dict:
    """JPEG block, scaled 2x; the scene is rotated 180 degrees so image top-left = TL."""
    im = Image.fromarray(np.ascontiguousarray(rgb[::-1, ::-1] if rotate else rgb))
    im = im.resize((im.width * 2, im.height * 2), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


class Run:
    """Run state: limits, the pieces placed so far, and what each result tells Claude."""

    def __init__(self, args):
        self.args, self.t0 = args, time.perf_counter()
        self.actions = 0
        self.placed: Counter[Step] = Counter()   # finished steps, checker-rejected ones excluded
        self.log: list[tuple[Step | None, str]] = []

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def out_of_budget(self) -> bool:
        return self.actions >= self.args.max_actions or self.elapsed() > self.args.time_limit

    def record(self, a: Attempt) -> None:
        self.actions += a.action != "look"
        if a.step is not None and a.outcome == "done" and a.placed is not False:
            self.placed[a.step] += 1
        if a.action != "look":
            self.log.append((a.step, f"{a.action}: {a.summary()}"))

    def status(self) -> str:
        tally = ", ".join(f"{s} x{n}" for s, n in self.placed.items()) or "nothing yet"
        return (f"[placed so far: {tally}] [{self.elapsed():.0f}s of {self.args.time_limit:.0f}s used, "
                f"{self.actions} of {self.args.max_actions} actions used]")

    def content(self, text: str, a: Attempt | None) -> list[dict]:
        blocks = [{"type": "text", "text": f"{text} {self.status()}"}]
        if a is not None and not self.args.no_images:
            if a.scene is not None:
                blocks += [{"type": "text", "text": "scene:"}, image_block(a.scene, rotate=True)]
            if a.wrist is not None:
                blocks += [{"type": "text", "text": "wrist:"}, image_block(a.wrist, rotate=False)]
        return blocks


def bad_input(name: str, inp: dict) -> str | None:
    """Why a tool call does not fit its schema (local models have no strict decoding), else None."""
    tool = next((t for t in TOOLS if t["name"] == name), None)
    if tool is None:
        return f"unknown tool {name!r}"
    props = tool["input_schema"]["properties"]
    if set(inp) != set(props):
        return f"{name} takes exactly {sorted(props) or 'no inputs'}"
    for k, v in inp.items():
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"{k} must be one of {props[k]['enum']}"
        if props[k]["type"] == "string" and not isinstance(v, str):
            return f"{k} must be a string"
    return None


def is_local(model: str) -> bool:
    return model.startswith(OLLAMA_PREFIX)


def ollama_api(url: str, path: str, body: dict | None = None, timeout: float = 5) -> dict:
    """GET (body None) or POST JSON to the Ollama server."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def ollama_connect(url: str) -> None:
    """Make sure the server answers; for the default localhost URL, open the gpu1 tunnel if it does not."""
    try:
        ollama_api(url, "/api/version", timeout=2)
        return
    except OSError:
        pass
    if urllib.parse.urlparse(url).hostname not in ("localhost", "127.0.0.1"):
        raise RuntimeError(f"Ollama not reachable at {url}")
    print("ollama: not reachable, opening the tunnel to gpu1 ...", flush=True)
    try:
        r = subprocess.run([str(OLLAMA_TUNNEL)], capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        raise RuntimeError("opening the tunnel to gpu1 timed out (is gpu1 running and its IP unchanged?)")
    for line in (r.stdout + r.stderr).splitlines():
        print(f"    {line}", flush=True)
    try:
        ollama_api(url, "/api/version", timeout=5)
    except OSError as e:
        raise RuntimeError(f"Ollama still not reachable at {url} ({e}); is gpu1 running? "
                           f"(scripts/ollama_tunnel.sh, ~/ollama-serve.sh in its Jupyter container)")


def ollama_warmup(url: str, model: str) -> None:
    """Connect (opening the tunnel if needed) and load the model into GPU memory (cold load ~75 s)."""
    ollama_connect(url)
    try:
        caps = ollama_api(url, "/api/show", {"model": model}, 10).get("capabilities", [])
    except OSError as e:
        raise RuntimeError(f"Ollama has no model {model!r} ({e}); pull it on gpu1: ollama pull {model}")
    if not {"tools", "vision"} <= set(caps):
        print(f"warning: {model} lacks {sorted({'tools', 'vision'} - set(caps))} (capabilities {caps})")
    t = time.perf_counter()
    if not any(m.get("name") == model for m in ollama_api(url, "/api/ps").get("models", [])):
        print(f"ollama: loading {model} into GPU memory (~75 s the first time) ...", flush=True)
    ollama_api(url, "/api/generate", {"model": model, "keep_alive": -1}, 300)
    print(f"ollama: {model} loaded ({time.perf_counter() - t:.0f}s)")


async def call_tool(robot, run: Run, name: str, inp: dict) -> list[dict]:
    if why := bad_input(name, inp):
        return run.content(f"Not run: {why}.", None)
    if name == "look":
        return run.content("Current view.", await robot.look())
    if run.out_of_budget():
        return run.content("Not run: time or action limit reached. Call finish.", None)
    if name == "add_topping":
        s = Step(inp["ingredient"], inp["target"])
        print(f"[{run.actions + 1}] add_topping {s}")
        a = await robot.add_topping(s, want_scene=True)
    elif name == "pick":
        print(f"[{run.actions + 1}] pick {inp['ingredient']}")
        a = await robot.pick(inp["ingredient"])
    else:
        print(f"[{run.actions + 1}] place {inp['target']}")
        a = await robot.place(inp["target"])
    print(f"    {a.summary()}")
    run.record(a)
    return run.content(f"{name}: {a.summary()}.", a)


async def agent_loop(robot, run: Run, recipe: str) -> str:
    """Claude's (or the local model's) loop; returns why it stopped."""
    args = run.args
    if is_local(args.model):
        # Ollama's Anthropic-compatible /v1/messages: no tool_choice, caching or fallbacks; thinking only
        # above --effort low (it adds seconds per decision).
        who, nudges = "local", 0
        client = anthropic.AsyncAnthropic(base_url=args.ollama_url, api_key="ollama", timeout=args.api_timeout)
        create = client.messages.create
        opts = {"model": args.model.removeprefix(OLLAMA_PREFIX),
                "thinking": {"type": "disabled"} if args.effort == "low" else
                {"type": "enabled", "budget_tokens": 2048}}
    else:
        who = "claude"
        client = anthropic.AsyncAnthropic(api_key=apikey.load(), timeout=args.api_timeout)
        create = client.beta.messages.create
        opts = {"model": args.model, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "output_config": {"effort": args.effort}, "cache_control": {"type": "ephemeral"}}
        if args.model in FALLBACK_MODELS:
            opts |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    messages = [{"role": "user", "content": run.content(
        f"Task: {recipe}\nTime limit {args.time_limit:.0f}s, at most {args.max_actions} actions.",
        await robot.look())}]
    while True:
        try:
            resp = await create(max_tokens=8000, system=SYSTEM, tools=TOOLS, messages=messages, **opts)
        except anthropic.APIConnectionError as e:
            return f"API unreachable ({e})"
        except anthropic.RateLimitError as e:
            return f"API rate limited ({e.message})"
        except anthropic.APIStatusError as e:
            return f"API error {e.status_code} ({e.message})"
        if resp.stop_reason == "refusal":
            return "API refusal"
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                print(f"{who}: {b.text.strip()}")
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            if who == "local" and nudges < 2:   # small models sometimes answer in prose; ask once or twice
                nudges += 1
                messages.append({"role": "user", "content": "Call one of the tools now (finish when done)."})
                continue
            return "finished (no tool call)"
        nudges = 0
        results = []
        for u in uses:
            if u.name == "finish":
                print(f"{who} finished: {u.input.get('summary', '')}")
                return "finished"
            results.append({"type": "tool_result", "tool_use_id": u.id,
                            "content": await call_tool(robot, run, u.name, u.input)})
        messages.append({"role": "user", "content": results})


async def fallback(robot, run: Run, recipe: str) -> None:
    """API gone: place what is held, then run the rule-parsed recipe minus what is already placed."""
    try:
        todo = parse_recipe(recipe)
    except ValueError as e:
        print(f"fallback: {e}")
        return
    done, rest = run.placed.copy(), []
    for s in todo:
        if done[s] > 0:
            done[s] -= 1
        else:
            rest.append(s)
    todo = rest
    print(f"fallback: {len(todo)} steps left from the rule plan: {' '.join(map(str, todo))}")
    if robot.holding and todo:
        s = next((t for t in todo if t.ingredient == robot.holding), Step(robot.holding, "GAP"))
        a = await robot.place(s.target)
        run.record(a)
        print(f"    place held {s}: {a.summary()}")
        if s in todo:
            todo.remove(s)
    for s in todo:
        if run.out_of_budget():
            print("fallback: time or action limit reached")
            break
        print(f"[{run.actions + 1}] add_topping {s} (fallback)")
        a = await robot.add_topping(s)
        print(f"    {a.summary()}")
        run.record(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recipe", nargs="+", help="the task/recipe text exactly as the judges give it")
    p.add_argument("--time-limit", type=float, default=480.0, help="seconds for the whole run (Ch1 180, Ch2/Ch3 480)")
    p.add_argument("--max-actions", type=int, default=24, help="hard cap on skill calls, retries included")
    p.add_argument("--model", default="claude-opus-5-5",
                   help=f"Claude model, or {OLLAMA_PREFIX}<name> for a local Ollama model (e.g. {LOCAL_MODEL})")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                   help="Claude's effort per decision; low keeps the gaps between steps short "
                        "(local models: low = no thinking, else thinking on)")
    p.add_argument("--api-timeout", type=float, default=60.0, help="seconds per model request")
    p.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama server for ollama: models (env OLLAMA_URL)")
    p.add_argument("--no-fallback", action="store_true", help="stop if the API fails instead of finishing by rules")
    p.add_argument("--no-images", action="store_true", help="send Claude text results only")
    add_robot_args(p)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


async def main_async() -> None:
    args = parse_args()
    recipe = " ".join(args.recipe)
    if is_local(args.model):
        try:
            ollama_warmup(args.ollama_url, args.model.removeprefix(OLLAMA_PREFIX))
        except RuntimeError as e:
            raise SystemExit(f"ollama: {e}")
    elif not apikey.load():
        raise SystemExit("no Anthropic API key: export ANTHROPIC_API_KEY or save it in the web UI, "
                         "or run the fixed sequence with run_challenge.py")
    robot = robot_from_args(args, check_default=True)
    print(f"task {recipe!r}, {args.time_limit:.0f}s, model {args.model}, "
          f"checker {'on' if robot.cfg.check else 'off'}")

    logging.basicConfig(level=logging.WARNING)
    if not args.sim:
        preflight(args.server)
    await robot.connect()
    run = Run(args)
    try:
        why = await agent_loop(robot, run, recipe)
        print(f"agent stopped: {why}")
        if why.startswith("API") and not args.no_fallback:
            await fallback(robot, run, recipe)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted (Stop / Ctrl-C)")
    finally:
        print(f"\nsummary ({run.elapsed():.0f}s total, {run.actions} actions):")
        for s, text in run.log:
            print(f"  {str(s or ''):16s} {text}")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass  # Stop / Ctrl-C: the summary is already printed


if __name__ == "__main__":
    main()
