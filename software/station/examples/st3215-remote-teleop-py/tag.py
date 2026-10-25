"""Add recording tags to the station from the command line.

Same effect as the station viewer's TAG button: the tag marks the current
`inference-states` queue pointer. Counters, layout and operator persist in a
JSON state file between calls.

A round is one pizza: 4 cheese episodes (one per quadrant) and 6 pepperoni
episodes (one per quadrant plus 2 extras), in the order the round plan says.
The plan also picks the pizza position (P0 = the challenge 1-2 spot, P1-P4 =
shifted spots) and the box order (AB = normal, BA = swapped) from a fixed
schedule, so the data mixes layouts from the start.

    python tag.py start            # cheese new | s2 | R1 | E1 | P0 | AB | MINSEON | START
    python tag.py end              # ... | END, then advance to the next episode
    python tag.py fail regrasp     # ... | FAIL regrasp, the episode is reused
    python tag.py round            # skip to the next round
    python tag.py layout P2 BA     # override this round's pizza position / box order
    python tag.py order PC         # this round: pepperoni first (CP = cheese first)
    python tag.py operator JOHAN   # change operator
    python tag.py set --session f1 --round 1 --episode 1
    python tag.py plan             # list this round's episodes
    python tag.py status
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from software.station.shared.station_py import new_station_client, send_commands
from target.gen_python.protobuf.station import commands as station_commands, drivers

STATE_PATH = Path.home() / "datasets" / "tag_state.json"
INFERENCE_QUEUE = "inference-states"  # the queue the viewer's TAG button points into

# Episode order within an ingredient: E1 top-left, E2 top-right, E3 bottom-right,
# E4 bottom-left. The tag leaves the quadrant out; it follows from the episode.
QUADRANTS = ["TL", "TR", "BR", "BL"]
# Two extra pepperoni slices per round. Alternating the pairs keeps every
# quadrant at the same count over two rounds.
EXTRA_PEPPERONI = [("TL", "BR"), ("TR", "BL")]
# 20-round layout cycle: P0 AB (the challenge 1-2 layout) 8x, P0 BA 2x, and the
# shifted spots P1-P4 with both box orders 10x.
LAYOUT_CYCLE = [
    "P0 AB", "P1 AB", "P0 AB", "P2 BA", "P0 BA", "P3 AB", "P0 AB", "P4 BA", "P0 AB", "P1 BA",
    "P0 AB", "P2 AB", "P0 AB", "P3 BA", "P0 BA", "P4 AB", "P0 AB", "P1 AB", "P0 AB", "P2 BA",
]

# ch3 minimum set (state "schedule": "ch3"): 10 takes per round, 5 of the first ingredient
# (TL TR BR BL + extra), then 5 of the second. The pizza moves to a new grid cell (FL F FR / L C R /
# NL N NR, C = P0) and the boxes are set before every take; "+K" = candy bowl on the table.
CH3_SCHEDULE = {
    4: ("CP", "", "TL", "FL FR NR NL C F R N L FR", "AB BA AB BA AB BA AB BA AB BA"),
    5: ("PC", "+K", "TR", "C NL FR R N L F NR FL NL", "BA AB BA AB BA AB BA AB BA AB"),
    6: ("CP", "", "BR", "F C L FR NL R FL N NR R", "BA AB BA AB BA AB BA AB BA AB"),
    7: ("PC", "", "BL", "NR L C NL FR N R FL F N", "AB BA AB BA AB BA AB BA AB BA"),
    8: ("CP", "+K", "TL", "L FR C NL F NR FL R N F", "AB BA AB BA AB BA AB BA AB BA"),
    9: ("PC", "", "TR", "N F R FL C NL NR FR L C", "AB BA AB BA AB BA AB BA AB BA"),
}

DEFAULT_STATE = {
    "session": "f1", "round": 1, "episode": 1, "operator": "UNKNOWN",
    "pos": "P0", "boxes": "AB", "order": "CP",
    "env": "", "only": "", "fixed_layout": "", "in_episode": False, "ended": 0, "failed": 0,
    "ended_by_ingredient": {"cheese": 0, "pepperoni": 0},
}


def load_state() -> dict:
    if STATE_PATH.exists():
        return {**DEFAULT_STATE, **json.loads(STATE_PATH.read_text())}
    return start_round(dict(DEFAULT_STATE), 1)


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def start_round(state: dict, number: int) -> dict:
    """Move to round `number` with the scheduled layout and stage order."""
    layout = state.get("fixed_layout") or LAYOUT_CYCLE[(number - 1) % len(LAYOUT_CYCLE)]
    pos, boxes = layout.split()
    order = "PC" if number % 3 == 0 else "CP"  # every third round pepperoni first
    state.update(round=number, episode=1, in_episode=False, pos=pos, boxes=boxes, order=order,
                 custom_plan=[])
    if sched := ch3_round(state):
        state["order"] = sched[0]
    return state


def ch3_round(state: dict) -> tuple | None:
    if state.get("schedule") == "ch3":
        return CH3_SCHEDULE.get(state["round"])
    return None


# "pizza4" schedule: 4 takes per round, boxes as usual, the pizza moves to a new spot every take
# (top-left, top-right, bottom-right, bottom-left from the robot's view), cheese and pepperoni
# alternate, and the target quarters follow the usual TL TR BR BL order.
PIZZA_SPOTS = ["TL", "TR", "BR", "BL"]


def pizza4_quadrants(round_number: int) -> list[str]:
    return list(QUADRANTS)  # the user's usual order every round (TL TR BR BL)


def round_plan(state: dict) -> list[tuple[str, str]]:
    if state.get("schedule") == "pizza4":
        # Two circles per round. Cheese and pepperoni alternate, cheese first on the first circle
        # (E1 cheese, E2 pepperoni, ...) and pepperoni first on the second (E5 pepperoni, E6 cheese, ...).
        quadrants = pizza4_quadrants(state["round"])
        return [(("cheese", "pepperoni")[(i + lap) % 2], q)
                for lap in range(2) for i, q in enumerate(quadrants)]
    if sched := ch3_round(state):
        order, _, cheese_extra, _, _ = sched
        first, second = ("cheese", "pepperoni") if order == "CP" else ("pepperoni", "cheese")
        extra = {"cheese": cheese_extra, "pepperoni": "CENTER"}
        return [(ing, q) for ing in (first, second) for q in QUADRANTS + [extra[ing]]]
    if state.get("custom_plan"):  # this round only, e.g. candy rounds with 3 bowls
        return [tuple(step) for step in state["custom_plan"]]
    if state["only"]:  # single-ingredient rounds: one episode per quadrant
        order = state.get("quadrants") or QUADRANTS
        return [(state["only"], q) for q in order * state.get("only_reps", 1)]
    cheese = [("cheese", q) for q in QUADRANTS]
    extras = EXTRA_PEPPERONI[(state["round"] - 1) % len(EXTRA_PEPPERONI)]
    pepperoni = [("pepperoni", q) for q in QUADRANTS + list(extras)]
    return cheese + pepperoni if state["order"] == "CP" else pepperoni + cheese


def current_step(state: dict) -> tuple[str, str]:
    return round_plan(state)[state["episode"] - 1]


def step_layout(state: dict) -> tuple[str, str]:
    """Pizza position and box order for the current take (ch3: per take, else per round)."""
    if state.get("schedule") == "pizza4":
        return "P" + PIZZA_SPOTS[(state["episode"] - 1) % 4], state["boxes"]
    if sched := ch3_round(state):
        _, candy, _, cells, boxes = sched
        i = state["episode"] - 1
        return cells.split()[i], boxes.split()[i] + candy
    return state["pos"], state["boxes"]


def label(state: dict, marker: str) -> str:
    ingredient, _ = current_step(state)
    name = f"{ingredient} {state['env']}" if state["env"] else ingredient
    pos, boxes = step_layout(state)
    return (f"{name} | {state['session']} | R{state['round']} | E{state['episode']} | "
            f"{pos} | {boxes} | {state['operator']} | {marker}")


def setup_line(state: dict) -> str:
    boxes = {"AB": "normal", "BA": "SWAPPED"}.get(state["boxes"], "3 bowls left-to-right")
    first = state["only"] or ("cheese" if state["order"] == "CP" else "pepperoni")
    if state.get("custom_plan"):
        first = state["custom_plan"][0][0]
    if state.get("schedule") == "pizza4":
        return (f"SET UP R{state['round']}: clean pizza, boxes as usual, {round_plan(state)[0][0]}; "
                f"pizza moves top-left, top-right, bottom-right, bottom-left (one spot per take)")
    if sched := ch3_round(state):
        first = round_plan(state)[0][0]
        candy = "candy bowl ON the table" if sched[1] else "no candy bowl"
        return f"SET UP R{state['round']}: clean pizza, {first} first, {candy}; pizza cell + boxes change every take"
    return f"SET UP R{state['round']}: clean pizza at {state['pos']}, boxes {boxes} ({state['boxes']}), {first}{'' if state['only'] else ' first'}"


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def encode_add_tag(ptr: bytes, tag: str) -> bytes:
    """inference_tags.Command{type=CT_ADD_TAG, inference_queue_ptr, tag}.

    No Python binding is generated for inference_tags, so encode by hand.
    CT_ADD_TAG is 0 and therefore omitted (proto3 default).
    """
    tag_bytes = tag.encode()
    return (b"\x12" + _varint(len(ptr)) + ptr
            + b"\x1a" + _varint(len(tag_bytes)) + tag_bytes)


async def latest_pointer(client) -> bytes:
    qr = client.read_from_tail(INFERENCE_QUEUE, (1).to_bytes(8, "little"), 1, 1, 1)
    ptr = None
    while True:
        entry = await qr.data.get()
        if entry is None:
            break
        ptr = bytes(entry.ID.ID)
    if ptr is None:
        raise RuntimeError(f"no entries in {INFERENCE_QUEUE}: is the station streaming?")
    return ptr


async def add_tag(tag: str) -> int:
    client = await new_station_client("localhost", logging.getLogger("tag"))
    ptr = await latest_pointer(client)
    cmd = station_commands.DriverCommand(
        type=drivers.StationCommandType.STC_INFERENCE_TAG_COMMAND,
        body=encode_add_tag(ptr, tag),
    )
    await send_commands(client, [cmd])
    return int.from_bytes(ptr, "little")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["start", "end", "fail", "round", "layout", "order",
                                      "operator", "set", "plan", "status"])
    p.add_argument("value", nargs="*")
    p.add_argument("--session")
    p.add_argument("--env", help="environment label added after the ingredient, e.g. new")
    p.add_argument("--only", choices=["", "cheese", "pepperoni"],
                   help="record rounds of 4 episodes of one ingredient ('' = mixed 10-episode rounds)")
    p.add_argument("--quadrants", help="with --only: quadrant order, e.g. TL,TR,BR,BL")
    p.add_argument("--reps", type=int, help="with --only: passes over the 4 quadrants per round")
    p.add_argument("--fixed-layout", help="use this layout every round, e.g. 'P0 AB' ('' = schedule)")
    p.add_argument("--custom-plan",
                   help="this round only: ingredient:quadrant list, e.g. cheese:TL,cheese:TR,candy:TL")
    p.add_argument("--schedule", choices=["", "ch3", "pizza4"],
                   help="ch3 = per-take pizza cell and box order (R4-R9); pizza4 = 4 pizza spots per round")
    p.add_argument("--round", type=int)
    p.add_argument("--episode", type=int)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    state = load_state()
    round_before = state["round"]

    if args.action == "set":
        if args.session is not None:
            state["session"] = args.session
        if args.env is not None:
            state["env"] = args.env
        if args.only is not None:
            state["only"] = args.only
        if args.quadrants is not None:
            state["quadrants"] = [q.strip().upper() for q in args.quadrants.split(",")]
        if args.reps is not None:
            state["only_reps"] = args.reps
        if args.fixed_layout is not None:
            state["fixed_layout"] = args.fixed_layout.upper()
        if args.round is not None:
            start_round(state, args.round)
        if args.schedule is not None:
            state["schedule"] = args.schedule
        if args.custom_plan is not None:
            state["custom_plan"] = [s.strip().split(":") for s in args.custom_plan.split(",") if s.strip()]
        if args.episode is not None:
            state["episode"] = args.episode
        state["in_episode"] = False
    elif args.action == "operator":
        state["operator"] = " ".join(args.value).upper()
    elif args.action == "round":
        start_round(state, state["round"] + 1)
    elif args.action == "layout":
        pos, boxes = (v.upper() for v in args.value)
        # 3 bowls: their contents left to right, P = pepperoni, C = cheese, K = candy (e.g. CKP).
        three = len(boxes) == 3 and sorted(boxes) == ["C", "K", "P"]
        if pos not in {f"P{i}" for i in range(5)} or not (boxes in ("AB", "BA") or three):
            sys.exit("usage: tag.py layout P0..P4 AB|BA|<3 bowls, e.g. CKP>")
        if state["episode"] > 1:
            print(f"WARNING: changing the layout in the middle of R{state['round']}")
        state.update(pos=pos, boxes=boxes)
    elif args.action == "order":
        order = args.value[0].upper()
        if order not in ("CP", "PC"):
            sys.exit("usage: tag.py order CP|PC")
        if state["episode"] > 1 or state["in_episode"]:
            sys.exit(f"R{state['round']} has already started; the order can only change at E1")
        state["order"] = order
    elif args.action == "plan":
        for i, (ingredient, quadrant) in enumerate(round_plan(state), start=1):
            mark = ">" if i == state["episode"] else " "
            print(f" {mark} E{i:<2} {ingredient:9s} {quadrant}")
    elif args.action in ("start", "end", "fail"):
        if args.action == "start":
            marker = "START"
        elif args.action == "end":
            marker = "END"
        else:
            marker = "FAIL " + " ".join(args.value) if args.value else "FAIL"
        tag = label(state, marker)
        frame = asyncio.run(add_tag(tag))
        print(f"tagged frame={frame}: {tag}")
        if args.action == "start":
            state["in_episode"] = True
            if state.get("round_started_for") != state["round"]:
                state.update(round_started_for=state["round"], round_started_at=time.time())
        elif args.action == "end":
            state["in_episode"] = False
            state["ended"] += 1
            ingredient, _ = current_step(state)
            counts = state["ended_by_ingredient"]
            counts[ingredient] = counts.get(ingredient, 0) + 1
            if state["episode"] >= len(round_plan(state)):
                if state.get("round_started_for") == state["round"]:
                    secs = time.time() - state["round_started_at"]
                    print(f"round R{state['round']} took {int(secs // 60)}m {int(secs % 60):02d}s")
                start_round(state, state["round"] + 1)
            else:
                state["episode"] += 1
        else:
            state["in_episode"] = False  # retry the same episode number
            state["failed"] += 1

    save_state(state)
    if state["round"] != round_before or args.action in ("layout", "order", "set"):
        print(setup_line(state))
    status = "RECORDING" if state["in_episode"] else "IDLE"
    ingredient, quadrant = current_step(state)
    counts = state["ended_by_ingredient"]
    pos, boxes = step_layout(state)
    print(f"status: {status} | R{state['round']} E{state['episode']}/{len(round_plan(state))} "
          f"{ingredient} {quadrant} | {pos} {boxes} {state['order']} | "
          f"op {state['operator']} | session {state['session']} | "
          f"ended {state['ended']} (cheese {counts.get('cheese', 0)}, "
          f"pepperoni {counts.get('pepperoni', 0)}) | failed {state['failed']}")


if __name__ == "__main__":
    main()
