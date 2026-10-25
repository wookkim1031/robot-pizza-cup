"""Fixed-sequence runs: the challenge presets and the step loop with checker retries and time budgets."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from pizza.tasks import INGREDIENTS, Step, default_steps, parse_recipe


@dataclass
class Challenge:
    title: str
    steps: list[Step] | None             # None: parsed from the recipe text given at evaluation
    time: dict[str, float]               # seconds per ingredient (retries included); a step that won't fit is skipped
    check: bool = True                   # change checker + retries for pepperoni (fixed pizza position only)
    time_total: float = 0.0              # with steps=None: split over the parsed stages by step count

    def plan(self, recipe_text: str | None) -> tuple[list[Step], dict[str, float]]:
        if self.steps is not None:
            return list(self.steps), dict(self.time)
        if not recipe_text:
            raise SystemExit(f"{self.title}: give the recipe text")
        steps = parse_recipe(recipe_text)
        n = {i: sum(s.ingredient == i for s in steps) for i in INGREDIENTS}
        return steps, {i: round(self.time_total * c / len(steps)) for i, c in n.items() if c}


# Official limits: Ch1 3 min per attempt, Ch2/Ch3 8 min. The budgets leave ~10 s of margin.
CHALLENGES = {
    "ch1-cheese": Challenge("Ch1 precision topping: cheese", default_steps("cheese"), {"cheese": 170}),
    "ch1-pepperoni": Challenge("Ch1 precision topping: 6 pepperoni", default_steps("pepperoni"), {"pepperoni": 170}),
    "ch2": Challenge("Ch2 recipe assembly: cheese, then pepperoni", default_steps("cheese", "pepperoni"),
                     {"cheese": 200, "pepperoni": 270}),
    "ch2a": Challenge("Ch2 recipe assembly, reversed: pepperoni, then cheese", default_steps("pepperoni", "cheese"),
                      {"pepperoni": 270, "cheese": 200}),
    # Ch3: the pizza moves in the frame, and the checker's sector boxes are fixed pixels -> checker off.
    "ch3a": Challenge("Ch3 part A: new layout, known recipe", default_steps("cheese", "pepperoni"),
                      {"cheese": 200, "pepperoni": 270}, check=False),
    "ch3b": Challenge("Ch3 part B: new layout, new recipe (text)", None, {}, check=False, time_total=470),
}


class Budget:
    """Per-ingredient time limits for a run; a step (or retry) that would not fit is skipped."""

    DEFAULT_ATTEMPT = 30.0  # seconds, until an attempt of that ingredient has been timed

    def __init__(self, limits: dict[str, float]):
        self.limit = {i: limits.get(i, 0.0) for i in INGREDIENTS}
        self.spent = {i: 0.0 for i in INGREDIENTS}
        self.times: dict[str, list[float]] = {i: [] for i in INGREDIENTS}

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


@dataclass
class RunLog:
    rows: list[tuple[Step, str]] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def add(self, step: Step, text: str) -> None:
        self.rows.append((step, text))

    def print(self, budget: Budget | None = None) -> None:
        spent = f"; {', '.join(f'{i} {s:.0f}s' for i, s in budget.spent.items() if s)}" if budget else ""
        print(f"\nsummary ({time.perf_counter() - self.t0:.0f}s total{spent}):")
        for s, text in self.rows:
            print(f"  {s.ingredient:9s} {s.target:6s} {text}")


async def run_steps(robot, steps: list[Step], budget: Budget, retries: int, log: RunLog,
                    confirm: bool = False) -> None:
    """Run the steps in order. A checked step that the checker says missed is repeated up to `retries` times."""
    for n, s in enumerate(steps, 1):
        if not budget.fits(s.ingredient):
            print(f"[{n}/{len(steps)}] SKIPPED {s}: {budget.status(s.ingredient)}")
            log.add(s, f"skipped, {budget.status(s.ingredient)}")
            continue
        if confirm:
            await asyncio.get_running_loop().run_in_executor(None, input, f"[{n}/{len(steps)}] ENTER to run {s} ")
        print(f"[{n}/{len(steps)}] {s}")
        tries, made = 1 + (retries if robot.checks(s) else 0), 0
        for k in range(tries):
            if k and not budget.fits(s.ingredient):
                print(f"    no retry: {budget.status(s.ingredient)}")
                break
            with budget.attempt(s.ingredient):
                a = await robot.add_topping(s)
            made += 1
            print(f"    attempt {made}: {a.summary()}")
            retry = a.placed is False and made < tries
            if getattr(robot, "check_log", None) and a.check_seq:
                robot.check_log.write("decision", {"check": a.check_seq, "step": n, "steps": len(steps),
                                                   "attempt": made, "attempts_max": tries, "retry": retry,
                                                   "gave_up": a.placed is False and not retry})
            if a.placed is not False:
                break
            if made < tries:
                print(f"    no {s.ingredient} in {s.target}, retrying ({made}/{retries})")
        else:
            if tries > 1:
                a.text += f" -> gave up after {tries} attempts"
        log.add(s, f"[{made} attempt{'s' * (made > 1)}] {a.summary()}" if tries > 1 else a.summary())
