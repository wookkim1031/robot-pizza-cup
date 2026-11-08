"""PizzaRobot: the action surface shared by the challenge runner and the LLM agent.

    add_topping(step)   one portion/slice from its bowl onto a target, back home (+ checker verdict)
    pick(ingredient)    grasp skill only: ends with the topping in the gripper
    place(target)       place skill only: the held topping onto a target, back home (+ checker verdict)
    look()              scene + wrist camera images, arm not moved

Change checker v1 (fixed sector boxes) or v2 (--failure-checker v2: the pizza is located once at connect,
arm at home, and the boxes follow it; smolvla.failure_checker_v2). With a checker log dir every verdict is
also written for the web UI (pizza.check_log).

Every action returns an Attempt. SimRobot has the same surface without a station (--sim).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pizza import skills
from pizza.check_log import CheckLog
from pizza.skills import DriveConfig, Policy
from pizza.station import FOLLOWER_BUS, Station
from pizza.tasks import INGREDIENTS, Step

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/smolvla-grasp4-2026-10-25-step006000"  # split cheese + pepperoni
DEFAULT_CHECK_MODEL = ROOT / "checkpoints/change-checker-2026-10-25/change.pt"
CHECKED_TOPPINGS = {"pepperoni"}  # the change checker finds 99% of slices, but only 51% of cheese


@dataclass
class RobotConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    checkpoint_cheese: Path | None = None      # optional specialist for cheese steps
    checkpoint_pepperoni: Path | None = None   # optional specialist for pepperoni steps
    check: bool = True                         # change checker after pepperoni steps
    check_model: Path = DEFAULT_CHECK_MODEL
    check_thr: float = 0.65                    # p(added) above this counts as placed
    check_settle: float = 0.5                  # seconds at home before the checker's frame
    failure_checker: str = "v2"                # "v1" fixed sector boxes, "v2" boxes around the located pizza
    checker_max_offset: float = 40.0           # v2: located pizza further than this (px) -> v1 boxes
    checker_log: Path | None = None            # dir for checker events + overlays (web UI)
    drive: DriveConfig = field(default_factory=DriveConfig)
    server: str = "localhost"
    bus_serial: str = FOLLOWER_BUS
    camera_order: str = "auto"

    def checkpoint_for(self, ingredient: str) -> Path:
        return getattr(self, f"checkpoint_{ingredient}") or self.checkpoint


@dataclass
class Attempt:
    action: str                      # "add_topping", "pick", "place", "look"
    step: Step | None
    outcome: str                     # "done", "grasped", "grasp failed", "timeout", "ok", "refused"
    seconds: float = 0.0
    text: str = ""
    placed: bool | None = None       # checker verdict; None = not checked
    p: float | None = None
    elsewhere: dict[str, float] = field(default_factory=dict)
    check_seq: int | None = None     # the verdict's event in the checker log
    scene: np.ndarray | None = None
    wrist: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in ("done", "grasped", "ok") and self.placed is not False

    def summary(self) -> str:
        s = self.text
        if self.placed is not None:
            s += f"; checker: {'ADDED' if self.placed else 'NOT added'} in {self.step.target} (p={self.p:.2f})"
            if self.elsewhere:
                s += f", also appeared in {self.elsewhere}"
        elif self.step is not None and self.action != "pick":
            s += "; not checked"
        return s


class PizzaRobot:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.station = Station(cfg.server, cfg.bus_serial, camera_order=cfg.camera_order)
        self.policies: dict[str, Policy] = {}
        self.checker = None
        self.v2 = cfg.failure_checker == "v2"
        self.check_log = CheckLog(cfg.checker_log) if cfg.checker_log else None
        self.scene: np.ndarray | None = None   # at-home scene after the last action = the next before-frame
        self.holding: str | None = None
        self._before: np.ndarray | None = None  # before-frame of a pick/place pair

    async def connect(self, ingredients=INGREDIENTS) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded: dict[Path, Policy] = {}
        for ing in ingredients:
            path = self.cfg.checkpoint_for(ing)
            if path not in loaded:
                print(f"loading {path} on {device} ...")
                loaded[path] = Policy(path, device)
            self.policies[ing] = loaded[path]
            print(f"  {ing}: {path.name}")
        if self.cfg.check and CHECKED_TOPPINGS & set(ingredients):
            self._load_checker(device)
        await self.station.connect()
        if self.v2 and self.checker is not None:
            self.start_checker(await self._scene())

    def _load_checker(self, device) -> None:
        if self.v2:
            from smolvla.failure_checker_v2 import FailureCheckerV2
            self.checker = FailureCheckerV2(self.cfg.check_model, thr=self.cfg.check_thr, device=device,
                                            max_offset_px=self.cfg.checker_max_offset)
        else:
            from smolvla.change_checker import ChangeChecker
            self.checker = ChangeChecker(self.cfg.check_model, device)
        print(f"change checker {self.cfg.failure_checker}: {self.cfg.check_model.name} (thr {self.cfg.check_thr}) "
              f"for {sorted(CHECKED_TOPPINGS)}")

    def start_checker(self, scene: np.ndarray) -> dict:
        """v2: locate the pizza on an at-home scene frame and set the sector boxes for the run."""
        st = self.checker.start(scene)
        where = f"pizza at {st['center']}, {st['offset_px']}px from the reference" if st["found"] else "pizza not found"
        print(f"checker v2: {where} -> {st['boxes_source']} boxes"
              + (f" ({st['reason']})" if st["reason"] else "") + f"  [{st['ms']:.0f} ms]")
        if self.check_log:
            self.check_log.write("start", st, self.checker.render(scene))
        return st

    async def localize(self) -> dict:
        """Dry run: read one scene frame (the arm is not moved) and run the v2 localization on it."""
        self._load_checker("cpu")
        await self.station.connect_readonly()
        return self.start_checker(await self.station.scene(self.cfg.check_settle))

    def checks(self, step: Step) -> bool:
        if self.checker is None:
            return False
        boxes = self.checker.boxes if self.v2 else self.checker.roi
        return step.ingredient in CHECKED_TOPPINGS and step.target in boxes

    async def _scene(self) -> np.ndarray:
        self.scene = await self.station.scene(self.cfg.check_settle)
        return self.scene

    def _verdict(self, a: Attempt, before: np.ndarray, after: np.ndarray) -> None:
        if self.v2:
            res = self.checker.check_step(before, after, a.step.target, a.step.ingredient)
            a.placed, a.p, a.elsewhere = res["ok"], res["p"], res["elsewhere"]
            if self.check_log:
                a.check_seq = self.check_log.write("check", res, self.checker.render(after, res))
            return
        a.placed, a.p, a.elsewhere = self.checker.check_step(before, after, a.step.target, a.step.ingredient,
                                                             self.cfg.check_thr)
        if self.check_log:
            a.check_seq = self.check_log.write("check", {
                "version": "v1", "supported": True, "target": a.step.target, "topping": a.step.ingredient,
                "p": round(float(a.p), 3), "ok": bool(a.placed), "threshold": self.cfg.check_thr, "per_sector": {},
                "elsewhere": a.elsewhere, "boxes_source": "fixed"})

    async def add_topping(self, step: Step, want_scene: bool = False) -> Attempt:
        if self.holding:
            return Attempt("add_topping", step, "refused", text=f"holding {self.holding}: place it first")
        checked = self.checks(step)
        before = (self.scene if self.scene is not None else await self._scene()) if checked else None
        r = await skills.add_topping(self.station, self.policies[step.ingredient], step, self.cfg.drive)
        a = Attempt("add_topping", step, r.outcome, r.seconds, r.text)
        if checked or want_scene:
            a.scene = await self._scene()
        else:
            await self.station.parked()
            self.scene = None
        if checked:
            self._verdict(a, before, a.scene)
        return a

    async def pick(self, ingredient: str) -> Attempt:
        if self.holding:
            return Attempt("pick", None, "refused", text=f"already holding {self.holding}")
        if not getattr(self.cfg.drive, f"split_{ingredient}"):
            return Attempt("pick", None, "refused", text=f"the {ingredient} checkpoint has no separate grasp skill")
        self._before = self.scene if self.scene is not None else (await self._scene() if self.checker else None)
        self.scene = None
        r = await skills.grasp(self.station, self.policies[ingredient], ingredient, self.cfg.drive)
        frame = await self.station.parked()
        a = Attempt("pick", None, r.outcome, r.seconds, r.text, wrist=self.station.wrist_rgb(frame))
        a.scene = self.station.scene_rgb(frame)
        self.holding = ingredient if r.ok else None
        return a

    async def place(self, target: str) -> Attempt:
        if not self.holding:
            return Attempt("place", None, "refused", text="not holding anything: pick first")
        step = Step(self.holding, target)
        r = await skills.place(self.station, self.policies[step.ingredient], step, self.cfg.drive)
        self.holding = None
        a = Attempt("place", step, r.outcome, r.seconds, r.text, scene=await self._scene())
        if self.checks(step) and self._before is not None:
            self._verdict(a, self._before, a.scene)
        return a

    async def look(self) -> Attempt:
        frame = await self.station.parked()
        return Attempt("look", None, "ok", scene=self.station.scene_rgb(frame), wrist=self.station.wrist_rgb(frame))


class SimRobot:
    """No station: steps take 20-50 s of pretend time, slices land 80% of the time, no images."""

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.holding: str | None = None

    async def connect(self, ingredients=INGREDIENTS) -> None:
        print("sim: no robot, no policy")

    def checks(self, step: Step) -> bool:
        return self.cfg.check and step.ingredient in CHECKED_TOPPINGS and step.target != "GAP"

    def _attempt(self, action: str, step: Step | None, outcome: str = "done") -> Attempt:
        t = random.uniform(20, 50)
        a = Attempt(action, step, outcome, t, f"{outcome} in {t:.0f}s (simulated)")
        if step is not None and action != "pick" and self.checks(step):
            a.placed = random.random() < 0.8
            a.p = 0.9 if a.placed else 0.1
        return a

    async def add_topping(self, step: Step, want_scene: bool = False) -> Attempt:
        if self.holding:
            return Attempt("add_topping", step, "refused", text=f"holding {self.holding}: place it first")
        return self._attempt("add_topping", step)

    async def pick(self, ingredient: str) -> Attempt:
        ok = random.random() < 0.9
        self.holding = ingredient if ok else None
        return self._attempt("pick", None, "grasped" if ok else "grasp failed")

    async def place(self, target: str) -> Attempt:
        if not self.holding:
            return Attempt("place", None, "refused", text="not holding anything: pick first")
        step, self.holding = Step(self.holding, target), None
        return self._attempt("place", step)

    async def look(self) -> Attempt:
        return Attempt("look", None, "ok")
