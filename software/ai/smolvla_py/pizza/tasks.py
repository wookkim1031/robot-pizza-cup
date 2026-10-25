"""Pizza vocabulary: ingredients, targets, the exact prompts the policy was trained on, recipe parsing.

Single source for the challenge runner and the agent. The prompt texts must match the training
datasets word for word (scripts/split_grasp_place.py writes the grasp/place ones into the data).

Targets are named from the robot/operator side, which is rotated 180 degrees against the scene
camera: TL = image bottom-right, TR = image bottom-left, and so on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INGREDIENTS = ("cheese", "pepperoni")
QUADRANTS = ("TL", "TR", "BR", "BL")  # clockwise, the order the demos were recorded in
TARGETS = (*QUADRANTS, "CENTER", "GAP")

WHERE = {"TL": "on the upper-left quadrant of the pizza", "TR": "on the upper-right quadrant of the pizza",
         "BR": "on the lower-right quadrant of the pizza", "BL": "on the lower-left quadrant of the pizza",
         "CENTER": "in the center of the pizza", "GAP": "on the largest empty area of the pizza"}
_FULL = {"cheese": "Spread one portion of cheese", "pepperoni": "Place one pepperoni slice"}
_GRASP = {"cheese": "Pick up one portion of cheese from the bowl.",
          "pepperoni": "Pick up one pepperoni slice from the bowl."}
_PLACE = {"cheese": "Spread the cheese you are holding", "pepperoni": "Place the pepperoni slice you are holding"}

# Default amounts, as the rounds were recorded: cheese two clockwise laps, pepperoni 6 slices.
DEFAULT_TARGETS = {"cheese": QUADRANTS * 2, "pepperoni": (*QUADRANTS, "CENTER", "GAP")}


def full_task(ingredient: str, target: str) -> str:
    """One-prompt step (checkpoints trained before the grasp/place split)."""
    return f"{_FULL[ingredient]} {WHERE[target]} and return to the start position."


def grasp_task(ingredient: str) -> str:
    return _GRASP[ingredient]


def place_task(ingredient: str, target: str) -> str:
    return f"{_PLACE[ingredient]} {WHERE[target]} and return to the start position."


@dataclass(frozen=True)
class Step:
    ingredient: str
    target: str

    def __post_init__(self):
        if self.ingredient not in INGREDIENTS or self.target not in TARGETS:
            raise ValueError(f"unknown step {self.ingredient} {self.target}")

    def __str__(self) -> str:
        return f"{self.ingredient} {self.target}"


def default_steps(*ingredients: str) -> list[Step]:
    return [Step(i, t) for i in ingredients for t in DEFAULT_TARGETS[i]]


# ---------------------------------------------------------------------------
# Recipe text -> steps (rules; the Ch3 Part B fallback when no LLM is available)

_SYNONYMS = {"cheese": "cheese", "mozzarella": "cheese", "pepperoni": "pepperoni", "salami": "pepperoni"}
_REGIONS = {
    "right half": ("TR", "BR"), "left half": ("TL", "BL"),
    "top half": ("TL", "TR"), "upper half": ("TL", "TR"),
    "bottom half": ("BR", "BL"), "lower half": ("BR", "BL"),
    "right side": ("TR", "BR"), "left side": ("TL", "BL"), "top side": ("TL", "TR"), "bottom side": ("BR", "BL"),
    "center": ("CENTER",), "middle": ("CENTER",),
}
for _q, _words in {"TL": ("top", "upper", "left"), "TR": ("top", "upper", "right"),
                   "BR": ("bottom", "lower", "right"), "BL": ("bottom", "lower", "left")}.items():
    for _v in _words[:2]:
        for _sep in ("-", " "):
            _REGIONS[f"{_v}{_sep}{_words[2]}"] = (_q,)
_NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def _count(text: str) -> int | None:
    m = re.search(r"\b(\d+|" + "|".join(_NUMBERS) + r")\b", text)
    if not m:
        return None
    return int(m.group(1)) if m.group(1).isdigit() else _NUMBERS[m.group(1)]


def parse_recipe(text: str) -> list[Step]:
    """Clauses naming an ingredient start a stage (in order); each may give a count and a region.

    "4 pepperoni on the right half, then spread cheese over the whole pizza"
    -> pepperoni TR BR TR BR, cheese TL TR BR BL TL TR BR BL
    No region: the ingredient's default targets. No count: as many as the region's default.
    """
    names = "|".join(_SYNONYMS)
    stages: list[list[str]] = []  # [ingredient, clause text]
    for clause in re.split(r"[,;:.]|\bthen\b|\band\b|\bafter that\b|\bfollowed by\b", text.lower()):
        m = re.search(rf"\b({names})\b", clause)
        if m:
            stages.append([_SYNONYMS[m.group(1)], clause])
        elif stages:
            stages[-1][1] += " " + clause
    if not stages:
        raise ValueError(f"no known ingredient in {text!r} (know {sorted(_SYNONYMS)})")
    steps: list[Step] = []
    for ingredient, clause in stages:
        region = next((_REGIONS[r] for r in sorted(_REGIONS, key=len, reverse=True) if r in clause), None)
        targets = region or DEFAULT_TARGETS[ingredient]
        if region and ingredient == "cheese" and len(region) < 4:
            targets = region * 2  # cheese spreads: default 2 portions per allowed quadrant
        n = _count(clause) or len(targets)
        steps += [Step(ingredient, targets[k % len(targets)]) for k in range(n)]
    return steps
