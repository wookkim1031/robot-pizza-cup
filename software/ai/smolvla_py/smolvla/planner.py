# Copyright 2026 Norma Core contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-level planner: one recipe prompt -> a list of placement steps.

Each step carries the exact prompt the SmolVLA policy was trained on, e.g.
"put the cheese on the top-left quarter of the pizza". The planner sits above
the policy (it only picks prompts); the policy produces every joint command.

Two backends:
  - rules (default): ingredient order, piece counts and regions parsed with
    regexes. Deterministic, covers the challenge recipe wording.
  - llm: an open-source instruct LLM writes the step list as JSON; anything it
    gets wrong (unknown ingredient/region, bad JSON) falls back to rules.

    plan("make a pizza: cheese, then 4 pepperoni on the right half")
    -> cheese TL, TR, BR, BL, TL, TR, BR, BL; pepperoni TR, BR, TR, BR
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from smolvla.quarter_checker import QUARTERS

INGREDIENTS = {"cheese": "cheese", "pepperoni": "pepperoni", "salami": "pepperoni"}
CENTER, GAP = "center", "gap"
REGIONS = {
    "right half": ["top-right", "bottom-right"],
    "left half": ["top-left", "bottom-left"],
    "top half": ["top-left", "top-right"],
    "bottom half": ["bottom-right", "bottom-left"],
    **{q: [q] for q in QUARTERS},
    **{q.replace("-", " "): [q] for q in QUARTERS},
}
# Default piece plans match how the rounds were recorded.
DEFAULT_TARGETS = {
    "cheese": list(QUARTERS) * 2,                     # two clockwise laps
    "pepperoni": list(QUARTERS) + [CENTER, GAP],      # 6 pieces: quarters, center, emptiest spot
}
NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


@dataclass
class Step:
    ingredient: str
    target: str  # a quarter, CENTER or GAP

    @property
    def prompt(self) -> str:
        if self.target == CENTER:
            return f"put the {self.ingredient} in the center of the pizza"
        if self.target == GAP:
            return f"put the {self.ingredient} where the pizza has the most empty space"
        return f"put the {self.ingredient} on the {self.target} quarter of the pizza"

    @property
    def generic_prompt(self) -> str:
        return f"cover the pizza with {self.ingredient}"


def _count(text: str) -> int | None:
    m = re.search(r"\b(\d+|" + "|".join(NUMBERS) + r")\b", text)
    if not m:
        return None
    return int(m.group(1)) if m.group(1).isdigit() else NUMBERS[m.group(1)]


def plan_rules(prompt: str) -> list[Step]:
    """Split into clauses; a clause naming an ingredient starts a stage, later clauses add to it.

    Each stage may give a count ("6", "four") and a region ("right half", "top-left").
    No region: the ingredient's default targets, cycled to the count.
    """
    names = "|".join(INGREDIENTS)
    stages: list[list] = []  # [ingredient, text]
    for clause in re.split(r"[,;:.]|\bthen\b|\band\b", prompt.lower()):
        m = re.search(rf"\b({names})\b", clause)
        if m:
            stages.append([INGREDIENTS[m.group(1)], clause])
        elif stages:
            stages[-1][1] += " " + clause
    if not stages:
        raise ValueError(f"no known ingredient in {prompt!r} (know {sorted(INGREDIENTS)})")
    steps: list[Step] = []
    for ingredient, text in stages:
        region = next((REGIONS[r] for r in sorted(REGIONS, key=len, reverse=True) if r in text), None)
        targets = region or DEFAULT_TARGETS[ingredient]
        count = _count(text) or len(targets)
        steps += [Step(ingredient, targets[k % len(targets)]) for k in range(count)]
    return steps


LLM_SYSTEM = (
    "You plan pizza topping for a robot arm. Reply with JSON only: a list of steps, each "
    '{"ingredient": "cheese"|"pepperoni", "target": "top-left"|"top-right"|"bottom-right"|'
    '"bottom-left"|"center"|"gap"}. One step = one piece. Keep the recipe\'s ingredient order. '
    "Spread pieces clockwise over the allowed quarters (top-left, top-right, bottom-right, "
    "bottom-left). If no count is given: cheese = 8 pieces (two laps), pepperoni = 6 pieces "
    "(four quarters, center, gap)."
)


def plan_llm(prompt: str, model_id: str, device: str = "cuda") -> list[Step]:
    """Ask an open-source instruct LLM for the step list; fall back to rules on any problem."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        msgs = [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=600, do_sample=False)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        del model
        torch.cuda.empty_cache()
        raw = json.loads(reply[reply.index("["): reply.rindex("]") + 1])
        steps = [Step(INGREDIENTS[s["ingredient"]], s["target"]) for s in raw]
        if not steps or any(s.target not in (*QUARTERS, CENTER, GAP) for s in steps):
            raise ValueError(f"bad targets in {raw}")
        return steps
    except Exception as e:  # noqa: BLE001 - any LLM failure means: use the rules
        print(f"planner: LLM plan failed ({e}); using rules")
        return plan_rules(prompt)


def describe(steps: list[Step]) -> str:
    return "\n".join(f"  {i + 1:>2}. {s.prompt}" for i, s in enumerate(steps))
