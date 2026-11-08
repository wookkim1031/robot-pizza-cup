# [TEAM-N]-[TEAM-NAME]: write-up

## What we built

One SmolVLA checkpoint drives the ElRobot arm for both toppings. A small sequencer above it
(`software/ai/smolvla_py/scripts/run_challenge.py`) turns a challenge or recipe into steps
(cheese x8 laps over the four quadrants; pepperoni x6: four quadrants, center, largest empty area),
sends one prompt per step, and moves on when the gripper has released and the arm is back near
home. Every joint command comes from the policy; the sequencer only picks prompts, decides "done",
and retries a pepperoni step when an image change checker sees no new slice.

Each topping is two skills, chosen by prompt: a shared **grasp** ("Pick up one pepperoni slice from
the bowl.") and a per-target **place** ("Place the pepperoni slice you are holding on the
upper-left quadrant of the pizza and return to the start position."). We got these by splitting
every recorded demo at the lift-off after the gripper closes.

## Data

274 teleop demos (8 datasets) recorded at the event with the leader/follower arms, 10 fps, scene + wrist
cameras at 224 px, one topping placement per episode, home pose to home pose. A CLI tagger
(`tag.py`) marked start/end/fail live, and an exporter dropped failed takes and takes with frame gaps.
Later sets are "grasp-focused": slow straight descent, short stop, close, straight lift.

## What we tried

- **ACT** (lerobot), one model per ingredient and then one per quadrant: reached the bowl but
  placed in the wrong quadrant or missed the box.
- **SmolVLA, one prompt per whole episode** (joint model, 20k steps): found the bowl and the
  quadrant, but hovered at the bowl or grabbed badly. Re-planning every frame made the arm barely
  move; playing 1 s of each 50-step chunk fixed that.
- **Grasp-focused demos** + fine-tune: grabs became reliable, but the policy copied the pause after
  closing and stalled. We skip the still prefix of a plan while holding ("hold kick").
- **Grasp/place split**: the biggest win. The grasp skill no longer has to know the target, and the
  place skill always starts in the air with the topping in hand.
- **Change checker** (ResNet18 on before/after crops of the pizza): detects a missed pepperoni
  slice so the sequencer retries the step. It does not work for cheese.
- **Recipe parsing for Ch3 Part B**: rules turn text like "4 pepperoni on the right half, then
  cheese over the whole pizza" into steps; an optional LLM agent (Claude API, or Qwen via Ollama)
  was built but is not used by default.

## What worked

- grasp5-2 (step 3000), Ch2 run: all 8 cheese portions grasped and spread in 134 s
  (7-9 s per grasp, 5-12 s per place), well inside the cheese budget.
- Pepperoni: grasps succeed almost every time; most misses happen at the release and are caught by
  the checker and retried (3 slices confirmed in 87 s in a Ch1 run, one of them on the second attempt).

## What did not work (yet)

- Cheese sometimes comes in from the side (old demo style) instead of top-down.
- The checker can't judge cheese, and it assumes the fixed pizza spot, so it is off for Ch3.
- Moving pizza layouts (Ch3) were only partly covered by data.
