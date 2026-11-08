# Robot Pizza Cup — Team 3 (RoboIT)

This repository is our fork of [norma-core](https://github.com/norma-core/norma-core) from the Robot Pizza Cup
hackathon (Cologne, 2026-10-23 to 25): a SmolVLA policy on the ElRobot arm (8 joints, 10 Hz) that puts cheese
and pepperoni on a pizza. The original NormaCore README follows below.

- **Our code:** `software/ai/smolvla_py/` (pizza skills, sequencer, agent, web UI, `scripts/run_challenge.py`),
  `software/ai/act_py/` (ACT pipeline and dataset export), `software/station/examples/st3215-remote-teleop-py/`
  (teleop changes, `tag.py`). Everything we changed: [`HACKATHON_CHANGES.md`](HACKATHON_CHANGES.md).
- **Frozen commit** (the one in the submission form): `29496c09b9dfa5115d1e6c8367d2b8b30a3bd153`. The commit after it
  adds the last fixes from the demo laptop, the hand-off notes, `submission/` and `sim-setting/`.
- **Submission:** [`submission/SUBMISSION.md`](submission/SUBMISSION.md), write-up [`submission/WRITEUP.md`](submission/WRITEUP.md).

## Models and datasets on Hugging Face: [RoboIT1234](https://huggingface.co/RoboIT1234)

| Model | What |
|---|---|
| [`RoboIT1234/TEAM-3-RoboIT-smolvla-grasp5-2-2026-10-25`](https://huggingface.co/RoboIT1234/TEAM-3-RoboIT-smolvla-grasp5-2-2026-10-25) | **Submitted model.** SmolVLA fine-tuned from `lerobot/smolvla_base`, checkpoint grasp5-2 step 3000. One checkpoint for both toppings; grasp and place skills chosen by prompt. |
| [`RoboIT1234/smolvla-grasp5-2-var-2026-10-25`](https://huggingface.co/RoboIT1234/smolvla-grasp5-2-var-2026-10-25) | Follow-up: grasp5-2 trained further on real layout-variation rounds plus 760 MuJoCo demos on randomized layouts. Handles moderate pizza shifts; with the trays swapped it still reaches for the usual tray. |

Both are in the `smolvla_py` format (`config.json`, `model.safetensors`, `stats.safetensors`), not a `lerobot`
policy folder; load them with `software/ai/smolvla_py` from this repository. Prompts and inputs are on the model cards.

```bash
cd software/ai/smolvla_py
hf download RoboIT1234/TEAM-3-RoboIT-smolvla-grasp5-2-2026-10-25 --local-dir checkpoints/grasp5-2
uv run python scripts/run_challenge.py ch2 --checkpoint checkpoints/grasp5-2
```

Datasets (LeRobot v3, cameras `top` = scene and `front` = wrist, 8 joint states and actions):

| Dataset | Episodes | Content |
|---|---|---|
| [`RoboIT1234/TEAM-3-RoboIT-cheese-r1r10-2026-10-24`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-cheese-r1r10-2026-10-24) | 63 | cheese, full episodes, rounds 1–10 |
| [`RoboIT1234/TEAM-3-RoboIT-pepperoni-r1r10-2026-10-24`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-pepperoni-r1r10-2026-10-24) | 60 | pepperoni, full episodes, rounds 1–10 |
| [`RoboIT1234/TEAM-3-RoboIT-pepperoni-on-cheese-r1r5-2026-10-24`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-pepperoni-on-cheese-r1r5-2026-10-24) | 30 | pepperoni placed on cheese, rounds 1–5 |
| [`RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-r1r3-2026-10-25`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-r1r3-2026-10-25) | 18 | pepperoni grasp-focused demos |
| [`RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-on-cheese-r1r2-2026-10-25`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-on-cheese-r1r2-2026-10-25) | 12 | pepperoni grasp-focused demos on cheese |
| [`RoboIT1234/TEAM-3-RoboIT-cheese-grasp-r1r3-2026-10-25`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-cheese-grasp-r1r3-2026-10-25) | 11 | cheese grasp-focused demos |
| [`RoboIT1234/TEAM-3-RoboIT-cheese-grasp-new-2026-10-25`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-cheese-grasp-new-2026-10-25) | 40 | cheese, new top-down grasp demos |
| [`RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-new-2026-10-25`](https://huggingface.co/datasets/RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-new-2026-10-25) | 40 | pepperoni, new top-down grasp demos |

Checkpoints, training outputs and raw recordings are not in this repository.

---

# NormaCore

### The Unified Toolkit for Physical System Development & Operations

**NormaCore** is a unified toolkit designed to facilitate the development and deployment of physical systems. From complex robotics to distributed sensor networks and hobby projects, the system provides a solid foundation to manage them all. To achieve this goal, the platform combines a unified API, high-performance data pipelines, and visual tooling to help you build and manage your entire ecosystem as one.

**Developer experience sits at the heart of our design philosophy.**

To fully realize the potential of this approach, we had to build a lot from scratch, rethinking traditional solutions from a practical perspective. This includes not just software, but complete hardware systems like our **7+1 DoF robotic arm** with a **parallel jaw gripper** - tools designed to open up a whole new dimension of applications for home and research robotics without significant cost or investment.

## What's inside

| Project | Path | Description |
|---|---|---|
| **ElRobot** | [`hardware/elrobot/`](hardware/elrobot/) | Fully 3D-printed 7+1 DoF robotic arm for imitation learning |
| **Parallel Jaw Gripper** | [`hardware/pgripper/`](hardware/pgripper/) | Modular gripper for the SO-101 arm |
| **Lamp** | [`hardware/lamp/`](hardware/lamp/) | Fully 3D-printed 5 DoF robotic desk lamp with an optional LED ring and web UI |
| **Station** | [`software/station/bin/station/`](software/station/bin/station/) | Real-time robotics platform — data collection, inference, control. Single binary, web UI |
| **SmolVLA fine-tune** | [`software/ai/smolvla_py/`](software/ai/smolvla_py/) | Train + deploy a [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) policy on the SO-101 arm |
| **Gremlin** | [`shared/gremlin_go/`](shared/gremlin_go/) · [`shared/gremlin_py/`](shared/gremlin_py/) | High-performance Protobuf SDK for Go and Python — used across the station + drivers stack |

**Website:** [normacore.dev](https://normacore.dev)

**Follow us:**
- 🐦 [X/Twitter](https://x.com/norma_core_dev)
- 🎥 [YouTube](https://www.youtube.com/@normacoredev)
- 💼 [LinkedIn](https://www.linkedin.com/company/normacore/)
- 📢 [Reddit](https://www.reddit.com/r/NormaCore/)

**Join & Contribute:**
- 💬 [Discord](https://discord.gg/Z4Ytw3QfHP) - Chat with the community
- 🐙 [GitHub](https://github.com/norma-core/norma-core) - Source code & issues
