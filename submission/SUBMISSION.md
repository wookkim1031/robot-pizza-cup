# Robot Pizza Cup submission: checklist and form answers

Guide: Notion "7 - Submission Guide". Placeholders: `[TEAM-N]`, `[TEAM-NAME]`, `[HF-ORG]`.

## Status

| Item | State | Where |
|---|---|---|
| Datasets (8, LeRobot v3, cameras `top`/`front`, dataset cards) | ready locally | `~/datasets/submission/<name>/` |
| Model (grasp5-2 step 3000 + model card) | ready locally | `~/datasets/submission/model/smolvla-grasp5-2-2026-10-25-step003000/` |
| Upload script | ready, dry-run tested | `submission/upload_to_hub.sh` |
| Write-up | draft | `submission/WRITEUP.md` |
| Demo video | waiting for phone footage | `submission/make_video.sh` |
| Frozen commit | made locally, **not pushed** | `29496c0` |
| Team number / name, HF org invite | **unknown** | from the organizers |
| HF write token | **missing** (laptop token `johan1031` is read-only) | `hf auth login` |

## Steps once the org invite arrives

1. Accept the org invite on huggingface.co with the account that will upload.
2. `software/ai/act_py/.venv/bin/hf auth login` with a **write** token.
3. Dry run, check the repo ids:
   ```bash
   HF_ORG=<org> TEAM_N=<n> TEAM_NAME=<name> ~/norma-core/submission/upload_to_hub.sh
   ```
4. Upload (about 15 GB; creates private repos, add `--public` if the organizers want them public):
   ```bash
   HF_ORG=<org> TEAM_N=<n> TEAM_NAME=<name> ~/norma-core/submission/upload_to_hub.sh --go
   ```
5. Video: `TEAM_N=<n> TEAM_NAME=<name> ~/norma-core/submission/make_video.sh phone.mp4 [start] [seconds]`
   (`SPEED=2` to fit a longer run).
6. Team lead fills in the submission form with the answers below.

## Form answers

**Dataset repo ids**
- `RoboIT1234/TEAM-3-RoboIT-cheese-r1r10-2026-10-24` (63 episodes)
- `RoboIT1234/TEAM-3-RoboIT-pepperoni-r1r10-2026-10-24` (60)
- `RoboIT1234/TEAM-3-RoboIT-pepperoni-on-cheese-r1r5-2026-10-24` (30)
- `RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-r1r3-2026-10-25` (18)
- `RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-on-cheese-r1r2-2026-10-25` (12)
- `RoboIT1234/TEAM-3-RoboIT-cheese-grasp-r1r3-2026-10-25` (11)
- `RoboIT1234/TEAM-3-RoboIT-cheese-grasp-new-2026-10-25` (40)
- `RoboIT1234/TEAM-3-RoboIT-pepperoni-grasp-new-2026-10-25` (40)

**Model repo id:** `RoboIT1234/TEAM-3-RoboIT-smolvla-grasp5-2-2026-10-25`
(`model.safetensors` sha256 `e0ff99dc401714f11d0f059f69b33b560888422408d4ee7df22d9368f2fa4232`)

**Frozen commit:** `29496c09b9dfa5115d1e6c8367d2b8b30a3bd153` (branch `hackathon/model-freeze-2026-10-25`, local only, not pushed yet)

**Video:** `TEAM-3-RoboIT.mp4` link `<link>`

**Write-up:** paste `WRITEUP.md` (short version below if the form field is small)

> One SmolVLA checkpoint (fine-tuned from lerobot/smolvla_base on 274 of our own teleop demos) runs
> both toppings. Each demo is split into a shared grasp skill and a per-target place skill, picked by
> prompt. A sequencer only chooses prompts, detects "done" and retries pepperoni steps an image change
> checker flags as missed. ACT and whole-episode SmolVLA hovered or grabbed badly; grasp-focused demos
> plus the grasp/place split made grasps reliable (8 cheese portions in 134 s in a Ch2 run).

## Freestyle declaration (only if we enter the Freestyle track)

- **Hardware:** the provided ElRobot leader/follower kit and its two cameras (scene + wrist, 320x240
  resized to 224); a laptop with an RTX 4080 for inference; training on the provided Nebius DevLab GPUs.
  No extra hardware.
- **Models:** `lerobot/smolvla_base` (with `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`); a
  torchvision ResNet18 change checker trained by a teammate on our own frames (check with them
  whether it started from ImageNet weights). Optional and off by
  default: an LLM recipe parser (Claude API or Qwen via Ollama) for Ch3 Part B.
- **External data:** none. All training data was recorded by our team at the event.

## Open questions for the organizers

- HF org name and invite link.
- Our team number and the exact team name to use in repo ids.
- Public or private repos.
