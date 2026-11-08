# Robot Pizza Cup: everything we changed (Oct 23–24, 2026)

Sources: all 11 Claude sessions in this project since the hackathon started, checked against
`git status` and the files on disk on Oct 24 at about 18:20. All times are **local (CEST)**.

Short version: we did **not** modify the station binary or any Rust/TS code. Our changes are the
teleop example (Python), one line-level fix in the SmolVLA trainer, a new `act_py` project
(our own ACT pipeline), a CLI tagger, camera/exposure settings, calibrations, datasets, and
training runs on the Nebius boxes. **Nothing has been committed yet.**

---

## 1. Changes to stock norma-core files

### Teleop example: `software/station/examples/st3215-remote-teleop-py/`
Branch `teleop/per-joint-range-overrides`. The changes are **staged but not committed**: the commit
on Oct 23 at 18:52 failed because no git user.name/email is set.

| File | Change | Why |
|---|---|---|
| `config.py` | Added `DEFAULT_RANGE_OVERRIDES_PATH`, `RangeOverrides`, `load_range_overrides()`, `save_range_overrides()` | Per-joint calibration that doesn't touch the motors |
| `config.py` | `per_motor_current_threshold` changed from `{}` (all 100) to **`{2: 250, 3: 250, 4: 250}`** | The lifting joints tripped overload protection in normal use (M2 peaked at 160, M4 tripped at 104). Stall is about 2.7 A. M3 got 250 by analogy and was never measured. |
| `main.py` | New `--range-overrides` flag. `_apply_range_override()` swaps in the custom min/max for each motor. Startup logs `Range override: bus … M… -> …` | Wires the overrides into mapping |
| `mirror.py` | Logs a warning when overload protection blocks a joint and an info line when it releases | Overload stops used to be silent |
| `README.md` | Documents `--range-overrides` and `record_range.py` | — |
| `.gitignore` (new) | Ignores `range_overrides.json` | Arcs are per robot |

### SmolVLA trainer: `software/ai/smolvla_py/scripts/train.py` (unstaged, uncommitted)
- Removed the hard-coded `STATE_DIM = 6` and `ACTION_DIM = 6`. Both now come from `train_ds.n_joints` (8 on our ElRobot).
- **Why it matters:** `action_dim` slices the loss, so the stock 6 silently dropped M7 and M8 (wrist roll and gripper) from training.
- The startup print now shows `joints: N`. `run_policy.py` is unchanged; it already reads the joint count from the arm.

---

## 2. New scripts (ours)

| Path | Git status | What it does |
|---|---|---|
| `…/st3215-remote-teleop-py/record_range.py` | staged | Records one joint's min/max while you move it by hand. Read-only to the motors. Samples at 50 Hz. `uv run python record_range.py --server localhost --bus <serial> --motor N [--clear]` writes to `range_overrides.json`. |
| `…/st3215-remote-teleop-py/tag.py` | untracked | CLI replacement for the station's TAG button. Posts to queue `inference-states`. Tag format: `<task> \| <session> \| R<n> \| E<n> \| <OP> \| START/END/FAIL <reason>`. 4 episodes per round, with ended/failed counters and a round timer. State file: `~/datasets/tag_state.json`. Commands: `start`, `end`, `fail <reason>`, `round`, `operator NAME`, `set …`, `status`. |
| `software/ai/act_py/` (whole project) | untracked | Our ACT pipeline. `pyproject.toml` pins `lerobot[dataset,training]==0.6.1` and Python `>=3.12,<3.13`, with a `uv.lock`. `.gitignore` covers `.venv/` and `outputs/`. |
| `act_py/scripts/convert_to_lerobot.py` | untracked | Converts norma parquet to a LeRobotDataset v3: state = `position_norm`, action = `goal_norm`, `images[N]` → `camN`, 10 fps. |
| `act_py/scripts/export_dataset.py` | untracked | End-to-end export. Reads the tags (FAIL drops a take, FAIL RETRO cancels an earlier END), runs `dataset-generator` per take, normalizes cameras to cam0 = scene and cam1 = wrist, then writes parquet + manifest + sha256, the LeRobot copy, an HF push, and copies to the VMs. Key flags: `--tag-task --session --min-round --max-round --task --name --vm (repeatable) --no-hub --existing --dry-run --hf-user johan1031 --public`. |
| `act_py/scripts/run_act.py` | untracked | ACT runner for the robot, ported from smolvla's `run_policy.py`. Engages safely (goal = present position, then speed/accel, then torque). Flags: `--bus-serial`, `--checkpoint`, `--auto`, `--dry-run`, `--max-ticks`, `--max-delta-ticks` (default 200; **use 1000**, since 200 blocked the arm), `--camera-order` (use `1,0` when the station sends wrist first), `--release-on-exit`. **The "act version 2" session is editing it right now** to run the 4 quadrant checkpoints in sequence, read image keys from each checkpoint, and add `--n-action-steps`. |
| `act_py/scripts/recover_raw_episodes.py` | untracked | Abandoned and never tested (an attempt to rebuild cheese R1–R2 from raw queues). Safe to delete. |
| `~/Downloads/camera_settings.sh` | outside repo | Sets manual exposure on both KD-USB cameras (defaults: exposure 300, gain 60, overridable with env vars). Run it with the station stopped. **Today's values differ (see §3).** |

Throwaway diagnostics such as `probe.py`, `which_moves.py`, `busdiag.py`, the camera overlay page and `grab_normvla.py` were only ever in Claude scratchpads, not the repo.

---

## 3. Settings and hardware configuration

**Arm roles.** Leader `5B61034559`, follower `5B3D050088`. They were swapped at first on both days.
`~/Downloads/debug_teleop.py` **still has them backwards** (we never edited it).

Teleop command:
```
uv run python main.py --leader-server localhost --leader-bus 5B61034559 --follower-server localhost --follower-bus 5B3D050088
```

**Calibration.** Done by the user in the station UI. Station calibration always resets and rewrites all 8 motors; there is no per-motor option.
- Oct 23, 18:07–18:16: leader recalibrated manually (no Auto) twice. M4 went from 1545–2551 to 902–3194.
- Oct 23, about 19:12–19:20: **both arms recalibrated**. Leader / follower:

  | | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 |
  |---|---|---|---|---|---|---|---|---|
  | Leader | 1108–2988 | 1003–3093 | 899–3197 | 898–3198 | 135–3961 | 1044–3052 | **2–4094** ⚠️ | 1414–2682 |
  | Follower | 1107–2989 | 994–3102 | 908–3188 | 909–3187 | 192–3904 | 1036–3060 | 146–3950 | 1266–2830 |

  Leader M7 came out as a full circle, which is unreliable. No later session recorded a fix.
- Range overrides: follower M7 was recorded at 131–3977, then **cleared** (user: "do not change m7"). `range_overrides.json` is `{}`, so no overrides are active.

**Cameras.** Exposure is set with v4l2 while the station is stopped. It is lost when a camera is unplugged or the laptop reboots.

| When | Scene cam | Wrist cam | Notes |
|---|---|---|---|
| Oct 23, 20:17 | exposure 300 / gain 60 | 300 / 60 | via `camera_settings.sh` |
| **Oct 24, 14:19–14:21 (current standard)** | **350 / gain 30** | **500 / gain 25** | `auto_exposure=1`, `exposure_dynamic_framerate=0`. Re-applied at 17:20 after a replug. |
| Oct 24, 17:30 (reverted) | 70 / 30 | 110 / 25 | only for the 640×480 trial |

- On Oct 23 the laptop's own webcam (`/dev/video0`) was changed by mistake, then reverted to auto.
- Physical changes: one arm camera moved off the shared USB hub to a direct laptop port (the hub starved one camera); the follower's M6–M7 cable was reseated.
- **Camera order flips between station restarts.** Right now cam0 = wrist and cam1 = scene. `export_dataset.py` normalizes it to scene first, so run the policy with `--camera-order 1,0` when needed.

**`~/Downloads/station.yaml`** is back to the original: `formats: []` (auto 320×240), `resize_target: 224`, `update-interval: 100ms` (10 fps recording).
- On Oct 24, 17:29–17:35 we tried 640×480 and reverted it. Backups: `station.yaml.bak-320x240` (original) and `station.yaml.640x480`.
- normvla is hardcoded to 224×224 in Rust (`normvla.rs:482`), and this machine has no cargo to rebuild it.
- The station binary is unchanged (0.1.0-beta.10), run as `./station --web --tcp 127.0.0.1:8888` from `~/Downloads`.
- Recording at 30 fps was proposed but never done.

**Machine and accounts**
- `~/.ssh/id_ed25519` was created Oct 24 at 12:23 for the Nebius boxes. Both hosts are in `known_hosts`.
- Hugging Face: logged in as `johan1031`, but the **token is read-only**, so pushes fail with 403. It is not confirmed whether this is the team account.
- Local venvs: `smolvla_py/.venv` (`uv sync`, torch 2.11 cu130, 4.8 GB) and `act_py/.venv` (Python 3.12, lerobot 0.6.1). System CUDA and the driver were not touched.
- Claude memory notes are in `~/.claude/projects/-home-johan-norma-core/memory/`. A stray copy from day 1 is in a scratch-workspace memory folder.

---

## 4. Nebius devlab (inside the Jupyter container as `jovyan`)

gpu1 = `81.85.4.50`, gpu2 = `81.85.0.200`. Each has 1× RTX PRO 6000 Blackwell.

**Setup**
- Both boxes:
  - uv installed in `~/.local/bin`; `~/models` and `~/logs` created.
  - Started `hf download lerobot/pi05_base` into `~/models/pi05_base` at 15:00. **Not verified to have finished.**
- `~/norma/act_py` (gpu1) and `~/norma/smolvla_py` (gpu2) were copied up with `uv sync --python 3.12`.
- Not done: accepting the PaliGemma licence and logging into HF on the boxes (π0.5 needs both).

**Datasets in `~/datasets/` on both boxes**
- `cheese-fixed-s1-2026-10-24`: the LeRobot conversion was done on gpu1.
- `salami-s3-2026-10-24`, plus its LeRobot copy.
- The quadrant data `~/act_data/<q>` on gpu2 is the teammate's.

**Training runs**

| Box | Run | Command essentials | Result |
|---|---|---|---|
| gpu1 | ACT `act-cheese-fixed-s1-2026-10-24` | `lerobot-train --policy.type=act --policy.chunk_size=20 --policy.n_action_steps=20 --dataset.image_transforms.enable=true --batch_size=32 --steps=30000 --num_workers=22`, written to `~/runs/…` | Finished 17:33, loss 0.068. `030000` downloaded to `act_py/outputs/act-cheese-fixed-s1-2026-10-24/`. Robot test: the first run missed the box; the second placed two cheeses in the same quadrant. |
| gpu2 | SmolVLA `smolvla-cheese-fixed-s1-2026-10-24` | `scripts/train.py --steps 20000 --batch-size 64 --num-workers 12 --warmup-steps 1000 --decay-steps 20000 --save-every 2000`, written to `~/runs/…` | At step 18k at 18:10 (loss about 0.02). **Not downloaded and not tested on the robot.** |
| gpu2 | 4× per-quadrant ACT (teammate, lerobot 0.3.3) | `~/outputs/act_<quadrant>`, chunk 100, 30k steps | Finished 18:04. Downloaded and **migrated to lerobot 0.6.1** in `act_py/outputs/act-quadrants-2026-10-24/<q>/`. Offline check: each model fits its own quadrant best. |

Local (laptop GPU) runs:
- `smolvla_py/checkpoints/smoke`: 200-step smoke test only.
- `act_py/outputs/act-cheese-pizza-2026-10-24`: stopped at 14.8k steps, trained on retired data.

**Not started:** SmolVLA on `cheese-quadrants.parquet` ("Path 1").

---

## 5. Datasets (`~/datasets/`)

| File | Episodes / frames | Task string | Status |
|---|---|---|---|
| `test-01.parquet` (Oct 23) | 1 | "put the cheese on the pizza" | test only |
| `cheese-01.parquet`, `cheese-01-good.parquet` (Oct 23) | 3 / 2 eps | "put **cheese** on the pizza" ⚠️ different string | old |
| `cheese-pizza-2026-10-24-clean.parquet` + LeRobot | 21 / 7046 | "put the cheese on the pizza" | **retired** (old layout / real dough) |
| `cheese-fixed-s1-2026-10-24.parquet` + LeRobot | 49 / 11,122 | "put the cheese on the pizza" | **main cheese set.** Rounds R3–R15. R1–R2 were lost to a frozen normvla stream. 3 takes dropped for frame gaps, 2 retro-failed. |
| `salami-s3-2026-10-24.parquet` + LeRobot | 21 / 8336 | "put the salami on the pizza" | **main salami set.** R1–R6. R1–R4 use the old exposure. The good "s4" take (counts as R4E4) is **not** included. R06E3 is twice the normal length; check it. |
| `cheese-pizza-2026-10-24-s1-r1r2-recovered.parquet` | 4 bytes | — | junk, delete |
| `tag_state*.json` | — | — | tagger state plus per-session backups (cheese-s1, salami s1-discarded, s2, s3) |

The station keeps only about 2 GB (about 2 hours) of normvla frames, so **export right after recording**.

---

## 6. Conventions we agreed on
- **Layout:** fixed for challenges 1–2; challenge 3 varies the pizza position and box order. Cheese and pepperoni ("salami") only.
- **Episodes:** one piece = one episode, 4 episodes per round (one per quadrant), no pizza reset within a round.
- **Tagging via chat:** `start` / `end` / `fail:<reason>` / `es` (end, then start) / `fs` (fail, then retry) / `operator NAME`. Claude replies with the tag and the status line. Labels are success/fail plus a reason.
- **Recording:** park the leader still for about 8 s between demos (the exporter splits on a 5 s pause). Keep hands out of frame and reset the same way every time.
- **Startup order:** replug cameras → set exposure → start the station → start teleop → check frames → tag. Restart the station after any USB replug.
- **Models:** ACT first, one model per ingredient. Code above the policy (sequencing, choosing a checkpoint) is allowed; hard-coded trajectories are not. Model freeze is **Sun Oct 25, 11:00**.

---

## 7. Loose ends
1. **Nothing is committed.** To commit, set a git identity first. Teleop changes are staged; `train.py` is unstaged; `act_py/` and `tag.py` are untracked. `smolvla_py/checkpoints/` and `smolvla.egg-info/` are not gitignored.
2. HF upload: started at about 18:18 in the salami session. It will fail with 403 unless someone runs `hf auth login` with a write token.
3. Leader M7 calibration (2–4094) needs redoing.
4. The SmolVLA checkpoint on gpu2 has not been downloaded.
5. `run_act.py` quadrant sequencing is in progress in the "act version 2" session.
6. Swap the serials in `~/Downloads/debug_teleop.py`.
7. Confirm the `pi05_base` download finished.
