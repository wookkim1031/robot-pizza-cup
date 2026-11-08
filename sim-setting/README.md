# Pizza rig: simulation setting

Everything we have on file for rebuilding the Robot Pizza Cup rig in simulation.
**What exists** is in this folder. **What's missing** (dimensions, camera poses,
intrinsics) was never measured and has to be measured on the rig; see the end of this file.

## 1. Robot model: ElRobot follower, 8 × Feetech ST3215

- `robot/elrobot_follower.urdf` with STL meshes in `robot/assets/` (copied from
  `hardware/elrobot/simulation/` in norma-core). Full CAD (STEP) is in
  `hardware/elrobot/STEP/`, and the whole arm is in `ElRobot.stp`.
- Joints: `rev_motor_01` … `rev_motor_07` are revolute. The gripper motor is
  `rev_motor_08` (revolute, 0 … 2.2028 rad), which drives two prismatic jaws:
  `rev_motor_08_1` (0 … 0.0255 m) and `rev_motor_08_2` (−0.0255 … 0 m).
- URDF limits, in rad:

| motor | URDF joint | lower | upper | span |
|---|---|---|---|---|
| m1 (base yaw) | rev_motor_01 | −1.5509 | 1.5509 | 3.102 |
| m2 | rev_motor_02 | −1.6122 | 1.6122 | 3.224 |
| m3 | rev_motor_03 | −1.7610 | 1.7610 | 3.522 |
| m4 | rev_motor_04 | −1.7533 | 1.7533 | 3.507 |
| m5 | rev_motor_05 | −2.6200 | 3.2520 | 5.872 |
| m6 | rev_motor_06 | −1.3775 | 1.7641 | 3.142 |
| m7 (wrist roll) | rev_motor_07 | −3.2014 | 2.7336 | 5.935 |
| m8 (gripper) | rev_motor_08 | 0 | 2.2028 | 2.203 |

- The station's 3D viewer places the URDF with base rpy = (−π/2, 0, −π/2) at the origin
  (`station-viewer/src/devices/elrobot/config.ts`).
- Effort limit is 2.94 N·m and velocity limit 4.71 rad/s per joint, as written in the URDF.

## 2. Calibration and the m1–m8 values (`calibration_follower.json`)

The calibration lives in the servos' EEPROM. The station writes an offset so that each
joint's range is centered on tick 2048, and every recorded frame carries the resulting
range. The ranges are identical in all 9 datasets recorded on 2026-10-24
(follower bus `5B3D050088`):

| motor | range_min | range_max | span (ticks) | span (rad, 4096 ticks/rev) | median start pose (norm) |
|---|---|---|---|---|---|
| m1 | 1107 | 2989 | 1882 | 2.887 | 0.50 |
| m2 | 994 | 3102 | 2108 | 3.234 | 0.01 |
| m3 | 908 | 3188 | 2280 | 3.497 | 0.48 |
| m4 | 909 | 3187 | 2278 | 3.494 | 0.96 |
| m5 | 192 | 3904 | 3712 | 5.694 | 0.98 |
| m6 | 1036 | 3060 | 2024 | 3.105 | 0.40 |
| m7 | 146 | 3950 | 3804 | 5.835 | 0.28 |
| m8 | 1266 | 2830 | 1564 | 2.399 | 0.87 (open) |

- **The m1–m8 values in datasets and policies** are
  `position_norm = (position_ticks − range_min) / (range_max − range_min)`, in [0, 1].
  This was checked exactly against the data. The state is the follower's `position_norm`;
  the action is `goal_norm`, the leader's command, which runs about 0.3 s ahead of the state.
- **Norm to URDF angle, as the station viewer does it:** `q = lower + norm × (upper − lower)`
  using the URDF limits above. The gripper is mirrored: `rev_motor_08 = upper − q`.
- **Physical alternative:** `q = (ticks − 2048) × 2π/4096`. The calibrated spans match the
  URDF spans to within 1% for m2, m3, m4 and m6, and to within 2–9% for m1, m5, m7 and m8.
  So the two mappings nearly agree, except m5–m7, whose URDF zero isn't the range center.
  **Sign and zero against the real arm haven't been checked.** To check, put the arm in the
  start pose above and compare it with the URDF rendered at those values.
- Gripper: norm about 0.03 means closed, about 0.9 means open.
- Robot control rate is 10 Hz (`update-interval: 100ms`). Goals are sent as raw ticks:
  `ticks = range_min + norm × (range_max − range_min)`.

## 3. Cameras

- Two Sonix "KD-USB Camera" units (USB id 2c7f:4a60). Since about 19:16 on 2026-10-24 they
  capture at 640×480 MJPG, 30 fps.
- **Every policy and dataset image is squashed to 224×224.** The 4:3 frame is resized
  without keeping the aspect ratio; this is hardcoded in station `normvla.rs`. A simulator
  should render 4:3 and then resize to 224×224 the same way.
- **Scene camera ("top"):** fixed in front of the table, facing the robot. It's raised and
  looks down at roughly 40–50°. The arm's base is on the far side of the pizza. See
  `reference_images/*_scene_224.jpg`.
- **Wrist camera:** on the gripper, looking along the jaws (the jaws are the dark band at
  the bottom of the image). The image appears rotated about 90° relative to the gripper
  axis. Mount CAD: `robot/CameraMount_*.stp`; which of the three variants is installed
  hasn't been confirmed.
- Exposure is manual (exposure 100, gain 40, auto white balance, 50 Hz anti-flicker).

## 4. Scene (from the reference images)

- From the scene camera: the pizza is in the middle of a hexagonal brown paper sheet,
  taped to a white table. The pepperoni bowl (red slices) is to the left of the pizza and
  the cheese tray (shredded white) to the right; the robot sits behind the pizza.
- The pizza in the 224×224 scene image: center at about (116, 142) px, radius about 35 px.
- **Quarter names are from the robot/operator side, rotated 180° from the camera view:**
  "top-left" is the bottom-right of the scene image.

## 5. Missing (needs a tape measure and/or a calibration board)

1. **Table:** size and height, plus the robot base's position and yaw on it.
2. **Pizza:** diameter and thickness, and its center relative to the robot base.
3. **Containers:** size of the pepperoni bowl and cheese tray (L×W×H) and their positions
   relative to the robot base. Pepperoni slice diameter and thickness, and the size of a
   cheese pinch.
4. **Scene camera pose:** position (x, y, z) and orientation relative to the robot base.
5. **Wrist camera pose:** which mount is used, plus its offset and rotation on the gripper.
6. **Camera intrinsics:** not calibrated. Take a checkerboard capture at 640×480 for each
   camera (`camera_setup.sh --snap DIR` saves frames).
7. **Leader arm:** not modelled. Only the follower matters for simulation. The leader's M7
   calibration is a full circle (2–4094) and still needs redoing.
