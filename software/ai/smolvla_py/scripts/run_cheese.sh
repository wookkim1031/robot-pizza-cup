#!/usr/bin/env bash
# One-click SmolVLA run on the local station (cheese-fixed-s1 checkpoint).
#
#   scripts/run_cheese.sh            # auto: predict + send continuously, Ctrl-C to stop
#   scripts/run_cheese.sh --step     # SPACE = one step, q = quit
#   TASK="..." scripts/run_cheese.sh # different prompt
#
# Stops teleop with SIGKILL first, so the follower keeps torque and holds its pose
# (a normal Ctrl-C of teleop disables torque and the arm drops). Start from the home pose.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-spread cheese evenly on pizza}"
CHECKPOINT="${CHECKPOINT:-checkpoints/smolvla-cheese-fixed-s1-2026-10-24/final}"
FOLLOWER_BUS=5B3D050088

mode=(--auto)
[[ "${1:-}" == "--step" ]] && mode=()

if ! (exec 3<>/dev/tcp/127.0.0.1/8888) 2>/dev/null; then
    echo "station is not running on :8888 - start it first (cd ~/Downloads && ./station --web --tcp 127.0.0.1:8888)"
    exit 1
fi

if pgrep -f "main.py --leader-server" >/dev/null; then
    echo "stopping teleop (follower keeps holding)"
    pkill -KILL -f "main.py --leader-server" || true
    sleep 1
fi

exec uv run python scripts/run_policy.py \
    --checkpoint "$CHECKPOINT" \
    --task "$TASK" \
    --bus-serial "$FOLLOWER_BUS" \
    --motor-ids 1,2,3,4,5,6,7,8 \
    --max-delta-ticks 1000 \
    --camera-order auto \
    "${mode[@]}"
