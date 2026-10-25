"""Teleop tuning constants and per-motor config."""

import json
from dataclasses import dataclass, field
from pathlib import Path

# How often to recompute follower goals from the latest leader state.
TELEOP_REFRESH_INTERVAL_S = 0.020  # 50 Hz

# Inference data older than this is considered stale and skipped.
MAX_DATA_AGE_NS = 100_000_000  # 100 ms

# Per-joint calibrated arcs that replace the station's range_min/range_max.
# Written by record_range.py. Keyed by bus serial so a leader/follower swap on
# the command line can't apply an arc to the wrong arm.
DEFAULT_RANGE_OVERRIDES_PATH = Path(__file__).parent / "range_overrides.json"

# bus serial -> motor id -> (range_min, range_max)
RangeOverrides = dict[str, dict[int, tuple[int, int]]]


def load_range_overrides(path: Path) -> RangeOverrides:
    """Read the overrides file. A missing file means no overrides."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        serial: {int(mid): (int(arc[0]), int(arc[1])) for mid, arc in motors.items()}
        for serial, motors in raw.items()
    }


def save_range_overrides(path: Path, overrides: RangeOverrides) -> None:
    raw = {
        serial: {str(mid): list(arc) for mid, arc in sorted(motors.items())}
        for serial, motors in sorted(overrides.items())
        if motors
    }
    path.write_text(json.dumps(raw, indent=2) + "\n")


@dataclass
class MotorConfig:
    """Per-motor tuning shared across all motors on the follower bus."""

    # Ignore movement smaller than this many encoder steps. Stops the follower
    # from twitching when the leader is "still" but jittering by 1-2 ticks.
    deadband: int = 20

    # Pad each end of the follower's calibrated arc by this much, so we never
    # command the follower into a hard stop.
    safety_margin: int = 20

    # Distance-proportional speed/accel envelope. The further the follower has
    # to travel, the faster it goes.
    min_speed: int = 300
    max_speed: int = 3300
    min_accel: int = 5
    max_accel: int = 100

    # Encoder resolution (4096 for ST3215).
    max_steps: int = 4096

    # Default current threshold (raw units). When the follower's measured
    # current exceeds this, freeze the goal at the present position to prevent
    # overload. 0 disables protection.
    current_threshold: int = 100

    # Per-motor overrides for current_threshold. Keyed by motor id.
    # The lifting joints carry the arm's weight and trip the default in normal
    # use (measured M2 peak 160 on fast moves, M4 tripped at 104). 250 (~1.6 A)
    # clears that but still stops well short of a stall (~2.7 A).
    per_motor_current_threshold: dict[int, int] = field(
        default_factory=lambda: {2: 250, 3: 250, 4: 250},
    )

    def get_current_threshold(self, motor_id: int) -> int:
        return self.per_motor_current_threshold.get(motor_id, self.current_threshold)
