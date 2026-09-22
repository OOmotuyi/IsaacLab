# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Released MicroDuck behavior contracts and demonstration command schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Behavior:
    """Playback contract for one released checkpoint.

    Args:
        robot: Converted robot asset stem.
        action_scale: Joint target offset scale [rad].
        period: Phase-command period [s]; zero for non-phase policies.
        active_time: One-shot policy duration [s]; zero for continuous policies.
        fallback: Policy used before and after a one-shot.
    """

    robot: str = "robot_groundcontact"
    action_scale: float = 1.0
    period: float = 0.0
    active_time: float = 0.0
    fallback: str = "alpha_stand"

    @property
    def rollers(self) -> bool:
        """Whether the articulation includes four passive wheels."""
        return self.robot == "robot_groundcontact_rollers"

    @property
    def root_height(self) -> float:
        """Initial trunk height [m]."""
        return 0.1385 if self.rollers else 0.12

    @property
    def cycle_time(self) -> float:
        """Demonstration duration before an explicitly requested reset [s]."""
        return 10.0 if not self.active_time else 1.0 + self.active_time + 4.0


# The release manifest overrides the roller ONNX action_scale metadata (1.0).
BEHAVIORS = {
    "alpha_walking": Behavior(robot="robot_walk"),
    "velstand": Behavior(robot="robot_allcollisions"),
    "alpha_stand": Behavior(),
    "alpha_sitstand": Behavior(),
    "alpha_ground_pick": Behavior(period=4.0, active_time=2.8),
    "ball_kick_left": Behavior(active_time=0.5),
    "ball_kick_right": Behavior(active_time=0.5),
    # Match the upstream simulator: 2 s permits a full roll before standing takes over.
    "roulade": Behavior(active_time=2.0),
    "roller": Behavior(robot="robot_groundcontact_rollers", action_scale=0.8),
    "roller_crouch": Behavior(
        robot="robot_groundcontact_rollers", action_scale=0.8, period=5.0, active_time=3.5, fallback="roller"
    ),
}


def policy_command(
    behavior: str, elapsed: float, velocity: tuple[float, float, float]
) -> tuple[str, tuple[float, ...]]:
    """Return active checkpoint and its 13 command values.

    Args:
        behavior: Released checkpoint name.
        elapsed: Time since the demonstration reset [s].
        velocity: Walking twist [m/s, m/s, rad/s]. For rollers these slots
            mean push/brake command, unused lateral command, and heading error [rad].
    """
    profile = BEHAVIORS[behavior]
    command = [0.0] * 13
    active = behavior
    if profile.active_time:
        phase_time = elapsed - 1.0  # Settle upright before triggering the behavior.
        if phase_time < 0.0 or phase_time >= profile.active_time:
            active = profile.fallback
        elif profile.period:
            angle = 2.0 * math.pi * phase_time / profile.period
            command[:2] = [math.cos(angle), math.sin(angle)]
    elif behavior == "alpha_sitstand":
        # The policy is trained on a binary posture flag, not a velocity ramp.
        command[0] = float(2.0 <= elapsed < 6.0)
    elif behavior in ("alpha_walking", "velstand", "roller"):
        command[:3] = velocity
    return active, tuple(command)
