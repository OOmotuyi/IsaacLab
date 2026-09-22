# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for the shared MicroDuck motor model."""

from isaaclab_tasks.contrib.microduck.mdp.controller import (
    DEFAULT_POSE,
    JOINT_NAMES,
    MicroDuckController,
    friction_budget,
    motor_torque,
)

__all__ = ["DEFAULT_POSE", "JOINT_NAMES", "MicroDuckController", "friction_budget", "motor_torque"]
