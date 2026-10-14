# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retention checks for a connector held exclusively by finger contact."""

from isaaclab.assets import CableObjectCfg
from isaaclab.utils import configclass


@configclass
class ContactCableCfg(CableObjectCfg):
    """Contact telemetry and failure thresholds, independent of physical constraints."""

    class_type = "isaaclab_tasks.contrib.usb_cable_routing_contact.cable:ContactUsbCable"

    acquisition_timeout_s: float = 0.35
    """Maximum time to establish bilateral fingertip contact after reset [s]."""

    contact_loss_timeout_s: float = 0.10
    """Continuous bilateral contact loss allowed before a drop [s]."""

    minimum_pad_force: float = 0.01
    """Minimum measured normal force on each fingertip to count contact [N]."""

    maximum_position_slip: float = 0.015
    """Maximum grasp-point translation relative to its reset/acquired reference [m]."""

    maximum_rotation_slip: float = 1.2
    """Maximum connector rotation relative to its reset/acquired reference [rad]."""
