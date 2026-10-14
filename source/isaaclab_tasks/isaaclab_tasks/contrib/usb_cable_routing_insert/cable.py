# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared compliant connector grasp with insertion fixture contact reporting."""

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab_tasks.contrib.usb_cable_routing.cable import (
    UsbCable,
)
from isaaclab_tasks.contrib.usb_cable_routing.cable import (
    grasp_torque as grasp_torque,
)


class InsertionCable(UsbCable):
    """Include the USB-C socket in measured robot-fixture collisions."""

    def _initialize_impl(self):
        super()._initialize_impl()
        model = NewtonManager.get_model()
        kinds = wp.to_torch(self._shape_kind)
        socket_shapes = [i for i, label in enumerate(model.shape_label) if "/Socket/" in label]
        kinds[torch.tensor(socket_shapes, device=self.device, dtype=torch.long)] = 2
