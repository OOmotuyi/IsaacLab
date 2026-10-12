# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an Isaac-trained object-conditioned body/jaw actor in the cleanup room."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import onnxruntime as ort
import torch

from isaaclab_tasks.contrib.microduck.constants import JOINT_NAMES
from isaaclab_tasks.contrib.microduck.grasp_contract import MicroDuckGraspState

if TYPE_CHECKING:
    import numpy as np
    from assisted_grasp import AssistedSockGrasp
    from controller import MicroDuckController

    from isaaclab.assets import DeformableObject, RigidObject


def load_grasp_policy(path: Path) -> ort.InferenceSession:
    """Load a trained 65-input/15-output actor with verified Isaac-only export metadata."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    if session.get_inputs()[0].shape != [1, 65] or session.get_outputs()[0].shape != [1, 15]:
        raise ValueError("Object grasp playback requires 65 observations and 15 actions.")
    expected = {
        "training_backend": "isaaclab_newton",
        "observation_contract": "grasp_proprio_object_v3",
        "action_contract": "absolute_body_offsets_and_jaw",
        "mujoco_policy_weights_used": "false",
        "joint_names": ",".join(JOINT_NAMES),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Object grasp policy has incompatible training or observation metadata.")
    if float(metadata.get("optimizer_steps", 0)) <= 0:
        raise ValueError("Object grasp policy has no recorded training updates.")
    return session


class MicroDuckGraspPlayback:
    """Use the training phase/features with actual bedroom object measurements."""

    def __init__(self, session: ort.InferenceSession, controller: MicroDuckController) -> None:
        self.session = session
        self.controller = controller
        self.robot = controller.robot
        self.head_id = self.robot.find_bodies("jaw_soft")[0][0]
        self.jaw_ids = self.robot.find_joints("mouth")[0]
        self.progress = MicroDuckGraspState(1, self.robot.device, 0.0395, 0.0395)
        self.previous = torch.zeros((1, 15), device=self.robot.device)
        self.origins = torch.zeros((1, 3), device=self.robot.device)
        self.target: str | None = None

    @property
    def ready(self) -> bool:
        """Return whether actual sustained contact and lift have passed the training criterion."""
        return bool(self.progress.contacts[0].all() and self.progress.lift_time[0] >= 0.30)

    def step(
        self,
        target: str,
        object_position: torch.Tensor,
        object_velocity: torch.Tensor,
        contacts: tuple[bool, bool],
        dt: float = 0.02,
        *,
        object_quaternion: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Return body offsets [rad] and jaw target [rad] from object state [m, m/s]."""
        if target != self.target:
            self.target = target
            self.progress.reset_progress()
            self.progress.object_quaternion.zero_()
            self.progress.object_quaternion[:, 3] = 1
            self.progress.start_height[:] = object_position[0, 2]
            self.progress.reach_height.copy_(self.progress.start_height)
            self.previous[:, :14] = self.controller.last_action
            self.previous[:, 14] = 1.0
            self.progress.pinch_local[:] = self.previous.new_tensor(
                [-0.0115, 0.0, -0.063] if target == "sock" else [-0.016, 0.0, -0.063]
            )
        data = self.robot.data
        self.progress.contacts[0] = torch.tensor(contacts, device=self.robot.device)
        self.progress.update_progress(
            data.body_link_pose_w.torch[:, self.head_id],
            object_position,
            object_velocity,
            self.origins,
            dt,
            object_quaternion,
        )
        features = self.progress.features(
            data.root_link_quat_w.torch,
            object_position,
            object_velocity,
            data.joint_pos.torch[:, self.jaw_ids],
            data.joint_vel.torch[:, self.jaw_ids],
        )
        observation = torch.cat((self.controller.observations()[:, :34], self.previous, features), dim=-1)
        if not torch.isfinite(observation).all():
            raise RuntimeError("Object grasp actor received non-finite observations.")
        actions = self.session.run(None, {self.session.get_inputs()[0].name: observation.cpu().numpy()})[0]
        raw = torch.as_tensor(actions, device=self.robot.device)
        if raw.shape != (1, 15) or not torch.isfinite(raw).all():
            raise RuntimeError("Object grasp actor returned invalid actions.")
        self.previous.copy_(raw)
        jaw = 0.2181661565 + 0.3054326191 * float(raw[0, 14].clamp(-1, 1))
        return raw[:, :14], jaw


def infer_cleanup_action(
    session: ort.InferenceSession,
    observation: torch.Tensor,
    jaw_target: float,
    *,
    grasp: MicroDuckGraspPlayback | None,
    policy_name: str,
    target: str,
    marker: RigidObject,
    sock: DeformableObject,
    marker_contacts: list[list[str]],
    sock_opposed: bool,
    assistance: AssistedSockGrasp | None,
) -> tuple[np.ndarray, float]:
    """Select legacy body inference or learned body/jaw control using measured room objects."""
    if grasp is None or policy_name != "alpha_ground_pick":
        return session.run(None, {session.get_inputs()[0].name: observation.cpu().numpy()})[0], jaw_target
    object_quaternion = None
    if target == "marker":
        object_position = marker.data.root_link_pos_w.torch
        object_velocity = marker.data.root_link_lin_vel_w.torch
        object_quaternion = marker.data.root_link_quat_w.torch
        contacts = (
            any("upper_mouth_pad" in name for pair in marker_contacts for name in pair),
            any("lower_mouth_" in name for pair in marker_contacts for name in pair),
        )
    else:
        nodes = sock.data.nodal_pos_w.torch[0]
        if assistance is not None and assistance.active:
            selected = assistance.ids
        else:
            selected = torch.where((nodes[:, 1] - nodes[:, 1].mean()).abs() < 0.012)[0]
        if len(selected) == 0:
            raise RuntimeError("No cloth nodes in the requested grasp region.")
        object_position = nodes[selected].mean(0, keepdim=True)
        object_velocity = sock.data.nodal_vel_w.torch[:, selected].mean(1)
        held = sock_opposed or bool(assistance and assistance.active)
        contacts = (held, held)
    action, jaw_target = grasp.step(
        target, object_position, object_velocity, contacts, object_quaternion=object_quaternion
    )
    if assistance is not None and target == "sock" and jaw_target < 0.25:
        from assisted_grasp import ASSISTED_JAW_ANGLE

        jaw_target = max(ASSISTED_JAW_ANGLE, jaw_target)
    return action.cpu().numpy(), jaw_target
