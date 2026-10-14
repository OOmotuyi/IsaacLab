# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton elastic rod with a fixed strain relief and breakable compliant grasp."""

import torch
import warp as wp
from isaaclab_newton.assets import CableObject
from isaaclab_newton.physics import NewtonManager
from newton import BodyFlags

from .assets import GRASP_LOCAL_Z, TCP_OFFSET

_GRASP_OFFSET = wp.constant(wp.vec3(*TCP_OFFSET))


@wp.kernel
def grasp_force(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    end_ids: wp.array(dtype=wp.int32),
    wrist_ids: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    tension: wp.array(dtype=float),
    force: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    if active[i] == 0:
        return
    end = end_ids[i]
    wrist = wrist_ids[i]
    target = wp.transform_point(poses[wrist], _GRASP_OFFSET)
    tip = wp.transform_point(poses[end], wp.vec3(0.0, 0.0, GRASP_LOCAL_Z))
    end_com = wp.transform_point(poses[end], body_com[end])
    wrist_com = wp.transform_point(poses[wrist], body_com[wrist])
    re = tip - end_com
    rw = target - wrist_com
    ve = wp.spatial_top(velocities[end]) + wp.cross(wp.spatial_bottom(velocities[end]), re)
    vw = wp.spatial_top(velocities[wrist]) + wp.cross(wp.spatial_bottom(velocities[wrist]), rw)
    delta = target - tip
    f = 400.0 * delta + 0.03 * (vw - ve)
    tension[i] = wp.length(f)
    # A finite holding force makes excessive tension a real drop, rather than a permanent weld.
    if wp.length(delta) > 0.015 or wp.length(f) > 6.0:
        active[i] = 0
        return
    wp.atomic_add(force, end, wp.spatial_vector(f, wp.cross(re, f)))
    wp.atomic_add(force, wrist, wp.spatial_vector(-f, wp.cross(rw, -f)))


@wp.kernel
def grasp_torque(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    inertia: wp.array(dtype=wp.mat33),
    ends: wp.array(dtype=wp.int32),
    wrists: wp.array(dtype=wp.int32),
    relative_quat: wp.array(dtype=wp.quat),
    active: wp.array(dtype=wp.int32),
    torque_norm: wp.array(dtype=float),
    dt: float,
    force: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    if active[i] == 0:
        return
    end, wrist = ends[i], wrists[i]
    q = wp.transform_get_rotation(pose[end])
    target = wp.transform_get_rotation(pose[wrist]) * relative_quat[i]
    error = target * wp.quat_inverse(q)
    if error[3] < 0.0:
        error = -error
    vector = wp.vec3(error[0], error[1], error[2])
    angle = 2.0 * wp.atan2(wp.length(vector), wp.max(error[3], 1.0e-8))
    world_error = wp.normalize(vector) * angle
    omega = wp.spatial_bottom(velocity[end]) - wp.spatial_bottom(velocity[wrist])
    local_error = wp.quat_rotate_inv(q, world_error)
    local_omega = wp.quat_rotate_inv(q, omega)
    # Implicit PD attenuation avoids explicit-spring instability at the light plug's inertia.
    kp, kd = 0.15, 0.00010
    regularizer = kd * dt + kp * dt * dt
    lhs = inertia[end] + wp.diag(wp.vec3(regularizer))
    local_torque = wp.inverse(lhs) * (inertia[end] * (kp * local_error - (kd + kp * dt) * local_omega))
    torque = wp.quat_rotate(q, local_torque)
    torque_norm[i] = wp.length(torque)
    if angle > 1.2 or wp.length(torque) > 0.04:
        active[i] = 0
        return
    wp.atomic_add(force, end, wp.spatial_vector(wp.vec3(0.0), torque))
    wp.atomic_add(force, wrist, wp.spatial_vector(wp.vec3(0.0), -torque))


class UsbCable(CableObject):
    """A passive rod; only its anchor and the pre-grasp boundary have prescribed constraints."""

    def _initialize_impl(self):
        super()._initialize_impl()
        model = NewtonManager.get_model()
        parents = wp.to_torch(self.root_view.get_attribute("joint_parent", model).contiguous())[:, 0]
        children = wp.to_torch(self.root_view.get_attribute("joint_child", model).contiguous())[:, 0]
        self.body_ids = torch.cat((parents[:, :1], children), dim=1).long()
        wp.to_torch(model.body_flags)[self.body_ids[:, 0]] = int(BodyFlags.KINEMATIC)
        self.grasp_active = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.grasp_relative_quat = wp.zeros(self.num_instances, dtype=wp.quat, device=self.device)
        self.grasp_torque = wp.zeros(self.num_instances, dtype=float, device=self.device)
        self.tension = wp.zeros(self.num_instances, dtype=float, device=self.device)
        self._end_ids = wp.from_torch(self.body_ids[:, -1].to(torch.int32).contiguous())
        wrist_ids = [i for i, label in enumerate(model.body_label) if label.endswith("/wrist_3_link")]
        if len(wrist_ids) != self.num_instances:
            raise RuntimeError(f"Expected one UR10e wrist per cable, got {len(wrist_ids)}.")
        self._wrist_ids = wp.array(wrist_ids, dtype=wp.int32, device=self.device)
        self.collisions = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        # The fixed robot base is intentionally bolted to the tabletop.
        kinds = [
            0
            if "/Robot/base_link/" in label
            else 1
            if "/Robot/" in label
            else 2
            if any(x in label for x in ("/Table/", "/Guide", "/Anchor/"))
            else 0
            for label in model.shape_label
        ]
        self._shape_kind = wp.array(kinds, dtype=wp.int32, device=self.device)
        NewtonManager.register_state_force_callback(self._apply_grasp)
        NewtonManager.register_post_step_callback(self._measure_collisions)

    def _measure_collisions(self):
        contacts = NewtonManager._solver.get_proxy_contacts("rigid", "cable")
        if contacts is not None:
            wp.launch(
                robot_fixture_contacts,
                contacts.rigid_contact_shape0.shape[0],
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    self._shape_kind,
                    NewtonManager.get_model().shape_world,
                    self.collisions,
                ],
                device=self.device,
            )

    def _apply_grasp(self, state):
        model = NewtonManager.get_model()
        wp.launch(
            grasp_force,
            self.num_instances,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                self._end_ids,
                self._wrist_ids,
                self.grasp_active,
                self.tension,
                state.body_f,
            ],
            device=self.device,
        )
        wp.launch(
            grasp_torque,
            self.num_instances,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_inertia,
                self._end_ids,
                self._wrist_ids,
                self.grasp_relative_quat,
                self.grasp_active,
                self.grasp_torque,
                NewtonManager.get_solver_dt(),
                state.body_f,
            ],
            device=self.device,
        )


@wp.kernel
def robot_fixture_contacts(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    world: wp.array(dtype=wp.int32),
    collision: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= count[0]:
        return
    a, b = shape0[i], shape1[i]
    if a < 0 or b < 0:
        return
    if (kind[a] == 1 and kind[b] == 2) or (kind[b] == 1 and kind[a] == 2):
        env = wp.max(world[a], world[b])
        if env >= 0:
            wp.atomic_max(collision, env, 1)
