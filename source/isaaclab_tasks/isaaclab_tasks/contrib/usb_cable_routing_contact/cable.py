# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-only connector retention and Newton VBD contact telemetry."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp
from isaaclab_newton.assets import CableObject
from isaaclab_newton.physics import NewtonManager
from newton import BodyFlags

from isaaclab_tasks.contrib.usb_cable_routing.assets import GRASP_LOCAL_Z

from .cable_cfg import ContactCableCfg

_ROBOT = wp.constant(1)
_FIXTURE = wp.constant(2)
_SOCKET = wp.constant(4)
_CONNECTOR = wp.constant(8)
_LEFT_PAD = wp.constant(16)
_RIGHT_PAD = wp.constant(32)
_HOUSING = wp.constant(64)


class ContactUsbCable(CableObject):
    """A fixed-end cable with no artificial force or torque retaining its free plug.

    Public Warp arrays are indexed by environment. ``pad_normal_forces`` has shape
    [N, 2] and units [N]. ``finger_contact_force`` and ``socket_contact_force_w``
    have shape [N, 3] and world-frame units [N]. Compatibility fields ``tension``
    and ``grasp_torque`` are the net finger-contact force [N] and torque [N m]
    magnitudes, not a cable tension sensor. ``socket_contact_force`` is the net
    socket-contact force magnitude [N]. ``position_slip`` [m], ``rotation_slip``
    [rad], ``contact_loss_time`` [s] and ``acquisition_time`` [s] are retention
    diagnostics. These report the final solver substep, not interval peak loads.

    ``grasp_active`` permits the bounded acquisition interval; ``grasp_acquired``
    is required separately for task success. ``grasp_dropped`` is sticky until
    reset. None of these flags imposes a physical constraint.
    """

    cfg: ContactCableCfg

    def reset_contact_grasp(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Record measured relative pose and clear retention history after reset FK.

        Args:
            env_ids: Environments to reset; None selects all instances.
        """
        if env_ids is None:
            indices = torch.arange(self.num_instances, device=self.device, dtype=torch.int32)
        else:
            indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32).contiguous()
        if indices.numel() == 0:
            return
        for array in self._scalar_zero_on_reset:
            wp.to_torch(array)[indices.long()] = 0
        for array in self._vector_zero_on_reset:
            wp.to_torch(array)[indices.long()] = 0
        wp.to_torch(self.grasp_active)[indices.long()] = 1
        wp.launch(
            _reset_reference,
            indices.numel(),
            inputs=[
                wp.from_torch(indices),
                NewtonManager.get_state_0().body_q,
                self._end_ids,
                self._wrist_ids,
                self.grasp_relative_position,
                self.grasp_relative_quat,
            ],
            device=self.device,
        )

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        model = NewtonManager.get_model()
        children = wp.to_torch(self.root_view.get_attribute("joint_child", model).contiguous())[:, 0]
        # The free root joint already includes the first segment as its child.
        self.body_ids = children.long()
        wp.to_torch(model.body_flags)[self.body_ids[:, 0]] = int(BodyFlags.KINEMATIC)
        self._end_ids = wp.from_torch(self.body_ids[:, -1].to(torch.int32).contiguous())
        wrists = [index for index, label in enumerate(model.body_label) if label.endswith("/wrist_3_link")]
        if len(wrists) != self.num_instances:
            raise RuntimeError(f"Expected one UR10e wrist per cable, got {len(wrists)}.")
        self._wrist_ids = wp.array(wrists, dtype=wp.int32, device=self.device)
        for name in ("grasp_active", "grasp_acquired", "grasp_dropped", "collisions"):
            setattr(self, name, wp.zeros(self.num_instances, dtype=wp.int32, device=self.device))
        for name in (
            "tension",
            "grasp_torque",
            "socket_contact_force",
            "position_slip",
            "rotation_slip",
            "contact_loss_time",
            "acquisition_time",
        ):
            setattr(self, name, wp.zeros(self.num_instances, dtype=float, device=self.device))
        for name in (
            "finger_contact_force",
            "socket_contact_force_w",
            "_finger_contact_torque",
            "grasp_relative_position",
        ):
            setattr(self, name, wp.zeros(self.num_instances, dtype=wp.vec3, device=self.device))
        self.grasp_relative_quat = wp.zeros(self.num_instances, dtype=wp.quat, device=self.device)
        self.pad_normal_forces = wp.zeros((self.num_instances, 2), dtype=float, device=self.device)
        self.pad_contact_count = wp.zeros((self.num_instances, 2), dtype=wp.int32, device=self.device)
        self._scalar_zero_on_reset = [
            self.grasp_acquired,
            self.grasp_dropped,
            self.collisions,
            self.tension,
            self.grasp_torque,
            self.socket_contact_force,
            self.position_slip,
            self.rotation_slip,
            self.contact_loss_time,
            self.acquisition_time,
        ]
        self._vector_zero_on_reset = [
            self.finger_contact_force,
            self.socket_contact_force_w,
            self._finger_contact_torque,
            self.pad_normal_forces,
            self.pad_contact_count,
        ]
        kinds = []
        for label in model.shape_label:
            kind = 0
            if "/Robot/" in label and "/Robot/base_link/" not in label:
                kind |= int(_ROBOT)
            if any(part in label for part in ("/Table/", "/Guide", "/Anchor/", "/Socket/")):
                kind |= int(_FIXTURE)
            if "/Socket/" in label:
                kind |= int(_SOCKET)
            if "/Cable/" in label and "mesh_edge_capsule_" not in label:
                kind |= int(_CONNECTOR)
            if label.endswith("/blue_usb_c_housing"):
                kind |= int(_HOUSING)
            if "/left_inner_finger/" in label and "fingertips" in label:
                kind |= int(_LEFT_PAD)
            if "/right_inner_finger/" in label and "fingertips" in label:
                kind |= int(_RIGHT_PAD)
            kinds.append(kind)
        if not any(kind & int(_LEFT_PAD) for kind in kinds) or not any(kind & int(_RIGHT_PAD) for kind in kinds):
            raise RuntimeError("Contact grasp requires both Robotiq fingertip collision meshes.")
        self._shape_kind = wp.array(kinds, dtype=wp.int32, device=self.device)
        NewtonManager.register_post_step_callback(self._measure_contact_grasp)

    def _measure_contact_grasp(self) -> None:
        coupler = NewtonManager._solver
        contacts = coupler.get_proxy_contacts("rigid", "cable")
        if contacts is None or contacts.rigid_contact_force is None:
            raise RuntimeError("Contact grasp requires VBD proxy contacts with harvested force buffers.")
        for array in (
            self.pad_normal_forces,
            self.pad_contact_count,
            self.finger_contact_force,
            self.socket_contact_force_w,
            self._finger_contact_torque,
        ):
            array.zero_()
        view = coupler.view("cable")
        state = coupler.entry_state("cable", "output")
        wp.launch(
            _reduce_contacts,
            contacts.rigid_contact_shape0.shape[0],
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_force,
                self._shape_kind,
                view.shape_body,
                view.shape_world,
                state.body_q,
                self.pad_normal_forces,
                self.pad_contact_count,
                self.finger_contact_force,
                self._finger_contact_torque,
                self.socket_contact_force_w,
                self.collisions,
            ],
            device=self.device,
        )
        dt = NewtonManager.get_dt()
        if NewtonManager.handles_decimation():
            dt *= NewtonManager._decimation
        wp.launch(
            _update_retention,
            self.num_instances,
            inputs=[
                NewtonManager.get_state_0().body_q,
                self._end_ids,
                self._wrist_ids,
                self.pad_normal_forces,
                self.finger_contact_force,
                self._finger_contact_torque,
                self.socket_contact_force_w,
                dt,
                self.cfg.minimum_pad_force,
                self.cfg.acquisition_timeout_s,
                self.cfg.contact_loss_timeout_s,
                self.cfg.maximum_position_slip,
                self.cfg.maximum_rotation_slip,
                self.grasp_relative_position,
                self.grasp_relative_quat,
                self.grasp_active,
                self.grasp_acquired,
                self.grasp_dropped,
                self.tension,
                self.grasp_torque,
                self.socket_contact_force,
                self.position_slip,
                self.rotation_slip,
                self.contact_loss_time,
                self.acquisition_time,
            ],
            device=self.device,
        )

    def _clear_callbacks(self) -> None:
        NewtonManager.unregister_post_step_callback(self._measure_contact_grasp)
        super()._clear_callbacks()


@wp.kernel
def _reset_reference(
    env_ids: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.transform),
    ends: wp.array(dtype=wp.int32),
    wrists: wp.array(dtype=wp.int32),
    relative_position: wp.array(dtype=wp.vec3),
    relative_quat: wp.array(dtype=wp.quat),
):
    env = env_ids[wp.tid()]
    end, wrist = poses[ends[env]], poses[wrists[env]]
    point = wp.transform_point(end, wp.vec3(0.0, 0.0, GRASP_LOCAL_Z))
    relative_position[env] = wp.transform_point(wp.transform_inverse(wrist), point)
    relative_quat[env] = wp.quat_inverse(wp.transform_get_rotation(wrist)) * wp.transform_get_rotation(end)


@wp.kernel
def _reduce_contacts(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    offset0: wp.array(dtype=wp.vec3),
    offset1: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    force_on_body1: wp.array(dtype=wp.vec3),
    kinds: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    world: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.transform),
    pad_load: wp.array2d(dtype=float),
    pad_count: wp.array2d(dtype=wp.int32),
    finger_force: wp.array(dtype=wp.vec3),
    finger_torque: wp.array(dtype=wp.vec3),
    socket_force: wp.array(dtype=wp.vec3),
    collisions: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= count[0]:
        return
    a, b = shape0[i], shape1[i]
    if a < 0 or b < 0:
        return
    ka, kb = kinds[a], kinds[b]
    env = wp.max(world[a], world[b])
    force = force_on_body1[i]
    if env < 0 or wp.length(force) < 1.0e-7:
        return
    if ((ka & _ROBOT) != 0 and (kb & _FIXTURE) != 0) or ((kb & _ROBOT) != 0 and (ka & _FIXTURE) != 0):
        wp.atomic_max(collisions, env, 1)
    connector = b
    other_kind = ka
    contact_point = point1[i] + offset1[i]
    if (ka & _CONNECTOR) != 0:
        connector = a
        other_kind = kb
        contact_point = point0[i] + offset0[i]
        force = -force
    elif (kb & _CONNECTOR) == 0:
        return
    body = shape_body[connector]
    if body < 0:
        return
    if (other_kind & _SOCKET) != 0:
        wp.atomic_add(socket_force, env, force)
    if (other_kind & (_LEFT_PAD | _RIGHT_PAD)) != 0:
        point = wp.transform_point(poses[body], contact_point)
        grasp = wp.transform_point(poses[body], wp.vec3(0.0, 0.0, GRASP_LOCAL_Z))
        wp.atomic_add(finger_force, env, force)
        wp.atomic_add(finger_torque, env, wp.cross(point - grasp, force))
        if (kinds[connector] & _HOUSING) != 0:
            side = int(0)
            if (other_kind & _RIGHT_PAD) != 0:
                side = 1
            load = wp.abs(wp.dot(force, normal[i]))
            wp.atomic_add(pad_load, env, side, load)
            wp.atomic_add(pad_count, env, side, 1)


@wp.kernel
def _update_retention(
    poses: wp.array(dtype=wp.transform),
    ends: wp.array(dtype=wp.int32),
    wrists: wp.array(dtype=wp.int32),
    pad_load: wp.array2d(dtype=float),
    finger_force: wp.array(dtype=wp.vec3),
    finger_torque: wp.array(dtype=wp.vec3),
    socket_force: wp.array(dtype=wp.vec3),
    dt: float,
    minimum_pad_force: float,
    acquisition_timeout: float,
    contact_loss_timeout: float,
    max_slip: float,
    max_rotation: float,
    relative_position: wp.array(dtype=wp.vec3),
    relative_quat: wp.array(dtype=wp.quat),
    active: wp.array(dtype=wp.int32),
    acquired: wp.array(dtype=wp.int32),
    dropped: wp.array(dtype=wp.int32),
    tension: wp.array(dtype=float),
    torque: wp.array(dtype=float),
    socket_force_norm: wp.array(dtype=float),
    position_slip: wp.array(dtype=float),
    rotation_slip: wp.array(dtype=float),
    loss_time: wp.array(dtype=float),
    acquisition_time: wp.array(dtype=float),
):
    env = wp.tid()
    tension[env] = wp.length(finger_force[env])
    torque[env] = wp.length(finger_torque[env])
    socket_force_norm[env] = wp.length(socket_force[env])
    end, wrist = poses[ends[env]], poses[wrists[env]]
    point = wp.transform_point(end, wp.vec3(0.0, 0.0, GRASP_LOCAL_Z))
    position = wp.transform_point(wp.transform_inverse(wrist), point)
    quat = wp.quat_inverse(wp.transform_get_rotation(wrist)) * wp.transform_get_rotation(end)
    position_slip[env] = wp.length(position - relative_position[env])
    rotation_slip[env] = 2.0 * wp.acos(wp.clamp(wp.abs(wp.dot(quat, relative_quat[env])), 0.0, 1.0))
    if active[env] == 0 or dropped[env] != 0:
        return
    bilateral = pad_load[env, 0] >= minimum_pad_force and pad_load[env, 1] >= minimum_pad_force
    escaped = position_slip[env] > max_slip or rotation_slip[env] > max_rotation
    if not wp.isfinite(position_slip[env]) or not wp.isfinite(rotation_slip[env]):
        escaped = True
    if acquired[env] == 0:
        acquisition_time[env] += dt
        if bilateral and not escaped:
            acquired[env] = 1
            relative_position[env] = position
            relative_quat[env] = quat
            position_slip[env] = 0.0
            rotation_slip[env] = 0.0
        elif acquisition_time[env] >= acquisition_timeout:
            escaped = True
    else:
        if bilateral:
            loss_time[env] = 0.0
        else:
            loss_time[env] += dt
        if loss_time[env] >= contact_loss_timeout:
            escaped = True
    if escaped:
        dropped[env] = 1
        active[env] = 0
