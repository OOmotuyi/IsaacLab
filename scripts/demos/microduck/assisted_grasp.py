# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit, opt-in cloth grasp assistance for the room cleanup demonstration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from newton import Model
    from newton._src.solvers.coupled.model_view import ModelView

    from isaaclab.assets import DeformableObject


ASSISTED_JAW_ANGLE = 0.18  # Jaw opening for the fabric aperture [rad].


class AssistedSockGrasp:
    """Hold at most two nearby cloth nodes between the partially closed jaws.

    Attachment is kinematic and does not return its holding force to the robot.
    A short seating motion draws the nearby patch into the open jaw aperture.
    Unpinned cloth nodes retain their normal deformable simulation and collisions.
    """

    def __init__(self, sock: DeformableObject, cloth_view: ModelView, model: Model):
        if (
            cloth_view.particle_count != sock.max_sim_vertices_per_body
            or model.particle_count != cloth_view.particle_count
        ):
            raise RuntimeError("Assistance requires the single-sock particle layout.")
        self.cloth_view = cloth_view
        self.model = model
        if cloth_view.shape_flags.ptr == model.shape_flags.ptr:
            raise RuntimeError("Assistance requires a separate cloth collision mask.")
        self.mouth_shapes = torch.tensor(
            [
                i
                for i, name in enumerate(cloth_view.shape_label)
                if any(part in name for part in ("upper_mouth_pad", "lower_mouth_", "top_head_shell"))
            ],
            dtype=torch.long,
            device=sock.data.nodal_pos_w.torch.device,
        )
        self.original_mouth_flags = wp.to_torch(cloth_view.shape_flags)[self.mouth_shapes].clone()
        self.sock = sock
        self.targets = sock.data.nodal_kinematic_target.torch.clone()
        self.ids = torch.empty(0, dtype=torch.long, device=self.targets.device)
        self.local = self.targets.new_empty((0, 3))
        self.previous = self.local.clone()
        self.velocity = self.local.clone()
        self.events: list[dict] = []
        self.seat_start = 0.0
        self.seat_from = self.local.clone()
        self.active_seconds = 0.0
        self.maximum_error_m = 0.0

    @property
    def active(self) -> bool:
        """Return whether a cloth patch is attached."""
        return self.ids.numel() > 0

    @property
    def mouth_contacts_suppressed(self) -> bool:
        """Return the legacy collision-mask status; beak contact now stays enabled."""
        return False

    def update(
        self,
        head_pose: torch.Tensor,
        *,
        allow_attach: bool,
        jaw_target: float,
        jaw_angle: float,
        jaw_velocity: float,
        time_s: float,
        dt: float,
    ) -> bool:
        """Update a patch from head pose [m, xyzw], jaw angles [rad], jaw speed [rad/s], and time [s].

        Returns:
            Whether particle dynamic flags changed and solver views need synchronization.
        """
        changed = False
        if self.active and (jaw_target >= 0.25 or jaw_angle > 0.35):
            released_ids = self.ids.clone()
            self.targets[0, released_ids, 3] = 1.0
            self.sock.write_nodal_kinematic_target_to_sim_index(self.targets)
            self.sock.write_data_to_sim()
            velocities = self.sock.data.nodal_vel_w.torch.clone()
            velocities[0, released_ids] = self.velocity
            self.sock.write_nodal_velocity_to_sim_index(velocities)
            self.events.append({"event": "release", "time_s": time_s, "node_count": len(released_ids)})
            self.ids = self.ids[:0]
            self._sync_dynamic_flags()
            return True

        if not self.active and allow_attach and jaw_target < 0.25 and jaw_angle < 0.25 and abs(jaw_velocity) < 0.25:
            nodes = self.sock.data.nodal_pos_w.torch[0]
            pinch = head_pose[:3] + quat_apply(head_pose[None, 3:], nodes.new_tensor([[-0.0115, 0.0, -0.063]]))[0]
            tip = head_pose[:3] + quat_apply(head_pose[None, 3:], nodes.new_tensor([[-0.00809334, 0.0, -0.0777383]]))[0]
            distances = torch.minimum((nodes - pinch).norm(dim=-1), (nodes - tip).norm(dim=-1))
            closest = int(distances.argmin())
            if float(distances[closest]) <= 0.014:
                # Keep a local patch without snapping it to a new grasp location.
                patch_distance = (nodes - nodes[closest]).norm(dim=-1)
                candidates = torch.where(patch_distance <= 0.008)[0]
                self.ids = candidates[torch.argsort(patch_distance[candidates])[:2]]
                self.previous = nodes[self.ids].clone()
                self.local = quat_apply_inverse(
                    head_pose[None, 3:].expand(len(self.ids), -1), self.previous - head_pose[:3]
                )
                self.seat_from = self.local.clone()
                self.seat_start = time_s
                # Seat the fabric between the actual pads, not in front of the tip.
                center = self.local.mean(0)
                # Preserve spacing and clear the lower lip as the patch seats.
                self.local[:, 0] += -0.011 - center[0]
                self.local[:, 2] += -0.0695 - center[2]
                self.velocity = torch.zeros_like(self.previous)
                self.targets[0, self.ids, 3] = 0.0
                self.events.append(
                    {
                        "event": "attach",
                        "time_s": time_s,
                        "node_ids": self.ids.tolist(),
                        "nearest_beak_distance_m": float(distances[closest]),
                        "jaw_angle_rad": jaw_angle,
                        "jaw_velocity_rad_s": jaw_velocity,
                        "initial_patch_head_m": self.seat_from.tolist(),
                        "seated_patch_head_m": self.local.tolist(),
                        "maximum_seating_displacement_m": float((self.local - self.seat_from).norm(dim=-1).max()),
                    }
                )
                changed = True

        if self.active:
            fraction = min(1.0, max(0.0, (time_s - self.seat_start) / 0.35))
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            local = torch.lerp(self.seat_from, self.local, blend)
            positions = head_pose[:3] + quat_apply(head_pose[None, 3:].expand(len(self.ids), -1), local)
            self.velocity = (positions - self.previous) / dt
            self.previous = positions.clone()
            self.targets[0, self.ids, :3] = positions
            self.sock.write_nodal_kinematic_target_to_sim_index(self.targets)
            self.sock.write_data_to_sim()
            self.active_seconds += dt
        if changed:
            self._sync_dynamic_flags()
        return changed

    def validate(self) -> None:
        """Check that the simulation honored the selected particle targets [m]."""
        if not bool((wp.to_torch(self.cloth_view.shape_flags)[self.mouth_shapes] == self.original_mouth_flags).all()):
            raise RuntimeError("Assisted grasp must retain beak/cloth collision flags.")
        if self.active:
            error = float((self.sock.data.nodal_pos_w.torch[0, self.ids] - self.previous).norm(dim=-1).max())
            self.maximum_error_m = max(self.maximum_error_m, error)
            if error > 1.0e-4:
                raise RuntimeError(f"Assisted cloth targets were not enforced: {error:.6f} m")

    def summary(self) -> dict:
        """Return explicit assistance usage and tracking measurements [s, m]."""
        return {
            "type": "kinematic sock patch; holding force is not returned to robot",
            "events": self.events,
            "active_seconds": self.active_seconds,
            "maximum_tracking_error_m": self.maximum_error_m,
            "active_at_end": self.active,
            "mouth_collisions_suppressed_while_attached": False,
            "seating_duration_s": 0.35,
            "holding_jaw_angle_rad": ASSISTED_JAW_ANGLE,
            "release_collision_delay_s": 0.0,
            "mouth_contact_restore_times_s": [],
            "mouth_collision_shapes": [self.cloth_view.shape_label[i] for i in self.mouth_shapes.tolist()],
            "mouth_collisions_restored_at_end": bool(
                (wp.to_torch(self.cloth_view.shape_flags)[self.mouth_shapes] == self.original_mouth_flags).all()
            ),
            "all_nodes_dynamic_at_end": bool(
                (self.targets[..., 3] == 1).all()
                and (wp.to_torch(self.cloth_view.particle_inv_mass) > 0).all()
                and ((wp.to_torch(self.cloth_view.particle_flags) & 1) != 0).all()
            ),
            "maximum_pinned_nodes": max((len(e.get("node_ids", [])) for e in self.events), default=0),
        }

    def _sync_dynamic_flags(self) -> None:
        print(f"ASSISTED SOCK GRASP {self.events[-1]}", flush=True)
        # The cloth entry owns every scene particle in its original order.
        for name in ("particle_inv_mass", "particle_flags"):
            source = getattr(self.model, name)
            destination = getattr(self.cloth_view, name)
            if source.ptr != destination.ptr:
                wp.copy(destination, source)


def label_assisted_frame(frame: np.ndarray, active: bool, *, body_skills_trained: bool = False) -> np.ndarray:
    """Add a persistent, explicit assistance label to an RGB video frame."""
    from PIL import Image, ImageDraw, ImageFont

    overlay = Image.fromarray(frame)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, overlay.width, 66), fill=(24, 30, 38))
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    label = "ACTIVE" if active else "released / inactive"
    draw.text((14, 6), "ASSISTED DEMO | sock attachment: " + label, font=font, fill=(255, 208, 96))
    caption = (
        "Isaac Lab-trained body skills | Scripted cleanup; marker contact-only"
        if body_skills_trained
        else "Marker: contact-only | Scripted cleanup; no new policy training"
    )
    draw.text((14, 35), caption, font=font, fill="white")
    return np.asarray(overlay)
