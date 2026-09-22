# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured-state sequencer for the MicroDuck room cleanup demonstration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from bin_geometry import BinGeometry
from marker_assets import MOUTH_CLOSED, MOUTH_OPEN


@dataclass
class PickupState:
    """Measured object bounds [m], speed [m/s], and contact state."""

    center: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    speed: float
    opposed_contacts: bool
    mouth_distance: float
    assisted_grasp: bool = False
    surface_points: np.ndarray | None = None


@dataclass
class BasketReach:
    """Gradually extend the beak while preserving its orientation above the rim."""

    strength: float = 0.0

    def apply(
        self,
        action: np.ndarray,
        head_anchor: np.ndarray | None,
        root: np.ndarray,
        command: list[float],
        sequence: CleanupSequence,
        dt: float = 0.02,
    ) -> None:
        """Blend head targets [rad] near the basket, with a 0.4 [s] transition."""
        active = sequence.bin_geometry.shape != "box" and sequence.phase in {"carry", "drop"}
        desired = float(np.clip((root[0] - sequence.bin_center[0] + 0.23) / 0.08, 0, 1)) if active else 0.0
        self.strength += float(np.clip(desired - self.strength, -dt / 0.4, dt / 0.4))
        reference = np.zeros(4) if head_anchor is None else head_anchor[0].copy()
        release_pitch = command[4] - reference[1] if sequence.phase == "drop" else 0.0
        reference += np.array([-0.18, 0.18 + release_pitch, 0.0, 0.0])
        action[:, 5:9] = (1 - self.strength) * action[:, 5:9] + self.strength * reference


@dataclass
class CleanupSequence:
    """Sequence body skills using measured object and optional assistance state."""

    mode: str = "cleanup"
    first_object: str = "marker"
    sock_y: float = 0.26
    sock_folded: bool = False
    sock_close_time_s: float = 0.7
    """Time after pickup begins to command jaw closure [s]."""
    bin_center: np.ndarray = field(default_factory=lambda: np.array([0.38, 0.13, 0.0]))
    bin_geometry: BinGeometry = field(default_factory=BinGeometry)
    phase: str = "settle"
    phase_start: float = 0.0
    release_root: np.ndarray | None = None
    transport_root: np.ndarray | None = None
    release_clearances: dict[str, float] = field(default_factory=dict)
    target: str = "marker"
    position_integral: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw_integral: float = 0.0
    held_time: float = 0.0
    grip_time: float = 0.0
    lift_start: float | None = None
    close_start: float | None = None
    settled_time: float = 0.0
    arrival_time: float = 0.0
    lifted: dict[str, bool] = field(default_factory=lambda: {"marker": False, "sock": False})
    released: dict[str, bool] = field(default_factory=lambda: {"marker": False, "sock": False})
    completed: dict[str, bool] = field(default_factory=lambda: {"marker": False, "sock": False})

    def __post_init__(self) -> None:
        if not math.isfinite(self.sock_close_time_s) or self.sock_close_time_s < 0:
            raise ValueError("sock_close_time_s must be finite and nonnegative.")
        self.target = self.first_object if self.mode == "cleanup" else "sock" if self.mode == "sock" else "marker"

    def inside_bin(self, state: PickupState, clearance: float = 0.003) -> bool:
        """Require the entire object footprint inside the bin [m]."""
        return self.bin_geometry.contains(
            state.minimum, state.maximum, self.bin_center, clearance, state.surface_points
        )

    def settled_in_bin(self, state: PickupState, clearance: float = -0.001) -> bool:
        """Check complete object containment [m] and low motion [m/s]."""
        return bool(
            self.inside_bin(state, clearance=clearance)
            and state.minimum[2] > self.bin_center[2] + self.bin_geometry.floor_height - 0.003
            and state.maximum[2] < self.bin_center[2] + self.bin_geometry.height
            and state.speed < 0.05
        )

    def _transition(self, phase: str, t: float) -> None:
        self.phase, self.phase_start = phase, t
        self.arrival_time = 0.0
        self.position_integral[:] = 0.0
        self.yaw_integral = 0.0

    def _navigate(self, root: np.ndarray, goal: np.ndarray, yaw: float, max_speed: float = 0.32) -> list[float]:
        error = goal[:2] - root[:2]
        self.position_integral = np.clip(
            self.position_integral + 0.02 * np.where(abs(error) < 0.06, error, 0.0), -0.12, 0.12
        )
        self.yaw_integral = float(np.clip(self.yaw_integral - 0.02 * yaw, -0.3, 0.3))
        velocity = 4.0 * error + 2.0 * self.position_integral
        c, s = math.cos(yaw), math.sin(yaw)
        return [
            float(np.clip(c * velocity[0] + s * velocity[1], -max_speed, max_speed)),
            float(np.clip(-s * velocity[0] + c * velocity[1], -min(0.3, max_speed), min(0.3, max_speed))),
            float(np.clip(-3 * yaw + 2 * self.yaw_integral, -0.9, 0.9)),
        ] + [0.0] * 10

    def _sock_pick(self, t: float, obj: PickupState) -> tuple[float, float, bool]:
        """Return pickup clock [s], jaw angle [rad], and lift completion."""
        elapsed = t - self.phase_start
        if self.close_start is None and elapsed >= self.sock_close_time_s:
            self.close_start = elapsed
        self.grip_time = (
            self.grip_time + 0.02
            if self.close_start is not None and (obj.opposed_contacts or obj.assisted_grasp)
            else 0.0
        )
        if self.lift_start is None and elapsed >= 1.70 and self.grip_time >= 0.12:
            self.lift_start = elapsed
        phase_time = min(elapsed, 0.6) if self.lift_start is None else 2.0 + (elapsed - self.lift_start) * 0.5
        jaw = MOUTH_CLOSED if self.close_start is not None else MOUTH_OPEN
        finished = self.lift_start is not None and elapsed - self.lift_start >= 1.60
        if elapsed > 3.0 and self.lift_start is None:
            self._transition("grasp_failed", t)
        return phase_time, jaw, finished

    def _prepare_pick(self, t: float, root: np.ndarray, yaw: float, obj: PickupState) -> tuple[str, list[float]]:
        """Align the base [m, rad], then let the standing policy settle."""
        policy, command = "alpha_stand", [0.0] * 13
        elapsed = t - self.phase_start
        if self.phase == "settle":
            if self.target == "sock":
                goal = obj.center[:2] - np.array([0.069, 0.006])
                ready = np.linalg.norm(goal - root[:2]) < 0.008 and abs(yaw) < 0.06
                if ready:
                    self._transition("prepare_pick", t)
                else:
                    policy = "native_walking"
                    command = self._navigate(root, goal, yaw)
            elif elapsed >= 1.0:
                self._transition("pick", t)
        elif self.phase == "prepare_pick":
            # Finish the standing transition before checking alignment again.
            if elapsed >= 1.0:
                goal = obj.center[:2] - np.array([0.069, 0.006])
                ready = np.linalg.norm(goal - root[:2]) < 0.02 and abs(yaw) < 0.18
                self.grip_time, self.lift_start, self.close_start = 0.0, None, None
                self._transition("pick" if ready else "settle", t)
        return policy, command

    def _align_carry(self, t: float, root: np.ndarray, yaw: float, obj: PickupState) -> list[float]:
        """Align the held marker over the basket lane before forward travel [m, rad]."""
        goal = (root[:2] if self.transport_root is None else self.transport_root).copy()
        goal[1] = root[1] + self.bin_center[1] - obj.center[1]
        command = self._navigate(root, goal, yaw, max_speed=0.10)
        aligned = abs(obj.center[1] - self.bin_center[1]) < 0.012 and abs(yaw) < 0.10
        self.arrival_time = self.arrival_time + 0.02 if aligned else 0.0
        if self.arrival_time >= 0.16:
            self._transition("carry", t)
        elif obj.minimum[2] < 0.04:
            self._transition("grasp_failed", t)
        return command

    def step(
        self, t: float, root: np.ndarray, yaw: float, objects: dict[str, PickupState]
    ) -> tuple[str, list[float], float]:
        """Return skill, 13-value command, and jaw angle [rad] every 0.02 [s]."""
        obj = objects[self.target]
        zero = [0.0] * 13
        policy, command = ("native_walking" if self.phase in {"drop", "recover", "done"} else "alpha_stand"), zero
        jaw = (
            MOUTH_OPEN
            if self.phase
            in {"settle", "prepare_pick", "pick", "drop", "recover", "retreat", "sidestep", "align", "done"}
            else MOUTH_CLOSED
        )
        elapsed = t - self.phase_start
        held = (
            obj.center[2] > 0.085
            and obj.minimum[2] > 0.015
            and obj.mouth_distance < 0.10
            and (obj.opposed_contacts or obj.assisted_grasp)
        )
        self.held_time = self.held_time + 0.02 if held else 0.0
        self.lifted[self.target] |= self.held_time >= 0.3

        if self.phase in {"settle", "prepare_pick"}:
            policy, command = self._prepare_pick(t, root, yaw, obj)
        elif self.phase == "pick":
            policy = "alpha_ground_pick"
            if self.target == "sock":
                phase_time, jaw, finished = self._sock_pick(t, obj)
            else:
                phase_time = min(elapsed, 1.4) + max(0.0, elapsed - 2.4)
                jaw = MOUTH_CLOSED if elapsed >= 1.5 else MOUTH_OPEN
                finished = phase_time >= 2.8
            command[:2] = [math.cos(2 * math.pi * phase_time / 4), math.sin(2 * math.pi * phase_time / 4)]
            if finished:
                self._transition("lift_check", t)
        elif self.phase == "lift_check":
            if elapsed >= 0.7 and self.lifted[self.target] and held:
                self.transport_root = root[:2].copy()
                next_phase = "clear_support" if self.target == "sock" and not self.sock_folded else "carry"
                if self.target == "marker" and self.bin_geometry.shape != "box":
                    next_phase = "align_carry"
                self._transition(next_phase, t)
            elif elapsed > 3.0:
                self._transition("grasp_failed", t)
        elif self.phase in {"clear_support", "carry_lane"}:
            policy = "native_walking"
            carry_y = 0.12 if self.bin_geometry.shape == "box" else self.bin_center[1] - 0.01
            goal = np.array([-0.07, self.sock_y - 0.006 if self.phase == "clear_support" else carry_y])
            command = self._navigate(root, goal, yaw)
            arrived = root[0] < -0.055 if self.phase == "clear_support" else abs(root[1] - goal[1]) < 0.012
            self.arrival_time = self.arrival_time + 0.02 if arrived else 0.0
            if self.arrival_time >= 0.16:
                self._transition("carry_lane" if self.phase == "clear_support" else "carry", t)
            elif obj.center[2] < 0.04 or root[2] < 0.065:
                self._transition("grasp_failed", t)
        elif self.phase == "align_carry":
            policy, command = "native_walking", self._align_carry(t, root, yaw, obj)
        elif self.phase == "carry":
            if (
                self.inside_bin(
                    obj,
                    clearance=0.006
                    if self.bin_geometry.shape != "box"
                    else 0.025
                    if self.target == "marker"
                    else 0.015,
                )
                and obj.minimum[2]
                > self.bin_center[2] + self.bin_geometry.height + (0.008 if self.bin_geometry.shape != "box" else 0.0)
                and (
                    self.bin_geometry.shape == "box"
                    or (
                        (self.target == "marker" or obj.center[0] > self.bin_center[0] - 0.035)
                        and abs(obj.center[1] - self.bin_center[1]) < 0.02
                    )
                )
            ):
                self.release_clearances[self.target] = float(
                    obj.minimum[2] - self.bin_center[2] - self.bin_geometry.height
                )
                self.released[self.target] = True
                self.release_root = root[:2].copy()
                self._transition("drop", t)
                policy = "native_walking" if self.target == "marker" and self.bin_geometry.shape != "box" else policy
                jaw = MOUTH_OPEN
            elif obj.center[2] < 0.04 or root[2] < 0.065:
                self._transition("grasp_failed", t)
            else:
                policy = "native_walking"
                # Align the base with the carrying lane; measured bounds gate release.
                goal = root.copy()
                goal[0] += self.bin_center[0] - obj.center[0]
                goal[1] = (self.sock_y - 0.006 if self.sock_folded else 0.12) if self.target == "sock" else 0.0
                if self.bin_geometry.shape != "box":
                    goal[1] = (
                        root[1] + self.bin_center[1] - obj.center[1]
                        if self.target == "marker"
                        else self.bin_center[1] - 0.01
                    )
                command = self._navigate(
                    root,
                    goal,
                    yaw,
                    max_speed=min(
                        0.32
                        if obj.assisted_grasp
                        else 0.2
                        if self.bin_geometry.shape != "box" or self.target == "sock"
                        else 0.3,
                        0.5 * elapsed,
                    ),
                )
        elif self.phase == "drop":
            goal = root if self.release_root is None else self.release_root
            command = self._navigate(root, goal, yaw)
            amplitude = 0.90 if self.target == "marker" else 0.15
            command[4] = amplitude * min(1.0, elapsed / 1.0) * max(0.0, 1.0 - max(0.0, elapsed - 1.6) / 1.0)
            settled = self.settled_in_bin(obj)
            self.settled_time = self.settled_time + 0.02 if settled else 0.0
            if self.settled_time >= 0.5 and self.lifted[self.target] and elapsed >= 2.6:
                self.completed[self.target] = True
                if self.mode == "cleanup" and not all(self.completed.values()):
                    self._transition("recover", t)
                else:
                    self._transition("done", t)
        elif self.phase == "recover":
            if self.target == "sock":
                policy = "alpha_stand"
            if elapsed >= 1.0:
                self._transition("retreat", t)
        elif self.phase in {"retreat", "sidestep", "align"}:
            next_y = 0.0 if self.target == "sock" else self.sock_y - 0.006
            if self.phase == "retreat":
                # Back past the waiting object before crossing into its pickup lane.
                goal = np.array([-0.035, 0.12 if self.target == "sock" else 0.0])
            elif self.phase == "sidestep":
                goal = np.array([-0.035, next_y])
            else:
                goal = np.array([0.0, next_y])
            distance = float(np.linalg.norm(goal - root[:2]))
            policy = "native_walking"
            command = self._navigate(
                root, goal, yaw, max_speed=0.4 if self.target == "sock" and self.phase == "retreat" else 0.32
            )
            if self.phase == "retreat":
                arrived = root[0] < -0.02
            elif self.phase == "sidestep":
                arrived = distance < 0.025
            else:
                arrived = distance < 0.012 and abs(yaw) < 0.10
            self.arrival_time = self.arrival_time + 0.02 if arrived else 0.0
            if self.arrival_time >= 0.16:
                if self.phase == "align":
                    self.target = "marker" if self.target == "sock" else "sock"
                    self.held_time = self.settled_time = 0.0
                    self._transition("settle", t)
                else:
                    self._transition("sidestep" if self.phase == "retreat" else "align", t)
        return policy, command, jaw
