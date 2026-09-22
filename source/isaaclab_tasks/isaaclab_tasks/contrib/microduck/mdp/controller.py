# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck policy contract and BAM M6 controller for Newton MJWarp.

Equations and fitted parameters follow Rhoban/bam at
62bd8ce12154340be97e06f7f41a0ca8f116d967 (Apache-2.0).
"""

from __future__ import annotations

import math

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverMuJoCo

from isaaclab.assets import Articulation

from ..constants import DEFAULT_POSE, JOINT_NAMES


def motor_torque(
    error: torch.Tensor, velocity: torch.Tensor, voltage: torch.Tensor, params: dict[str, float]
) -> torch.Tensor:
    """Compute XL330 torque [N·m] from error [rad], speed [rad/s], and supply [V]."""
    duty = (error * 200.0 * 4096.0 / (2.0 * math.pi * 256.0 * 885.0)).clamp(-1.0, 1.0)
    return params["kt"] * (voltage * duty - params["kt"] * velocity) / params["R"]


def friction_budget(
    motor: torch.Tensor,
    external: torch.Tensor,
    velocity: torch.Tensor,
    params: dict[str, float],
    training_model: bool = False,
) -> torch.Tensor:
    """Compute BAM M6 dry friction [N·m] from loads [N·m] and speed [rad/s]."""
    p = params
    stribeck = torch.exp(-(velocity.abs() / p["dtheta_stribeck"]).pow(p["alpha"]))
    dry = p["friction_base"] + stribeck * p["friction_stribeck"]
    dry = dry + (external * p["load_friction_external"] - motor * p["load_friction_motor"]).abs()
    dry = (
        dry
        + stribeck * (external * p["load_friction_external_stribeck"] - motor * p["load_friction_motor_stribeck"]).abs()
    )
    if training_model:
        drive = motor.abs() > external.abs()
        quadratic = torch.where(
            drive,
            p["load_friction_external_quad"] * external.square(),
            p["load_friction_motor_quad"] * motor.square(),
        )
        return dry + stribeck * quadratic
    opposed = external.sign() != motor.sign()
    quadratic = (external.abs() < motor.abs()) * p["load_friction_external_quad"] * external.square()
    quadratic += (external.abs() > motor.abs()) * p["load_friction_motor_quad"] * motor.square()
    return dry + stribeck * opposed * quadratic


class MicroDuckController:
    """Apply the upstream servo model at each 5 ms physics step.

    Uses Isaac Lab articulation commands and Newton's MJWarp constraint forces.
    The private solver handle is isolated here because Lab has no public per-DOF
    bias/constraint-force accessor. Targets Newton 1.6 / MJWarp 3.12.
    An explicit GPU MJWarp sub-solver may be supplied for a coupled cloth scene.
    """

    def __init__(
        self,
        robot: Articulation,
        params: dict[str, float],
        training_model: bool = False,
        voltage_limit: float = 7.4,
        *,
        solver: SolverMuJoCo | None = None,
    ):
        self.robot = robot
        self.params = params
        self.training_model = training_model
        self.voltage_limit = voltage_limit
        self.joint_ids, names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"Unexpected servo joints: {names}")
        solver = NewtonManager._solver if solver is None else solver
        if not isinstance(solver, SolverMuJoCo) or solver.use_mujoco_cpu:
            raise ValueError("MicroDuck playback requires Newton with the GPU MJWarp solver.")
        self.solver = solver
        model = solver.mj_model
        dofs = []
        for name in JOINT_NAMES:
            matches = [i for i in range(model.njnt) if model.joint(i).name.endswith("_" + name)]
            if len(matches) != 1:
                raise ValueError(f"Expected one MJWarp joint named {name}, found {matches}")
            dofs.append(int(model.jnt_dofadr[matches[0]]))
        self.dof_ids = torch.tensor(dofs, device=robot.device)
        wheel_dofs = [int(model.jnt_dofadr[i]) for i in range(model.njnt) if "_passive_" in model.joint(i).name]
        if wheel_dofs:
            # Match infer_policy.py: passive wheel bearing resistance [N·m].
            wp.to_torch(solver.mjw_model.dof_frictionloss)[:, wheel_dofs] = 0.003
        self.home = torch.tensor(DEFAULT_POSE, device=robot.device).repeat(robot.num_instances, 1)
        self.previous_torque = torch.zeros_like(self.home)
        self.previous_applied_torque = torch.zeros_like(self.home)
        self.target = self.home.clone()
        self.last_action = torch.zeros_like(self.home)
        self.battery = torch.full((robot.num_instances, 1), 7.4, device=robot.device)
        self.sag = torch.full_like(self.battery, 0.1)
        self.friction_scale = torch.ones_like(self.home)
        self.command = torch.zeros((robot.num_instances, 13), device=robot.device)
        # Write the persistent Lab property: the first step synchronizes it into MJWarp.
        robot.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff=params["friction_viscous"], joint_ids=self.joint_ids
        )
        newton_dofs = wp.to_torch(solver.mjc_dof_to_newton_dof)[:, self.dof_ids].long()
        for name, value in (
            ("solreffriction", [-5.0e4, -2.0e2]),
            ("solimpfriction", [0.99, 0.9999, 0.001, 0.5, 2.0]),
        ):
            wp.to_torch(getattr(solver.model.mujoco, name))[newton_dofs] = torch.tensor(value, device=robot.device)
        wp.to_torch(solver.mjw_model.dof_solref)[:, self.dof_ids] = torch.tensor([-5.0e4, -2.0e2], device=robot.device)
        wp.to_torch(solver.mjw_model.dof_solimp)[:, self.dof_ids] = torch.tensor(
            [0.99, 0.9999, 0.001, 0.5, 2.0], device=robot.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear action and motor history after resetting the articulation."""
        env_ids = slice(None) if env_ids is None else env_ids
        self.previous_torque[env_ids] = 0.0
        self.previous_applied_torque[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.target[env_ids] = self.home[env_ids]

    def observations(self) -> torch.Tensor:
        """Return the upstream 61-value input, including commands and last action."""
        data = self.robot.data
        return torch.cat(
            (
                data.root_ang_vel_b.torch,
                data.projected_gravity_b.torch,
                data.joint_pos.torch[:, self.joint_ids] - self.home,
                data.joint_vel.torch[:, self.joint_ids],
                self.last_action,
                self.command,
            ),
            dim=-1,
        )

    def set_action(self, action: torch.Tensor, scale: float = 1.0) -> None:
        """Set policy offsets [rad] for the next 20 ms control interval."""
        if action.shape != self.home.shape or not torch.isfinite(action).all():
            raise ValueError("Policy must return finite actions with shape (num_envs, 14).")
        self.last_action.copy_(action)
        self.target.copy_(self.home + action * scale)

    def apply(self) -> None:
        """Submit motor effort and load-dependent friction for one physics step."""
        p = self.params
        data = self.solver.mjw_data
        velocity = self.robot.data.joint_vel.torch[:, self.joint_ids]
        position = self.robot.data.joint_pos.torch[:, self.joint_ids]
        voltage = (self.battery - self.sag * self.previous_torque.abs().sum(-1, keepdim=True)).clamp(min=6.0)
        torque = motor_torque(self.target - position, velocity, voltage, p)
        efc = data.efc
        kinds, ids, forces = (wp.to_torch(getattr(efc, name)) for name in ("type", "id", "force"))
        valid = torch.arange(kinds.shape[1], device=position.device)[None, :] < wp.to_torch(data.nefc)[:, None]
        friction_rows = valid & (kinds == 1)  # mjCNSTR_FRICTION_DOF
        friction_force = torch.zeros_like(wp.to_torch(data.qfrc_bias))
        friction_force.scatter_add_(
            1, torch.where(friction_rows, ids, 0).long(), torch.where(friction_rows, forces, 0.0)
        )
        external = (-wp.to_torch(data.qfrc_bias) + wp.to_torch(data.qfrc_constraint) - friction_force)[:, self.dof_ids]
        budget = (
            friction_budget(self.previous_applied_torque, external, velocity, p, training_model=self.training_model)
            * self.friction_scale
        )
        # Keep the Newton source buffer consistent when reset-time armature changes trigger synchronization.
        self.robot.data.joint_friction_coeff.torch[:, self.joint_ids] = budget
        wp.to_torch(self.solver.mjw_model.dof_frictionloss)[:, self.dof_ids] = budget
        limit = self.voltage_limit * p["kt"] / p["R"]
        applied = torque.clamp(-limit, limit)
        self.robot.set_joint_effort_target_index(target=applied, joint_ids=self.joint_ids)
        self.previous_applied_torque.copy_(applied)
        self.previous_torque.copy_(torque if self.training_model else applied)

    def self_collision(self) -> torch.Tensor:
        """Return whether each world has a robot/robot contact."""
        data = self.solver.mjw_data
        geoms = wp.to_torch(data.contact.geom)
        world = wp.to_torch(data.contact.worldid)
        valid = torch.arange(geoms.shape[0], device=self.robot.device) < wp.to_torch(data.nacon)[0]
        # The walking scene contains one robot and the global floor (body zero).
        body = torch.as_tensor(self.solver.mj_model.geom_bodyid, device=self.robot.device)
        hit = valid & (geoms >= 0).all(-1) & (body[geoms.clamp_min(0)] > 0).all(-1)
        counts = torch.zeros(self.robot.num_instances, device=self.robot.device)
        counts.scatter_add_(0, world.clamp(0, self.robot.num_instances - 1).long(), hit.float())
        return (counts > 0).float()
