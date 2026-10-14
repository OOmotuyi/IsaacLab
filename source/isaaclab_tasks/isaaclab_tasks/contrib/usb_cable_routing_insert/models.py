# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO residual corrections around a learned route and compliant insertion feedback."""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul

from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import SCurveCommandCfg

from .assets import SOCKET_QUAT
from .mdp import select_socket_orientation


class InsertionGaussianDistribution(GaussianDistribution):
    """Reduce Gaussian exploration tenfold after routing, retaining correct PPO likelihoods."""

    uses_insertion_precision = True
    exploration_scale: torch.Tensor | None = None

    def update(self, mlp_output: torch.Tensor) -> None:
        """Construct the state-conditioned Gaussian used by sampling and PPO log probabilities."""
        super().update(mlp_output)
        if self.exploration_scale is not None:
            self._distribution = torch.distributions.Normal(self.mean, self.std * self.exploration_scale)


class _ResidualInsertionHead(torch.nn.Module):
    def __init__(self, residual: torch.nn.Module, routing_checkpoint: str | None):
        super().__init__()
        self.residual = residual
        self.normalizer = EmpiricalNormalization(195)
        self.routing = MLP(175, 3, [128, 128, 64], "elu")
        self.routing_normalizer = EmpiricalNormalization(175)
        if routing_checkpoint:
            state = torch.load(routing_checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
            self.routing.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")})
            self.routing_normalizer.load_state_dict(
                {k.removeprefix("obs_normalizer."): v for k, v in state.items() if k.startswith("obs_normalizer.")}
            )
        self.routing.requires_grad_(False)
        self.register_buffer("routing_indices", torch.cat((torch.arange(172), torch.arange(189, 192))))
        self.register_buffer("lift", torch.tensor(SCurveCommandCfg().lift_targets))
        self.register_buffer("seat", torch.tensor(SCurveCommandCfg().seat_targets))
        self.register_buffer("socket_quat", torch.tensor(SOCKET_QUAT).unsqueeze(0))
        self.register_buffer("flip", torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
        self.register_buffer("residual_limits", torch.tensor([0.05, 0.05, 0.05, 0.03, 0.03, 0.03]))
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        tip = state[:, 144:147]
        routing_state = state.index_select(1, self.routing_indices).clone()
        guide = (3 * state[:, 150]).round().long().clamp(0, 2)
        waypoint = torch.where((state[:, 153] > 0.5)[:, None], self.seat[guide], self.lift[guide])
        routing_state[:, 147:150] = waypoint - tip
        routing_action = self.routing(self.routing_normalizer(routing_state))
        routed = state[:, 150] >= state[:, 151]
        align, approach = state[:, 187] > 0.5, state[:, 188] > 0.5
        local = state[:, 172:175]
        orientation = state[:, 175:179]
        desired = select_socket_orientation(orientation, state[:, 179:183], self.socket_quat.expand_as(orientation))
        local_error = axis_angle_from_quat(quat_mul(desired, quat_inv(orientation)))
        error = quat_apply(self.socket_quat.expand_as(orientation), local_error)
        depth = torch.where(state[:, 183] > 0.5, 0.006, -0.008)
        # Socket local axes are world y, z, x respectively.
        socket_move = torch.stack((depth - local[:, 2], -local[:, 0], -local[:, 1]), dim=-1)
        move = torch.where(approach[:, None], socket_move, self.seat[-1] - tip)
        translation = move / move.norm(p=2, dim=-1, keepdim=True).clamp(min=0.003) * 0.5
        near = approach & (local[:, 2].abs() < 0.02)
        translation = torch.where(near[:, None], (move / 0.001 * 0.05).clamp(-0.15, 0.15), translation)
        rotation = torch.where(align[:, None], (error * 0.08 / 0.025).clamp(-0.3, 0.3), 0.0)
        baseline = torch.cat((translation, rotation), dim=-1)
        precision = near & (local[:, :2].square().sum(dim=-1) < 0.005**2)
        limits = torch.where(precision[:, None], self.residual_limits * 0.1, self.residual_limits)
        residual = torch.tanh(self.residual(self.normalizer(state))) * limits
        route = torch.cat((routing_action, torch.zeros_like(routing_action)), dim=-1)
        return torch.where(routed[:, None], baseline + residual, route)


class ResidualInsertionModel(MLPModel):
    """Embed a frozen routing actor and learn bounded insertion corrections with PPO.

    The checkpoint includes the routing network and its normalization. Insertion
    feedback uses measured cable/plug state; it does not prescribe cable motion.
    Residual means are bounded to 0.05 translation-action units and 0.03 rotation-
    action units, corresponding to 50 micrometers and 0.00075 radians per step.
    Both correction limits decrease tenfold near the socket.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        routing_checkpoint: str | None = None,
    ):
        if obs_normalization:
            raise ValueError("The residual head normalizes its learned features internally.")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, False, distribution_cfg)
        if self.obs_dim != 195 or output_dim != 6:
            raise ValueError("Residual insertion requires 195 state values and six Cartesian actions.")
        self.mlp = _ResidualInsertionHead(self.mlp, routing_checkpoint)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Retain raw feedback coordinates and condition exploration on measured routing progress."""
        state = super().get_latent(obs, masks, hidden_state)
        if isinstance(self.distribution, InsertionGaussianDistribution):
            routed = state[:, 150] >= state[:, 151]
            self.distribution.exploration_scale = torch.where(routed, 0.1, 1.0)[:, None]
        return state

    def update_normalization(self, obs: TensorDict) -> None:
        """Update only the residual features; preserve the frozen routing normalization."""
        self.mlp.normalizer.update(torch.cat([obs[name] for name in self.obs_groups], dim=-1))
