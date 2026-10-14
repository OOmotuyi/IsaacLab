# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional residual PPO recipe that retains the learned S-curve routing policy."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .env_cfg import UsbInsertPPORunnerCfg


@configclass
class ResidualInsertionActorCfg(RslRlMLPModelCfg):
    """Frozen S-curve policy plus a trainable residual insertion head."""

    class_name = "isaaclab_tasks.contrib.usb_cable_routing_insert.models:ResidualInsertionModel"
    hidden_dims = [128, 128, 64]
    activation = "elu"
    obs_normalization = False
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
        class_name="isaaclab_tasks.contrib.usb_cable_routing_insert.models:InsertionGaussianDistribution",
        init_std=0.02,
    )
    routing_checkpoint: str | None = None
    """Routing checkpoint for initial training; embedded weights suffice for subsequent playback."""


@configclass
class UsbInsertResidualPPORunnerCfg(UsbInsertPPORunnerCfg):
    """Conservative PPO refinement of the insertion feedback controller."""

    actor = ResidualInsertionActorCfg()
    clip_actions = None

    def __post_init__(self):
        self.algorithm.learning_rate = 0.0001
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0001
