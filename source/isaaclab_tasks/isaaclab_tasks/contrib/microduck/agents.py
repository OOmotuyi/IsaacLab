# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for MicroDuck training."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MicroDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50_000
    save_interval = 250
    experiment_name = "microduck_velocity"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckSkillPPORunnerCfg(MicroDuckPPORunnerCfg):
    """Conservative PPO fine-tuning from the released deterministic actor."""

    max_iterations = 600
    save_interval = 50
    experiment_name = "microduck_cleanup_skills"
    init_at_random_ep_len = False

    def __post_init__(self):
        self.actor.obs_normalization = False
        self.actor.distribution_cfg.init_std = 0.025
        self.algorithm.learning_rate = 1e-6
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.num_learning_epochs = 4
        self.algorithm.max_grad_norm = 0.2
