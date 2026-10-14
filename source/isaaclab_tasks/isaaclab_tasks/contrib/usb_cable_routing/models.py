# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional PPO model for refinement of an actor initialized from demonstrations."""

from rsl_rl.models import MLPModel
from tensordict import TensorDict


class FixedNormalizationModel(MLPModel):
    """Preserve loaded observation statistics while PPO updates the policy weights.

    Use this model only with a checkpoint containing initialized normalization.
    Its checkpoint and inference formats match the standard RSL-RL MLP model.
    """

    def update_normalization(self, obs: TensorDict) -> None:
        """Keep demonstration normalization fixed during policy refinement."""
