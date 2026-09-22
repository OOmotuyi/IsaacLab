# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load the released MicroDuck actor for verified initialization and training priors."""

from pathlib import Path

import torch
from torch import nn


class MicroDuckSourceActor(nn.Module):
    """Evaluate the released 61-input, 14-output ELU policy without ONNX Runtime."""

    def __init__(self, path: str | Path):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        values = {item.name: torch.from_numpy(numpy_helper.to_array(item).copy()) for item in model.graph.initializer}
        self.register_buffer("mean", values["obs_normalizer._mean"])
        self.register_buffer("scale", values["onnx::Div_24"])
        if self.mean.shape != (1, 61) or not bool((self.scale > 0).all()):
            raise ValueError("Expected the released 61-value MicroDuck normalization.")
        self.mlp = nn.Sequential(
            nn.Linear(61, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 14),
        )
        self.mlp.load_state_dict(
            {name.removeprefix("mlp."): value for name, value in values.items() if name.startswith("mlp.")}
        )
        self.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return fourteen joint offsets [rad] for the released observation contract."""
        return self.mlp((observation - self.mean) / self.scale)

    def folded_state_dict(self) -> dict[str, torch.Tensor]:
        """Fold fixed observation normalization into the first linear layer."""
        state = {name: value.clone() for name, value in self.mlp.state_dict().items()}
        state["0.weight"] /= self.scale
        state["0.bias"] -= state["0.weight"] @ self.mean[0]
        return {"mlp." + name: value for name, value in state.items()}
