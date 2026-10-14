# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transfer a compatible S-curve policy into a direct or residual insertion checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def initialize(routing_path: Path, template_path: Path, output: Path):
    """Copy shared observations and translation outputs, with fresh insertion optimizer state."""
    source = torch.load(routing_path, map_location="cpu", weights_only=False)
    target = torch.load(template_path, map_location="cpu", weights_only=False)
    # S-curve: 172 state values + 3 previous translations.
    # Insertion: those 172 values + 17 socket/grasp/phase values + 6 previous actions.
    indices = torch.cat((torch.arange(172), torch.arange(189, 192)))
    for group in ("actor_state_dict", "critic_state_dict"):
        old, new = source[group], target[group]
        if old["mlp.0.weight"].shape != (128, 175) or new["mlp.0.weight"].shape != (128, 195):
            raise ValueError("Expected the maintained 175-observation routing and 195-observation insertion MLPs.")
        for key, value in new.items():
            previous = old[key]
            if value.shape == previous.shape:
                value.copy_(previous)
            elif key == "mlp.0.weight":
                value.zero_()
                value[:, indices] = previous
            elif key.startswith("obs_normalizer."):
                value.fill_(0.0 if key.endswith("_mean") else 1.0)
                value[:, indices] = previous
            elif group == "actor_state_dict" and key in ("mlp.6.weight", "mlp.6.bias"):
                value.zero_()
                value[:3] = previous
            elif key == "distribution.std_param":
                value.fill_(0.15)
                value[:3] = previous
            else:
                raise ValueError(f"Unsupported checkpoint field: {group}.{key}")
        # Keep shared statistics but let the new socket features adapt promptly.
        new["obs_normalizer.count"].fill_(128)
    target["optimizer_state_dict"]["state"] = {}
    target["iter"] = 0
    metadata = {
        "method": "PPO parameter transfer; no scripted actions or demonstrations",
        "routing_checkpoint": str(routing_path),
        "routing_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(),
        "insertion_template": str(template_path),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "shared_input_indices": indices.tolist(),
        "new_rotation_std": 0.15,
    }
    target["infos"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


def update_residual_routing(routing_path: Path, template_path: Path, output: Path) -> None:
    """Replace embedded routing weights while preserving trained insertion corrections."""
    source = torch.load(routing_path, map_location="cpu", weights_only=False)
    target = torch.load(template_path, map_location="cpu", weights_only=False)
    old, new = source["actor_state_dict"], target["actor_state_dict"]
    copied = []
    for key, value in old.items():
        if key.startswith("mlp."):
            destination = "mlp.routing." + key.removeprefix("mlp.")
        elif key.startswith("obs_normalizer."):
            destination = "mlp.routing_normalizer." + key.removeprefix("obs_normalizer.")
        else:
            continue
        if destination not in new or new[destination].shape != value.shape:
            raise ValueError(f"Incompatible embedded routing field: {destination}")
        new[destination].copy_(value)
        copied.append(destination)
    expected = {key for key in new if key.startswith(("mlp.routing.", "mlp.routing_normalizer."))}
    if not copied or set(copied) != expected:
        raise ValueError("The template must contain the complete residual insertion routing network.")
    target["optimizer_state_dict"]["state"] = {}
    metadata = {
        "method": "Updated frozen routing actor; preserved trained insertion residual and critic",
        "routing_checkpoint": str(routing_path),
        "routing_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(),
        "insertion_template": str(template_path),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "copied_actor_fields": copied,
        "insertion_training_iteration": target["iter"],
    }
    target["infos"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing_checkpoint", type=Path, required=True)
    parser.add_argument("--insertion_template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--update_residual_routing", action="store_true", help="Preserve a residual template's trained insertion head."
    )
    args = parser.parse_args()
    transfer = update_residual_routing if args.update_residual_routing else initialize
    transfer(args.routing_checkpoint, args.insertion_template, args.output)
