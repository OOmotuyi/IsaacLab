# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initialize a routing actor from physically successful demonstrations before PPO refinement."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--action_std", type=float, default=0.20)
    parser.add_argument(
        "--ignore_previous_actions", action="store_true", help="Fit state feedback without previous-action inputs."
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(args.demonstrations, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    good = [row for row in data["records"] if row["success"]]
    if data["observations"].ndim == 3:
        data["observations"] = torch.cat([data["observations"][: row["step"] + 1, row["environment"]] for row in good])
        data["actions"] = torch.cat([data["actions"][: row["step"] + 1, row["environment"]] for row in good])
    input_dim = state["mlp.0.weight"].shape[1]
    action_dim = state["mlp.6.weight"].shape[0]
    if data["observations"].shape[1] != input_dim or data["actions"].shape[1] != action_dim:
        raise ValueError("Demonstration observations/actions do not match the checkpoint.")
    actor = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, action_dim),
    ).to(device)
    actor.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")})
    inputs = ((data["observations"] - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)).to(device)
    if args.ignore_previous_actions:
        inputs[:, -action_dim:] = 0.0
        with torch.no_grad():
            actor[0].weight[:, -action_dim:] = 0.0
    labels = data["actions"].clamp(-1, 1).to(device)
    successes = [row for row in data["records"] if row["success"]]
    if len(successes) < 2:
        raise ValueError("Require at least two complete successful physical trajectories.")
    # Hold out a whole trajectory; adjacent frames must not leak across this split.
    split = len(inputs) - successes[-1]["step"] - 1
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    best_loss, best = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(split, device=device)
        for indices in order.split(256):
            loss = torch.nn.functional.mse_loss(actor(inputs[indices]), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(actor(inputs[split:]), labels[split:]))
        if validation < best_loss:
            best_loss, best = validation, copy.deepcopy(actor.state_dict())
        if epoch % 20 == 0:
            history.append({"epoch": epoch, "validation_mse": validation})
            print(history[-1], flush=True)
    for key, value in best.items():
        state["mlp." + key] = value.cpu()
    state["distribution.std_param"].fill_(args.action_std)
    checkpoint["optimizer_state_dict"]["state"] = {}
    for group in checkpoint["optimizer_state_dict"]["param_groups"]:
        group["lr"] = 3e-4
    metadata = {
        "method": "behavior cloning initialization followed by PPO; final evaluation uses the learned actor only",
        "source_checkpoint": str(args.checkpoint),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "demonstrations": str(args.demonstrations),
        "demonstration_sha256": hashlib.sha256(args.demonstrations.read_bytes()).hexdigest(),
        "trajectories": len(successes),
        "samples": len(inputs),
        "training_samples": split,
        "validation_mse": best_loss,
        "ignore_previous_actions": args.ignore_previous_actions,
        "history": history,
    }
    checkpoint["infos"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output, best_loss)


if __name__ == "__main__":
    main()
