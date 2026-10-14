# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect benchmark verification quarantine without launching a simulator."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from server import DirectBridge


class TestDirectServer(unittest.TestCase):
    """Reject physical commands after a paired initial-state comparison fails."""

    def test_benchmark_reset_requires_fresh_process(self):
        """Reject a reset even after native termination to avoid cached reset history."""
        bridge = DirectBridge.__new__(DirectBridge)
        bridge.evaluation_cases = {1: "case_001", 2: "case_002"}
        for terminal in (None, {"success": False}):
            with self.subTest(terminal=terminal):
                bridge.terminal = terminal
                with self.assertRaisesRegex(ValueError, "fresh simulator process"):
                    bridge.execute({"op": "reset", "seed": 2, "rationale": "Check process isolation."})

    def test_failed_verification_blocks_move_and_hold(self):
        """A previously verified bridge must quarantine a mismatched subsequent case."""
        with tempfile.TemporaryDirectory(prefix="astra-server-test-") as temporary:
            bridge = DirectBridge.__new__(DirectBridge)
            bridge.evaluation_cases = {1: "case_001"}
            bridge.initial_state_verified = True
            bridge.case_id = "case_001"
            bridge.reference_dir = Path(temporary) / "reference"
            bridge.episode_dir = Path(temporary)
            bridge.terminal = None
            bridge.variant = "scurve"
            position = torch.zeros((1, 3))
            quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
            bridge.env = SimpleNamespace(
                device="cpu",
                step_dt=1 / 30,
                action_manager=SimpleNamespace(total_action_dim=3),
                scene=SimpleNamespace(env_origins=position),
            )
            bridge.command = SimpleNamespace(
                completed=torch.tensor([0]),
                robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=position))),
            )
            bridge.arm = SimpleNamespace(_target_valid=torch.tensor([False]), cfg=SimpleNamespace(scale=0.01))
            bridge.tcp_pose = Mock(return_value=(position, quaternion))
            bridge.frame_pose = Mock(return_value=(position, quaternion))
            bridge.step = Mock(side_effect=AssertionError("Physics stepped after failed verification."))
            with (
                patch("benchmark_state.save_initial_state", return_value={}),
                patch(
                    "benchmark_state.assert_matching_initial_states", side_effect=ValueError("Initial states differ.")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Initial states differ"):
                    bridge.verify_initial_state()

            comparison = json.loads((bridge.episode_dir / "initial_state_comparison.json").read_text())
            self.assertFalse(comparison["matched"])
            self.assertIn("Initial states differ", comparison["error"])
            for operation in ("move", "hold"):
                with self.subTest(operation=operation):
                    request = {"op": operation, "steps": 1, "rationale": "Check quarantined-case rejection."}
                    if operation == "move":
                        request.update(frame="tcp", target_position_m=[0.5, -0.2, 0.2])
                    with self.assertRaisesRegex(ValueError, "quarantined"):
                        bridge.execute(request)
                    bridge.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
