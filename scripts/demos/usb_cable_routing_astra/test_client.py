# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the direct-control client's public protocol without starting a simulator."""

import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


class TestDirectClient(unittest.TestCase):
    """Protect state-changing command deduplication and request validation."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="astra-client-test-")
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name)
        (self.run_dir / "commands").mkdir()
        (self.run_dir / "responses").mkdir()
        self.client = Path(__file__).with_name("client.py")

    def _run(self, request: dict, request_id: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(self.client),
                "--run_dir",
                str(self.run_dir),
                "--request",
                json.dumps(request),
                "--request_id",
                request_id,
                "--timeout_s",
                "0.01",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def test_timeout_resume_does_not_repeat_consumed_movement(self):
        """A missing reply never permits another physical execution of the same request."""
        request_id = str(uuid.uuid4())
        request = {"op": "hold", "steps": 1, "rationale": "Explicit command for the isolated protocol test."}
        command_path = self.run_dir / "commands" / f"{request_id}.json"
        first = self._run(request, request_id)
        self.assertEqual(first.returncode, 124, first.stderr)
        self.assertEqual(json.loads(command_path.read_text()), request)

        command_path.unlink()  # Simulate a server consuming the command before its response is ready.
        pending = self._run(request, request_id)
        self.assertEqual(pending.returncode, 124, pending.stderr)
        self.assertFalse(command_path.exists())

        response = {"status": "ok", "executed_steps": 1, "observation": {"done": False}}
        (self.run_dir / "responses" / f"{request_id}.json").write_text(json.dumps(response))
        complete = self._run(request, request_id)
        self.assertEqual(complete.returncode, 0, complete.stderr)
        self.assertEqual(json.loads(complete.stdout), response)
        self.assertFalse(command_path.exists())

        altered = self._run({**request, "steps": 2}, request_id)
        self.assertEqual(altered.returncode, 1)
        self.assertIn("different request", altered.stderr)
        self.assertFalse(command_path.exists())

    def test_socket_camera_request_matches_server_contract(self):
        """Astra can request the socket close-up through the supported client."""
        request_id = str(uuid.uuid4())
        request = {"op": "observe", "camera": "socket"}
        result = self._run(request, request_id)
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertEqual(json.loads((self.run_dir / "commands" / f"{request_id}.json").read_text()), request)

    def test_invalid_physical_commands_are_not_published(self):
        """Reject missing rationales, nonfinite poses, and out-of-contract motion bounds."""
        valid = {
            "op": "move",
            "frame": "tcp",
            "target_position_m": [0.5, -0.2, 0.2],
            "steps": 1,
            "rationale": "Validate request bounds without executing a simulation.",
        }
        for override in (
            {"rationale": None},
            {"steps": True},
            {"steps": 91},
            {"target_position_m": [float("nan"), 0, 0]},
            {"target_quat_xyzw": [0, 0, 0, 0]},
            {"speed_m_s": 0.16},
            {"angular_speed_rad_s": 0.51},
            {"stop_on_stage_change": "false"},
        ):
            with self.subTest(override=override):
                result = self._run({**valid, **override}, str(uuid.uuid4()))
                self.assertEqual(result.returncode, 1)
        self.assertEqual(list((self.run_dir / "commands").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
