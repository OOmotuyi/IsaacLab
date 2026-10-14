# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect paired-state matching without starting Newton or executing a policy."""

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from benchmark_state import assert_matching_initial_states, capture_initial_state, load_evaluation_cases


def _cfg(**values):
    return SimpleNamespace(**values, to_dict=lambda: values)


class TestBenchmarkState(unittest.TestCase):
    """Check exact physical matching, observation matching, and inactive target handling."""

    def setUp(self):
        def zeros(*shape):
            return np.zeros(shape, dtype=np.float32)

        identity = np.array([[0, 0, 0, 1]], dtype=np.float32)
        self.model = SimpleNamespace(
            body_mass=np.ones(3, dtype=np.float32),
            body_inertia=zeros(3, 3, 3),
            body_com=zeros(3, 3),
            joint_target_ke=np.array([10, 20], dtype=np.float32),
            joint_target_kd=zeros(2),
            shape_material_mu=np.array([0.4, 0.5], dtype=np.float32),
            body_label=["robot", "cable0", "cable1"],
            shape_label=["table", "cable"],
            joint_label=["joint0", "joint1"],
        )
        self.state = SimpleNamespace(body_q=zeros(3, 7), body_qd=zeros(3, 6), joint_q=zeros(2), joint_qd=zeros(2))
        cable = SimpleNamespace(
            data=SimpleNamespace(segment_pose_w=zeros(1, 40, 7), segment_velocity_w=zeros(1, 40, 6)),
            grasp_relative_quat=identity.copy(),
            grasp_active=np.ones(1, dtype=np.int32),
            tension=zeros(1),
            grasp_torque=zeros(1),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(joint_pos=zeros(1, 6), joint_vel=zeros(1, 6)),
            actuators=SimpleNamespace(target_command=SimpleNamespace(position=zeros(1, 6))),
            joint_names=[f"joint{i}" for i in range(6)],
        )
        self.command = SimpleNamespace(
            robot=robot,
            cable=cable,
            dof_ids=np.array([[0, 1]]),
            material=np.array([[1, 0.4]], dtype=np.float32),
            guide_positions=zeros(1, 3, 3),
            guide_yaws=zeros(1, 3),
            tool_quat=identity.copy(),
            home_q=zeros(1, 6),
            completed=np.zeros(1, dtype=np.int64),
            guide_count=np.array([3]),
            cfg=_cfg(guide_count=3, curriculum=False),
        )
        self.arm = SimpleNamespace(
            _target_valid=np.array([False]),
            _processed_actions=np.array([[9, 9, 9]], dtype=np.float32),
            target_quat=np.array([[9, 9, 9, 9]], dtype=np.float32),
            _compute_frame_pose=lambda: (zeros(1, 3), identity.copy()),
            cfg=_cfg(scale=0.01),
        )
        self.env = SimpleNamespace(
            num_envs=1,
            episode_length_buf=np.zeros(1, dtype=np.int64),
            command_manager=SimpleNamespace(get_term=lambda _: self.command),
            action_manager=SimpleNamespace(get_term=lambda _: self.arm),
            scene=SimpleNamespace(env_origins=zeros(1, 3)),
            spec=SimpleNamespace(id="paired-test-task"),
            physics_dt=1 / 120,
            step_dt=1 / 30,
            obs_buf={"policy": zeros(1, 175)},
            cfg=SimpleNamespace(
                episode_length_s=60,
                sim=SimpleNamespace(physics=_cfg(num_substeps=8)),
                terminations=_cfg(success="unchanged"),
                scene=SimpleNamespace(cable=SimpleNamespace(spawn=SimpleNamespace(physics_material=_cfg(density=1)))),
            ),
            step=lambda _: self.fail("State capture must not advance physics."),
            reset=lambda **_: self.fail("State capture must not reset the environment."),
        )
        newton = types.ModuleType("isaaclab_newton")
        physics = types.ModuleType("isaaclab_newton.physics")
        physics.NewtonManager = SimpleNamespace(get_model=lambda: self.model, get_state_0=lambda: self.state)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"isaaclab_newton": newton, "isaaclab_newton.physics": physics}).start()

    def test_exact_round_trip_ignores_inactive_stale_targets(self):
        """Ignored prior targets must not reject a physically identical reset."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.json"
            reference = capture_initial_state(self.env, path)
            self.arm._processed_actions[:] = -5
            self.arm.target_quat[:] = -3
            candidate = capture_initial_state(self.env)
            result = assert_matching_initial_states(path, candidate)
            self.assertTrue(result["matched"])
            self.assertEqual(reference["state_sha256"], candidate["state_sha256"])
            self.arm._target_valid[:] = True
            with self.assertRaisesRegex(ValueError, "effective_cartesian"):
                assert_matching_initial_states(reference, capture_initial_state(self.env))

    def test_friction_difference_is_rejected_even_with_equal_geometry(self):
        """A paired-case claim cannot rely on poses or seed labels alone."""
        reference = capture_initial_state(self.env)
        self.model.shape_material_mu[1] = 0.45
        with self.assertRaisesRegex(ValueError, "shape_friction"):
            assert_matching_initial_states(reference, capture_initial_state(self.env))

    def test_policy_input_mismatch_is_separate_from_physical_state(self):
        """Equal physics does not conceal stale or changed initial policy observations."""
        reference = capture_initial_state(self.env)
        self.env.obs_buf["policy"][0, 0] = 1
        candidate = capture_initial_state(self.env)
        self.assertEqual(reference["physical_state_sha256"], candidate["physical_state_sha256"])
        self.assertNotEqual(reference["policy_observation_sha256"], candidate["policy_observation_sha256"])
        with self.assertRaisesRegex(ValueError, "policy_observation"):
            assert_matching_initial_states(reference, candidate)

    def test_modified_array_values_cannot_reuse_a_stored_digest(self):
        """Validate stored array bytes instead of trusting possibly stale hash fields."""
        reference = capture_initial_state(self.env)
        candidate = copy.deepcopy(reference)
        candidate["arrays"]["newton_body_pose"]["values"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "array integrity"):
            assert_matching_initial_states(reference, candidate)

    def test_noninitial_state_is_rejected(self):
        """Only pre-action states qualify as matching initial configurations."""
        self.env.episode_length_buf[0] = 1
        with self.assertRaisesRegex(ValueError, "before any action"):
            capture_initial_state(self.env)


class TestEvaluationManifest(unittest.TestCase):
    """Reject manifests that would change seeds or collapse distinct scored cases."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="usb-cases-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cases.json"
        self.valid = {"case_id": "scurve_001", "variant": "scurve", "seed": 123}

    def _write(self, cases):
        self.path.write_text(json.dumps({"cases": cases}))

    def test_smoke_case_and_same_seed_for_different_tasks_are_valid(self):
        """Keep explicit manifest order and allow small infrastructure smoke runs."""
        insert = {"case_id": "insert_001", "variant": "insert", "seed": self.valid["seed"]}
        self._write([insert, self.valid])
        self.assertEqual(load_evaluation_cases(self.path, "scurve"), [self.valid])
        self.assertEqual(load_evaluation_cases(self.path, "insert"), [insert])

    def test_seed_type_range_and_duplicate_cases_are_rejected(self):
        """Never coerce a seed or silently collapse duplicate identifiers/configurations."""
        for seed in (True, "123", 123.0, -1, 2**32):
            with self.subTest(seed=seed):
                self._write([{**self.valid, "seed": seed}])
                with self.assertRaisesRegex(ValueError, "seed"):
                    load_evaluation_cases(self.path, "scurve")
        for duplicate, message in (
            ({**self.valid, "seed": 124}, "Duplicate case_id"),
            ({**self.valid, "case_id": "scurve_002"}, "Repeated seed"),
        ):
            with self.subTest(duplicate=duplicate):
                self._write([self.valid, duplicate])
                with self.assertRaisesRegex(ValueError, message):
                    load_evaluation_cases(self.path, "scurve")

    def test_case_identifier_cannot_escape_its_output_directory(self):
        """Case identifiers are directory components, not arbitrary paths."""
        for identifier in ("", ".", "..", "../other", "/absolute", "nested/path", "nested\\path"):
            with self.subTest(identifier=identifier):
                self._write([{**self.valid, "case_id": identifier}])
                with self.assertRaisesRegex(ValueError, "case_id"):
                    load_evaluation_cases(self.path, "scurve")


if __name__ == "__main__":
    unittest.main()
