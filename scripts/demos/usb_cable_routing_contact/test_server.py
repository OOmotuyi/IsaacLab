# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect the public controller information boundary without launching physics."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

_SERVER_SPEC = importlib.util.spec_from_file_location(
    "_contact_server_under_test", Path(__file__).with_name("server.py")
)
_SERVER = importlib.util.module_from_spec(_SERVER_SPEC)
_SERVER_SPEC.loader.exec_module(_SERVER)


def _measurements(variant):
    shapes = {**_SERVER._MEASUREMENT_SHAPES, "last_action": (6 if variant == "insert" else 3,)}
    if variant == "insert":
        shapes.update(socket_position_m=(3,), socket_quat_xyzw=(4,))
    values = {name: torch.zeros((1, *shape)) for name, shape in shapes.items()}
    for name, value in values.items():
        if name.endswith("quat_xyzw"):
            value[:, 3] = 1
    return values


class _ForbiddenTaskState:
    def __getattribute__(self, name):
        raise AssertionError(f"The public TCP servo read privileged task state: {name}")


class TestContactServer(unittest.TestCase):
    """Reject privileged observations and execute moves independently of hidden task progress."""

    def test_extra_privileged_measurement_rejected(self):
        """A shared observation change cannot silently expose evaluator-only signals."""
        for variant in ("scurve", "insert"):
            values = _measurements(variant)
            serialized = _SERVER.serialize_measurements(values, variant)
            self.assertEqual(set(serialized), set(values))
            for forbidden in ("cable_points_m", "grasp_active", "minimum_bend_radius_m", "completed", "friction"):
                with self.subTest(variant=variant, forbidden=forbidden):
                    with self.assertRaisesRegex(ValueError, "public allowlist"):
                        _SERVER.serialize_measurements({**values, forbidden: torch.ones((1, 1))}, variant)

    def test_private_recorder_distinguishes_pinch_and_socket_load(self):
        """Contact-task insertion audit retains socket load separately from fingertip force."""
        bridge = _SERVER.ContactBridge.__new__(_SERVER.ContactBridge)
        bridge.command = SimpleNamespace(
            cable=SimpleNamespace(
                grasp_acquired=torch.tensor([1]),
                grasp_dropped=torch.tensor([0]),
                pad_normal_forces=torch.tensor([[2.0, 2.2]]),
                pad_contact_count=torch.tensor([[2, 3]]),
                position_slip=torch.tensor([0.001]),
                rotation_slip=torch.tensor([0.02]),
                socket_contact_force=torch.tensor([0.25]),
                socket_contact_force_w=torch.tensor([[0.25, 0.0, 0.0]]),
            )
        )
        with (
            patch.object(_SERVER._RECORDER.DirectBridge, "measure", return_value={"grasp_force_n": 0.5}),
            patch.object(_SERVER.wp, "to_torch", side_effect=lambda value: value),
        ):
            record = bridge.measure()
        self.assertEqual(record["grasp_force_n"], 0.5)
        self.assertEqual(record["socket_contact_force_n"], 0.25)
        self.assertEqual(record["socket_contact_force_w_n"], [0.25, 0.0, 0.0])
        self.assertEqual(record["pad_contact_count"], [2, 3])
        self.assertTrue(record["contact_grasp_acquired"])
        self.assertFalse(record["contact_grasp_dropped"])
        self.assertIn("fingertip", record["grasp_force_n_semantics"])

    def test_terminal_publishes_final_success_without_diagnostics_or_paths(self):
        """A terminal response retains the allowed pre-reset measurements, not private evaluator data."""
        with tempfile.TemporaryDirectory(prefix="contact-server-test-") as temporary:
            bridge = _SERVER.ContactBridge.__new__(_SERVER.ContactBridge)
            bridge.variant = "insert"
            bridge.render_images = False
            bridge.public_directory = Path(temporary)
            bridge.terminal = {
                "success": False,
                "grasp_force_n": 100.0,
                "terminations": {"dropped": True},
                "episode_directory": "/private/evaluator/trace",
            }
            bridge._actor_measurements = _SERVER.serialize_measurements(_measurements("insert"), "insert")
            bridge._read_measurements = Mock(side_effect=AssertionError("Read post-reset state."))
            result = bridge.observation(render_image=False)
            self.assertEqual(set(result), {"measurements", "measurement_model", "done", "final_success"})
            self.assertTrue(result["done"])
            self.assertFalse(result["final_success"])
            self.assertNotIn("/private/", json.dumps(result))
            self.assertNotIn("terminations", result)
            self.assertNotIn("grasp_force_n", result["measurements"])
            bridge._read_measurements.assert_not_called()

    def test_controller_trajectory_aligns_actions_and_pre_reset_terminal(self):
        """Private shared vectors retain N+1 states for N actions, including the native terminal."""
        for variant, width, action_width in (("scurve", 55, 3), ("insert", 65, 6)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(prefix="contact-trajectory-") as temporary:
                values = _measurements(variant)
                directory = Path(temporary)
                visualizer = SimpleNamespace(
                    _viewer=SimpleNamespace(renderer=SimpleNamespace(window=SimpleNamespace(set_visible=Mock())))
                )
                env = SimpleNamespace(
                    command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace()),
                    action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace()),
                    sim=SimpleNamespace(visualizers=[visualizer]),
                    spec=SimpleNamespace(id="contact-test"),
                    step_dt=0.05,
                    termination_manager=SimpleNamespace(
                        dones=torch.tensor([False]),
                        active_terms=["success"],
                        get_term=lambda _: torch.tensor([False]),
                    ),
                )

                def reset(_):
                    values["tcp_position_m"].fill_(99)

                env._reset_idx = reset
                step_count = 0

                def step(action):
                    nonlocal step_count
                    step_count += 1
                    values["tcp_position_m"].fill_(step_count)
                    values["last_action"][:] = action
                    terminal = step_count == 2
                    if terminal:
                        env.termination_manager.dones[:] = True
                        env._reset_idx(torch.tensor([0]))
                    return None, None, torch.tensor([terminal]), torch.tensor([False]), {}

                env.step = step
                with (
                    patch.object(
                        _SERVER.NewtonManager,
                        "get_state",
                        return_value=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.zeros((1, 7)))),
                    ),
                    patch.object(_SERVER.NewtonManager, "get_model", return_value=SimpleNamespace(body_label=["fake"])),
                    patch.object(_SERVER.ContactBridge, "_read_measurements", return_value=values),
                    patch.object(
                        _SERVER.ContactBridge,
                        "measure",
                        lambda bridge: {
                            "time_s": len(bridge.actions) * env.step_dt,
                            "minimum_bend_radius_m": 0.02,
                        },
                    ),
                ):
                    bridge = _SERVER.ContactBridge(env, directory, variant, 1)
                    path = bridge.episode_dir / "controller_trajectory.npz"
                    with np.load(path) as initial:
                        self.assertEqual(initial["observations"].shape, (1, width))
                        self.assertEqual(initial["actions"].shape, (0, action_width))
                    first = torch.full((1, action_width), 0.1)
                    second = torch.full((1, action_width), -0.2)
                    self.assertFalse(bridge.step(first))
                    self.assertTrue(bridge.step(second))
                    bridge.save()
                    result = bridge.observation(render_image=False)
                self.assertEqual(result["measurements"]["tcp_position_m"], [2.0, 2.0, 2.0])
                self.assertTrue(bool((values["tcp_position_m"] == 99).all()))
                with np.load(path) as data:
                    self.assertEqual(data["observations"].shape, (3, width))
                    np.testing.assert_array_equal(data["actions"], np.concatenate((first.numpy(), second.numpy())))
                    np.testing.assert_allclose(data["observation_time_s"], [0, 0.05, 0.1])
                    np.testing.assert_allclose(data["action_time_s"], [0, 0.05])
                    fields = data["observation_fields"].tolist()
                    offset = sum(values[name].numel() for name in fields[: fields.index("tcp_position_m")])
                    np.testing.assert_array_equal(
                        data["observations"][:, offset : offset + 3], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
                    )
                    self.assertEqual(set(fields), set(values))

    def test_tcp_servo_uses_constant_scale_without_task_state(self):
        """Insertion servoing uses only allowed command pose, with no cable frame or stage stopping."""
        bridge = _SERVER.ContactBridge.__new__(_SERVER.ContactBridge)
        bridge.variant = "insert"
        bridge.render_images = False
        bridge.terminal = None
        bridge.command = _ForbiddenTaskState()
        bridge.env = SimpleNamespace(device="cpu", step_dt=1 / 30, action_manager=SimpleNamespace(total_action_dim=6))
        bridge.arm = SimpleNamespace(cfg=SimpleNamespace(scale=0.01, insertion_scale=1e-8, rotation_scale=0.02))
        values = _measurements("insert")
        bridge._read_measurements = lambda: values
        bridge.save = Mock()
        bridge.observation = Mock(return_value={"done": False})
        actions = []

        def step(action):
            actions.append(action.clone())
            values["commanded_tcp_position_m"] += action[:, :3] * bridge.arm.cfg.scale
            return False

        bridge.step = step
        result = bridge.execute(
            {
                "op": "move",
                "frame": "tcp",
                "target_position_m": [0.009, 0.0, 0.0],
                "steps": 3,
                "speed_m_s": 0.15,
                "rationale": "Move the TCP to the measured target.",
            }
        )
        self.assertEqual(result["executed_steps"], 3)
        self.assertEqual(result["stop_reason"], "duration")
        torch.testing.assert_close(torch.stack(actions)[:, 0, 0], torch.tensor([0.5, 0.4, 0.0]))
        torch.testing.assert_close(values["commanded_tcp_position_m"], torch.tensor([[0.009, 0.0, 0.0]]))

    def test_camera_disabled_for_matched_vector_inputs(self):
        """Astra cannot request an extra image modality in the default comparison mode."""
        bridge = _SERVER.ContactBridge.__new__(_SERVER.ContactBridge)
        bridge.variant = "insert"
        bridge.render_images = False
        bridge.terminal = None
        bridge.observation = Mock(return_value={"done": False})
        for camera in ("routing", "socket"):
            with self.subTest(camera=camera):
                with self.assertRaisesRegex(ValueError, "Camera access is disabled"):
                    bridge.execute({"op": "observe", "camera": camera})
        bridge.observation.assert_not_called()
        bridge.render_images = True
        bridge.execute({"op": "observe", "camera": "socket"})
        bridge.observation.assert_called_once_with("socket", render_image=True)

    def test_hidden_frame_stage_stopping_and_reset_rejected(self):
        """Requests cannot activate legacy privileged servo options or reset a case."""
        bridge = _SERVER.ContactBridge.__new__(_SERVER.ContactBridge)
        bridge.variant = "insert"
        bridge.render_images = False
        bridge.terminal = None
        bridge.step = Mock(side_effect=AssertionError("Rejected request advanced physics."))
        for request in (
            {"op": "reset"},
            {"op": "move", "frame": "plug_front"},
            {"op": "move", "frame": "tip"},
            {"op": "hold", "stop_on_stage_change": True},
        ):
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    bridge.execute(request)
        bridge.step.assert_not_called()


class TestContactEvaluation(unittest.TestCase):
    """Protect recurrent inference and one-attempt evaluation independently of physics."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "_contact_evaluator_under_test", Path(__file__).with_name("evaluate_rl.py")
        )
        cls.evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluator)

    @staticmethod
    def training_config():
        return {
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "clip_actions": 1.0,
            "actor": {
                "class_name": "RNNModel",
                "hidden_dims": [8],
                "activation": "elu",
                "obs_normalization": True,
                "rnn_type": "gru",
                "rnn_hidden_dim": 8,
                "rnn_num_layers": 1,
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.3},
            },
        }

    def test_checkpoint_and_export_preserve_recurrent_actions_and_reset(self):
        from rsl_rl.models import RNNModel
        from tensordict import TensorDict

        config = self.training_config()
        actor_cfg = {key: value for key, value in config["actor"].items() if key != "class_name"}
        torch.manual_seed(741)
        observation = torch.randn(1, 55)
        actor = RNNModel(
            TensorDict({"policy": observation}, batch_size=[1]), config["obs_groups"], "actor", 3, **actor_cfg
        )
        actor.eval()
        with tempfile.TemporaryDirectory(prefix="contact-policy-test-") as temporary:
            checkpoint, exported = Path(temporary) / "model.pt", Path(temporary) / "policy.pt"
            torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
            torch.jit.script(actor.as_jit()).save(str(exported))
            policies = [
                self.evaluator._load_policy(checkpoint, "checkpoint", config, 55, 3, "cpu"),
                self.evaluator._load_policy(exported, "policy", config, 55, 3, "cpu"),
            ]
            with torch.inference_mode():
                outputs = []
                for policy in policies:
                    first = self.evaluator._policy_action(policy, observation, 55, 3)
                    second = self.evaluator._policy_action(policy, observation, 55, 3)
                    self.assertFalse(torch.equal(first, second))
                    policy.reset()
                    torch.testing.assert_close(self.evaluator._policy_action(policy, observation, 55, 3), first)
                    outputs.append(torch.stack((first, second)))
                torch.testing.assert_close(outputs[0], outputs[1])
                clipped = self.evaluator._policy_action(lambda _: torch.tensor([[2.0, -3.0, 0.5]]), observation, 55, 3)
                torch.testing.assert_close(clipped, torch.tensor([[1.0, -1.0, 0.5]]))
                with self.assertRaisesRegex(ValueError, "restricted shape"):
                    self.evaluator._policy_action(policies[0], torch.zeros(1, 175), 55, 3)

    def test_evaluation_stops_after_first_worker_error_without_retry(self):
        import yaml

        with tempfile.TemporaryDirectory(prefix="contact-evaluation-test-") as temporary:
            directory = Path(temporary)
            manifest, artifact, training = directory / "cases.json", directory / "model.pt", directory / "agent.yaml"
            cases = [{"case_id": f"scurve_{i}", "variant": "scurve", "seed": i} for i in (1, 2)]
            manifest.write_text(json.dumps({"cases": cases}))
            artifact.write_bytes(b"not-loaded-in-parent")
            training.write_text(yaml.safe_dump(self.training_config()))
            output = directory / "results"
            argv = [
                "evaluate_rl.py",
                "--manifest",
                str(manifest),
                "--variant",
                "scurve",
                "--checkpoint",
                str(artifact),
                "--training_config",
                str(training),
                "--output_dir",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(self.evaluator, "_source_hashes", return_value={}),
                patch.object(self.evaluator.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as worker,
            ):
                with self.assertRaisesRegex(RuntimeError, "status 7"):
                    self.evaluator.main()
            self.assertEqual(worker.call_count, 1)
            self.assertIn("--worker_case_id", worker.call_args.args[0])
            self.assertEqual(worker.call_args.args[0][-1], "scurve_1")
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["completed_cases"], 0)
            self.assertEqual(progress["error"]["case_id"], "scurve_1")
            self.assertFalse((output / "evaluation.json").exists())
            config = self.training_config()
            config["clip_actions"] = None
            training.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "action clipping"):
                self.evaluator._read_training_config(training)

    def test_shared_task_configuration_preserves_direct_defaults(self):
        for variant in ("scurve", "insert"):
            cfg = _SERVER.build_task_cfg(variant, 741)
            self.assertEqual(cfg.scene.num_envs, 1)
            self.assertEqual(cfg.seed, 741)
            self.assertEqual(cfg.commands.routing.guide_count, 3)
            self.assertFalse(cfg.commands.routing.curriculum)
            self.assertTrue(cfg.commands.routing.randomize)
            self.assertTrue(cfg.commands.routing.randomize_guides)
            self.assertTrue(cfg.sim.physics.use_cuda_graph)
            self.assertEqual(len(cfg.sim.visualizer_cfgs), 1)
            development = _SERVER.build_task_cfg(variant, 742, randomize=False, use_cuda_graph=False)
            self.assertFalse(development.commands.routing.randomize)
            self.assertFalse(development.commands.routing.randomize_guides)
            self.assertFalse(development.sim.physics.use_cuda_graph)


class TestContactDirectEvaluation(unittest.TestCase):
    """Match one private initial state before exposing direct-controller input."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "_contact_direct_evaluator_under_test", Path(__file__).with_name("evaluate_direct.py")
        )
        cls.evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluator)

    def make_arguments(self, root):
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"cases": [{"case_id": "route_a", "variant": "scurve", "seed": 42}]}))
        metadata = root / "controller.json"
        metadata.write_text(json.dumps({"controller": "declared direct", "model": "declared model", "prompt": "test"}))
        reference = root / "reference/route_a"
        reference.mkdir(parents=True)
        (reference / "initial_state.json").write_text("{}")
        return SimpleNamespace(
            manifest=manifest,
            variant="scurve",
            case_id="route_a",
            output_dir=root / "output",
            initial_reference_dir=reference.parent,
            controller_metadata=metadata,
        )

    def test_case_metadata_and_reference_are_frozen_without_retries(self):
        with tempfile.TemporaryDirectory(prefix="direct-provenance-") as temporary:
            args = self.make_arguments(Path(temporary))
            with patch.object(self.evaluator, "_source_hashes", return_value={"source": "original"}):
                case, provenance, directory = self.evaluator._prepare_case(args)
                self.assertEqual(case, {"case_id": "route_a", "variant": "scurve", "seed": 42})
                self.assertIn("not verified", provenance["model_identity_and_context_isolation"])
                self.assertTrue((directory / "case.json").is_file())
                self.evaluator._assert_frozen(provenance)
                with self.assertRaises(FileExistsError):
                    self.evaluator._prepare_case(args)
                for name in ("manifest", "controller_metadata", "initial_reference"):
                    path = Path(provenance[name])
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    with self.subTest(changed=name), self.assertRaisesRegex(RuntimeError, f"Frozen {name}"):
                        self.evaluator._assert_frozen(provenance)
                    path.write_bytes(original)
            with patch.object(self.evaluator, "_source_hashes", return_value={"source": "changed"}):
                with self.assertRaisesRegex(RuntimeError, "sources changed"):
                    self.evaluator._assert_frozen(provenance)

    def test_initial_mismatch_prevents_public_inputs_after_one_reset(self):
        with tempfile.TemporaryDirectory(prefix="direct-initial-mismatch-") as temporary:
            args = self.make_arguments(Path(temporary))
            env = SimpleNamespace(reset=Mock())
            factory = Mock()
            with (
                patch.object(self.evaluator, "_source_hashes", return_value={}),
                patch.object(self.evaluator._COMMON._STATE, "capture_initial_state", return_value={}) as capture,
                patch.object(
                    self.evaluator._COMMON._STATE,
                    "assert_matching_initial_states",
                    side_effect=ValueError("Initial configurations do not match exactly: newton_body_pose"),
                ),
            ):
                case, provenance, directory = self.evaluator._prepare_case(args)
                with self.assertRaisesRegex(ValueError, "do not match") as caught:
                    self.evaluator._run_case(env, case, provenance, directory, factory)
                self.evaluator._record_error(case, directory, caught.exception)
                with self.assertRaises(FileExistsError):
                    self.evaluator._prepare_case(args)
            env.reset.assert_called_once_with(seed=42)
            capture.assert_called_once_with(env, directory / "initial_state.json")
            factory.assert_not_called()
            self.assertFalse((directory / "commands").exists())
            self.assertFalse((directory / "public").exists())
            self.assertFalse((directory / "result.json").exists())
            self.assertFalse(json.loads((directory / "infrastructure_error.json").read_text())["completed"])

    def test_terminal_and_early_close_end_one_attempt(self):
        for early_close in (False, True):
            with self.subTest(early_close=early_close), tempfile.TemporaryDirectory(prefix="direct-terminal-") as temp:
                args = self.make_arguments(Path(temp))
                events = []
                env = SimpleNamespace(reset=Mock(side_effect=lambda **_: events.append("reset")))
                initial = {"physical_state_sha256": "physical", "policy_observation_sha256": "policy"}
                bridge = SimpleNamespace(
                    terminal=None,
                    actions=[],
                    sequence=0,
                    observation=Mock(side_effect=lambda: events.append("publish")),
                    save=Mock(),
                    episode_dir=Path(temp) / "private_episode",
                )

                def factory(env, directory, variant, seed, **kwargs):
                    events.append("bridge")
                    self.assertEqual(kwargs["case_id"], "route_a")
                    self.assertFalse(kwargs["render_images"])
                    self.assertFalse(kwargs["rl_policy_loaded"])
                    for index, operation in enumerate(("close" if early_close else "hold", "reset")):
                        (directory / "commands" / f"{index}.json").write_text(json.dumps({"op": operation}))
                    return bridge

                def execute(request):
                    events.append("execute")
                    if early_close:
                        return {"status": "closed", "observation": {"done": False}}
                    bridge.actions.append([0, 0, 0])
                    bridge.terminal = {"success": False, "terminations": {"time_out": True}, "time_s": 24.0}
                    return {"status": "ok", "observation": {"done": True, "final_success": False}}

                bridge.execute = Mock(side_effect=execute)
                with (
                    patch.object(self.evaluator, "_source_hashes", return_value={}),
                    patch.object(
                        self.evaluator._COMMON._STATE,
                        "capture_initial_state",
                        side_effect=lambda *_: events.append("capture") or initial,
                    ),
                    patch.object(
                        self.evaluator._COMMON._STATE,
                        "assert_matching_initial_states",
                        side_effect=lambda *_: events.append("match") or {"matched": True},
                    ),
                ):
                    case, provenance, directory = self.evaluator._prepare_case(args)
                    if early_close:
                        with self.assertRaisesRegex(RuntimeError, "incomplete") as caught:
                            self.evaluator._run_case(env, case, provenance, directory, factory)
                        self.evaluator._record_error(case, directory, caught.exception)
                        self.assertFalse((directory / "result.json").exists())
                    else:
                        result = self.evaluator._run_case(env, case, provenance, directory, factory)
                        self.assertEqual(result["status"], "completed")
                        self.assertEqual(result["physical_state_sha256"], "physical")
                        self.assertEqual(result["terminations"], {"time_out": True})
                self.assertEqual(events, ["reset", "capture", "match", "bridge", "publish", "execute"])
                bridge.execute.assert_called_once()
                bridge.save.assert_called_once()
                self.assertFalse((directory / "responses/1.json").exists())
                response = json.loads((directory / "responses/0.json").read_text())
                self.assertEqual(set(response), {"status", "observation", "request_id"})


if __name__ == "__main__":
    unittest.main()
