# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expose only the contact-task actor measurements to direct Astra control.

Object poses are ideal simulated pose-estimator placeholders, not vision estimates.
Full-state trajectories and native diagnostics remain in private evaluator records.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from PIL import Image

from isaaclab.app import launch_simulation
from isaaclab.utils.math import axis_angle_from_quat, quat_inv, quat_mul
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.contrib.usb_cable_routing_contact.observations import (
    CONTROLLER_FIELDS,
    INSERTION_FIELDS,
    controller_measurements,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# Import the frozen recorder under a distinct name; never modify its implementation.
_RECORDER_PATH = Path(__file__).resolve().parents[1] / "usb_cable_routing_astra" / "server.py"
_RECORDER_SPEC = importlib.util.spec_from_file_location("_usb_assisted_private_recorder", _RECORDER_PATH)
_RECORDER = importlib.util.module_from_spec(_RECORDER_SPEC)
_RECORDER_SPEC.loader.exec_module(_RECORDER)

_MEASUREMENT_SHAPES = {
    "arm_joint_position_rad": (6,),
    "arm_joint_velocity_rad_s": (6,),
    "gripper_drive_position_rad": (1,),
    "gripper_drive_velocity_rad_s": (1,),
    "tcp_position_m": (3,),
    "tcp_quat_xyzw": (4,),
    "commanded_tcp_position_m": (3,),
    "commanded_tcp_quat_xyzw": (4,),
    "connector_position_m": (3,),
    "connector_quat_xyzw": (4,),
    "guide_positions_m": (3, 3),
    "guide_yaw_sin_cos": (3, 2),
    "requested_guide_count": (1,),
    "time_remaining_s": (1,),
}
_CAMERAS = {
    "routing": ((0.86, 0.16, 0.48), (0.53, -0.15, 0.10)),
    "socket": ((0.663, -0.40, 0.095), (0.668, -0.15, 0.065)),
}


def serialize_measurements(measurements: dict[str, torch.Tensor], variant: str) -> dict[str, list]:
    """Validate the shared actor allowlist and serialize one environment in SI units."""
    if variant not in ("scurve", "insert"):
        raise ValueError("variant must be scurve or insert.")
    expected = {**_MEASUREMENT_SHAPES, "last_action": (6 if variant == "insert" else 3,)}
    if variant == "insert":
        expected.update(socket_position_m=(3,), socket_quat_xyzw=(4,))
    if set(measurements) != set(expected):
        raise ValueError("Shared controller measurement fields differ from the public allowlist.")
    result = {}
    for name, shape in expected.items():
        value = measurements[name]
        if not isinstance(value, torch.Tensor) or value.shape != (1, *shape) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid shape or nonfinite shared measurement: {name}.")
        result[name] = value[0].tolist()
    return result


class ContactBridge(_RECORDER.DirectBridge):
    """Use shared actor measurements for a bounded TCP servo, with separate private recording."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        directory: Path,
        variant: str,
        seed: int,
        *,
        render_images: bool = False,
        controller: str = "Astra direct shared-measurement TCP servo + IK; physical contact grasp",
        rl_policy_loaded: bool = False,
        case_id: str | None = None,
    ):
        self.public_directory = directory
        self.image_directory = directory / "images"
        self.image_directory.mkdir(exist_ok=True)
        self._actor_measurements: dict[str, list] = {}
        self._controller_observations: list[np.ndarray] = []
        self._controller_fields = CONTROLLER_FIELDS + (INSERTION_FIELDS if variant == "insert" else ())
        private_directory = directory / "private_evaluator"
        private_directory.mkdir(exist_ok=True)
        super().__init__(
            env,
            private_directory,
            variant,
            seed,
            controller=controller,
            rl_policy_loaded=rl_policy_loaded,
            case_id=case_id,
            render_images=render_images,
        )
        self.observation_mode = "shared actor measurements; ideal simulated object pose-estimator placeholders"
        self.save()

    def _read_measurements(self) -> dict[str, torch.Tensor]:
        values = controller_measurements(self.env)
        serialize_measurements(values, self.variant)
        return values

    def measure(self) -> dict:
        """Record private contact diagnostics [N, m, rad] without exposing them to either controller."""
        result = super().measure()
        cable = self.command.cable
        result.update(
            contact_grasp_acquired=bool(wp.to_torch(cable.grasp_acquired)[0]),
            contact_grasp_dropped=bool(wp.to_torch(cable.grasp_dropped)[0]),
            pad_normal_forces_n=wp.to_torch(cable.pad_normal_forces)[0].tolist(),
            pad_contact_count=wp.to_torch(cable.pad_contact_count)[0].tolist(),
            contact_position_slip_m=float(wp.to_torch(cable.position_slip)[0]),
            contact_rotation_slip_rad=float(wp.to_torch(cable.rotation_slip)[0]),
            socket_contact_force_n=float(wp.to_torch(cable.socket_contact_force)[0]),
            socket_contact_force_w_n=wp.to_torch(cable.socket_contact_force_w)[0].tolist(),
            grasp_force_n_semantics="Magnitude of the net fingertip contact resultant, not the socket load.",
        )
        return result

    def save(self) -> None:
        """Save private traces under the registered contact-task physics and success contract."""
        super().save()
        path = self.episode_dir / "success.json"
        result = json.loads(path.read_text())
        result.pop("task_physics_or_success_modified", None)
        result["physics_and_success_contract"] = (
            "Registered contact-grasp task definition; distinct from the assisted-grasp benchmark."
        )
        result["contact_force_semantics"] = {
            "grasp_force_n": "Magnitude of the net fingertip contact resultant [N].",
            "socket_contact_force_n": "Magnitude of the net connector/socket contact resultant [N].",
        }
        result["private_evaluator_record"] = True
        _RECORDER.write_json(path, result)
        if len(self._controller_observations) != len(self.actions) + 1:
            raise RuntimeError("Controller recording requires one more observation than actions.")
        times = np.asarray([row["time_s"] for row in self.trace], dtype=np.float64)
        np.savez_compressed(
            self.episode_dir / "controller_trajectory.npz",
            observations=np.stack(self._controller_observations),
            actions=np.asarray(self.actions, dtype=np.float32).reshape(-1, 6 if self.variant == "insert" else 3),
            observation_time_s=times,
            action_time_s=times[:-1],
            observation_fields=np.asarray(self._controller_fields),
        )

    def snapshot(self) -> None:
        """Record private evaluator state and preserve allowed measurements before automatic reset."""
        super().snapshot()
        measurements = self._read_measurements()
        self._actor_measurements = serialize_measurements(measurements, self.variant)
        observation = torch.cat([measurements[name].flatten(1) for name in self._controller_fields], dim=-1)
        self._controller_observations.append(observation[0].detach().cpu().numpy().copy())

    def observation(self, camera: str = "routing", *, render_image: bool | None = None) -> dict:
        """Publish only actor-accessible measurements and optional fixed-camera simulated RGB."""
        if camera not in _CAMERAS or (camera == "socket" and self.variant != "insert"):
            raise ValueError("Use the fixed routing camera, or the socket camera for insertion.")
        image_requested = self.render_images if render_image is None else render_image
        if image_requested and not self.render_images:
            raise ValueError("Camera access is disabled; --images is development inspection mode only.")
        if self.terminal is None:
            self._actor_measurements = serialize_measurements(self._read_measurements(), self.variant)
        observation = {
            "measurements": self._actor_measurements,
            "measurement_model": {
                "robot": "Simulated joint encoders and robot kinematics.",
                "object_poses": "Ideal simulated pose-estimator placeholders; no image-based estimator or pose noise.",
                "rgb": "Development inspection with --images only; omitted from matched vector-input evaluations.",
                "timing": "Physics pauses between model commands; real-time reasoning is not implemented.",
            },
            "done": self.terminal is not None,
        }
        if self.terminal is not None:
            observation["final_success"] = bool(self.terminal["success"])
        elif image_requested:
            self.visualizer.set_camera_view(*_CAMERAS[camera])
            self.visualizer._state = NewtonManager.get_state_0()
            pixels = self.visualizer.render_rgb_array()
            path = self.image_directory / f"observation_{self.sequence:04d}_{camera}.png"
            Image.fromarray(pixels).save(path)
            observation["rgb_image"] = {
                "path": str(path.resolve()),
                "camera": camera,
                "source": "Simulated fixed-camera Newton GL RGB; no depth or segmentation channel.",
            }
        _RECORDER.write_json(self.public_directory / "observation.json", observation)
        return observation

    def execute(self, request: dict) -> dict:
        """Execute explicit TCP targets [m, xyzw] without cable-state or hidden-stage feedback."""
        if not isinstance(request, dict):
            raise ValueError("A command must be a JSON object.")
        allowed = {
            "op",
            "rationale",
            "steps",
            "frame",
            "target_position_m",
            "target_quat_xyzw",
            "speed_m_s",
            "angular_speed_rad_s",
            "camera",
        }
        if set(request) - allowed:
            raise ValueError("Unsupported request field; stage stopping and private diagnostics are unavailable.")
        operation = request.get("op", "observe")
        if operation == "reset":
            raise ValueError("Reset is forbidden; each case requires a fresh simulator process.")
        if operation not in ("observe", "move", "hold", "close"):
            raise ValueError("Supported operations are observe, move, hold, and close.")
        if request.get("frame", "tcp") != "tcp":
            raise ValueError("Only the tcp frame is supported; connector servoing is not provided.")
        if "camera" in request and not self.render_images:
            raise ValueError("Camera access is disabled; --images is development inspection mode only.")
        camera = request.get("camera", "routing")
        if camera not in _CAMERAS or (camera == "socket" and self.variant != "insert"):
            raise ValueError("Use the fixed routing camera, or the socket camera for insertion.")
        render_image = "camera" in request or self.render_images
        if operation in ("observe", "close"):
            if operation == "close":
                self.save()
            return {
                "status": "closed" if operation == "close" else "ok",
                "observation": self.observation(camera, render_image=render_image),
            }
        if not isinstance(request.get("rationale"), str) or not request["rationale"].strip():
            raise ValueError("A brief model-authored rationale is required for state-changing requests.")
        if self.terminal is not None:
            raise ValueError("Episode has terminated; inspect its final result and close this case.")
        steps = request.get("steps", 30)
        if type(steps) is not int or not 1 <= steps <= 90:
            raise ValueError("steps must be an integer from 1 through 90.")
        speed = float(request.get("speed_m_s", 0.10))
        angular_speed = float(request.get("angular_speed_rad_s", 0.30))
        if (
            not math.isfinite(speed)
            or not 0 < speed <= 0.15
            or not math.isfinite(angular_speed)
            or not 0 < angular_speed <= 0.5
        ):
            raise ValueError("Require speed_m_s in (0,.15] and angular_speed_rad_s in (0,.5].")
        goal = rotation_goal = None
        if operation == "move":
            goal = torch.tensor(
                [_RECORDER.bounded_vector(request.get("target_position_m"), 3, "target_position_m")],
                device=self.env.device,
            )
            if "target_quat_xyzw" in request:
                if self.variant != "insert":
                    raise ValueError(
                        "Only insertion accepts TCP rotation goals; routing holds its initial orientation."
                    )
                rotation_goal = torch.tensor(
                    [_RECORDER.bounded_vector(request["target_quat_xyzw"], 4, "target_quat_xyzw")],
                    device=self.env.device,
                )
                norm = float(rotation_goal.norm())
                if not 0.99 < norm < 1.01:
                    raise ValueError("target_quat_xyzw must be normalized.")
                rotation_goal /= norm
        executed = 0
        stop_reason = "duration"
        for _ in range(steps):
            action = torch.zeros((1, self.env.action_manager.total_action_dim), device=self.env.device)
            if operation == "move":
                measurements = self._read_measurements()
                delta = goal - measurements["commanded_tcp_position_m"]
                limit = speed * self.env.step_dt
                delta *= (limit / delta.norm(dim=-1, keepdim=True).clamp(min=limit)).clamp(max=1.0)
                action[:, :3] = delta / self.arm.cfg.scale
                if rotation_goal is not None:
                    previous_quat = measurements["commanded_tcp_quat_xyzw"]
                    error = axis_angle_from_quat(quat_mul(rotation_goal, quat_inv(previous_quat)))
                    angular_limit = angular_speed * self.env.step_dt
                    error *= (angular_limit / error.norm(dim=-1, keepdim=True).clamp(min=angular_limit)).clamp(max=1.0)
                    action[:, 3:] = error / self.arm.cfg.rotation_scale
            executed += 1
            if self.step(action):
                stop_reason = "episode_terminated"
                break
        self.save()
        return {
            "status": "ok",
            "executed_steps": executed,
            "stop_reason": stop_reason,
            "observation": self.observation(camera, render_image=render_image),
        }


def task_id(variant: str) -> str:
    """Return the registered contact task for a supported variant."""
    if variant not in ("scurve", "insert"):
        raise ValueError("variant must be scurve or insert.")
    suffix = "SCurve-Insert-Contact" if variant == "insert" else "SCurve-Contact"
    return f"IsaacContrib-Route-USB-{suffix}-UR10e"


def build_task_cfg(
    variant: str, seed: int, *, randomize: bool = True, use_cuda_graph: bool = True
) -> ManagerBasedRLEnvCfg:
    """Build identical single-case physics and randomized guides for both controllers."""
    task = task_id(variant)
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.guide_count = 3
    cfg.commands.routing.randomize = randomize
    cfg.commands.routing.randomize_guides = randomize
    cfg.sim.physics.use_cuda_graph = use_cuda_graph
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            eye=_CAMERAS["routing"][0],
            lookat=_CAMERAS["routing"][1],
            window=WindowCfg(size=(1280, 720)),
            headless=False,
            enable_picking=False,
            enable_markers=False,
            enable_live_plots=False,
            streaming_view=False,
        )
    ]
    return cfg


def main() -> None:
    """Serve one contact task through the existing local file protocol; never reset a scored case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("scurve", "insert"), required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=920001)
    parser.add_argument(
        "--images",
        action="store_true",
        help="Development inspection: enable additional RGB input; excluded from matched vector-input evaluations.",
    )
    parser.add_argument("--no_graph", action="store_true", help="Disable CUDA graphs for debugging.")
    parser.add_argument(
        "--no_randomization", action="store_true", help="Disable cable and guide randomization for development."
    )
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run_dir must be absent or empty; existing cases are never overwritten.")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("commands", "responses"):
        (args.run_dir / name).mkdir(exist_ok=True)
    task = task_id(args.variant)
    cfg = build_task_cfg(args.variant, args.seed, randomize=not args.no_randomization, use_cuda_graph=not args.no_graph)
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset(seed=args.seed)
            bridge = ContactBridge(env, args.run_dir, args.variant, args.seed, render_images=args.images)
            bridge.observation()
            print(f"CONTACT_ASTRA_BRIDGE_READY {args.run_dir.resolve()}", flush=True)
            with torch.inference_mode():
                while True:
                    pending = [
                        path
                        for path in sorted((args.run_dir / "commands").glob("*.json"))
                        if not (args.run_dir / "responses" / path.name).exists()
                    ]
                    if not pending:
                        time.sleep(0.1)
                        continue
                    path = pending[0]
                    bridge.sequence += 1
                    request = None
                    try:
                        request = json.loads(path.read_text())
                        result = bridge.execute(request)
                    except (ValueError, TypeError, KeyError) as error:
                        result = {"status": "error", "error": str(error)}
                    result["request_id"] = path.stem
                    _RECORDER.write_json(args.run_dir / "responses" / path.name, result)
                    with (args.run_dir / "decisions.jsonl").open("a") as stream:
                        stream.write(json.dumps({"request": request, "response": result}) + "\n")
                    print(f"COMMAND {path.stem} {result['status']}", flush=True)
                    if result["status"] == "closed":
                        break
        finally:
            env.close()


if __name__ == "__main__":
    main()
