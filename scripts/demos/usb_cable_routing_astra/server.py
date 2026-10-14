# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expose paused Newton environments to direct model-selected Cartesian commands.

No learned policy, optimizer, automatic guide planner, or task-stage controller is loaded.
A model selects each bounded command; the existing robot IK executes the requested pose.
"""

from __future__ import annotations

import argparse
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

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


def write_json(path: Path, value: dict | list) -> None:
    """Atomically publish a response or measured observation."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bounded_vector(value: object, size: int, name: str) -> list[float]:
    """Validate a finite numeric vector without accepting booleans."""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers.")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain only finite numbers.")
    return [float(v) for v in value]


class DirectBridge:
    """Execute explicit Cartesian targets [m, xyzw] and record their physical outcomes."""

    def __init__(
        self,
        env,
        directory: Path,
        variant: str,
        seed: int,
        *,
        controller: str = "gpt-6-astra direct tool-selected Cartesian commands + existing IK",
        rl_policy_loaded: bool = False,
        render_images: bool = True,
        case_id: str | None = None,
        evaluation_cases: dict[int, str] | None = None,
        reference_dir: Path | None = None,
    ):
        self.env = env
        self.directory = directory
        self.variant = variant
        self.seed = seed
        self.controller = controller
        self.rl_policy_loaded = rl_policy_loaded
        self.observation_mode = (
            "trained policy observation vector; full simulator state recorded for audit"
            if rl_policy_loaded
            else "measured full simulator state, optional image"
        )
        self.render_images = render_images
        self.evaluation_cases = evaluation_cases
        self.initial_state_verified = evaluation_cases is None
        self.reference_dir = reference_dir
        self.case_id = evaluation_cases[seed] if evaluation_cases else case_id
        self.command = env.command_manager.get_term("routing")
        self.arm = env.action_manager.get_term("arm")
        self.visualizer = env.sim.visualizers[0]
        self.visualizer._viewer.renderer.window.set_visible(False)
        self.visualizer._runtime_headless = True
        self.episode = 0
        self.sequence = 0
        self.terminal: dict | None = None
        self.poses: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.trace: list[dict] = []
        self.episode_dir = directory / "episode_000"
        self.episode_dir.mkdir(exist_ok=True)
        original_reset = env._reset_idx

        def capture_terminal(ids):
            if self.trace and bool(env.termination_manager.dones[0]):
                self.snapshot()
                self.terminal = {
                    **self.trace[-1],
                    "success": bool(env.termination_manager.get_term("success")[0]),
                    "terminations": {
                        name: bool(env.termination_manager.get_term(name)[0])
                        for name in env.termination_manager.active_terms
                    },
                }
            original_reset(ids)

        env._reset_idx = capture_terminal
        self.snapshot()
        self.verify_initial_state()
        self.save()

    def verify_initial_state(self) -> None:
        """Verify a benchmark case against its paired RL state before any action."""
        if self.evaluation_cases is None:
            return
        from benchmark_state import assert_matching_initial_states, save_initial_state

        self.initial_state_verified = False
        try:
            state = save_initial_state(self.env, self.episode_dir / "initial_state.json")
            reference = self.reference_dir / self.case_id / "initial_state.json"
            comparison = assert_matching_initial_states(reference, state)
        except (ValueError, TypeError, KeyError, OSError) as error:
            write_json(self.episode_dir / "initial_state_comparison.json", {"matched": False, "error": str(error)})
            raise
        write_json(self.episode_dir / "initial_state_comparison.json", comparison)
        self.initial_state_verified = True

    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return measured TCP position [m] and quaternion in the environment frame."""
        pos, quat = self.arm._compute_frame_pose()
        return pos + self.command.robot.data.root_pos_w.torch - self.env.scene.env_origins, quat

    def frame_pose(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a measured controlled point [m] and orientation (xyzw)."""
        if name == "tcp":
            return self.tcp_pose()
        if name == "tip":
            return self.command.tip(), self.command.cable.data.segment_pose_w.torch[:, -1, 3:]
        if name == "plug_front" and self.variant == "insert":
            return self.command.plug_pose()
        raise ValueError("frame must be tcp, tip, or (for insertion) plug_front.")

    def measure(self) -> dict:
        """Read task state, cable points [m], grasp loads [N, N m], and guide crossings."""
        c = self.command
        tcp, quat = self.tcp_pose()
        points = c.points()[0]
        root = c.robot.data.root_pos_w.torch - self.env.scene.env_origins
        target = self.arm._processed_actions + root if bool(self.arm._target_valid[0]) else tcp
        nearest = (points[:, 0, None] - c.guide_positions[0, None, :, 0]).abs().argmin(dim=0)
        radius = float(c.minimum_bend_radius[0])
        result = {
            "step": len(self.actions),
            "time_s": len(self.actions) * self.env.step_dt,
            "time_remaining_s": self.env.max_episode_length_s - len(self.actions) * self.env.step_dt,
            "tcp_position_m": tcp[0].tolist(),
            "tcp_quat_xyzw": quat[0].tolist(),
            "commanded_tcp_position_m": target[0].tolist(),
            "tip_m": c.tip()[0].tolist(),
            "plug_quat_xyzw": c.cable.data.segment_pose_w.torch[0, -1, 3:].tolist(),
            "cable_points_m": points.tolist(),
            "guides_m": c.guide_positions[0].tolist(),
            "guide_yaws_rad": c.guide_yaws[0].tolist(),
            "nearest_cable_to_guides_m": points[nearest].tolist(),
            "guides_completed": int(c.completed[0]),
            "guides_retained": c.seated[0].tolist(),
            "lowering_gate": bool(c.lowering[0]),
            "guide_dwell_steps": int(c.dwell[0]),
            "success_dwell_steps": int(c.success_dwell[0]),
            "skip": bool(c.skip[0]),
            "minimum_bend_radius_m": radius if math.isfinite(radius) else None,
            "stretch_ratio": float(c.stretch()[0]),
            "grasp_active": bool(wp.to_torch(c.cable.grasp_active)[0]),
            "grasp_force_n": float(wp.to_torch(c.cable.tension)[0]),
            "grasp_torque_nm": float(wp.to_torch(c.cable.grasp_torque)[0]),
            "wrist_height_m": float(c.robot.data.body_link_pos_w.torch[0, c.wrist_body_ids[0], 2]),
            "collision_candidate_steps": int(c.episode_peaks[0, 2]),
            "material_stiffness_scale": float(c.material[0, 0]),
            "friction": float(c.material[0, 1]),
        }
        if self.variant == "insert":
            plug_pos, plug_quat = c.plug_pose()
            result.update(
                plug_front_m=plug_pos[0].tolist(),
                plug_quat_xyzw=plug_quat[0].tolist(),
                socket_position_m=c.socket_position[0].tolist(),
                socket_quat_xyzw=c.socket_quat[0].tolist(),
                socket_relative_front_m=c.local_front[0].tolist(),
                depth_m=float(c.local_front[0, 2]),
                alignment_error_rad=float(c.angle_error[0]),
                route_mastered=bool(c.route_mastered[0]),
                ready_to_align=bool(c.ready_to_align[0]),
                ready_to_approach=bool(c.ready_to_approach[0]),
                inserting=bool(c.inserting[0]),
                entered=bool(c.entered[0]),
                rotation_unlock_tip_target_m=c.seat_targets[-1].tolist(),
                plug_relative_to_tcp_quat_xyzw=quat_mul(quat_inv(quat), plug_quat)[0].tolist(),
            )
        return result

    def snapshot(self) -> None:
        """Save every actual control-step pose, including the terminal pose before reset."""
        self.poses.append(NewtonManager.get_state_0().body_q.numpy().copy())
        self.trace.append(self.measure())

    def save(self) -> None:
        """Persist measured replay states, all model commands, and unmodified task results."""
        np.savez_compressed(
            self.episode_dir / "trajectory.npz",
            body_pose_m_xyzw=np.stack(self.poses),
            actions=np.array(self.actions),
            body_labels=np.array(NewtonManager.get_model().body_label),
            time_s=np.array([row["time_s"] for row in self.trace]),
        )
        write_json(self.episode_dir / "trace.json", self.trace)
        write_json(
            self.episode_dir / "success.json",
            {
                "task": self.env.spec.id if self.env.spec else self.variant,
                "variant": self.variant,
                "seed": self.seed,
                "controller": self.controller,
                "observation_mode": self.observation_mode,
                "rl_policy_loaded": self.rl_policy_loaded,
                "evaluation_case_id": self.case_id,
                "task_physics_or_success_modified": False,
                "simulation_dt_s": self.env.step_dt,
                "recorded_states": len(self.poses),
                "policy_steps": len(self.actions),
                "minimum_bend_radius_over_episode_m": min(
                    row["minimum_bend_radius_m"] for row in self.trace if row["minimum_bend_radius_m"] is not None
                ),
                "terminal": self.terminal,
            },
        )

    def observation(self, camera: str = "routing", *, render_image: bool | None = None) -> dict:
        """Publish measured state and a current renderer snapshot without advancing time."""
        state = dict(self.terminal) if self.terminal else self.measure()
        if not self.terminal and (self.render_images if render_image is None else render_image):
            if camera == "socket" and self.variant == "insert":
                eye, target = (0.663, -0.40, 0.095), (0.668, -0.15, 0.065)
            else:
                eye, target = (0.86, 0.16, 0.48), (0.53, -0.15, 0.10)
            self.visualizer.set_camera_view(eye, target)
            self.visualizer._state = NewtonManager.get_state_0()
            pixels = self.visualizer.render_rgb_array()
            image_path = self.episode_dir / f"observation_{self.sequence:04d}.png"
            Image.fromarray(pixels).save(image_path)
            state["image_path"] = str(image_path.resolve())
        state.update(
            episode=self.episode,
            episode_directory=str(self.episode_dir.resolve()),
            done=self.terminal is not None,
            terminal=self.terminal,
            variant=self.variant,
            seed=self.seed,
            evaluation_case_id=self.case_id,
            initial_state_verified=self.initial_state_verified,
        )
        write_json(self.directory / "observation.json", state)
        return state

    def step(self, action: torch.Tensor) -> bool:
        """Advance one control interval using a model-requested low-level servo action."""
        self.actions.append(action[0].cpu().numpy().copy())
        _, _, terminated, truncated, _ = self.env.step(action)
        done = bool(terminated[0] | truncated[0])
        if not done:
            self.snapshot()
        return done

    def execute(self, request: dict) -> dict:
        """Execute one explicit model request and return fresh measured feedback."""
        op = request.get("op", "observe")
        if op in ("move", "hold", "reset") and (
            not isinstance(request.get("rationale"), str) or not request["rationale"].strip()
        ):
            raise ValueError("A brief model-authored rationale is required for state-changing requests.")
        if op == "observe":
            return {
                "status": "ok",
                "observation": self.observation(
                    request.get("camera", "routing"), render_image="camera" in request or self.render_images
                ),
            }
        if op == "reset":
            if self.evaluation_cases is not None:
                raise ValueError("Each benchmark case requires a fresh simulator process; reset is forbidden.")
            if "seed" in request and (type(request["seed"]) is not int or request["seed"] < 0):
                raise ValueError("seed must be a nonnegative integer.")
            next_seed = int(request.get("seed", self.seed))
            self.save()
            self.episode += 1
            self.seed = next_seed
            self.terminal = None
            self.poses, self.actions, self.trace = [], [], []
            self.episode_dir = self.directory / f"episode_{self.episode:03d}"
            self.episode_dir.mkdir(exist_ok=False)
            self.env.reset(seed=self.seed)
            self.snapshot()
            self.verify_initial_state()
            self.save()
            return {"status": "reset", "observation": self.observation()}
        if op == "close":
            self.save()
            return {"status": "closed", "observation": self.observation()}
        if op not in ("move", "hold"):
            raise ValueError(f"Unknown operation: {op}")
        if not self.initial_state_verified:
            raise ValueError("Initial-state verification failed; this benchmark case is quarantined.")
        if self.terminal:
            raise ValueError("Episode has terminated. Inspect its result, then explicitly reset if desired.")
        steps = request.get("steps", 30)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 90:
            raise ValueError("steps must be an integer from 1 through 90.")
        if not isinstance(request.get("stop_on_stage_change", True), bool):
            raise ValueError("stop_on_stage_change must be a boolean.")
        frame = request.get("frame", "tcp")
        speed = float(request.get("speed_m_s", 0.10))
        angular_speed = float(request.get("angular_speed_rad_s", 0.30))
        if not 0 < speed <= 0.15 or not 0 < angular_speed <= 0.5:
            raise ValueError("Require speed_m_s in (0,.15] and angular_speed_rad_s in (0,.5].")
        goal = rotation_goal = None
        if op == "move":
            goal = torch.tensor(
                [bounded_vector(request.get("target_position_m"), 3, "target_position_m")], device=self.env.device
            )
            self.frame_pose(frame)
            if "target_quat_xyzw" in request:
                if self.variant != "insert":
                    raise ValueError("The routing task fixes orientation; only insertion accepts rotation goals.")
                rotation_goal = torch.tensor(
                    [bounded_vector(request["target_quat_xyzw"], 4, "target_quat_xyzw")], device=self.env.device
                )
                norm = float(rotation_goal.norm())
                if not 0.99 < norm < 1.01:
                    raise ValueError("target_quat_xyzw must be normalized.")
                rotation_goal /= norm
        initial_completed = int(self.command.completed[0])
        executed = 0
        stop_reason = "duration"
        for _ in range(steps):
            action = torch.zeros((1, self.env.action_manager.total_action_dim), device=self.env.device)
            if op == "move":
                tcp, tcp_quat = self.tcp_pose()
                point, point_quat = self.frame_pose(frame)
                desired_tcp = tcp + goal - point
                root = self.command.robot.data.root_pos_w.torch - self.env.scene.env_origins
                previous = self.arm._processed_actions + root if bool(self.arm._target_valid[0]) else tcp
                delta = desired_tcp - previous
                limit = speed * self.env.step_dt
                delta *= (limit / delta.norm(dim=-1, keepdim=True).clamp(min=limit)).clamp(max=1.0)
                scale = self.arm.cfg.scale
                if self.variant == "insert" and int(self.command.completed[0]) >= int(self.command.guide_count[0]):
                    scale = self.arm.cfg.insertion_scale
                action[:, :3] = delta / scale
                if rotation_goal is not None:
                    relative = quat_mul(quat_inv(tcp_quat), point_quat)
                    desired_quat = quat_mul(rotation_goal, quat_inv(relative))
                    previous_quat = self.arm.target_quat if bool(self.arm._target_valid[0]) else tcp_quat
                    error = axis_angle_from_quat(quat_mul(desired_quat, quat_inv(previous_quat)))
                    angular_limit = angular_speed * self.env.step_dt
                    error *= (angular_limit / error.norm(dim=-1, keepdim=True).clamp(min=angular_limit)).clamp(max=1.0)
                    action[:, 3:] = error / self.arm.cfg.rotation_scale
            executed += 1
            if self.step(action):
                stop_reason = "episode_terminated"
                break
            if request.get("stop_on_stage_change", True) and int(self.command.completed[0]) != initial_completed:
                stop_reason = "guide_progress"
                break
        self.save()
        return {
            "status": "ok",
            "executed_steps": executed,
            "stop_reason": stop_reason,
            "observation": self.observation(
                request.get("camera", "routing"), render_image="camera" in request or self.render_images
            ),
        }


def task_id(variant: str) -> str:
    """Return the existing task registration for a supported cable variant."""
    if variant not in ("scurve", "insert"):
        raise ValueError("variant must be scurve or insert.")
    return (
        "IsaacContrib-Route-USB-SCurve-Insert-UR10e" if variant == "insert" else "IsaacContrib-Route-USB-SCurve-UR10e"
    )


def build_task_cfg(variant: str, seed: int, render_images: bool = True) -> ManagerBasedRLEnvCfg:
    """Build identical single-environment physics for both evaluation controllers.

    The visualizer is retained for optional observation images. ``render_images``
    controls observation capture in the bridge, not the physics configuration.
    """
    cfg = load_cfg_from_registry(task_id(variant), "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.guide_count = 3
    cfg.sim.physics.use_cuda_graph = True
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            eye=(0.86, 0.16, 0.48),
            lookat=(0.53, -0.15, 0.10),
            focal_length=24.0,
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
    """Run one paused environment and process local file-based model requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("scurve", "insert"), required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=650001)
    parser.add_argument(
        "--no_images", action="store_true", help="Render observation images only when a camera is requested."
    )
    parser.add_argument(
        "--evaluation_manifest", type=Path, help="Frozen case seeds; requires one process per case and disables resets."
    )
    parser.add_argument(
        "--reference_dir", type=Path, help="Paired RL case directories containing verified initial states."
    )
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("commands", "responses"):
        (args.run_dir / name).mkdir(exist_ok=True)
    if (args.run_dir / "observation.json").exists():
        parser.error("run_dir already contains a bridge session; choose a new directory.")
    cases = None
    if args.evaluation_manifest is not None:
        if args.reference_dir is None:
            parser.error("--evaluation_manifest requires --reference_dir.")
        from benchmark_state import load_evaluation_cases

        selected_cases = load_evaluation_cases(args.evaluation_manifest, args.variant)
        cases = {case["seed"]: case["case_id"] for case in selected_cases}
        if args.seed not in cases:
            parser.error("--seed must be a case seed in the frozen evaluation manifest.")
    cfg = build_task_cfg(args.variant, args.seed, render_images=not args.no_images)
    task = task_id(args.variant)
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset(seed=args.seed)
            bridge = DirectBridge(
                env,
                args.run_dir,
                args.variant,
                args.seed,
                render_images=not args.no_images,
                evaluation_cases=cases,
                reference_dir=args.reference_dir,
            )
            bridge.observation()
            print(f"ASTRA_BRIDGE_READY {args.run_dir.resolve()}", flush=True)
            with torch.inference_mode():
                while True:
                    pending = [
                        p
                        for p in sorted((args.run_dir / "commands").glob("*.json"))
                        if not (args.run_dir / "responses" / p.name).exists()
                    ]
                    if not pending:
                        time.sleep(0.1)
                        continue
                    path = pending[0]
                    bridge.sequence += 1
                    request = None
                    try:
                        request = json.loads(path.read_text())
                        if not isinstance(request, dict):
                            raise ValueError("A command must be a JSON object.")
                        result = bridge.execute(request)
                    except (ValueError, TypeError, KeyError) as error:
                        result = {"status": "error", "error": str(error)}
                    result["request_id"] = path.stem
                    write_json(args.run_dir / "responses" / path.name, result)
                    with (args.run_dir / "decisions.jsonl").open("a") as stream:
                        stream.write(json.dumps({"request": request, "response": result}) + "\n")
                    print(f"COMMAND {path.stem} {result['status']}", flush=True)
                    if result["status"] == "closed":
                        break
        finally:
            env.close()


if __name__ == "__main__":
    main()
