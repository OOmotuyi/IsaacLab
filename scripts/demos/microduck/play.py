# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play the ten released MicroDuck policies in Isaac Lab with Newton MJWarp."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from behaviors import BEHAVIORS, policy_command
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import VisualizerCfg

if TYPE_CHECKING:
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

    from isaaclab.sim import SimulationContext


def _create_video_writer(sim: SimulationContext, path: Path | None) -> FFMPEG_VideoWriter | None:
    if path is None:
        return None
    if not sim.visualizers:
        raise ValueError("--record requires --viz newton_gl.")
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = sim.visualizers[0].render_rgb_array()
    return FFMPEG_VideoWriter(str(path), (frame.shape[1], frame.shape[0]), fps=50, codec="libx264")


def parse_args() -> argparse.Namespace:
    """Parse and validate the demonstration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    parser.add_argument("--behavior", choices=BEHAVIORS, default="alpha_walking")
    parser.add_argument("--policy", type=Path, help="Override the selected behavior's checkpoint.")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration [s]; 0 runs until closed.")
    parser.add_argument(
        "--velocity",
        type=float,
        nargs=3,
        default=(0.30, 0.0, 0.0),
        metavar=("VX", "VY", "WZ"),
        help="Walking twist [m/s, m/s, rad/s]; for rollers: push/brake, 0, heading error [rad].",
    )
    parser.add_argument("--start_pose", choices=("standing", "prone"), default="standing")
    parser.add_argument("--repeat", action="store_true", help="Reset and repeat the demonstration on a fixed schedule.")
    parser.add_argument("--real_time", action="store_true", help="Pace playback to wall time.")
    parser.add_argument("--metrics", type=Path, help="Write rollout metrics as JSON.")
    parser.add_argument("--trace", type=Path, help="Write policy-rate states and commands as JSON lines.")
    parser.add_argument("--record", type=Path, help="Record a Newton viewer MP4 (requires the video extra).")
    parser.add_argument("--offscreen", action="store_true", help="Render with a headless Newton GL visualizer.")
    parser.add_argument(
        "--pickup_object", action="store_true", help="Add a physical block to the alpha_ground_pick scene."
    )
    parser.add_argument("--free_camera", action="store_true", help="Disable the following camera.")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.duration < 0 or not all(math.isfinite(v) for v in (*args.velocity, args.duration)):
        parser.error("Duration and velocity must be finite, and duration must be nonnegative.")
    if args.start_pose == "prone" and args.behavior not in ("alpha_stand", "velstand"):
        parser.error("--start_pose prone is supported for alpha_stand and velstand.")
    if args.pickup_object and args.behavior != "alpha_ground_pick":
        parser.error("--pickup_object is supported for alpha_ground_pick.")
    return args


def load_policies(paths: dict[str, Path]) -> dict:
    """Load checkpoints after checking the released observation/action contract."""
    import onnxruntime as ort
    from controller import JOINT_NAMES

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 1
    sessions = {}
    for name, path in paths.items():
        session = ort.InferenceSession(str(path), sess_options=session_opts, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
            raise ValueError(f"{name}: expected a 61-observation, 14-action ONNX policy.")
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("joint_names", "").split(",") != list(JOINT_NAMES):
            raise ValueError(f"{name}: joint metadata does not match the MicroDuck contract.")
        if (
            metadata.get("observation_names")
            != "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
            or float(metadata.get("action_scale", "nan")) != 1.0
        ):
            raise ValueError(f"{name}: unsupported observation layout or export action scale.")
        sessions[name] = session
    return sessions


def _motion_metrics(
    twist_sum: list[float], squared_error: list[float], speed_sum: float, samples: int, walking: bool
) -> dict[str, list[float] | float | None]:
    """Summarize root-link motion [m/s, m/s, rad/s] and horizontal ground speed [m/s]."""
    result = {"mean_body_twist": None, "mean_ground_speed_m_s": None}
    if samples:
        result["mean_body_twist"] = [value / samples for value in twist_sum]
        result["mean_ground_speed_m_s"] = speed_sum / samples
        if walking:
            result["twist_rmse"] = [math.sqrt(value / samples) for value in squared_error]
    return result


def _simulation_cfg(args: argparse.Namespace) -> SimulationCfg:
    """Configure Newton physics and optional offscreen rendering."""
    cfg = SimulationCfg(
        device=args.device or "cuda:0",
        dt=0.005,
        render_interval=4,
        use_newton_actuators=False,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(integrator="implicitfast", njmax=600, nconmax=128, cone="pyramidal"),
            use_cuda_graph=True,
        ),
        default_visualizer_cfg=VisualizerCfg(eye=(0.6, -0.7, 0.4), lookat=(0.0, 0.0, 0.12), focal_length=20.0),
    )
    if args.offscreen:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        cfg.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True)]
    return cfg


def main() -> None:
    """Run a single-robot demonstration with the selected behavior."""
    args = parse_args()
    profile = BEHAVIORS[args.behavior]
    asset = args.asset_dir.resolve() / profile.robot / f"{profile.robot}.usda"
    policy_paths = {args.behavior: args.policy or args.asset_dir / f"{args.behavior}.onnx"}
    if profile.active_time:
        policy_paths[profile.fallback] = args.asset_dir / f"{profile.fallback}.onnx"
    if not asset.is_file() or not all(path.is_file() for path in policy_paths.values()):
        raise FileNotFoundError("Run scripts/demos/microduck/prepare.py first.")
    params = json.loads((args.asset_dir / "bam_m6.json").read_text())
    cfg = _simulation_cfg(args)
    with launch_simulation(cfg, args):
        import numpy as np
        import torch
        from controller import DEFAULT_POSE, JOINT_NAMES, MicroDuckController
        from pickup import PickupTarget

        from pxr import Gf, UsdPhysics

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
        from isaaclab.cloner import make_clone_plan, replicate
        from isaaclab.sim import SimulationContext
        from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

        sessions = load_policies(policy_paths)
        sim = SimulationContext(cfg)
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        )
        ground_cfg.func("/World/ground", ground_cfg)
        sim_utils.DistantLightCfg(intensity=2500.0).func("/World/light", sim_utils.DistantLightCfg(intensity=2500.0))
        actuators = {
            "servos": ImplicitActuatorCfg(
                joint_names_expr=list(JOINT_NAMES),
                stiffness=0.0,
                damping=0.0,
                armature=params["armature"],
                friction=params["friction_base"],
                joint_effort_limit=7.4 * params["kt"] / params["R"],
            )
        }
        joint_pos = dict(zip(JOINT_NAMES, DEFAULT_POSE))
        if profile.rollers:
            joint_pos["passive_.*"] = 0.0
            actuators["wheels"] = ImplicitActuatorCfg(
                joint_names_expr=["passive_.*"],
                stiffness=0.0,
                damping=0.0,
                armature=0.0001,
                friction=0.003,
                joint_effort_limit=float("inf"),
            )
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=str(asset), variants={"Physics": "mujoco"}),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_pos, joint_vel={".*": 0.0}),
                actuators=actuators,
            )
        )
        ball = None
        if args.behavior.startswith("ball_kick"):
            ball = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/Ball",
                    spawn=sim_utils.SphereCfg(
                        radius=0.035,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
                        physics_material=RigidBodyMaterialBaseCfg(static_friction=0.5, dynamic_friction=0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.09, 0.042, 0.035)),
                )
            )
            # Source ball.xml specifies a hollow sphere, not a solid sphere.
            UsdPhysics.MassAPI(sim.stage.GetPrimAtPath("/World/Ball")).CreateDiagonalInertiaAttr(
                Gf.Vec3f(1.225e-5, 1.225e-5, 1.225e-5)
            )
        pickup = PickupTarget(robot, args.asset_dir / "source" / "robot" / f"{profile.robot}.xml", args.pickup_object)
        plan = make_clone_plan((AssetBaseCfg(prim_path="/World"),), ((),), 1, shared_assets=(0,))
        sim.set_clone_plan(plan)
        replicate(plan)
        sim.reset()
        controller = MicroDuckController(robot, params)
        wheel_ids = robot.find_joints("passive_.*")[0] if profile.rollers else []
        ball_start = None

        def place_ball() -> None:
            nonlocal ball_start
            if ball is None:
                return
            pose = robot.data.root_link_pose_w.torch[0].cpu().tolist()
            x, y, z, qx, qy, qz, qw = pose
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            off_y = 0.042 if args.behavior.endswith("left") else -0.042
            ball_start = [
                x + math.cos(yaw) * 0.09 - math.sin(yaw) * off_y,
                y + math.sin(yaw) * 0.09 + math.cos(yaw) * off_y,
                0.035,
            ]
            ball.write_root_pose_to_sim_index(
                root_pose=torch.tensor([[*ball_start, 0.0, 0.0, 0.0, 1.0]], device=cfg.device)
            )
            ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=cfg.device))
            ball.reset()

        def reset() -> None:
            if args.start_pose == "prone":
                root_pose = [0.0, 0.0, 0.07, 0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
            else:
                root_pose = [0.0, 0.0, profile.root_height, 0.0, 0.0, 0.0, 1.0]
            robot.write_root_pose_to_sim_index(root_pose=torch.tensor([root_pose], device=cfg.device))
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=cfg.device))
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_pos.torch))
            robot.reset()
            controller.reset()
            sim.forward()
            robot.update(cfg.dt)
            place_ball()
            pickup.reset()

        reset()
        trace = None
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = args.trace.open("w")
        writer = _create_video_writer(sim, args.record)
        viewers = tuple(sim.visualizers)
        count = sample_count = resets = 0
        episode_step = 0
        minimum_height = float("inf")
        max_wheel_speed = max_ball_speed = 0.0
        velocity_sum = np.zeros(3)
        twist_sum = np.zeros(3)
        twist_squared_error = np.zeros(3)
        speed_sum = 0.0
        telemetry_count = 0
        playback_samples = []
        active_counts = {}
        previous_policy = None
        completed = False
        start = time.perf_counter()
        print(f"Playing {args.behavior} with Newton MJWarp; velocity={args.velocity}", flush=True)
        try:
            with torch.inference_mode():
                while (
                    (not viewers or any(v.is_running() and not v.is_closed for v in viewers))
                    and not sim.is_stopped()
                    and (args.duration == 0.0 or count * cfg.dt < args.duration)
                ):
                    if args.repeat and episode_step * cfg.dt >= profile.cycle_time:
                        reset()
                        episode_step = 0
                        previous_policy = None
                        resets += 1
                        print(f"Demonstration reset {resets}", flush=True)
                    if count % 4 == 0:
                        active, command = policy_command(args.behavior, episode_step * cfg.dt, tuple(args.velocity))
                        if active != previous_policy:
                            print(f"{count * cfg.dt:.2f}s: {active}", flush=True)
                            if active == args.behavior and ball is not None:
                                place_ball()
                            previous_policy = active
                        controller.command[0] = torch.tensor(command, device=cfg.device)
                        observation = controller.observations()
                        if not torch.isfinite(observation).all():
                            raise RuntimeError("Non-finite robot state.")
                        session = sessions[active]
                        action = session.run(None, {session.get_inputs()[0].name: observation.cpu().numpy()})[0]
                        controller.set_action(
                            torch.as_tensor(action, device=cfg.device), BEHAVIORS[active].action_scale
                        )
                        pose = robot.data.root_link_pose_w.torch[0]
                        xyz = pose[:3].cpu().tolist()
                        if not args.free_camera:
                            sim.set_camera_view(
                                (xyz[0] + 0.6, xyz[1] - 0.7, xyz[2] + 0.35), (xyz[0], xyz[1], xyz[2] + 0.02)
                            )
                        height = float(pose[2])
                        minimum_height = min(minimum_height, height)
                        velocity_sum += robot.data.root_lin_vel_b.torch[0].cpu().numpy()
                        sample_count += 1
                        active_counts[active] = active_counts.get(active, 0) + 1
                        if wheel_ids:
                            max_wheel_speed = max(
                                max_wheel_speed, float(robot.data.joint_vel.torch[0, wheel_ids].abs().max())
                            )
                        if ball is not None:
                            max_ball_speed = max(max_ball_speed, float(ball.data.root_lin_vel_w.torch.norm()))
                        pickup_state = pickup.sample()
                        if trace:
                            trace.write(
                                json.dumps(
                                    {
                                        "time_s": count * cfg.dt,
                                        "episode": resets,
                                        "policy": active,
                                        "command": command,
                                        "root_pose": pose.cpu().tolist(),
                                        "gravity": robot.data.projected_gravity_b.torch[0].cpu().tolist(),
                                        "joint_pos": robot.data.joint_pos.torch[0, controller.joint_ids].cpu().tolist(),
                                        "pickup_object": pickup_state,
                                        "ball_position": ball.data.root_link_pos_w.torch[0].cpu().tolist()
                                        if ball
                                        else None,
                                    }
                                )
                                + "\n"
                            )
                        # Recovery, sitting and rolling intentionally violate walking's upright guard.
                        if args.behavior == "alpha_walking" and (
                            height < 0.055 or float(robot.data.projected_gravity_b.torch[0, 2]) > -0.3
                        ):
                            raise RuntimeError(f"Robot fell at {count * cfg.dt:.2f} s (height={height:.3f} m).")
                    controller.apply()
                    robot.write_data_to_sim()
                    sim.step(render=count % 4 == 0)
                    robot.update(cfg.dt)
                    pickup.update(cfg.dt)
                    if ball is not None:
                        ball.update(cfg.dt)
                    if count % 4 == 0:
                        # Sample the same root-link velocity and rendered state as native evaluation.
                        data = robot.data
                        twist = torch.cat((data.root_link_lin_vel_b.torch[0, :2], data.root_ang_vel_b.torch[0, 2:3]))
                        measured = twist.cpu().numpy()
                        speed = float(data.root_link_lin_vel_w.torch[0, :2].norm())
                        twist_sum += measured
                        twist_squared_error += (measured - np.asarray(args.velocity)) ** 2
                        speed_sum += speed
                        telemetry_count += 1
                        if args.record:
                            playback_samples.append(
                                {
                                    "time_s": count * cfg.dt,
                                    "simulation_time_s": (count + 1) * cfg.dt,
                                    "command": list(command[:3]),
                                    "body_twist": measured.tolist(),
                                    "ground_speed_m_s": speed,
                                }
                            )
                    if writer is not None and count % 4 == 0:
                        writer.write_frame(sim.visualizers[0].render_rgb_array())
                    count += 1
                    episode_step += 1
                    if args.real_time:
                        time.sleep(max(0.0, start + count * cfg.dt - time.perf_counter()))
                completed = True
        except KeyboardInterrupt:
            pass
        finally:
            if writer is not None:
                writer.close()
            if trace is not None:
                trace.close()
            metrics = {
                "behavior": args.behavior,
                "completed": completed,
                "simulated_seconds": count * cfg.dt,
                "resets": resets,
                "command": list(args.velocity),
                "start_pose": args.start_pose,
                "minimum_root_height_m": minimum_height if sample_count else None,
                "final_root_position_m": robot.data.root_link_pos_w.torch[0].cpu().tolist(),
                "final_projected_gravity": robot.data.projected_gravity_b.torch[0].cpu().tolist(),
                "mean_body_velocity_m_s": (velocity_sum / sample_count).tolist() if sample_count else None,
                "policy_seconds": {name: samples * 0.02 for name, samples in active_counts.items()},
                "pickup_object": pickup.metrics(),
                "max_wheel_speed_rad_s": max_wheel_speed if wheel_ids else None,
                "max_ball_speed_m_s": max_ball_speed if ball else None,
                "ball_displacement_m": (
                    (ball.data.root_link_pos_w.torch[0].cpu().numpy() - ball_start).tolist() if ball else None
                ),
            }
            metrics.update(
                _motion_metrics(
                    twist_sum.tolist(),
                    twist_squared_error.tolist(),
                    speed_sum,
                    telemetry_count,
                    args.behavior == "alpha_walking",
                )
            )
            print(json.dumps(metrics, indent=2), flush=True)
            metrics["playback_samples"] = playback_samples
            if args.metrics:
                args.metrics.parent.mkdir(parents=True, exist_ok=True)
                args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
            sim.clear_instance()


if __name__ == "__main__":
    main()
