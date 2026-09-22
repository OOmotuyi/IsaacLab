# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe physical marker pickup and bin release with pretrained body policies.

The mouth and task sequencer are scripted. This is not a newly trained
pick-and-place policy, and it never attaches or teleports the marker.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from marker_assets import BIN_FLOOR, BIN_HEIGHT, BIN_INNER, MARKER_LENGTH, MARKER_RADIUS, MOUTH_CLOSED, MOUTH_OPEN
from play import _create_video_writer, _simulation_cfg, load_policies

from isaaclab.app import add_launcher_args, launch_simulation


def parse_args() -> argparse.Namespace:
    """Parse object placement [m], jaw timing [s], and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    parser.add_argument("--scene_dir", type=Path, default=Path("logs/microduck/marker_pick/assets_lip"))
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/marker_pick/probe"))
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--marker_x", type=float, default=0.068)
    parser.add_argument("--support_height", type=float, default=0.030)
    parser.add_argument("--close_time", type=float, default=2.50)
    parser.add_argument("--approach_hold", type=float, default=1.0, help="Pause at the crouch phase [s].")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--real_time", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if not all(
        math.isfinite(v)
        for v in (args.duration, args.marker_x, args.support_height, args.close_time, args.approach_hold)
    ):
        parser.error("Numeric options must be finite.")
    if args.duration <= 0 or args.support_height < 0 or args.close_time < 1 or args.approach_hold < 0:
        parser.error("Require duration > 0, support_height >= 0, and close_time >= 1.")
    return args


def has_opposed_mouth_contacts(contact_names: list[list[str]]) -> bool:
    """Require penetrating marker contacts on both upper and lower gripping surfaces."""
    return any("upper_mouth_pad" in name for pair in contact_names for name in pair) and any(
        "lower_mouth_" in name for pair in contact_names for name in pair
    )


def main() -> None:
    """Run contact-based pickup, transport, release, and measured bin settlement."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _simulation_cfg(args)
    cfg.physics.solver_cfg.cone = "elliptic"
    cfg.physics.solver_cfg.impratio = 10.0
    cfg.default_visualizer_cfg.eye = (0.65, -0.65, 0.40)
    cfg.default_visualizer_cfg.lookat = (0.14, 0.0, 0.10)
    with launch_simulation(cfg, args):
        import numpy as np
        import torch
        import warp as wp
        from controller import DEFAULT_POSE, JOINT_NAMES, MicroDuckController
        from isaaclab_newton.sim.schemas import MujocoCollisionCfg

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        sim = SimulationContext(cfg)
        ground = sim_utils.GroundPlaneCfg(
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        )
        ground.func("/World/Ground", ground)
        light = sim_utils.DistantLightCfg(intensity=2500)
        light.func("/World/Light", light)
        bin_center = np.array([0.38, 0.0, 0.0])
        bin_cfg = sim_utils.UsdFileCfg(usd_path=str((args.scene_dir / "bin.usda").resolve()))
        bin_cfg.func("/World/Bin", bin_cfg, translation=tuple(bin_center))
        for index, y in enumerate((0.006,)):
            if args.support_height > 0:
                support = sim_utils.CuboidCfg(
                    size=(0.024, 0.016, args.support_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.40)),
                )
                support.func(f"/World/Support{index}", support, translation=(args.marker_x, y, args.support_height / 2))
        params = json.loads((args.asset_dir / "bam_m6.json").read_text())
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str((args.scene_dir / "robot_marker/robot_marker.usda").resolve()),
                    variants={"Physics": "mujoco"},
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={**dict(zip(JOINT_NAMES, DEFAULT_POSE)), "mouth": MOUTH_OPEN},
                    joint_vel={".*": 0},
                ),
                actuators={
                    "servos": ImplicitActuatorCfg(
                        joint_names_expr=list(JOINT_NAMES),
                        stiffness=0,
                        damping=0,
                        armature=params["armature"],
                        friction=params["friction_base"],
                        joint_effort_limit=7.4 * params["kt"] / params["R"],
                    ),
                    "mouth": ImplicitActuatorCfg(
                        joint_names_expr=["mouth"],
                        stiffness=1.0,
                        damping=0.008,
                        armature=0.0001,
                        friction=0.002,
                        joint_effort_limit=0.05,
                    ),
                },
            )
        )
        initial_marker = [args.marker_x, 0.006, args.support_height + MARKER_RADIUS + 0.0005]
        marker = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Marker",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str((args.scene_dir / "marker.usda").resolve()),
                    collision_props=[
                        MujocoCollisionCfg(
                            priority=1, condim=6, solref=(0.01, 1.0), solimp=(0.99, 0.9999, 0.001, 0.5, 2.0)
                        )
                    ],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(initial_marker)),
            )
        )
        policies = load_policies(
            {name: args.asset_dir / f"{name}.onnx" for name in ("alpha_stand", "alpha_ground_pick", "alpha_walking")}
        )
        sim.reset()
        controller = MicroDuckController(robot, params)
        jaw_ids = robot.find_joints("mouth")[0]
        head_id = robot.find_bodies("jaw_soft")[0][0]
        robot.write_root_pose_to_sim_index(root_pose=torch.tensor([[0, 0, 0.12, 0, 0, 0, 1]], device=cfg.device))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=cfg.device))
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_pos.torch))
        robot.reset()
        controller.reset()
        sim.forward()
        robot.update(cfg.dt)
        marker.update(cfg.dt)
        model = controller.solver.mj_model
        marker_geoms = [i for i in range(model.ngeom) if "Marker" in model.geom(i).name]
        if not marker_geoms:
            raise RuntimeError("Marker collider was not imported into the Newton model.")
        contact_contract = {
            model.geom(i).name: {
                "condim": int(model.geom_condim[i]),
                "friction": model.geom_friction[i].tolist(),
                "solref": model.geom_solref[i].tolist(),
                "solimp": model.geom_solimp[i].tolist(),
            }
            for i in range(model.ngeom)
            if i in marker_geoms or "mouth_pad" in model.geom(i).name
        }
        (args.output_dir / "contact_model.json").write_text(json.dumps(contact_contract, indent=2) + "\n")
        writer = _create_video_writer(sim, args.record)
        viewers = tuple(sim.visualizers)
        trace = (args.output_dir / "trace.jsonl").open("w")
        pick_end = 3.8 + args.approach_hold
        released = False
        held_time = settled_time = max_height = 0.0
        ever_lifted = success = False
        previous_phase = ""
        count = 0
        wall_start = time.perf_counter()
        try:
            with torch.inference_mode():
                while count * cfg.dt < args.duration and (not viewers or any(not v.is_closed for v in viewers)):
                    t = count * cfg.dt
                    if count % 4 == 0:
                        marker_pos = marker.data.root_link_pos_w.torch[0]
                        mouth = robot.data.body_link_pos_w.torch[:, head_id] + quat_apply(
                            robot.data.body_link_quat_w.torch[:, head_id],
                            marker_pos.new_tensor([[-0.00809334, 0, -0.0777383]]),
                        )
                        position = marker_pos.cpu().numpy()
                        mouth_distance = float((marker_pos - mouth[0]).norm())
                        max_height = max(max_height, float(position[2]))
                        contacts = controller.solver.mjw_data.contact
                        geoms = wp.to_torch(contacts.geom)
                        valid = (
                            torch.arange(geoms.shape[0], device=cfg.device)
                            < wp.to_torch(controller.solver.mjw_data.nacon)[0]
                        )
                        hit = (
                            valid
                            & (wp.to_torch(contacts.dist) <= 0.0)
                            & torch.isin(geoms, torch.tensor(marker_geoms, device=cfg.device)).any(-1)
                        )
                        other = geoms[hit].cpu().tolist()
                        contact_names = [[model.geom(g).name for g in pair] for pair in other]
                        both_jaws = has_opposed_mouth_contacts(contact_names)
                        lifted = position[2] > initial_marker[2] + 0.05 and mouth_distance < 0.06 and both_jaws
                        held_time = held_time + 0.02 if lifted else 0.0
                        ever_lifted |= held_time >= 0.3
                        axis = (
                            quat_apply(
                                marker.data.root_link_quat_w.torch,
                                marker_pos.new_tensor([[0, 1, 0]]),
                            )[0]
                            .cpu()
                            .numpy()
                        )
                        half_extent = abs(axis) * MARKER_LENGTH / 2 + MARKER_RADIUS * np.sqrt(
                            np.maximum(0, 1 - axis**2)
                        )
                        inside_xy = bool(
                            np.all(
                                abs(position[:2] - bin_center[:2]) + half_extent[:2] < np.asarray(BIN_INNER) / 2 - 0.003
                            )
                        )
                        command = [0.0] * 13
                        policy = "alpha_stand"
                        jaw_target = MOUTH_OPEN if t < args.close_time or released else MOUTH_CLOSED
                        phase = "settle"
                        if 1 <= t < pick_end:
                            policy = "alpha_ground_pick"
                            phase_time = min(t - 1, 1.4) + max(0.0, t - 2.4 - args.approach_hold)
                            angle = 2 * math.pi * phase_time / 4
                            command[:2] = [math.cos(angle), math.sin(angle)]
                            phase = "approach" if t < args.close_time else "close_and_lift"
                        elif t >= pick_end:
                            phase = "hold_check"
                            if ever_lifted and lifted and t > pick_end + 0.7 and not released:
                                phase = "transport"
                                policy = "alpha_walking"
                                yaw = float(
                                    torch.atan2(
                                        2
                                        * (
                                            robot.data.root_link_quat_w.torch[0, 3]
                                            * robot.data.root_link_quat_w.torch[0, 2]
                                            + robot.data.root_link_quat_w.torch[0, 0]
                                            * robot.data.root_link_quat_w.torch[0, 1]
                                        ),
                                        1
                                        - 2
                                        * (
                                            robot.data.root_link_quat_w.torch[0, 1] ** 2
                                            + robot.data.root_link_quat_w.torch[0, 2] ** 2
                                        ),
                                    )
                                )
                                command[:3] = [0.3, 0, float(np.clip(-2 * yaw - 2 * position[1], -0.6, 0.6))]
                                if inside_xy and position[2] - half_extent[2] > BIN_FLOOR + BIN_HEIGHT:
                                    released = True
                                    policy = "alpha_stand"
                                    command = [0.0] * 13
                                    jaw_target = MOUTH_OPEN
                            if released:
                                phase = "release"
                        bottom = position[2] - half_extent[2]
                        settled = (
                            released
                            and inside_xy
                            and BIN_FLOOR - 0.002 < bottom < BIN_FLOOR + 0.006
                            and float(marker.data.root_lin_vel_w.torch.norm()) < 0.05
                        )
                        settled_time = settled_time + 0.02 if settled else 0.0
                        success = ever_lifted and settled_time >= 0.5
                        if phase != previous_phase:
                            print(
                                f"{t:.2f}s: {phase}; marker={position.tolist()}; jaw contacts={both_jaws}", flush=True
                            )
                            previous_phase = phase
                        controller.command[0] = torch.tensor(command, device=cfg.device)
                        observation = controller.observations()
                        if not torch.isfinite(observation).all() or not torch.isfinite(marker_pos).all():
                            raise RuntimeError("Non-finite robot or marker state")
                        session = policies[policy]
                        action = session.run(None, {session.get_inputs()[0].name: observation.cpu().numpy()})[0]
                        controller.set_action(torch.as_tensor(action, device=cfg.device))
                        robot.actuators.target_command.set_position_index(
                            value=torch.tensor([[jaw_target]], device=cfg.device), joint_ids=jaw_ids
                        )
                        trace.write(
                            json.dumps(
                                {
                                    "time_s": t,
                                    "phase": phase,
                                    "policy": policy,
                                    "marker_position_m": position.tolist(),
                                    "mouth_position_m": mouth[0].tolist(),
                                    "marker_in_head_m": quat_apply_inverse(
                                        robot.data.body_link_quat_w.torch[:, head_id],
                                        marker_pos[None] - robot.data.body_link_pos_w.torch[:, head_id],
                                    )[0].tolist(),
                                    "mouth_distance_m": mouth_distance,
                                    "mouth_angle_rad": float(robot.data.joint_pos.torch[0, jaw_ids[0]]),
                                    "marker_quaternion_xyzw": marker.data.root_link_quat_w.torch[0].tolist(),
                                    "marker_velocity_m_s": marker.data.root_lin_vel_w.torch[0].tolist(),
                                    "contact_distance_m": wp.to_torch(contacts.dist)[hit].tolist(),
                                    "both_jaw_contacts": both_jaws,
                                    "contacts": contact_names,
                                    "lifted": bool(lifted),
                                    "released": released,
                                    "inside_bin_xy": inside_xy,
                                    "bottom_m": float(bottom),
                                    "success": success,
                                    "root_pose": robot.data.root_link_pose_w.torch[0].tolist(),
                                }
                            )
                            + "\n"
                        )
                    controller.apply()
                    robot.write_data_to_sim()
                    sim.step(render=count % 4 == 0)
                    robot.update(cfg.dt)
                    marker.update(cfg.dt)
                    if writer is not None and count % 4 == 0:
                        writer.write_frame(viewers[0].render_rgb_array())
                    count += 1
                    if args.real_time:
                        time.sleep(max(0, wall_start + count * cfg.dt - time.perf_counter()))
        finally:
            trace.close()
            if writer is not None:
                writer.close()
            result = {
                "controller": "released body policies with scripted jaw and task sequencer",
                "physics_dt_s": cfg.dt,
                "policy_dt_s": 4 * cfg.dt,
                "final_root_pose": robot.data.root_link_pose_w.torch[0].tolist(),
                "final_projected_gravity": robot.data.projected_gravity_b.torch[0].tolist(),
                "new_policy_trained": False,
                "grasp_attachment": False,
                "simulated_seconds": count * cfg.dt,
                "initial_marker_position_m": initial_marker,
                "maximum_marker_center_height_m": max_height,
                "verified_lift": bool(ever_lifted),
                "released_over_bin": released,
                "settled_in_bin": bool(success),
                "final_marker_position_m": marker.data.root_link_pos_w.torch[0].tolist(),
                "parameters": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
            sim.clear_instance()


if __name__ == "__main__":
    main()
