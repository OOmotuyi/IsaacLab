# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a rigid marker and a deformable sock in a child bedroom.

The default mouth, carrying head pose, and task sequencer are scripted. The optional
object-grasp actor learns body and jaw control; navigation and release remain scripted.
Optional, explicitly labeled assistance pins a small sock patch
while the jaw is closed. The marker always uses contact-only grasping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from bin_geometry import BinGeometry, cylinder_surface_points
from cleanup_recording import CleanupRecording
from marker_assets import MARKER_LENGTH, MARKER_RADIUS, MOUTH_OPEN
from play import _create_video_writer, _simulation_cfg, load_policies

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.contrib.microduck.constants import MARKER_SUPPORT_SIZE, MARKER_SUPPORT_Y

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab_contrib.coupling import CouplerProxyCfg


def parse_args() -> argparse.Namespace:
    """Parse object placement [m], duration [s], and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    parser.add_argument(
        "--walking_policy",
        type=Path,
        default=Path("logs/rsl_rl/microduck_velocity/2026-09-18_16-52-41_native_walking_v2/exported/policy.onnx"),
    )
    parser.add_argument("--standing_policy", type=Path, help="Override the standing body actor.")
    parser.add_argument("--pickup_policy", type=Path, help="Override the phase-conditioned pickup body actor.")
    parser.add_argument("--grasp_policy", type=Path, help="Use a trained object-conditioned body/jaw actor.")
    parser.add_argument("--scene_dir", type=Path, default=Path("logs/microduck/marker_pick/assets_lip"))
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/room_cleanup/probe_01"))
    parser.add_argument("--room_dir", type=Path, default=Path("logs/microduck/room_cleanup/assets"))
    parser.add_argument("--mode", choices=("cleanup", "marker", "sock", "navigate"), default="cleanup")
    parser.add_argument("--first_object", choices=("marker", "sock"), default="marker")
    parser.add_argument("--physics_substeps", type=int, help="Defaults to 8 for learned grasping and 4 otherwise.")
    parser.add_argument("--cloth_iterations", type=int, default=30)
    parser.add_argument(
        "--coupling_relaxation",
        type=float,
        default=1.0,
        help="Cloth force feedback relaxation [0, 1]; zero disables feedback to the robot.",
    )
    parser.add_argument(
        "--sock_close_time", type=float, default=0.7, help="Scripted sock jaw-closing time after pickup starts [s]."
    )
    parser.add_argument("--sock_y", type=float, default=0.26)
    parser.add_argument("--start_xy", type=float, nargs=2, help="Override the initial robot position [m].")
    parser.add_argument("--sock_shape", choices=("folded", "flat"), default="flat")
    parser.add_argument("--align_beak", action="store_true", help="Enable experimental measured head-yaw correction.")
    parser.add_argument(
        "--assisted_grasp", action="store_true", help="Explicitly pin a nearby sock patch while the jaw is closed."
    )
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--marker_x", type=float, default=0.068)
    parser.add_argument("--support_height", type=float, default=0.030)
    parser.add_argument(
        "--record_states",
        action="store_true",
        help="Save measured body and cloth poses at 50 Hz for offline RTX rendering.",
    )
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--live_state", type=Path, help="Atomically publish measured states for a live multi-room viewer."
    )
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--real_time", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.physics_substeps is None:
        args.physics_substeps = 8 if args.grasp_policy else 4
    if not all(math.isfinite(v) for v in (args.duration, args.marker_x, args.support_height, args.sock_y)):
        parser.error("Numeric options must be finite.")
    if not math.isfinite(args.coupling_relaxation) or not 0.0 <= args.coupling_relaxation <= 1.0:
        parser.error("coupling_relaxation must be within [0, 1].")
    if args.start_xy is not None and not all(math.isfinite(v) for v in args.start_xy):
        parser.error("start_xy must be finite.")
    if args.duration <= 0 or args.support_height < 0 or min(args.physics_substeps, args.cloth_iterations) < 1:
        parser.error("Require positive duration and solver counts, and nonnegative support_height.")
    return args


def has_opposed_mouth_contacts(contact_names: list[list[str]], *, upper_shell_contact: bool = False) -> bool:
    """Require penetrating marker contacts on both upper and lower gripping surfaces."""
    return (upper_shell_contact or any("upper_mouth_pad" in name for pair in contact_names for name in pair)) and any(
        "lower_mouth_" in name for pair in contact_names for name in pair
    )


def cloth_collision_pipeline(model, *, marker_contact: bool = False):
    """Give VBD cloth contacts, optionally scaling marker contact to 1000 [N/m], 0.5 [N*s/m]."""
    import warp as wp
    from newton import CollisionPipeline, ShapeFlags

    if marker_contact:
        # MuJoCo solref gains are mass-normalized; their imported damping is too large for VBD.
        # Clone the destination arrays so MJWarp retains its physical grasp contact settings.
        indices = [i for i, label in enumerate(model.shape_label) if "/Marker/" in label]
        model.shape_material_ke = wp.clone(model.shape_material_ke)
        model.shape_material_kd = wp.clone(model.shape_material_kd)
        wp.to_torch(model.shape_material_ke)[indices] = 1000.0
        wp.to_torch(model.shape_material_kd)[indices] = 0.5
    # Keep this view's collision mask separate from the rigid solver's model.
    model.shape_flags = wp.clone(model.shape_flags)
    wp.to_torch(model.shape_flags).bitwise_and_(~int(ShapeFlags.COLLIDE_SHAPES))
    return CollisionPipeline(
        model,
        shape_pairs_filtered=wp.empty(0, dtype=wp.vec2i, device=model.device),
        soft_contact_gap=0.004,
        soft_contact_max=8192,
        enable_rigid_soft_full_surface_contact=True,
    )


def _basket_cloth_collision_pipeline(model):
    return cloth_collision_pipeline(model, marker_contact=True)


def _prepare_robot_sdfs(stage: Usd.Stage) -> None:
    """Cook full-surface cloth collision fields for the robot meshes."""
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg

    from pxr import Usd, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    for prim in list(Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"))):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            sim_utils.schemas.apply_mesh_collision_properties(
                str(prim.GetPath()), [NewtonSDFCollisionCfg(sdf_max_resolution=64)], stage=stage
            )


def _configure_assisted_beak_proxy(cfg: CouplerProxyCfg, *, enabled: bool) -> None:
    """Retain beak collisions while excluding their assisted feedback to the robot."""
    if enabled:
        cfg.proxies.append(
            cfg.proxies[0].replace(
                bodies=[r"/World/Robot/.*jaw_soft"],
                proxy_relaxation=0.0,
                mass_scale=1000.0,
            )
        )
        cfg.proxies[0].bodies = [r"/World/Robot/(?!.*(?:jaw_soft|lower_beak)).*$", r"/World/Marker"]


def _load_body_policies(args: argparse.Namespace):
    """Load body actors and verify the training provenance embedded in their exports."""
    policy_paths = {name: args.asset_dir / f"{name}.onnx" for name in ("alpha_stand", "alpha_ground_pick")}
    if args.standing_policy:
        policy_paths["alpha_stand"] = args.standing_policy
    if args.pickup_policy:
        policy_paths["alpha_ground_pick"] = args.pickup_policy
    if args.grasp_policy:
        from grasp_playback import load_grasp_policy

        policy_paths.pop("alpha_ground_pick")
        policies = load_policies(policy_paths)
        policies["alpha_ground_pick"] = load_grasp_policy(args.grasp_policy)
        policy_paths["alpha_ground_pick"] = args.grasp_policy
        stand = policies["alpha_stand"].get_modelmeta().custom_metadata_map
        if stand.get("training_backend") != "isaaclab_newton" or stand.get("initialization") != "random":
            raise ValueError("Object grasp mode requires --standing_policy from scratch Isaac training.")
    else:
        policies = load_policies(policy_paths)
    policy_provenance = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": policies[name].get_modelmeta().custom_metadata_map,
        }
        for name, path in policy_paths.items()
        if name in {"alpha_stand", "alpha_ground_pick"}
    }
    native_body_skills = all(
        item["metadata"].get("training_backend") == "isaaclab_newton"
        and float(item["metadata"].get("optimizer_steps", 0)) > 0
        for item in policy_provenance.values()
    )
    return policies, policy_provenance, native_body_skills


def adapt_skill_command(command: list[float], metadata: dict[str, str]) -> list[float]:
    """Supply the trained reach-depth fraction without changing phase timing."""
    adapted = command.copy()
    if metadata.get("command_contract") == "phase_cos_sin_depth":
        adapted[2] = 1.0
    return adapted


def main() -> None:
    """Run contact-based pickup, transport, release, and measured bin settlement."""
    args = parse_args()
    start_with_sock = args.mode == "sock" or (args.mode == "cleanup" and args.first_object == "sock")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "room_cleanup.py",
            "cleanup_sequence.py",
            "grasp_playback.py",
            "room_assets.py",
            "beak_grasp.py",
            "assisted_grasp.py",
            "bin_geometry.py",
            "cleanup_recording.py",
            "bedroom_materials.py",
        )
    }
    bin_geometry = BinGeometry.load(args.room_dir)
    cfg = _simulation_cfg(args)
    cfg.physics.solver_cfg.cone = "elliptic"
    cfg.physics.solver_cfg.impratio = 10.0
    from isaaclab_newton.physics import NewtonCfg, NewtonSoftContactCfg, VBDSolverCfg

    from isaaclab.utils import configclass

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

    @configclass
    class SockVBDSolverCfg(VBDSolverCfg):
        # Smaller smoothing velocity reduces creep in a sustained fabric grasp [m/s].
        friction_epsilon: float = 0.0005

    @configclass
    class SockProxyMappingCfg(CouplerProxyMappingCfg):
        proxy_relaxation: float = args.coupling_relaxation

    rigid_cfg = cfg.physics.solver_cfg
    rigid_cfg.iterations = 100
    coupled_cfg = CouplerProxyCfg(
        entries=[
            # Empty explicit shape ownership keeps static shapes visible in both views.
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=rigid_cfg,
                bodies=[r"/World/Robot", r"/World/Marker"],
                include_body_shapes=False,
            ),
            CouplerEntryCfg(
                name="cloth",
                solver_cfg=SockVBDSolverCfg(
                    iterations=args.cloth_iterations,
                    rigid_body_particle_contact_buffer_size=4096,
                    particle_enable_self_contact=True,
                    particle_self_contact_radius=0.0005,
                    particle_self_contact_margin=0.001,
                    particle_rest_shape_contact_exclusion_radius=0.003,
                    particle_collision_detection_interval=1 if args.assisted_grasp else 5,
                ),
                all_particles=True,
                include_static_shapes=True,
                substeps=1,
            ),
        ],
        proxies=[
            SockProxyMappingCfg(
                source="rigid",
                destination="cloth",
                bodies=[r"/World/Robot", r"/World/Marker"],
                collide_interval=1,
                collision_pipeline=_basket_cloth_collision_pipeline
                if bin_geometry.shape != "box"
                else cloth_collision_pipeline,
            )
        ],
        iterations=1,
    )
    _configure_assisted_beak_proxy(coupled_cfg, enabled=args.assisted_grasp)
    cfg.physics = NewtonCfg(
        solver_cfg=coupled_cfg,
        use_cuda_graph=True,
        num_substeps=args.physics_substeps,
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=1000.0, soft_contact_kd=0.01, soft_contact_mu=4.0),
    )
    cfg.default_visualizer_cfg.eye = (0.70, -0.62, 0.48)
    cfg.default_visualizer_cfg.lookat = (0.16, 0.09, 0.10)
    with launch_simulation(cfg, args):
        import numpy as np
        import torch
        import warp as wp
        from assisted_grasp import ASSISTED_JAW_ANGLE, AssistedSockGrasp, label_assisted_frame
        from beak_grasp import BeakGraspController
        from cleanup_sequence import BasketReach, CleanupSequence, PickupState
        from controller import DEFAULT_POSE, JOINT_NAMES, MicroDuckController
        from isaaclab_newton.physics import NewtonManager
        from isaaclab_newton.sim.schemas import (
            MujocoCollisionCfg,
            NewtonDeformableBodyPropertiesCfg,
            NewtonSDFCollisionCfg,
        )
        from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import (
            Articulation,
            ArticulationCfg,
            DeformableObject,
            DeformableObjectCfg,
            RigidObject,
            RigidObjectCfg,
        )
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
        room_cfg = sim_utils.UsdFileCfg(usd_path=str((args.room_dir / "children_room.usda").resolve()))
        room_cfg.func("/World/Room", room_cfg)
        sock_asset = "child_sock_folded.usda" if args.sock_shape == "folded" else "child_sock.usda"
        sock = DeformableObject(
            DeformableObjectCfg(
                prim_path="/World/Sock",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str((args.room_dir / sock_asset).resolve()),
                    deformable_props=NewtonDeformableBodyPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.35, 0.48)),
                    physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                        density=0.6,
                        particle_radius=0.001,
                        tri_ke=500,
                        tri_ka=500,
                        tri_kd=0.001,
                        edge_ke=0.05,
                        edge_kd=0.001,
                    ),
                ),
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=(args.marker_x, args.sock_y - 0.007, 0.018)
                    if args.sock_shape == "folded"
                    else (args.marker_x + 0.004, args.sock_y - 0.060, 0.043),
                    rot=(0, 0, 0, 1) if args.sock_shape == "folded" else (0, 0, 0.70710678, 0.70710678),
                ),
            )
        )
        support = sim_utils.CuboidCfg(
            size=(0.024, 0.016, 0.030) if args.sock_shape == "folded" else (0.042, 0.006, 0.036),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.5 if args.sock_shape == "folded" else 0.1,
                dynamic_friction=0.5 if args.sock_shape == "folded" else 0.1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.66, 0.35)),
        )
        for side in () if args.sock_shape == "folded" else (-1, 1):
            support.func(
                f"/World/SockSupport{side + 1}",
                support,
                translation=(args.marker_x, args.sock_y, 0.015)
                if args.sock_shape == "folded"
                else (args.marker_x, args.sock_y + side * 0.055, 0.018),
            )
        bin_center = np.array([0.38, 0.13, 0.0])
        bin_cfg = sim_utils.UsdFileCfg(usd_path=str((args.room_dir / "cleanup_bin.usda").resolve()))
        bin_cfg.func("/World/Bin", bin_cfg, translation=tuple(bin_center))
        marker_support_y = tuple(0.006 + offset for offset in MARKER_SUPPORT_Y) if args.grasp_policy else (0.006,)
        marker_support_xy = MARKER_SUPPORT_SIZE[:2] if args.grasp_policy else (0.024, 0.016)
        for index, y in enumerate(marker_support_y):
            if args.support_height > 0:
                support = sim_utils.CuboidCfg(
                    size=(*marker_support_xy, args.support_height),
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
                        stiffness=3.0 if start_with_sock else 1.0,
                        damping=0.06 if start_with_sock else 0.008,
                        armature=0.0001,
                        friction=0.002,
                        joint_effort_limit=0.15 if start_with_sock else 0.05,
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
                        NewtonSDFCollisionCfg(sdf_max_resolution=64),
                        MujocoCollisionCfg(
                            priority=1, condim=6, solref=(0.01, 1.0), solimp=(0.99, 0.9999, 0.001, 0.5, 2.0)
                        ),
                    ],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(initial_marker)),
            )
        )
        policies, policy_provenance, native_body_skills = _load_body_policies(args)
        import onnxruntime as ort

        native_path = args.walking_policy
        native_options = ort.SessionOptions()
        native_options.intra_op_num_threads = 1
        policies["native_walking"] = ort.InferenceSession(
            str(native_path), sess_options=native_options, providers=["CPUExecutionProvider"]
        )
        from pxr import Usd, UsdGeom

        _prepare_robot_sdfs(sim.stage)
        sim.reset()

        sock_stage = Usd.Stage.Open(str((args.room_dir / sock_asset).resolve()))
        colors = np.asarray(UsdGeom.Mesh(sock_stage.GetPrimAtPath("/Sock/Textile")).GetDisplayColorAttr().Get())
        if len(colors) == NewtonManager._model.tri_count:
            wp.to_torch(NewtonManager._model.tri_color).copy_(torch.tensor(colors, device=cfg.device))
        controller = MicroDuckController(robot, params, solver=NewtonManager._solver.solver("rigid"))
        from grasp_playback import MicroDuckGraspPlayback, infer_cleanup_action

        grasp_actor = MicroDuckGraspPlayback(policies["alpha_ground_pick"], controller) if args.grasp_policy else None
        jaw_ids = robot.find_joints("mouth")[0]
        head_id = robot.find_bodies("jaw_soft")[0][0]
        robot.write_root_pose_to_sim_index(
            root_pose=torch.tensor(
                [
                    [
                        args.start_xy[0] if args.start_xy else 0.24 if args.mode == "navigate" else 0,
                        args.start_xy[1] if args.start_xy else args.sock_y - 0.006 if start_with_sock else 0,
                        0.12,
                        0,
                        0,
                        0,
                        1,
                    ]
                ],
                device=cfg.device,
            )
        )
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=cfg.device))
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_pos.torch))
        robot.reset()
        controller.reset()
        sim.forward()
        robot.update(cfg.dt)
        marker.update(cfg.dt)
        sock.update(cfg.dt)
        sim.stage.Export(str(args.output_dir / "scene.usda"))
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
        if viewers:
            from PIL import Image

            sim.set_camera_view(eye=(2.0, -2.5, 1.8), target=(-0.03, 0.35, 0.35))
            sim.render()
            Image.fromarray(viewers[0].render_rgb_array()).save(args.output_dir / "room_overview.jpg")
            sim.set_camera_view(eye=cfg.default_visualizer_cfg.eye, target=cfg.default_visualizer_cfg.lookat)
            sim.render()
        trace = (args.output_dir / "trace.jsonl").open("w")
        sequence = CleanupSequence(
            mode=args.mode,
            first_object=args.first_object,
            sock_y=args.sock_y,
            sock_folded=args.sock_shape == "folded",
            sock_close_time_s=args.sock_close_time,
            object_conditioned_pick=grasp_actor is not None,
            bin_geometry=bin_geometry,
            bin_center=bin_center,
        )
        if args.mode == "navigate":
            sequence.phase = "retreat"
        max_height = 0.0
        coupler = NewtonManager._solver
        cloth_view = coupler.view("cloth")
        marker_cloth_material = {
            label: {
                "stiffness_n_m": float(cloth_view.shape_material_ke.numpy()[i]),
                "damping_ns_m": float(cloth_view.shape_material_kd.numpy()[i]),
            }
            for i, label in enumerate(cloth_view.shape_label)
            if "/Marker/" in label
        }
        node_frames = []
        recording = CleanupRecording(args.record_states, robot.body_names, args.live_state)
        bin_geom_ids = [i for i in range(model.ngeom) if "Bin/" in model.geom(i).name]
        robot_geom_ids = [i for i in range(model.ngeom) if "Robot/" in model.geom(i).name]
        maximum_robot_bin_penetration = 0.0
        previous_phase = ""
        count = 0
        target_history = controller.home.unsqueeze(0).repeat(4, 1, 1)
        current_target = controller.home.clone()
        previous_policy = ""
        head_anchor = None
        basket_reach = BasketReach()
        grasp_point = None
        beak = BeakGraspController(
            controller.solver.mj_model, controller.dof_ids.cpu().numpy(), np.asarray(DEFAULT_POSE)
        )
        assistance = AssistedSockGrasp(sock, cloth_view, NewtonManager._model) if args.assisted_grasp else None
        sock_jaw_profile = start_with_sock
        wall_start = time.perf_counter()
        failure = None
        objects = {}
        root = np.full(7, np.nan)
        ever_lifted = released = False
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
                        grip_position = robot.data.body_link_pos_w.torch[:, head_id] + quat_apply(
                            robot.data.body_link_quat_w.torch[:, head_id],
                            marker_pos.new_tensor([[-0.0115, 0, -0.063]]),
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
                        marker_surface = cylinder_surface_points(position, axis, MARKER_LENGTH, MARKER_RADIUS)
                        inside_xy = bin_geometry.contains(
                            position - half_extent, position + half_extent, bin_center, points=marker_surface
                        )
                        bin_hits = valid & torch.isin(geoms, torch.tensor(bin_geom_ids, device=cfg.device)).any(-1)
                        bin_hits &= torch.isin(geoms, torch.tensor(robot_geom_ids, device=cfg.device)).any(-1)
                        distances = wp.to_torch(contacts.dist)
                        maximum_robot_bin_penetration = max(
                            maximum_robot_bin_penetration,
                            float(-torch.cat((distances[bin_hits], distances.new_zeros(1))).min()),
                        )
                        nodes = sock.data.nodal_pos_w.torch[0]
                        soft = coupler.get_proxy_contacts("rigid", "cloth")
                        nsoft = min(int(wp.to_torch(soft.soft_contact_count)[0]), soft.soft_contact_shape.shape[0])
                        shape_ids = wp.to_torch(soft.soft_contact_shape)[:nsoft].long()
                        particle_ids = wp.to_torch(soft.soft_contact_indices)[:nsoft].long()
                        weights = wp.to_torch(soft.soft_contact_barycentric)[:nsoft]
                        valid_soft = (shape_ids >= 0) & (particle_ids >= 0).any(-1)
                        shape_ids = shape_ids[valid_soft]
                        contact_points = (
                            nodes[particle_ids[valid_soft].clamp_min(0)] * weights[valid_soft, :, None]
                        ).sum(1)
                        bodies = wp.to_torch(cloth_view.shape_body)[shape_ids].long()
                        poses = wp.to_torch(coupler.entry_state("cloth").body_q)[bodies.clamp_min(0)]
                        local_points = wp.to_torch(soft.soft_contact_body_pos)[:nsoft][valid_soft]
                        world_points = torch.where(
                            (bodies >= 0)[:, None], poses[:, :3] + quat_apply(poses[:, 3:], local_points), local_points
                        )
                        normals = wp.to_torch(soft.soft_contact_normal)[:nsoft][valid_soft]
                        separation = ((contact_points - world_points) * normals).sum(-1) - 0.001
                        active = separation <= 0.0005
                        active_shapes = shape_ids[active].cpu().tolist()
                        sock_contact_names = [cloth_view.shape_label[i] for i in active_shapes]
                        inverse_head = robot.data.body_link_quat_w.torch[:, head_id].clone()
                        inverse_head[:, :3] *= -1
                        head_points = quat_apply(
                            inverse_head.expand(len(active_shapes), -1),
                            contact_points[active] - robot.data.body_link_pos_w.torch[:, head_id],
                        )
                        front_contacts = (head_points[:, 2] < -0.045).cpu().tolist()
                        upper_shell_contact = any(
                            front and "top_head_shell" in name
                            for front, name in zip(front_contacts, sock_contact_names)
                        )
                        sock_opposed = has_opposed_mouth_contacts(
                            [sock_contact_names], upper_shell_contact=upper_shell_contact
                        )
                        sock_center = nodes.mean(0).cpu().numpy()
                        objects = {
                            "marker": PickupState(
                                position,
                                position - half_extent,
                                position + half_extent,
                                float(marker.data.root_lin_vel_w.torch.norm()),
                                both_jaws,
                                mouth_distance,
                                surface_points=marker_surface,
                            ),
                            "sock": PickupState(
                                sock_center,
                                nodes.amin(0).cpu().numpy(),
                                nodes.amax(0).cpu().numpy(),
                                float(sock.data.nodal_vel_w.torch[0].norm(dim=-1).mean()),
                                sock_opposed,
                                float((nodes - grip_position[0]).norm(dim=-1).min()),
                                assisted_grasp=assistance is not None and assistance.active,
                                surface_points=nodes.cpu().numpy(),
                            ),
                        }
                        root = robot.data.root_link_pose_w.torch[0].cpu().numpy()
                        qx, qy, qz, qw = root[3:]
                        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                        if sequence.phase != "pick" and (root[2] < 0.065 or 1 - 2 * (qx * qx + qy * qy) < 0.45):
                            sequence.phase, sequence.phase_start = "robot_fell", t
                        policy, command, jaw_target = sequence.step(
                            t,
                            root[:3],
                            yaw,
                            objects,
                            learned_grasp_ready=bool(
                                grasp_actor and grasp_actor.target == sequence.target and grasp_actor.ready
                            ),
                        )
                        policy = (
                            "alpha_ground_pick"
                            if grasp_actor is not None and sequence.phase == "lift_check"
                            else policy
                        )
                        # Leave room for the seated fabric instead of closing through it.
                        jaw_target = (
                            ASSISTED_JAW_ANGLE
                            if assistance is not None and sequence.target == "sock" and jaw_target < 0.0
                            else jaw_target
                        )
                        strong_grip = sequence.target == "sock"
                        if strong_grip != sock_jaw_profile:
                            sock_jaw_profile = strong_grip
                            robot.write_joint_stiffness_to_sim_index(
                                stiffness=3.0 if strong_grip else 1.0, joint_ids=jaw_ids
                            )
                            robot.write_joint_damping_to_sim_index(
                                damping=0.06 if strong_grip else 0.008, joint_ids=jaw_ids
                            )
                            robot.write_joint_effort_limit_to_sim_index(
                                limits=0.15 if strong_grip else 0.05,
                                joint_ids=jaw_ids,
                            )
                        phase = sequence.target + ":" + sequence.phase
                        lifted = sequence.lifted["marker"]
                        ever_lifted = lifted
                        released = sequence.released["marker"]
                        success = all(sequence.completed.values())
                        bottom = position[2] - half_extent[2]
                        if count % 20 == 0:
                            node_frames.append(nodes.cpu().numpy().copy())
                        if phase != previous_phase:
                            print(
                                f"{t:.2f}s: {phase}; marker={position.tolist()}; "
                                f"sock={sock_center.tolist()}; contacts={both_jaws}/{sock_opposed}",
                                flush=True,
                            )
                            previous_phase = phase
                        if sequence.phase == "recover" or (
                            assistance is not None and assistance.active and policy == "native_walking"
                        ):
                            head_anchor = None
                        if policy == "native_walking" and head_anchor is not None:
                            release_pitch = command[4] if sequence.phase == "drop" else 0.0
                            command[3:7] = head_anchor[0].tolist()
                            command[4] += release_pitch
                        command = adapt_skill_command(command, policies[policy].get_modelmeta().custom_metadata_map)
                        controller.command[0] = torch.tensor(command, device=cfg.device)
                        observation = controller.observations()
                        if not all(torch.isfinite(value).all() for value in (observation, marker_pos, nodes)):
                            raise RuntimeError("Non-finite robot, marker, or cloth state")
                        controller.training_model = policy == "native_walking"
                        session = policies[policy]
                        action, jaw_target = infer_cleanup_action(
                            session,
                            observation,
                            jaw_target,
                            grasp=grasp_actor,
                            policy_name=policy,
                            target=sequence.target,
                            marker=marker,
                            sock=sock,
                            marker_contacts=contact_names,
                            sock_opposed=sock_opposed,
                            assistance=assistance,
                        )
                        raw_head_action = action[0, 5:9].tolist()
                        if (
                            grasp_actor is None
                            and policy == "alpha_ground_pick"
                            and sequence.target == "sock"
                            and args.align_beak
                        ):
                            if grasp_point is None:
                                section = nodes[(nodes[:, 1] - objects["sock"].center[1]).abs() < 0.012]
                                grasp_point = (section.mean(0) if len(section) else nodes.mean(0)).cpu().numpy()
                            beak.apply(
                                wp.to_torch(controller.solver.mjw_data.qpos)[0].cpu().numpy(),
                                action,
                                grip_position[0].cpu().numpy(),
                                grasp_point,
                            )
                        if policy == "alpha_ground_pick" or (policy == "alpha_walking" and sequence.target == "marker"):
                            head_anchor = action[:, 5:9].copy()
                        elif head_anchor is not None and (
                            policy != "native_walking"
                            or assistance is None
                            and sequence.target == "sock"
                            and sequence.phase in {"lift_check", "clear_support", "carry_lane", "carry"}
                        ):
                            # Keep the grasp pose while the body skill controls the legs.
                            action[:, 5:9] = head_anchor
                        basket_reach.apply(action, head_anchor, root, command, sequence)
                        controller.set_action(torch.as_tensor(action, device=cfg.device))
                        current_target.copy_(controller.target)
                        if policy != previous_policy:
                            target_history[:] = current_target
                            previous_policy = policy
                        robot.actuators.target_command.set_position_index(
                            value=torch.tensor([[jaw_target]], device=cfg.device), joint_ids=jaw_ids
                        )
                        recording.append(
                            t, robot, marker, nodes, bool(assistance and assistance.active), phase=phase, policy=policy
                        )
                        trace.write(
                            json.dumps(
                                {
                                    "time_s": t,
                                    "phase": phase,
                                    "policy": policy,
                                    "command": command,
                                    "raw_head_action": raw_head_action,
                                    "learned_grasp_stage": int(grasp_actor.progress.stage[0]) if grasp_actor else None,
                                    "learned_jaw_target_rad": jaw_target
                                    if grasp_actor and policy == "alpha_ground_pick"
                                    else None,
                                    "basket_reach_fraction": basket_reach.strength,
                                    "applied_head_action": action[0, 5:9].tolist(),
                                    "actual_velocity_m_s": robot.data.root_link_lin_vel_b.torch[0].tolist(),
                                    "actual_yaw_rate_rad_s": float(robot.data.root_ang_vel_b.torch[0, 2]),
                                    "marker_position_m": position.tolist(),
                                    "sock_center_m": sock.data.root_pos_w.torch[0].tolist(),
                                    "sock_jaw_contacts": sock_opposed,
                                    "sock_assistance_active": assistance is not None and assistance.active,
                                    "sock_assisted_node_count": len(assistance.ids) if assistance else 0,
                                    "sock_beak_contacts_suppressed": bool(
                                        assistance and assistance.mouth_contacts_suppressed
                                    ),
                                    "sock_upper_shell_grip": upper_shell_contact,
                                    "sock_contact_shapes": sock_contact_names,
                                    "completed": sequence.completed.copy(),
                                    "objects_in_bin": {
                                        name: sequence.settled_in_bin(obj) for name, obj in objects.items()
                                    },
                                    "sock_bounds_m": [nodes.amin(0).tolist(), nodes.amax(0).tolist()],
                                    "mouth_position_m": mouth[0].tolist(),
                                    "grip_position_m": grip_position[0].tolist(),
                                    "grip_target_m": beak.target.tolist() if beak.target is not None else None,
                                    "mouth_exit_direction_w": quat_apply(
                                        robot.data.body_link_quat_w.torch[:, head_id],
                                        torch.tensor([[0.0, 0.0, -1.0]], device=cfg.device),
                                    )[0].tolist(),
                                    "marker_in_head_m": quat_apply_inverse(
                                        robot.data.body_link_quat_w.torch[:, head_id],
                                        marker_pos[None] - robot.data.body_link_pos_w.torch[:, head_id],
                                    )[0].tolist(),
                                    "mouth_distance_m": mouth_distance,
                                    "mouth_angle_rad": float(robot.data.joint_pos.torch[0, jaw_ids[0]]),
                                    "head_pose_m_xyzw": robot.data.body_link_pose_w.torch[0, head_id].tolist(),
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
                    target_history[count % 4] = current_target
                    controller.target.copy_(
                        target_history[(count - 3) % 4] if policy == "native_walking" else current_target
                    )
                    controller.apply()
                    robot.write_data_to_sim()
                    if assistance is not None:
                        assistance.update(
                            robot.data.body_link_pose_w.torch[0, head_id],
                            allow_attach=sequence.target == "sock" and sequence.phase == "pick",
                            jaw_target=jaw_target,
                            jaw_angle=float(robot.data.joint_pos.torch[0, jaw_ids[0]]),
                            jaw_velocity=float(robot.data.joint_vel.torch[0, jaw_ids[0]]),
                            time_s=t,
                            dt=cfg.dt,
                        )
                    sim.step(render=count % 4 == 0)
                    robot.update(cfg.dt)
                    marker.update(cfg.dt)
                    sock.update(cfg.dt)
                    if assistance is not None and count % 4 == 0:
                        assistance.validate()
                    if writer is not None and count % 4 == 0:
                        frame = viewers[0].render_rgb_array()
                        if assistance is not None:
                            frame = label_assisted_frame(
                                frame,
                                assistance.active,
                                body_skills_trained=native_body_skills,
                            )
                        writer.write_frame(frame)
                    count += 1
                    if sequence.phase in {"grasp_failed", "robot_fell"} or (
                        sequence.phase == "done" and t - sequence.phase_start >= 2.0
                    ):
                        break
                    if args.real_time:
                        time.sleep(max(0, wall_start + count * cfg.dt - time.perf_counter()))
        except (Exception, KeyboardInterrupt) as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            trace.close()
            if writer is not None:
                writer.close()
            np.savez_compressed(args.output_dir / "sock_nodes.npz", positions_m=np.asarray(node_frames), dt_s=0.1)
            recording.save(args.output_dir)
            result = {
                "objects_lifted": sequence.lifted,
                "objects_released": sequence.released,
                "objects_in_bin": {
                    name: bool(sequence.completed[name] and sequence.settled_in_bin(obj, clearance=-0.001))
                    for name, obj in objects.items()
                },
                "final_phase": sequence.phase,
                "sock_nodes": sock.max_sim_vertices_per_body,
                "head_frame_max_error_m": beak.maximum_frame_error if args.align_beak else None,
                "sock_center_m": objects["sock"].center.tolist() if objects else None,
                "controller": (
                    "Isaac-only standing, object-grasp body/jaw, and walking; scripted navigation and release"
                    if grasp_actor is not None
                    else "Isaac Lab-trained standing, pickup, and walking; scripted head, jaw, and sequence"
                    if native_body_skills
                    else "released or overridden body skills and native walking; scripted head, jaw, and sequence"
                ),
                "policy_provenance": policy_provenance,
                "walking_policy": str(native_path.resolve()),
                "walking_provenance": {
                    "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                    "metadata": policies["native_walking"].get_modelmeta().custom_metadata_map,
                },
                "native_body_skills": native_body_skills,
                "scratch_body_skills": all(
                    item["metadata"].get("initialization")
                    in {"random", "continued_isaac_scratch_standing", "random_geometric_output_bias"}
                    for item in policy_provenance.values()
                ),
                "new_cleanup_policy_trained": False,
                "learned_object_grasp_used": grasp_actor is not None,
                "physics_dt_s": cfg.dt,
                "policy_dt_s": 4 * cfg.dt,
                "robot_upright": bool(root[2] > 0.075 and 1 - 2 * (root[3] ** 2 + root[4] ** 2) > 0.7),
                "final_root_pose": root.tolist() if np.isfinite(root).all() else None,
                "bin_center_m": bin_center.tolist(),
                "bin_inner_size_m": [bin_geometry.width, bin_geometry.depth],
                "bin_geometry": vars(bin_geometry),
                "release_rim_clearance_m": sequence.release_clearances,
                "maximum_robot_bin_penetration_m": maximum_robot_bin_penetration,
                "failure": failure,
                "wall_seconds": time.perf_counter() - wall_start,
                "source_sha256": source_hashes,
                "new_policy_trained": native_body_skills,
                "grasp_attachment": bool(assistance and assistance.events),
                "assisted_grasp_enabled": args.assisted_grasp,
                "assistance": assistance.summary() if assistance is not None else None,
                "cloth_feedback_enabled": args.coupling_relaxation > 0.0,
                "marker_cloth_material": marker_cloth_material,
                "beak_cloth_feedback_enabled": not args.assisted_grasp and args.coupling_relaxation > 0.0,
                "beak_cloth_proxy_mass_scale": 1000.0 if args.assisted_grasp else 1.0,
                "simulated_seconds": count * cfg.dt,
                "initial_marker_position_m": initial_marker,
                "maximum_marker_center_height_m": max_height,
                "verified_lift": bool(ever_lifted),
                "released_over_bin": released,
                "settled_in_bin": bool(sequence.completed["marker"]),
                "cleanup_complete": bool(all(sequence.completed.values())),
                "final_marker_position_m": objects["marker"].center.tolist() if objects else None,
                "parameters": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
            }
            result["cleanup_complete"] = (
                failure is None
                and len(objects) == 2
                and all(result["objects_in_bin"].values())
                and result["robot_upright"]
            )
            (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
            sim.clear_instance()


if __name__ == "__main__":
    main()
