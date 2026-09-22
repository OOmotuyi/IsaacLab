# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build Endeavor, record the existing MicroDuck walking policy, and render with RTX.

Physics is recorded first, so offline rendering cannot change policy timing.
The environment USD contains no robot; the recorded USD references existing assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from endeavor_scene import PHOTO_URL, REFERENCE_URL, build_endeavor


def parse_args() -> argparse.Namespace:
    """Parse scene-building, policy-recording, and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "record", "render", "render_scene"))
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/endeavor"))
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    parser.add_argument("--policy", type=Path, help="Existing 61-input, 14-output alpha_walking ONNX checkpoint.")
    parser.add_argument("--duration", type=float, default=8.0, help="Rollout duration [s].")
    parser.add_argument("--velocity", type=float, nargs=3, default=(0.3, 0, 0), help="Twist [m/s, m/s, rad/s].")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--samples", type=int, default=64, help="Path-tracing samples per pixel.")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0 or not all(math.isfinite(v) for v in args.velocity):
        parser.error("Duration must be positive, and duration and velocity must be finite.")
    if min(args.width, args.height, args.fps, args.samples) < 1:
        parser.error("Resolution, FPS and samples must be positive.")
    if args.width % 2 or args.height % 2:
        parser.error("Video width and height must be even.")
    args.output_dir = args.output_dir.resolve()
    args.asset_dir = args.asset_dir.resolve()
    return args


def record_walk(args: argparse.Namespace) -> None:
    """Run the unchanged walking contract in Newton and save measured link poses."""
    import torch
    from behaviors import BEHAVIORS, policy_command
    from controller import DEFAULT_POSE, JOINT_NAMES, MicroDuckController
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from play import load_policies

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.app import launch_simulation
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, SimulationContext, UsdFileCfg

    environment_path = args.output_dir / "endeavor.usda"
    animation_path = args.output_dir / "walking.usda"
    if animation_path.exists():
        raise FileExistsError(animation_path)
    robot_path = args.asset_dir / "robot_walk/robot_walk.usda"
    policy_path = (args.policy or args.asset_dir / "alpha_walking.onnx").resolve()
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (robot_path, policy_path)}
    params = json.loads((args.asset_dir / "bam_m6.json").read_text())
    session = load_policies({"alpha_walking": policy_path})["alpha_walking"]
    cfg = SimulationCfg(
        dt=0.005,
        device="cuda:0",
        render_interval=4,
        use_newton_actuators=False,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(integrator="implicitfast", njmax=600, nconmax=128, cone="pyramidal"),
            use_cuda_graph=True,
        ),
    )
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene_cfg = UsdFileCfg(usd_path=str(environment_path))
        scene_cfg.func("/World/Endeavor", scene_cfg)
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=UsdFileCfg(usd_path=str(robot_path), variants={"Physics": "mujoco"}),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(JOINT_NAMES, DEFAULT_POSE))),
                actuators={
                    "servos": ImplicitActuatorCfg(
                        joint_names_expr=list(JOINT_NAMES),
                        stiffness=0.0,
                        damping=0.0,
                        armature=params["armature"],
                        friction=params["friction_base"],
                        joint_effort_limit=7.4 * params["kt"] / params["R"],
                    )
                },
            )
        )
        sim.reset()
        controller = MicroDuckController(robot, params)
        robot.write_root_pose_to_sim_index(root_pose=torch.tensor([[-1.0, 0.0, 0.12, 0, 0, 0, 1]], device=cfg.device))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=cfg.device))
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_pos.torch))
        robot.reset()
        controller.reset()
        sim.forward()
        robot.update(cfg.dt)

        animation = Usd.Stage.CreateNew(str(animation_path))
        animation.SetDefaultPrim(UsdGeom.Xform.Define(animation, "/World").GetPrim())
        UsdGeom.SetStageUpAxis(animation, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(animation, 1.0)
        animation.SetTimeCodesPerSecond(50.0)
        animation.SetFramesPerSecond(50.0)
        animation.SetStartTimeCode(0)
        animation.DefinePrim("/World/Endeavor", "Xform").GetReferences().AddReference("./endeavor.usda")
        render_robot = animation.DefinePrim("/World/Robot", "Xform")
        render_robot.GetReferences().AddReference(os.path.relpath(robot_path, args.output_dir))
        render_robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
        body_paths = {}
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath("/World/Robot")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.GetName() in body_paths:
                    raise ValueError(f"Ambiguous body name: {prim.GetName()}")
                body_paths[prim.GetName()] = str(prim.GetPath())
        body_ops = []
        for name in robot.body_names:
            prim = animation.OverridePrim(body_paths[name])
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "recorded")
            orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "recorded")
            xform.SetResetXformStack(True)
            body_ops.append((translate, orient))
        camera = UsdGeom.Camera.Define(animation, "/World/Camera")
        camera.CreateFocalLengthAttr(20.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr((0.02, 300))
        camera_transform = camera.AddTransformOp()
        records = []
        minimum_height = float("inf")
        completed = False
        steps = max(4, round(args.duration / cfg.dt) // 4 * 4)
        print(f"Recording {steps * cfg.dt:.2f} s of alpha_walking in Endeavor", flush=True)
        try:
            with torch.inference_mode():
                for step in range(steps):
                    if step % 4 == 0:
                        _, command = policy_command("alpha_walking", step * cfg.dt, tuple(args.velocity))
                        controller.command[0] = torch.tensor(command, device=cfg.device)
                        observation = controller.observations()
                        if not torch.isfinite(observation).all():
                            raise RuntimeError("Non-finite policy observation.")
                        action = session.run(None, {session.get_inputs()[0].name: observation.cpu().numpy()})[0]
                        controller.set_action(
                            torch.as_tensor(action, device=cfg.device), BEHAVIORS["alpha_walking"].action_scale
                        )
                    controller.apply()
                    robot.write_data_to_sim()
                    sim.step(render=False)
                    robot.update(cfg.dt)
                    if (step + 1) % 4:
                        continue
                    root = robot.data.root_link_pose_w.torch[0].cpu().tolist()
                    gravity = robot.data.projected_gravity_b.torch[0].cpu().tolist()
                    minimum_height = min(minimum_height, root[2])
                    if root[2] < 0.055 or gravity[2] > -0.3:
                        raise RuntimeError(f"Robot fell at {(step + 1) * cfg.dt:.2f} s.")
                    if not (-12 < root[0] < 12 and -12 < root[1] < 8):
                        raise RuntimeError("Robot left the validated flat walking area.")
                    frame = len(records)
                    for (translate, orient), pose in zip(body_ops, robot.data.body_link_pose_w.torch[0].cpu().tolist()):
                        x, y, z, qx, qy, qz, qw = pose
                        translate.Set(Gf.Vec3d(x, y, z), frame)
                        orient.Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)), frame)
                    x, y = root[:2]
                    camera_transform.Set(
                        Gf.Matrix4d()
                        .SetLookAt(Gf.Vec3d(x + 0.6, y - 1.05, 0.40), Gf.Vec3d(x, y + 0.03, 0.29), Gf.Vec3d(0, 0, 1))
                        .GetInverse(),
                        frame,
                    )
                    records.append({"time_s": (step + 1) * cfg.dt, "root_pose": root, "projected_gravity": gravity})
            completed = True
        finally:
            animation.SetEndTimeCode(max(0, len(records) - 1))
            animation.GetRootLayer().Save()
            metric = {
                "completed": completed,
                "policy": str(policy_path),
                "robot": str(robot_path),
                "environment": str(environment_path),
                "command": args.velocity,
                "physics_dt_s": cfg.dt,
                "policy_hz": 50,
                "samples": len(records),
                "duration_s": len(records) / 50,
                "resets": 0,
                "minimum_root_height_m": minimum_height,
                "start_root_position_m": records[0]["root_pose"][:3] if records else None,
                "end_root_position_m": records[-1]["root_pose"][:3] if records else None,
                "input_sha256": hashes,
                "inputs_unchanged": all(
                    hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items()
                ),
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(metric, indent=2) + "\n")
            (args.output_dir / "trajectory.json").write_text(json.dumps(records) + "\n")
            print(json.dumps(metric, indent=2), flush=True)
            sim.stop()
            sim.clear_instance()


def render(args: argparse.Namespace, environment_only: bool = False) -> None:
    """Path-trace the scene or measured policy motion without rerunning physics."""
    os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
    import ovrtx
    import ovstage
    from PIL import Image

    from pxr import Usd

    input_path = args.output_dir / ("endeavor.usda" if environment_only else "walking.usda")
    source = Usd.Stage.Open(str(input_path))
    if not environment_only and not json.loads((args.output_dir / "metrics.json").read_text())["completed"]:
        raise RuntimeError("The policy recording did not complete successfully.")
    camera_path = "/Endeavor/ReferenceCamera" if environment_only else "/World/Camera"
    usd = (
        source.ExportToString()
        + f"""
def Scope "Render" {{
    def RenderProduct "Camera" {{
        int2 resolution = ({args.width}, {args.height})
        rel camera = <{camera_path}>
        rel orderedVars = [</Render/LdrColor>]
        token omni:rtx:rendermode = "PathTracing"
        uint omni:rtx:pt:samplesPerPixel = {args.samples}
        bool omni:rtx:pt:denoising:enabled = true
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
    }}
    def RenderVar "LdrColor" {{
        string sourceName = "LdrColor"
    }}
}}
"""
    )
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            log_level="warn",
            log_file_path=str(args.output_dir / "render.log"),
            suppress_deprecation_warnings=True,
            texture_streaming_mode="synchronous",
        )
    )
    stage = ovstage.Stage("microduck.endeavor")
    renderer.attach_ovstage(stage)
    writer = None
    try:
        ovstage.population.open_usd_from_string(stage, usd, ordinal=1, time_code=0.0)
        stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        for _ in range(4):
            _render_pixels(renderer, 1, 1 / args.fps)
        if environment_only:
            image = Image.fromarray(_render_pixels(renderer, 1, 1 / args.fps)).convert("RGB")
            image.save(args.output_dir / "endeavor.png")
            image.save(args.output_dir / "endeavor.jpg", quality=92)
            print(f"Saved {args.output_dir / 'endeavor.png'}", flush=True)
            return
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        frames = math.ceil((source.GetEndTimeCode() + 1) / source.GetTimeCodesPerSecond() * args.fps)
        writer = FFMPEG_VideoWriter(
            str(args.output_dir / "microduck_endeavor.mp4"),
            (args.width, args.height),
            fps=args.fps,
            codec="libx264",
            preset="medium",
            ffmpeg_params=["-crf", "18"],
        )
        for frame in range(frames):
            ordinal = frame + 2
            ovstage.population.update_from_usd_time_async(stage, ordinal=ordinal, time_code=frame / args.fps).wait()
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            pixels = _render_pixels(renderer, ordinal, 1 / args.fps)[..., :3]
            writer.write_frame(np.ascontiguousarray(pixels))
            if frame == min(frames - 1, round(args.fps * 2)):
                still = Image.fromarray(pixels)
                still.save(args.output_dir / "microduck_endeavor.png")
                still.save(args.output_dir / "microduck_endeavor.jpg", quality=92)
            if frame % args.fps == 0:
                print(f"Rendered {frame + 1}/{frames} frames", flush=True)
        print(f"Saved {args.output_dir / 'microduck_endeavor.mp4'}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


def _render_pixels(renderer, ordinal: int, dt: float) -> np.ndarray:
    """Copy an RTX frame to CPU memory before releasing renderer-owned buffers."""
    import ovrtx

    outputs = renderer.step(render_products={"/Render/Camera"}, delta_time=dt, ordinal=ordinal)
    frame = outputs["/Render/Camera"].frames[0]
    mapped = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
    view = np.from_dlpack(mapped)
    pixels = view.copy()
    del view
    mapped.unmap()
    del mapped, frame, outputs
    return pixels


def main() -> None:
    """Run the selected phase without modifying the existing robot or policy."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "build":
        build_endeavor(args.output_dir / "endeavor.usda")
        (args.output_dir / "references.json").write_text(
            json.dumps(
                {
                    "page": REFERENCE_URL,
                    "photograph": PHOTO_URL,
                    "photographer": "Connie Zhou",
                    "method": "Manual USD reconstruction from published photographs; no generated imagery.",
                    "dimensions": "Estimated, not surveyed.",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved {args.output_dir / 'endeavor.usda'}", flush=True)
    elif args.mode == "record":
        record_walk(args)
    else:
        render(args, environment_only=args.mode == "render_scene")


if __name__ == "__main__":
    main()
