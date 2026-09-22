# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download pinned MicroDuck assets and convert the released policies' MJCFs to layered USD."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from behaviors import BEHAVIORS

ROBOT_REVISION = "cb70b792312d559a4da09064d92009079671815f"
POLICY_REVISION = "1b56c396825c052a4e26e95cf2b8d8298af9e9b4"
BAM_REVISION = "62bd8ce12154340be97e06f7f41a0ca8f116d967"


def download(url: str, destination: Path) -> None:
    """Download a public file to the requested local path."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        contents = response.read()
    destination.write_bytes(contents)


def repair_material_bindings(asset: Path) -> None:
    """Move invalid prototype physics bindings to the robot root.

    The importer references geometry from /Instances, where its absolute physics
    material target lies outside the reference scope. All original geoms share
    this material, so binding it on the robot root preserves its intended use.
    """
    from pxr import Sdf, Usd, UsdShade

    instances = Sdf.Layer.FindOrOpen(str(asset.parent / "payloads/instances.usda"))
    paths = []
    instances.Traverse(Sdf.Path.absoluteRootPath, lambda path: paths.append(path))
    for path in paths:
        spec = instances.GetObjectAtPath(path)
        if isinstance(spec, Sdf.RelationshipSpec) and spec.name == "material:binding:physics":
            spec.owner.RemoveProperty(spec)
    instances.Save()
    if asset.stem == "robot_groundcontact_rollers":
        from isaaclab_newton.sim.schemas import NewtonMeshCollisionCfg

        from isaaclab.sim.schemas import apply_mesh_collision_properties

        geometry = Usd.Stage.Open(instances.identifier)
        for prim in geometry.Traverse():
            if prim.GetTypeName() == "Mesh" and prim.GetName() == "tire":
                # A 64-vertex hull makes the small tires polygonal and inhibits rolling.
                apply_mesh_collision_properties(
                    str(prim.GetPath()), [NewtonMeshCollisionCfg(max_hull_vertices=-1)], stage=geometry
                )
        geometry.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(asset))
    root = stage.GetDefaultPrim()
    root.GetVariantSet("Physics").SetVariantSelection("mujoco")
    material = UsdShade.Material.Get(stage, root.GetPath().AppendPath("Physics/PhysicsMaterial"))
    if material:
        UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()


def main() -> None:
    """Fetch assets, run the maintained importer, and record source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    args = parser.parse_args()
    destination = args.asset_dir.resolve()
    source = destination / "source"
    robot_dir = source / "robot"
    if not (robot_dir / "robot_walk.xml").is_file():
        url = f"https://github.com/pollen-robotics/microduck_rl/archive/{ROBOT_REVISION}.tar.gz"
        print(f"Downloading MicroDuck at {ROBOT_REVISION}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            archive_bytes = response.read()
        prefix = f"microduck_rl-{ROBOT_REVISION}/src/mjlab_microduck/robot/microduck/"
        # Copy only ordinary files under the robot directory; never extract links.
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.startswith(prefix):
                    relative = Path(member.name[len(prefix) :])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"Unsafe archive path: {member.name}")
                    output = robot_dir / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(archive.extractfile(member).read())
    for filename in ("manifest.json", *(name + ".onnx" for name in BEHAVIORS)):
        download(
            f"https://huggingface.co/pollen-robotics/microduck-policies/resolve/{POLICY_REVISION}/{filename}",
            destination / filename,
        )
    download(
        f"https://raw.githubusercontent.com/Rhoban/bam/{BAM_REVISION}/bam/params/xl330/m6.json",
        destination / "bam_m6.json",
    )
    download(
        f"https://raw.githubusercontent.com/pollen-robotics/microduck_rl/{ROBOT_REVISION}/LICENSE",
        destination / "MICRODUCK_LICENSE",
    )
    converter = Path(__file__).resolve().parents[2] / "tools/convert_mjcf.py"
    for robot in sorted({profile.robot for profile in BEHAVIORS.values()}):
        asset = destination / robot / f"{robot}.usda"
        if not asset.is_file():
            subprocess.run(
                [sys.executable, str(converter), str(robot_dir / f"{robot}.xml"), str(destination / f"{robot}.usd")],
                check=True,
            )
        repair_material_bindings(asset)
        print(f"Ready: {asset}", flush=True)
    provenance = {
        "robot_repository": "https://github.com/pollen-robotics/microduck_rl",
        "robot_revision": ROBOT_REVISION,
        "policy_repository": "https://huggingface.co/pollen-robotics/microduck-policies",
        "policy_revision": POLICY_REVISION,
        "bam_repository": "https://github.com/Rhoban/bam",
        "bam_revision": BAM_REVISION,
        "model_license": "Creative Commons BY-SA-NC (see upstream README)",
        "policy_license": "Apache-2.0",
    }
    (destination / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
