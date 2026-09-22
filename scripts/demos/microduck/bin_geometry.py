# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared dimensions and containment checks for cleanup receptacles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BinGeometry:
    """Receptacle dimensions [m], shared by USD authoring and placement checks."""

    shape: str = "box"
    width: float = 0.18
    depth: float = 0.52
    height: float = 0.048
    floor_height: float = 0.003
    bottom_radius: float = 0.082
    top_radius: float = 0.094
    wire_radius: float = 0.00085
    rim_radius: float = 0.0015

    @classmethod
    def load(cls, directory: Path) -> BinGeometry:
        """Read dimensions [m], retaining the legacy tray for older asset directories."""
        path = directory / "bin_geometry.json"
        return cls(**json.loads(path.read_text())["geometry"]) if path.exists() else cls()

    def save(self, directory: Path) -> None:
        """Save the dimensions used to create the physical USD asset."""
        (directory / "bin_geometry.json").write_text(
            json.dumps(
                {
                    "geometry": asdict(self),
                    "lid": False,
                    "collision_model": "individual wire capsules and a solid bottom disk"
                    if self.shape != "box"
                    else "box walls",
                },
                indent=2,
            )
            + "\n"
        )

    def radius_at(self, height: np.ndarray | float) -> np.ndarray:
        """Return the interior envelope radius at height above the ground [m]."""
        fraction = np.clip((np.asarray(height) - self.floor_height) / (self.height - self.floor_height), 0, 1)
        return self.bottom_radius + fraction * (self.top_radius - self.bottom_radius) - self.rim_radius

    def contains(
        self,
        minimum: np.ndarray,
        maximum: np.ndarray,
        center: np.ndarray,
        clearance: float = 0.003,
        points: np.ndarray | None = None,
    ) -> bool:
        """Check an object's footprint against the interior envelope [m].

        Supplied surface points allow accurate tapered-basket checks for deforming cloth.
        Without points, the entire axis-aligned bounding box must fit.
        """
        if self.shape == "box":
            lower = center[:2] - np.array([self.width, self.depth]) / 2 + clearance
            upper = center[:2] + np.array([self.width, self.depth]) / 2 - clearance
            return bool(np.all(minimum[:2] > lower) and np.all(maximum[:2] < upper))
        if points is None:
            points = np.array([[x, y, minimum[2]] for x in (minimum[0], maximum[0]) for y in (minimum[1], maximum[1])])
        local = points - center
        radial = np.linalg.norm(local[:, :2], axis=-1)
        return bool(np.all(radial < self.radius_at(local[:, 2]) - clearance))


BASKET_GEOMETRY = BinGeometry(shape="tapered_wire", width=0.191, depth=0.191, height=0.130, floor_height=0.004)


def cylinder_surface_points(center: np.ndarray, axis: np.ndarray, length: float, radius: float) -> np.ndarray:
    """Return a conservative end-ring hull for a cylinder [m], shape [128, 3].

    The ring polygon circumscribes the circular cross-section, so checking all
    vertices inside a convex receptacle also contains the complete cylinder.
    """
    sides = 64
    axis = axis / np.linalg.norm(axis)
    reference = np.eye(3)[np.argmin(abs(axis))]
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.arange(sides) * 2 * np.pi / sides
    outer_radius = radius / np.cos(np.pi / sides)
    ring = outer_radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    return np.concatenate((center + length / 2 * axis + ring, center - length / 2 * axis + ring))
