# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nominal effective cable estimates, intended for calibration against measured video."""

import math

from .assets import CABLE_DIAMETER, CABLE_LENGTH

CABLE_BODY_MASS = 0.021
"""Flexible cable mass, including the fixed first segment [kg]."""
CONNECTOR_MASS = 0.004
"""Free USB-C connector mass, attached to the final segment [kg]."""
CABLE_DENSITY = CABLE_BODY_MASS / (math.pi * (CABLE_DIAMETER / 2) ** 2 * CABLE_LENGTH)
"""Effective cylindrical bulk density [kg/m^3]; builder masses are normalized exactly."""
STRETCH_MODULUS = 150e6
"""Effective axial elastic modulus [Pa]."""
BEND_MODULUS = 8e6
"""Effective bending elastic modulus [Pa]."""
TWIST_MODULUS = 2e6
"""Effective torsional shear modulus [Pa]."""
STATIC_FRICTION_TARGET = 0.50
"""Static friction calibration target; not separately supported by installed VBD."""
DYNAMIC_FRICTION = 0.40
"""Active VBD Coulomb friction coefficient, used for both sticking and sliding."""
RESTITUTION_TARGET = 0.05
"""Restitution calibration target; not applied by installed VBD."""
MIN_BEND_RADIUS = 0.015
"""Minimum allowed discrete centerline bend radius [m], enforced as a task limit."""
