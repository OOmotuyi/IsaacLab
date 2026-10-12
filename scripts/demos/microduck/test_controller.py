# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-contract tests for the MicroDuck playback controller."""

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from behaviors import BEHAVIORS, policy_command

SPEC = importlib.util.spec_from_file_location("microduck_controller", Path(__file__).with_name("controller.py"))
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def test_motor_saturation_and_back_emf():
    """Supply voltage bounds drive torque; motion produces electrical damping."""
    params = {"kt": 0.4, "R": 2.0}
    error = torch.tensor([0.0, 100.0, -100.0, 0.0, 0.1], dtype=torch.float64)
    velocity = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0], dtype=torch.float64)
    actual = controller.motor_torque(error, velocity, torch.tensor(7.0), params)
    # Stall torque = kt*V/R; back-EMF = -kt**2*speed/R.
    small_signal_gain = 0.4 * 7.0 / 2.0 * 200.0 * 4096.0 / (2.0 * math.pi * 256.0 * 885.0)
    expected = torch.tensor([0.0, 1.4, -1.4, -0.16, 0.1 * small_signal_gain], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("velocity,expected", [(0.0, 0.03), (100.0, 0.01)])
def test_friction_without_load(velocity, expected):
    """At rest static friction includes Stribeck; at speed only Coulomb remains."""
    params = {
        "friction_base": 0.01,
        "friction_stribeck": 0.02,
        "dtheta_stribeck": 1.0,
        "alpha": 2.0,
        "load_friction_external": 0.1,
        "load_friction_motor": 0.2,
        "load_friction_external_stribeck": 0.3,
        "load_friction_motor_stribeck": 0.4,
        "load_friction_external_quad": 0.5,
        "load_friction_motor_quad": 0.6,
    }
    actual = controller.friction_budget(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(velocity), params)
    assert actual.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "training_model,expected",
    [(False, [0.5, 0.25, 0.0, 0.0, 0.5]), (True, [0.5, 0.25, 0.5, 0.25, 0.5])],
)
def test_load_friction_direction_and_quadratic_boundary(training_model, expected):
    """Opposed loads enable the quadratic term; equal magnitudes disable it."""
    params = {
        "friction_base": 0.0,
        "friction_stribeck": 0.0,
        "dtheta_stribeck": 1.0,
        "alpha": 2.0,
        "load_friction_external": 0.0,
        "load_friction_motor": 0.0,
        "load_friction_external_stribeck": 0.0,
        "load_friction_motor_stribeck": 0.0,
        "load_friction_external_quad": 0.5,
        "load_friction_motor_quad": 0.25,
    }
    motor = torch.tensor([2.0, -1.0, 2.0, 1.0, -2.0])
    external = torch.tensor([-1.0, 2.0, 1.0, -1.0, 1.0])
    actual = controller.friction_budget(motor, external, torch.zeros_like(motor), params, training_model=training_model)
    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize("name,period,end", [("alpha_ground_pick", 4.0, 3.8), ("roller_crouch", 5.0, 4.5)])
def test_phase_command_and_handoff(name, period, end):
    """One-shots receive a unit-circle clock, then return to a zero-command holder."""
    fallback = BEHAVIORS[name].fallback
    assert policy_command(name, 0.0, (0.3, 0.0, 0.0)) == (fallback, (0.0,) * 13)
    active, command = policy_command(name, 1.0 + period / 4, (0.3, 0.0, 0.0))
    assert active == name
    assert command == pytest.approx((0.0, 1.0) + (0.0,) * 11, abs=1e-7)
    assert policy_command(name, end, (0.3, 0.0, 0.0)) == (fallback, (0.0,) * 13)


def test_kicks_roll_and_posture_commands():
    """Maneuvers ignore walking twist, and sitting returns to the same policy."""
    for name in ("ball_kick_left", "ball_kick_right", "roulade"):
        assert policy_command(name, 1.0, (0.3, 0.2, 1.0)) == (name, (0.0,) * 13)
        assert policy_command(name, 1.0 + BEHAVIORS[name].active_time, (0.3, 0.2, 1.0))[0] == "alpha_stand"
    assert policy_command("alpha_sitstand", 2.0, (0.3, 0.0, 0.0)) == ("alpha_sitstand", (1.0,) + (0.0,) * 12)
    assert policy_command("alpha_sitstand", 6.0, (0.3, 0.0, 0.0)) == ("alpha_sitstand", (0.0,) * 13)


def test_scaled_targets_preserve_raw_action_history():
    """Roller target scaling must not scale the previous-action observation."""
    instance = controller.MicroDuckController.__new__(controller.MicroDuckController)
    instance.home = torch.tensor(controller.DEFAULT_POSE).unsqueeze(0)
    instance.target = torch.zeros_like(instance.home)
    instance.last_action = torch.zeros_like(instance.home)
    action = torch.ones_like(instance.home)
    instance.set_action(action, scale=BEHAVIORS["roller"].action_scale)
    torch.testing.assert_close(instance.target - instance.home, torch.full_like(action, 0.8))
    torch.testing.assert_close(instance.last_action, action)


def test_viscous_friction_survives_first_physics_step(tmp_path):
    """Lab's deferred model synchronization must preserve the BAM motor fit."""
    demo = Path(__file__).resolve().parent
    repo = demo.parents[2]
    assets = repo / "logs/microduck/assets"
    if not (assets / "robot_walk/robot_walk.usda").is_file() or not torch.cuda.is_available():
        pytest.skip("Requires prepared MicroDuck assets and CUDA.")
    probe = tmp_path / "probe.py"
    probe.write_text(
        f"""import sys
sys.path.insert(0, {str(demo)!r})
import controller
import play
import torch
import warp as wp
original = controller.MicroDuckController.apply
calls = 0
def checked_apply(self):
    global calls
    if calls:
        actual = wp.to_torch(self.solver.mjw_model.dof_damping)[0, self.dof_ids]
        torch.testing.assert_close(actual, torch.full_like(actual, self.params["friction_viscous"]))
        reference = torch.tensor([-5.0e4, -2.0e2], device=self.robot.device)
        solref = wp.to_torch(self.solver.mjw_model.dof_solref)[0, self.dof_ids]
        torch.testing.assert_close(solref, reference.expand_as(solref))
        friction = wp.to_torch(self.solver.mjw_model.dof_frictionloss)[:, self.dof_ids]
        torch.testing.assert_close(friction, self.robot.data.joint_friction_coeff.torch[:, self.joint_ids])
    calls += 1
    original(self)
controller.MicroDuckController.apply = checked_apply
sys.argv = ["play.py", "--asset_dir", {str(assets)!r}, "--duration", "0.04"]
play.main()
assert calls == 8
"""
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=repo,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "yes"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_batched_training_reset_and_friction(tmp_path):
    """Selected resets preserve other robots and per-world motor friction."""
    repo = Path(__file__).resolve().parents[3]
    assets = repo / "logs/microduck/assets"
    if not (assets / "robot_walk/robot_walk.usda").is_file() or not torch.cuda.is_available():
        pytest.skip("Requires prepared MicroDuck assets and CUDA.")
    probe = tmp_path / "training_probe.py"
    probe.write_text(
        """import argparse
import torch
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.contrib.microduck.microduck_env_cfg import MicroDuckFlatEnvCfg
parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args(["--viz", "none"])
cfg = MicroDuckFlatEnvCfg()
cfg.scene.num_envs = 4
with launch_simulation(cfg.sim, args):
    from isaaclab_tasks.contrib.microduck.microduck_env import MicroDuckEnv
    env = MicroDuckEnv(cfg)
    with torch.inference_mode():
        obs, _ = env.reset()
        assert obs["policy"].shape == (4, 61)
        assert obs["critic"].shape == (4, 76)
        term = env.action_manager.get_term("joint_pos")
        controller = term.controller
        model = controller.solver.mj_model
        # Source MJCF uses mask 1 for feet/floor and mask 2 for legs/battery.
        for a in range(model.ngeom):
            for b in range(a + 1, model.ngeom):
                names = (model.geom(a).name, model.geom(b).name)
                group = [1 if "GroundPlane" in name or "_foot_collision/" in name else 2 for name in names]
                allowed = bool(model.geom_contype[a] & model.geom_conaffinity[b]
                               or model.geom_contype[b] & model.geom_conaffinity[a])
                assert allowed == (group[0] == group[1]), names
        friction = wp.to_torch(controller.solver.mjw_model.dof_frictionloss)
        assert friction.shape[0] == 4 and friction.stride(0) != 0
        actions = torch.zeros((4, 14), device=env.device)
        actions[:, 0] = torch.tensor([0.1, 0.2, -0.1, -0.2], device=env.device)
        for _ in range(8):
            obs, reward, done, timeout, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
        before = env.scene["robot"].data.joint_pos.torch.clone()
        battery = controller.battery.clone()
        other_action = term.raw_actions[[0, 2, 3]].clone()
        env.reset(env_ids=torch.tensor([1], device=env.device))
        torch.testing.assert_close(env.scene["robot"].data.joint_pos.torch[[0, 2, 3]], before[[0, 2, 3]])
        torch.testing.assert_close(term.raw_actions[[0, 2, 3]], other_action)
        torch.testing.assert_close(term.raw_actions[1], torch.zeros(14, device=env.device))
        torch.testing.assert_close(controller.battery, battery)
        torch.testing.assert_close(controller.previous_torque[1], torch.zeros(14, device=env.device))
        obs, reward, *_ = env.step(actions)
        assert torch.isfinite(reward).all()
        # Resuming the curriculum preserves all schedules; scaling changes only smoothing.
        from isaaclab_tasks.contrib.microduck.mdp.curricula import walking_schedule
        env.common_step_counter = 72_000
        mature = walking_schedule(env, None)
        mature_head = env.command_manager.get_term("head_pose").cfg.ranges
        env.common_step_counter = 0
        resumed = walking_schedule(env, None, step_offset=72_000)
        assert resumed == mature
        assert env.command_manager.get_term("head_pose").cfg.ranges == mature_head
        weaker = walking_schedule(env, None, step_offset=72_000, action_rate_scale=0.4)
        assert weaker["action_rate_weight"] == mature["action_rate_weight"] * 0.4
        assert weaker["standing_fraction"] == mature["standing_fraction"]
        assert weaker["head_bias_weight"] == mature["head_bias_weight"]
        assert env.command_manager.get_term("head_pose").cfg.ranges == mature_head
        # Independent MuJoCo calculation validates COM-frame and inertia-frame handling.
        import mujoco
        import numpy as np
        from isaaclab_tasks.contrib.microduck.mdp.rewards import angular_momentum
        solver = controller.solver
        reference = mujoco.MjData(model)
        for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
            getattr(model, name)[:] = wp.to_torch(getattr(solver.mjw_model, name))[0].cpu().numpy()
        mujoco.mj_setConst(model, reference)
        reference.qpos[:] = wp.to_torch(solver.mjw_data.qpos)[0].cpu().numpy()
        reference.qvel[:] = wp.to_torch(solver.mjw_data.qvel)[0].cpu().numpy()
        mujoco.mj_forward(model, reference)
        mujoco.mj_subtreeVel(model, reference)
        expected = np.square(reference.subtree_angmom[1]).sum()
        np.testing.assert_allclose(angular_momentum(env)[0].item(), expected, rtol=1e-4, atol=1e-8)
    env.close()
"""
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=repo,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "yes"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_marker_grasp_accepts_lower_lip_contact():
    """A marker touching the lower lip and upper pad is gripped; one-sided contact is not."""
    from marker_pick import has_opposed_mouth_contacts
    from room_cleanup import has_opposed_mouth_contacts as room_contacts

    upper = ["/World/Marker/Collision_93", "/World/Robot/jaw_soft/upper_mouth_pad_91"]
    lower = ["/World/Marker/Collision_93", "/World/Robot/jaw_soft/lower_beak/lower_mouth_lip_90"]
    assert has_opposed_mouth_contacts([upper, lower])
    assert not has_opposed_mouth_contacts([upper])
    assert not has_opposed_mouth_contacts([lower])
    # Cloth may touch the front of the upper shell instead of the added pad.
    assert room_contacts([lower], upper_shell_contact=True)
    assert not room_contacts([], upper_shell_contact=True)
    assert not room_contacts([lower])


@pytest.mark.parametrize("target", ["marker", "sock"])
def test_cleanup_transport_accepts_world_positions(target):
    """Transport accepts 3D world positions and emits a planar policy command."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(target=target, phase="carry")
    sequence.target = target
    position = np.array([0.10, 0.20, 0.15])
    obj = PickupState(position, position - 0.01, position + 0.01, 0.0, True, 0.02)
    policy, command, jaw = sequence.step(6.0, np.array([0.04, 0.20, 0.12]), 0.0, {target: obj})
    assert policy in {"native_walking", "alpha_walking"}
    assert np.shape(command) == (13,)
    assert np.isfinite(command).all()
    assert command[0] > 0
    assert command[1] < 0
    assert jaw < 0


def test_cleanup_release_retracts_head():
    """Settlement cannot interrupt head retraction before recovery begins."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState
    from marker_assets import MOUTH_OPEN

    sequence = CleanupSequence(phase="drop", phase_start=5.0)
    position = np.array([0.38, 0.13, 0.015])
    obj = PickupState(position, position - 0.01, position + 0.01, 0.0, False, 0.1)
    sequence.lifted["marker"] = True
    sequence.settled_time = 0.5
    _, release_command, jaw = sequence.step(5.4, np.array([0.24, 0.0, 0.12]), 0.0, {"marker": obj})
    assert release_command[4] > 0
    assert jaw == MOUTH_OPEN
    assert sequence.phase == "drop"
    _, settled_command, jaw = sequence.step(8.0, np.array([0.24, 0.0, 0.12]), 0.0, {"marker": obj})
    assert settled_command[4] == 0
    assert jaw == MOUTH_OPEN
    assert sequence.phase == "recover"


@pytest.mark.parametrize(
    ("offset", "speed", "expected"),
    [
        ((0.0, 0.0, 0.0), 0.0, True),
        ((0.0, 0.0, 0.10), 0.0, False),
        ((0.0, 0.0, 0.0), 0.10, False),
        ((0.085, 0.0, 0.0), 0.0, False),
        ((0.082, 0.0, 0.0), 0.0, True),
    ],
)
def test_cleanup_settlement_requires_containment_and_rest(offset, speed, expected):
    """An object above or straddling the bin, or still moving, is not placed."""
    import json

    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState
    from marker_assets import BIN_FLOOR

    sequence = CleanupSequence()
    center = sequence.bin_center + np.array([0.0, 0.0, BIN_FLOOR + 0.01]) + offset
    obj = PickupState(center, center - 0.008, center + 0.008, speed, False, 0.2)
    result = sequence.settled_in_bin(obj)
    assert result is expected
    assert json.loads(json.dumps({"settled": result}))["settled"] is expected


def test_cleanup_waits_for_pickup_heading():
    """The second pickup does not begin while the robot faces away from the sock."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(mode="sock")
    root = np.array([-0.001, sequence.sock_y - 0.006, 0.12])
    center = np.array([0.068, sequence.sock_y, 0.04])
    obj = PickupState(center, center - 0.01, center + 0.01, 0.0, False, 0.1)
    policy, command, _ = sequence.step(1.0, root, 0.25, {"sock": obj})
    assert sequence.phase == "settle"
    assert policy == "native_walking"
    assert command[2] < 0
    sequence.step(1.02, root, 0.0, {"sock": obj})
    for t in np.arange(1.04, 2.02, 0.02):
        policy, _, _ = sequence.step(float(t), root, 0.08, {"sock": obj})
        assert policy == "alpha_stand"
    sequence.step(2.04, root, 0.08, {"sock": obj})
    assert sequence.phase == "pick"


def test_cleanup_rejects_ungripped_sock():
    """A missing sock grasp times out without beginning transport."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(mode="sock", phase="pick")
    center = np.array([0.068, sequence.sock_y, 0.04])
    obj = PickupState(center, center - 0.01, center + 0.01, 0.0, False, 0.1)
    sequence.step(3.1, np.array([0.0, sequence.sock_y, 0.12]), 0.0, {"sock": obj})
    assert sequence.phase == "grasp_failed"
    assert not sequence.lifted["sock"]


@pytest.mark.parametrize("first", ["marker", "sock"])
def test_cleanup_collects_both_orders(first):
    """Either object can be collected first; the second placement ends cleanup."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(first_object=first, phase="drop")
    center = sequence.bin_center + np.array([0.0, 0.0, 0.015])
    obj = PickupState(center, center - 0.008, center + 0.008, 0.0, False, 0.2)
    objects = {"marker": obj, "sock": obj}
    sequence.lifted[first] = True
    sequence.settled_time = 0.5
    sequence.step(3.0, np.array([0.25, 0.0, 0.12]), 0.0, objects)
    assert sequence.completed[first]
    assert sequence.phase == "recover"

    sequence.phase = "retreat"
    lane_y = 0.12 if first == "sock" else 0.0
    _, retreat_command, _ = sequence.step(4.0, np.array([0.24, lane_y, 0.12]), 0.0, objects)
    assert retreat_command[0] < 0.0
    assert abs(retreat_command[1]) < 1e-12

    second = "sock" if first == "marker" else "marker"
    sequence.phase = "align"
    sequence.arrival_time = 0.2
    next_y = sequence.sock_y - 0.006 if second == "sock" else 0.0
    sequence.step(5.0, np.array([0.0, next_y, 0.12]), 0.0, objects)
    assert sequence.target == second
    assert sequence.phase == "settle"

    sequence.phase = "drop"
    sequence.phase_start = 6.0
    sequence.settled_time = 0.5
    sequence.lifted[second] = True
    sequence.step(9.0, np.array([0.25, 0.0, 0.12]), 0.0, objects)
    assert all(sequence.completed.values())
    assert sequence.phase == "done"


def test_assisted_sock_grasp_gates_tracks_and_releases():
    """Seat nearby cloth in the aperture without a jump or disabling collisions."""
    from types import SimpleNamespace

    import warp as wp
    from assisted_grasp import AssistedSockGrasp

    class Sock:
        def __init__(self):
            nodes = torch.tensor([[-0.0125, i * 0.001, -0.089] for i in range(20)])
            self.data = SimpleNamespace(
                nodal_pos_w=SimpleNamespace(torch=nodes[None].clone()),
                nodal_vel_w=SimpleNamespace(torch=torch.zeros_like(nodes[None])),
                nodal_kinematic_target=SimpleNamespace(torch=torch.cat((nodes, torch.ones(20, 1)), 1)[None]),
            )
            self.max_sim_vertices_per_body = len(nodes)

        def write_nodal_kinematic_target_to_sim_index(self, targets):
            self.targets = targets.clone()

        def write_data_to_sim(self):
            ids = self.targets[0, :, 3] == 0
            self.data.nodal_pos_w.torch[0, ids] = self.targets[0, ids, :3]

        def write_nodal_velocity_to_sim_index(self, velocities):
            self.data.nodal_vel_w.torch.copy_(velocities)

    sock = Sock()
    model = SimpleNamespace(
        particle_count=20,
        particle_inv_mass=wp.ones(20, device="cpu"),
        particle_flags=wp.ones(20, dtype=wp.int32, device="cpu"),
        shape_label=["/upper_mouth_pad", "/Ground", "/lower_mouth_lip", "/top_head_shell"],
        shape_flags=wp.full(4, 7, dtype=wp.int32, device="cpu"),
    )
    view = SimpleNamespace(**vars(model))
    view.shape_flags = wp.clone(model.shape_flags)
    grasp = AssistedSockGrasp(sock, view, model)
    pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    original = sock.data.nodal_pos_w.torch.clone()
    for allow, target, angle, velocity, x in (
        (False, 0.18, 0.18, 0.0, 0.0),
        (True, 0.5, 0.0, 0.0, 0.0),
        (True, 0.18, 0.5, 0.0, 0.0),
        (True, 0.18, 0.18, 0.0, 1.0),
        (True, 0.18, 0.18, -1.0, 0.0),
    ):
        pose[0] = x
        assert not grasp.update(
            pose, allow_attach=allow, jaw_target=target, jaw_angle=angle, jaw_velocity=velocity, time_s=0.0, dt=0.01
        )
        assert not grasp.active
    pose[0] = 0.0
    assert grasp.update(pose, allow_attach=True, jaw_target=0.18, jaw_angle=0.18, jaw_velocity=0.0, time_s=1.0, dt=0.01)
    assert 0 < len(grasp.ids) <= 2 < len(original[0])
    flags = wp.to_torch(view.shape_flags)
    assert bool((flags == 7).all())
    assert bool((wp.to_torch(model.shape_flags) == 7).all())
    torch.testing.assert_close(sock.data.nodal_pos_w.torch, original)
    ids = grasp.ids.clone()
    grasp.update(pose, allow_attach=False, jaw_target=0.18, jaw_angle=0.18, jaw_velocity=0.0, time_s=1.175, dt=0.175)
    halfway = sock.data.nodal_pos_w.torch[0, ids, 2].clone()
    assert bool(((halfway > -0.089) & (halfway < -0.0695)).all())
    grasp.update(pose, allow_attach=False, jaw_target=0.18, jaw_angle=0.18, jaw_velocity=0.0, time_s=1.35, dt=0.175)
    seated = sock.data.nodal_pos_w.torch.clone()
    # The prototype pad aperture at 0.18 rad contains the held patch.
    assert bool(((seated[0, ids, 0] > -0.016) & (seated[0, ids, 0] < -0.008)).all())
    assert bool(((seated[0, ids, 2] > -0.073) & (seated[0, ids, 2] < -0.065)).all())
    pose[0] += 0.01
    grasp.update(pose, allow_attach=False, jaw_target=0.18, jaw_angle=0.18, jaw_velocity=0.0, time_s=1.36, dt=0.01)
    expected = seated.clone()
    expected[0, ids, 0] += 0.01
    torch.testing.assert_close(sock.data.nodal_pos_w.torch, expected)
    grasp.validate()
    assert grasp.update(pose, allow_attach=False, jaw_target=0.5, jaw_angle=0.0, jaw_velocity=0.0, time_s=1.37, dt=0.01)
    assert not grasp.active
    assert not grasp.mouth_contacts_suppressed
    assert bool((flags == 7).all())
    grasp.update(pose, allow_attach=False, jaw_target=0.5, jaw_angle=0.5, jaw_velocity=0.0, time_s=1.68, dt=0.01)
    assert not grasp.mouth_contacts_suppressed
    assert bool((flags == 7).all())
    assert bool((sock.targets[0, :, 3] == 1).all())
    torch.testing.assert_close(sock.data.nodal_vel_w.torch[0, ids, 0], torch.ones(len(ids)))
    assert [e["event"] for e in grasp.events] == ["attach", "release"]


def test_cleanup_assisted_grasp_does_not_claim_physical_contacts():
    """An explicitly attached sock can lift without setting the physical contact flag."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(mode="sock", phase="pick")
    center = np.array([0.068, 0.26, 0.14])
    obj = PickupState(center, center - 0.01, center + 0.01, 0.0, False, 0.01, assisted_grasp=True)
    for t in np.arange(1.5, 1.9, 0.02):
        sequence.step(float(t), np.array([0.0, 0.254, 0.12]), 0.0, {"sock": obj})
    assert sequence.lift_start is not None
    assert sequence.lifted["sock"]
    assert not obj.opposed_contacts


@pytest.mark.parametrize("name", ["alpha_stand", "alpha_ground_pick"])
def test_source_actor_fold_preserves_released_outputs(name):
    """PPO initialization must preserve the released normalized ONNX action contract."""
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    from isaaclab_tasks.contrib.microduck.policy import MicroDuckSourceActor

    path = Path(__file__).resolve().parents[3] / "logs/microduck/assets" / f"{name}.onnx"
    if not path.is_file():
        pytest.skip("Requires the prepared released MicroDuck policy assets.")
    source = MicroDuckSourceActor(path)
    generator = torch.Generator().manual_seed(42)
    observations = source.mean + source.scale * torch.randn(16, 61, generator=generator)
    folded = torch.nn.Sequential(
        torch.nn.Linear(61, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 14),
    )
    folded.load_state_dict({name.removeprefix("mlp."): value for name, value in source.folded_state_dict().items()})
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = np.concatenate(
        [session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in observations.numpy()]
    )
    with torch.no_grad():
        np.testing.assert_allclose(folded(observations).numpy(), expected, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(source(observations).numpy(), expected, atol=2e-6, rtol=2e-5)


def test_pickup_phase_starts_immediately_and_resets_after_inference(tmp_path, monkeypatch):
    """A new pickup starts at phase zero without a stale command or a one-second hold."""
    import json
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.microduck.mdp import skills
    from isaaclab_tasks.contrib.microduck.skill_env_cfg import MicroDuckSkillCommandCfg

    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"phase_s": [0.0, 3.0], "reference": [[0, 0, 0.2, 0, 0, -1, 0.12]] * 2}))
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02, episode_length_buf=torch.zeros(2))
    cfg = MicroDuckSkillCommandCfg(skill="pickup", reference_path=str(reference))
    command = skills.MicroDuckSkillCommand(cfg, env)
    monkeypatch.setattr(skills, "skill_head_position", lambda _: torch.zeros(2, 3))
    command.reset(torch.arange(2))
    command.style[:] = torch.tensor([False, True])
    env.episode_length_buf[:] = 10
    with torch.inference_mode():
        command.compute(0.02)
    torch.testing.assert_close(command.phase, torch.tensor([0.2, 0.2]))
    env.episode_length_buf[:] = 50
    with torch.inference_mode():
        command.compute(0.02)
    torch.testing.assert_close(command.phase, torch.tensor([1.0, 0.6]))
    other = command.command[1].clone()
    command.reset(torch.tensor([0]))
    torch.testing.assert_close(command.command[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(command.command[1], other)
    assert command.phase[0] == 0


@pytest.mark.parametrize("close_time", [0.55, 0.7])
def test_cleanup_sock_closing_time_is_configurable(close_time):
    """Jaw timing can adapt to a body actor without claiming a successful grasp."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState
    from marker_assets import MOUTH_CLOSED, MOUTH_OPEN

    sequence = CleanupSequence(mode="sock", phase="pick", sock_close_time_s=close_time)
    position = np.array([0.068, 0.26, 0.04])
    obj = PickupState(position, position - 0.01, position + 0.01, 0.0, False, 0.1)
    _, _, before = sequence.step(close_time - 0.01, np.array([0.0, 0.26, 0.12]), 0.0, {"sock": obj})
    _, _, after = sequence.step(close_time + 0.01, np.array([0.0, 0.26, 0.12]), 0.0, {"sock": obj})
    assert before == MOUTH_OPEN
    assert after == MOUTH_CLOSED
    assert sequence.lift_start is None
    assert not sequence.lifted["sock"]


@pytest.mark.parametrize(("bottom", "releases"), [(0.129, False), (0.137, False), (0.15, True)])
def test_cleanup_basket_release_requires_rim_clearance(bottom, releases):
    """The entire object must clear the physical basket rim before the jaw opens."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY
    from cleanup_sequence import CleanupSequence, PickupState
    from marker_assets import MOUTH_OPEN

    sequence = CleanupSequence(phase="carry", bin_geometry=BASKET_GEOMETRY)
    center = sequence.bin_center + np.array([0.0, 0.0, bottom + 0.01])
    obj = PickupState(center, center - 0.01, center + 0.01, 0.0, True, 0.02)
    _, _, jaw = sequence.step(5.0, np.array([0.24, 0.13, 0.12]), 0.0, {"marker": obj})
    assert sequence.released["marker"] is releases
    assert (jaw == MOUTH_OPEN) is releases
    if releases:
        assert sequence.release_clearances["marker"] == pytest.approx(bottom - BASKET_GEOMETRY.height)


def test_cleanup_basket_rejects_rectangle_corner_and_accepts_cloth_surface():
    """Round containment uses all cloth vertices rather than its centroid or bounding rectangle."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY

    center = np.zeros(3)
    minimum = np.array([0.045, 0.045, 0.015])
    maximum = np.array([0.065, 0.065, 0.025])
    assert not BASKET_GEOMETRY.contains(minimum, maximum, center)
    points = np.array([[0.065, 0.045, 0.020], [0.045, 0.065, 0.020]])
    assert BASKET_GEOMETRY.contains(points.min(0), points.max(0), center, clearance=0, points=points)


def test_cleanup_basket_carry_targets_bin_center():
    """Both objects align with the narrower basket instead of using separate tray lanes."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(phase="carry", bin_geometry=BASKET_GEOMETRY)
    center = np.array([0.18, 0.0, 0.21])
    obj = PickupState(center, center - 0.01, center + 0.01, 0.0, True, 0.02)
    _, command, _ = sequence.step(5.0, np.array([0.10, 0.0, 0.12]), 0.0, {"marker": obj})
    assert command[1] > 0


def test_cleanup_marker_aligns_laterally_before_forward_transport():
    """The basket approach holds the starting x until the held object is in its lane."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(phase="lift_check", bin_geometry=BASKET_GEOMETRY)
    sequence.lifted["marker"] = True
    root = np.array([0.0, 0.0, 0.12])
    center = np.array([0.07, 0.0, 0.21])
    obj = PickupState(center, center - 0.01, center + 0.01, 0.0, True, 0.02)
    sequence.step(1.0, root, 0.0, {"marker": obj})
    assert sequence.phase == "align_carry"
    _, command, _ = sequence.step(1.02, root, 0.0, {"marker": obj})
    assert command[0] == 0
    assert 0 < command[1] <= 0.10
    assert sequence.phase == "align_carry"
    root[1] = sequence.bin_center[1]
    obj.center[1] = sequence.bin_center[1]
    for i in range(10):
        sequence.step(1.04 + i * 0.02, root, 0.0, {"marker": obj})
    assert sequence.phase == "carry"


def test_cleanup_basket_reach_is_gradual_and_keeps_legacy_heads():
    """A small opposed neck/head motion extends the beak without changing leg actions."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY
    from cleanup_sequence import BasketReach, CleanupSequence

    reach = BasketReach()
    sequence = CleanupSequence(phase="carry", bin_geometry=BASKET_GEOMETRY)
    root = np.array([0.25, 0.13, 0.12])
    action = np.zeros((1, 14))
    reach.apply(action, None, root, [0.0] * 13, sequence)
    assert 0 < reach.strength < 1
    assert action[0, 5] < 0 < action[0, 6]
    assert action[0, 5] + action[0, 6] == pytest.approx(0)
    for _ in range(25):
        action = np.zeros((1, 14))
        reach.apply(action, None, root, [0.0] * 13, sequence)
    assert reach.strength == 1
    assert action[0, 5:7] == pytest.approx([-0.18, 0.18])
    assert np.count_nonzero(action) == 2
    original = np.ones((1, 14))
    legacy = original.copy()
    BasketReach().apply(legacy, None, root, [0.0] * 13, CleanupSequence(phase="carry"))
    np.testing.assert_array_equal(legacy, original)


def test_cleanup_sock_does_not_release_at_front_edge():
    """A footprint that fits the opening is insufficient when its center hugs the front wall."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(mode="sock", phase="carry", bin_geometry=BASKET_GEOMETRY)
    center = sequence.bin_center + np.array([-0.06, 0.0, 0.20])
    obj = PickupState(center, center - 0.005, center + 0.005, 0.0, True, 0.02)
    assert sequence.inside_bin(obj)
    sequence.step(5.0, np.array([0.23, 0.13, 0.12]), 0.0, {"sock": obj})
    assert not sequence.released["sock"]


def test_cleanup_tilted_marker_release_uses_cylinder_hull():
    """A measured tilted cylinder fits the rim even when its empty box corners do not."""
    import numpy as np
    from bin_geometry import BASKET_GEOMETRY, cylinder_surface_points
    from cleanup_sequence import CleanupSequence, PickupState
    from marker_assets import MARKER_LENGTH, MARKER_RADIUS

    # Measured at the first safe release in the wire-basket rollout.
    center = np.array([0.33722275495529175, 0.14934605360031128, 0.19363471865653992])
    x, y, z, w = np.array([-0.17929135262966156, 0.000952998932916671, -0.37678200006484985, 0.9087845087051392])
    axis = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    axis /= np.linalg.norm(axis)
    half = abs(axis) * MARKER_LENGTH / 2 + MARKER_RADIUS * np.sqrt(np.maximum(0, 1 - axis**2))
    sequence = CleanupSequence(phase="carry", bin_geometry=BASKET_GEOMETRY)
    state = PickupState(center, center - half, center + half, 0.05, True, 0.04)
    assert not sequence.inside_bin(state, clearance=0.006)
    state.surface_points = cylinder_surface_points(center, axis, MARKER_LENGTH, MARKER_RADIUS)
    assert sequence.inside_bin(state, clearance=0.006)
    policy, _, _ = sequence.step(40.66, np.array([0.266485, 0.191388, 0.120577]), -0.16, {"marker": state})
    assert sequence.released["marker"]
    assert policy == "native_walking"


def test_cleanup_bin_horizontal_wire_gaps_are_smaller_than_marker(tmp_path):
    """The visible collision wires retain a marker whose diameter exceeds their vertical gaps."""
    import numpy as np
    from marker_assets import MARKER_RADIUS
    from room_assets import build_bin

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    build_bin(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path / "cleanup_bin.usda"))
    cache = UsdGeom.XformCache(0)
    levels = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Capsule) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        wire = UsdGeom.Capsule(prim)
        half = wire.GetHeightAttr().Get() / 2
        transform = cache.GetLocalToWorldTransform(prim)
        endpoints = np.array([transform.Transform(Gf.Vec3d(0, 0, z)) for z in (-half, half)])
        if abs(endpoints[1, 2] - endpoints[0, 2]) < 1e-8:
            levels[round(float(endpoints[:, 2].mean()), 8)] = wire.GetRadiusAttr().Get()
    rings = sorted(levels.items())
    gaps = [upper - lower - r_lower - r_upper for (lower, r_lower), (upper, r_upper) in zip(rings, rings[1:])]
    assert gaps
    assert max(gaps) < 2 * MARKER_RADIUS


def test_scratch_reach_curriculum_is_observed_and_resets_selected_robot(monkeypatch):
    """Geometric targets need no motion file and expose curriculum depth to the actor."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.microduck.mdp import skills
    from isaaclab_tasks.contrib.microduck.skill_env_cfg import MicroDuckPickupScratchEnvCfg

    cfg = MicroDuckPickupScratchEnvCfg()
    assert cfg.actions.joint_pos.teacher_path is None
    assert cfg.rewards.source_prior is None
    assert cfg.commands.twist.reference_path is None
    env = SimpleNamespace(
        num_envs=2, device="cpu", step_dt=0.02, episode_length_buf=torch.zeros(2), common_step_counter=0
    )
    command = skills.MicroDuckSkillCommand(cfg.commands.twist, env)
    monkeypatch.setattr(skills, "skill_head_position", lambda _: torch.zeros(2, 3))
    command.reset(torch.arange(2))
    command.style[:] = False
    env.episode_length_buf[:] = 70
    command.compute(env.step_dt)
    initial = command.reference_pose.clone()
    assert torch.count_nonzero(command.command[:, 2]) == 0
    env.common_step_counter = cfg.commands.twist.reach_ramp_steps[1]
    command.compute(env.step_dt)
    assert torch.all(command.command[:, 2] == 1)
    assert torch.all(command.reference_pose[:, 2] < initial[:, 2])
    torch.testing.assert_close(command.reference_pose[:, 3:6].norm(dim=-1), torch.ones(2))
    other = command.reference_pose[1].clone()
    command.reset(torch.tensor([0]))
    torch.testing.assert_close(command.reference_pose[0], initial[0])
    torch.testing.assert_close(command.reference_pose[1], other)
    cfg.play_mode()
    assert cfg.commands.twist.reach_ramp_steps is None


@pytest.mark.parametrize(
    "config_name", ["MicroDuckStandScratchEnvCfg", "MicroDuckPickupScratchEnvCfg", "MicroDuckMarkerGraspEnvCfg"]
)
def test_scratch_skills_reset_and_step_without_pretrained_policy(tmp_path, config_name):
    """Both tasks simulate finite random actions even when teacher loading is forbidden."""
    repo = Path(__file__).resolve().parents[3]
    robot = repo / "logs/microduck/marker_pick/assets_lip/robot_marker/robot_marker.usda"
    if not robot.is_file() or not torch.cuda.is_available():
        pytest.skip("Requires prepared cleanup robot and CUDA.")
    grasp_task = config_name == "MicroDuckMarkerGraspEnvCfg"
    module = "grasp_env_cfg" if grasp_task else "skill_env_cfg"
    actor_dim, critic_dim, action_dim = (65, 68, 15) if grasp_task else (61, 76, 14)
    probe = tmp_path / "scratch_probe.py"
    probe.write_text(
        f"""import argparse
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.contrib.microduck.{module} import {config_name}
from isaaclab_tasks.contrib.microduck.mdp import skills

def forbidden_teacher(*args, **kwargs):
    raise AssertionError("Scratch task attempted to load pretrained weights")

skills.MicroDuckSourceActor = forbidden_teacher
parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args(["--viz", "none"])
cfg = {config_name}()
cfg.scene.num_envs = 4
cfg.seed = 42
with launch_simulation(cfg.sim, args):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.contrib.microduck.microduck_env import MicroDuckGraspEnv
    env_type = MicroDuckGraspEnv if {grasp_task} else ManagerBasedRLEnv
    env = env_type(cfg)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            assert obs["policy"].shape == (4, {actor_dim})
            assert obs["critic"].shape == (4, {critic_dim})
            assert getattr(env.action_manager.get_term("joint_pos"), "teacher", None) is None
            assert "source_prior" not in env.reward_manager.active_terms
            print("RESET_BEAK_M", skills.skill_head_position(env).cpu().tolist())
            for _ in range(32):
                obs, reward, done, timeout, _ = env.step(torch.randn(4, {action_dim}, device=env.device) * 0.15)
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(obs["critic"]).all()
                assert reward.shape == done.shape == timeout.shape == (4,)
                assert torch.isfinite(reward).all()
            env.reset(env_ids=torch.tensor([1], device=env.device))
            assert torch.count_nonzero(env.action_manager.get_term("joint_pos").raw_actions[1]) == 0
            if {grasp_task}:
                model = env.action_manager.get_term("joint_pos").controller.solver.mj_model
                ground = [i for i in range(model.ngeom) if "GroundPlane" in model.geom(i).name]
                supports = [i for i in range(model.ngeom) if "/Support" in model.geom(i).name]
                markers = [i for i in range(model.ngeom) if "/Object/" in model.geom(i).name]
                assert ground and len(supports) == 2 and markers
                def allowed(a, b):
                    return bool(model.geom_contype[a] & model.geom_conaffinity[b]
                                or model.geom_contype[b] & model.geom_conaffinity[a])
                assert all(not allowed(a, b) for a in ground for b in supports)
                assert all(allowed(a, b) for a in markers for b in supports)
                pose = env.scene["object"].data.root_link_pose_w.torch.clone()
                pose[1, 0] = float("nan")
                env.scene["object"].write_root_pose_to_sim_index(root_pose=pose)
                action = torch.zeros(4, {action_dim}, device=env.device)
                obs, reward, done, _, _ = env.step(action)
                assert done[1] and torch.isfinite(reward).all()
                assert torch.isfinite(obs["policy"]).all()
                for _ in range(8):
                    obs, reward, done, _, _ = env.step(action)
                    assert not env.termination_manager.get_term("invalid").any()
                    assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
    finally:
        env.close()
"""
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=repo,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "yes"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scratch_cleanup_command_matches_trained_depth_contract():
    """Scratch reaching receives full depth; legacy commands remain unchanged."""
    from room_cleanup import adapt_skill_command

    command = [0.4, 0.8] + [0.0] * 11
    adapted = adapt_skill_command(command, {"command_contract": "phase_cos_sin_depth"})
    assert adapted[:2] == command[:2]
    assert adapted[2] == 1.0
    assert adapted[3:] == command[3:]
    assert command[2] == 0.0
    assert adapt_skill_command(command, {}) == command


def test_live_cleanup_state_is_published_without_enabling_recording(tmp_path):
    """A live viewer receives complete frames even when trajectory archival is off."""
    from types import SimpleNamespace

    import numpy as np
    from cleanup_recording import CleanupRecording

    bodies = torch.zeros(1, 2, 7)
    bodies[..., -1] = 1
    marker_pose = torch.tensor([[0.1, 0.2, 0.04, 0.0, 0.0, 0.0, 1.0]])
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=bodies)))
    marker = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=marker_pose)))
    nodes = torch.tensor([[0.1, 0.2, 0.03], [0.11, 0.2, 0.03]])
    path = tmp_path / "room/live.npz"
    recorder = CleanupRecording(False, ["trunk", "head"], path)
    recorder.append(0.02, robot, marker, nodes, False, phase="sock:pick", policy="alpha_ground_pick")
    with np.load(path) as frame:
        np.testing.assert_array_equal(frame["body_pose_m_xyzw"], bodies[0].numpy())
        np.testing.assert_array_equal(frame["cloth_points_m"], nodes.numpy())
        assert str(frame["phase"]) == "sock:pick"
        assert float(frame["time_s"]) == 0.02
    assert not path.with_suffix(".tmp").exists()
    assert recorder.times == []


def test_grasp_lift_reward_requires_opposed_contacts_and_bin_clearance():
    """Lifting without a real two-sided grasp earns no lift reward."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.microduck.mdp.grasp import grasp_lift_target, grasp_reward

    heights = torch.tensor([0.20, 0.16, 0.0395, 0.24])
    torch.testing.assert_close(grasp_lift_target(heights), torch.tensor([0.225, 0.21, 0.20, 0.25]))
    command = SimpleNamespace(
        refresh=lambda: None,
        start_height=torch.full((4,), 0.04),
        grasp_start_height=torch.full((4,), 0.04),
        stage=torch.full((4,), 2),
        cfg=SimpleNamespace(lift_distance=0.05),
        lift=torch.tensor([0.16, 0.16, 0.08, -0.01]),
        contacts=torch.tensor([[False, False], [True, False], [True, True], [True, True]]),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        scene={"robot": SimpleNamespace(data=None)},
        termination_manager=SimpleNamespace(get_term=lambda _: torch.zeros(4, dtype=torch.bool)),
    )
    torch.testing.assert_close(grasp_reward(env, "lift"), torch.tensor([0.0, 0.0, 0.5, 0.0]))
    # Recovering an object grasped below its support height must get immediate lift credit.
    command.grasp_start_height[3] = 0.02
    assert grasp_reward(env, "lift")[3] == pytest.approx(0.01 / 0.18)
    # Invalid physics is already a terminal failure and must not poison PPO returns.
    command.lift[3] = float("nan")
    env.termination_manager.get_term = lambda _: torch.tensor([False, False, False, True])
    torch.testing.assert_close(grasp_reward(env, "lift"), torch.tensor([0.0, 0.0, 0.5, 0.0]))


def test_grasp_jaw_reward_accepts_object_blocked_closure():
    """A squeeze against an object is valid even though the jaws cannot close fully."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.microduck.mdp.grasp import grasp_reward

    command = SimpleNamespace(
        stage=torch.tensor([2, 2, 0, 1]),
        distance=torch.zeros(4),
        jaw_ids=[0],
        refresh=lambda: None,
        contacts=torch.zeros(4, 2, dtype=torch.bool),
    )
    data = SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([[0.37], [0.37], [0.5235987756], [0.37]])))
    action = SimpleNamespace(jaw_target=torch.tensor([[-0.0872664626], [0.5235987756], [0.5235987756], [0.2181661565]]))
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_term=lambda _: command),
        action_manager=SimpleNamespace(get_term=lambda _: action),
        termination_manager=SimpleNamespace(get_term=lambda _: torch.zeros(4, dtype=torch.bool)),
    )
    reward = grasp_reward(env, "jaw")
    torch.testing.assert_close(reward[[0, 2]], torch.ones(2))
    assert reward[1] < 0.01
    assert reward[3] == pytest.approx(0.5)
    # Insertion must not reward hovering just outside the closing distance with an open jaw.
    command.stage[2] = 1
    command.distance[2] = 0.013
    assert grasp_reward(env, "jaw")[2] == 0
    # Real opposed contact keeps the squeeze valid after a marker settles at the lip.
    command.contacts[2] = True
    command.distance[2] = 0.02
    action.jaw_target[2, 0] = -0.0872664626
    assert grasp_reward(env, "jaw")[2] == pytest.approx(1.0)


def test_grasp_approach_rewards_forward_aperture_and_is_yaw_invariant():
    """Equal grip-point errors must not reward a backward-facing jaw equally."""
    from types import SimpleNamespace

    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    from isaaclab_tasks.contrib.microduck.mdp.grasp import grasp_jaw_orientation, grasp_reward

    zero = torch.zeros(3)
    yaw = quat_from_euler_xyz(zero, zero, torch.tensor([0.0, 0.8, 0.0]))
    head_local = quat_from_euler_xyz(zero, torch.tensor([-torch.pi / 2, -torch.pi / 2, torch.pi / 2]), zero)
    goal = torch.tensor([[0.068, 0.0, 0.215]]).repeat(3, 1)
    command = SimpleNamespace(
        head_id=0,
        goal=goal,
        pinch=goal.clone(),
        object_quaternion=yaw.clone(),
        refresh=lambda: None,
        reach_height=torch.full((3,), 0.215),
        start_height=torch.full((3,), 0.0395),
        stage=torch.zeros(3, dtype=torch.long),
        distance=torch.zeros(3),
        lift=torch.zeros(3),
    )
    data = SimpleNamespace(
        body_link_quat_w=SimpleNamespace(torch=quat_mul(yaw, head_local)[:, None]),
        root_link_quat_w=SimpleNamespace(torch=yaw),
    )

    class Scene(dict):
        env_origins = torch.zeros(3, 3)

    env = SimpleNamespace(
        scene=Scene(robot=SimpleNamespace(data=data)),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=SimpleNamespace(get_term=lambda _: torch.zeros(3, dtype=torch.bool)),
    )
    orientation = grasp_jaw_orientation(env)
    torch.testing.assert_close(orientation[:2], torch.ones(2))
    assert orientation[2] < 0.01
    approach = grasp_reward(env, "approach")
    torch.testing.assert_close(approach[0], approach[1])
    assert approach[0] > 9 * approach[2]
    # Turning the robot without turning the marker must lose alignment reward.
    command.object_quaternion[1] = command.object_quaternion[0]
    skewed = grasp_jaw_orientation(env)
    assert skewed[1] < 0.8 * skewed[0]
    # Floor pickup approaches from above, with the aperture pointing down.
    command.object_quaternion.copy_(yaw)
    command.goal[:, 2] = 0.0395
    data.body_link_quat_w.torch = yaw[:, None]
    torch.testing.assert_close(grasp_jaw_orientation(env), torch.ones(3))
    # An 8 mm error is substantial inside the 18 mm marker grasp aperture.
    command.pinch = command.goal - torch.tensor([0.008, 0.0, 0.0])
    broad = grasp_reward(env, "approach")
    command.reach_height[:] = command.start_height
    command.distance[:] = 0.008
    precise = grasp_reward(env, "approach")
    assert torch.all(precise < broad / 2)
    # Once held, the useful centering reward follows the object, not a moving lift goal.
    command.stage[:] = 2
    command.distance[:] = 0.002
    held = grasp_reward(env, "approach")
    command.pinch[:, 2] -= 0.04
    torch.testing.assert_close(grasp_reward(env, "approach"), held)
    # Pregrasp and insertion must credit the same measured object centering.
    command.stage[:] = 0
    command.goal[:, 2] += 0.018
    command.pinch[:] = command.goal - torch.tensor([0.0, 0.0, 0.020])
    command.distance[:] = 0.002
    pregrasp = grasp_reward(env, "approach") / (0.1 + 0.9 * grasp_jaw_orientation(env))
    command.stage[:] = 1
    command.goal[:, 2] -= 0.018
    inserted = grasp_reward(env, "approach") / (0.1 + 0.9 * grasp_jaw_orientation(env))
    torch.testing.assert_close(pregrasp, inserted)
    assert (pregrasp > 0.7).all()
    # A long marker needs stronger final roll alignment than a virtual reaching target.
    command.goal[:, 2] = 0.0395
    data.body_link_quat_w.torch = quat_from_euler_xyz(torch.full((3,), 0.15), zero, zero)[:, None]
    narrow = grasp_jaw_orientation(env)
    command.reach_height[:] = 0.215
    broad = grasp_jaw_orientation(env)
    assert (narrow < 0.9 * broad).all()


def test_grasp_progress_requires_contact_lift_and_dwell_and_resets_independently():
    """Reaching and one-sided touches cannot complete pickup; resets stay local."""
    from isaaclab_tasks.contrib.microduck.grasp_contract import MicroDuckGraspState

    state = MicroDuckGraspState(2, "cpu", 0.04, 0.04)
    origins = torch.zeros(2, 3)
    object_position = torch.tensor([[0.068, 0.0, 0.04]]).repeat(2, 1)
    velocity = torch.zeros_like(object_position)
    head = torch.zeros(2, 7)
    head[:, 6] = 1
    # A correctly placed grip point with an upside-down mouth must not advance.
    head[:, 3:] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    from isaaclab.utils.math import quat_apply

    head[:, :3] = object_position - quat_apply(head[:, 3:], state.pinch_local) - torch.tensor([0.0, 0.0, 0.018])
    for _ in range(20):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.stage.tolist() == [0, 0]
    assert not state.reach_seen.any()
    # Correct centering with excessive roll must not unlock insertion of a long marker.
    from isaaclab.utils.math import quat_from_euler_xyz

    head[:, 3:] = quat_from_euler_xyz(torch.full((2,), 0.15), torch.zeros(2), torch.zeros(2))
    exit_axis = quat_apply(head[:, 3:], torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1))
    head[:, :3] = object_position - quat_apply(head[:, 3:], state.pinch_local) - exit_axis * 0.018
    for _ in range(20):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.stage.tolist() == [0, 0]
    head[:, 3:] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    # With identity orientation, the aperture's approach point is 18 mm above the marker.
    head[:, :3] = object_position - state.pinch_local + torch.tensor([0.0, 0.0, 0.018])
    for _ in range(4):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.stage.tolist() == [1, 1]
    head[:, :3] = object_position - state.pinch_local
    state.contacts[:, 0] = True
    for _ in range(20):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert not state.success_seen.any()
    assert state.stage.tolist() == [1, 1]
    state.contacts[0, 1] = True
    for _ in range(6):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.stage.tolist() == [2, 1]
    assert not state.success_seen.any()
    # Contact acquisition must not jump the target 160 mm out of reach.
    assert 0.04 < state.goal[0, 2] < 0.045
    before_goal = state.goal.clone()
    state.update_progress(head, object_position, velocity, origins, 0.0)
    torch.testing.assert_close(state.goal, before_goal)
    # A stationary grasp must not leave the moving goal arbitrarily far overhead.
    for _ in range(100):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.goal[0, 2] <= object_position[0, 2] + 0.010001
    object_position[:, 2] = 0.20
    head[:, :3] = object_position - state.pinch_local
    for _ in range(16):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.success_seen.tolist() == [True, False]
    before = state.contact_time[0].clone()
    state.reset_progress(torch.tensor([1]))
    assert state.success_seen.tolist() == [True, False]
    assert state.stage.tolist() == [3, 0]
    torch.testing.assert_close(state.contact_time[0], before)
    assert not state.contacts[1].any()
    # A dropped object must return to approach even after a previous success.
    state.contacts[0] = False
    object_position[0, 2] = 0.04
    for _ in range(10):
        state.update_progress(head, object_position, velocity, origins, 0.02)
    assert state.stage[0] == 0
    assert state.lift_time[0] == 0
    assert state.goal[0, 2] < 0.06
    assert state.success_seen[0]  # Historical success is retained for evaluation.
    # A cylinder can settle at the lip, forward of the nominal open-mouth center.
    grip = MicroDuckGraspState(1, "cpu", 0.04, 0.04)
    obj = torch.tensor([[0.068, 0.0, 0.04]])
    head = torch.tensor([[0.084, 0.0, 0.116, 0.0, 0.0, 0.0, 1.0]])
    grip.contacts[:] = True
    for _ in range(20):
        grip.update_progress(head, obj, torch.zeros_like(obj), torch.zeros_like(obj), 0.02)
    assert grip.stage[0] == 2
    torch.testing.assert_close(grip.pinch, obj)
    assert grip.goal[0, 2] - grip.pinch[0, 2] > 0.009
    # The captured grasp frame must expose later slip; it must not follow the object.
    obj[:, 2] -= 0.005
    grip.update_progress(head, obj, torch.zeros_like(obj), torch.zeros_like(obj), 0.02)
    torch.testing.assert_close(grip.distance, torch.tensor([0.005]))


@pytest.mark.parametrize(
    "failed,final_dwell,neck_error,advances",
    [(True, 0.4, 0.0, False), (False, 0.0, 0.0, False), (False, 0.4, 0.0, True), (False, 0.4, 0.5, False)],
)
def test_grasp_curriculum_requires_survival_and_final_dwell(failed, final_dwell, neck_error, advances):
    """A brief reach before falling or drifting away must not unlock a harder goal."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.microduck.grasp_contract import MicroDuckGraspState
    from isaaclab_tasks.contrib.microduck.mdp.grasp import reset_grasp

    count = 256
    command = MicroDuckGraspState(count, "cpu", 0.0395, 0.2)
    command.cfg = SimpleNamespace(curriculum=True, minimum_object_height=0.0395)
    command.last_step = 1
    command.height_level, command.reach_level = 0.0395, 0.2
    command.recent_success = command.recent_reach = command.recent_episodes = 0
    command.reach_seen[:] = True
    command.reach_time[:] = final_dwell
    command.neck_error = torch.full((count,), neck_error)

    def asset():
        return SimpleNamespace(
            cfg=SimpleNamespace(spawn=SimpleNamespace(size=(0.024, 0.016, 0.03))),
            data=SimpleNamespace(
                default_root_state=SimpleNamespace(torch=torch.zeros(count, 13)),
                default_joint_pos=SimpleNamespace(torch=torch.zeros(count, 15)),
            ),
            write_root_pose_to_sim_index=lambda **kwargs: None,
            write_root_velocity_to_sim_index=lambda **kwargs: None,
            write_joint_state_to_sim_index=lambda **kwargs: None,
        )

    class Scene(dict):
        env_origins = torch.zeros(count, 3)

    env = SimpleNamespace(
        scene=Scene(robot=asset(), object=asset(), support=asset(), support_other=asset()),
        num_envs=count,
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=SimpleNamespace(terminated=torch.full((count,), failed)),
    )
    reset_grasp(env, torch.arange(count))
    assert (command.reach_level < 0.2) == advances
    assert command.recent_episodes == 0


def test_grasp_partial_reset_does_not_double_count_other_worlds_dwell():
    """Refreshing reset observations must not advance another world's contact timers."""
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.microduck.grasp_contract import MicroDuckGraspState
    from isaaclab_tasks.contrib.microduck.mdp.grasp import MicroDuckGraspCommand

    command = MicroDuckGraspCommand.__new__(MicroDuckGraspCommand)
    MicroDuckGraspState.__init__(command, 2, "cpu", 0.0395, 0.215)
    command._env = SimpleNamespace(
        common_step_counter=1, step_dt=0.02, num_envs=2, scene=SimpleNamespace(env_origins=torch.zeros(2, 3))
    )
    head = torch.tensor([[[0.005, 0.0, 0.231, 0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)]]]).repeat(2, 1, 1)
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=head), joint_pos=SimpleNamespace(torch=torch.full((2, 1), 0.27))
        )
    )
    command.neck_id = 0
    command.neck_error = torch.zeros(2)
    command.object = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=SimpleNamespace(torch=torch.tensor([[0.068, 0.0, 0.0395]]).repeat(2, 1)),
            root_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(2, 3)),
            root_link_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)),
        )
    )
    command.head_id = 0
    command.last_step = -1
    command._progress_step = torch.zeros(2, dtype=torch.long)
    command._reset_step = torch.full((2,), -1, dtype=torch.long)
    command._read_contacts = lambda: command.contacts.fill_(True)
    command.refresh()
    torch.testing.assert_close(command.reach_time, torch.tensor([0.02, 0.02]))
    command._resample_command(torch.tensor([0]))
    command.refresh()
    torch.testing.assert_close(command.reach_time, torch.tensor([0.0, 0.02]))
    torch.testing.assert_close(command.contact_time, torch.tensor([0.0, 0.02]))
    assert not command.contacts[0].any()
    command._env.common_step_counter = 2
    command.refresh()
    torch.testing.assert_close(command.reach_time, torch.tensor([0.02, 0.04]))


def test_grasp_features_use_robot_yaw_and_include_measured_jaw():
    """Playback's object observations share the task's sixteen-value SI contract."""
    from isaaclab_tasks.contrib.microduck.grasp_contract import MicroDuckGraspState

    state = MicroDuckGraspState(1, "cpu", 0.04, 0.04)
    state.pinch = torch.zeros(1, 3)
    state.goal = torch.tensor([[1.0, 0.0, 0.0]])
    quarter_turn = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    state.object_quaternion.copy_(quarter_turn)
    features = state.features(
        quarter_turn, state.goal * 2, state.goal * 3, torch.tensor([[0.2]]), torch.tensor([[0.4]])
    )
    assert features.shape == (1, 16)
    torch.testing.assert_close(
        features[0, :9], torch.tensor([0.0, -1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -3.0, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(features[0, 9:], torch.tensor([0.2, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0]))
    state.object_quaternion[:] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    misaligned = state.features(
        quarter_turn, state.goal * 2, state.goal * 3, torch.tensor([[0.2]]), torch.tensor([[0.4]])
    )
    assert misaligned[0, -1] == pytest.approx(-math.pi / 2)


def test_learned_cleanup_pick_waits_for_measured_grasp_completion():
    """The object-conditioned controller cannot finish pickup from a phase clock."""
    import numpy as np
    from cleanup_sequence import CleanupSequence, PickupState

    sequence = CleanupSequence(phase="pick", object_conditioned_pick=True)
    center = np.array([0.068, 0.0, 0.20])
    obj = PickupState(center, center - 0.009, center + 0.009, 0.0, True, 0.01)
    root = np.array([0.0, 0.0, 0.12])
    objects = {"marker": obj, "sock": obj}
    sequence.step(5.0, root, 0.0, objects)
    assert sequence.phase == "pick"
    sequence.step(5.02, root, 0.0, objects, learned_grasp_ready=True)
    assert sequence.phase == "lift_check"
    failed = CleanupSequence(phase="pick", object_conditioned_pick=True)
    failed.step(12.02, root, 0.0, objects)
    assert failed.phase == "grasp_failed"


def test_cleanup_grasp_inference_preserves_walking_and_tracks_assisted_cloth_patch():
    """Walking keeps its jaw command; sock feedback follows the two held cloth nodes."""
    from types import SimpleNamespace

    import numpy as np
    from grasp_playback import infer_cleanup_action

    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="obs")],
        run=lambda *_: [np.zeros((1, 14), dtype=np.float32)],
    )
    measured = []

    def grasp_step(target, position, velocity, contacts, *, object_quaternion=None):
        measured.append((target, position.clone(), velocity.clone(), contacts))
        return torch.ones(1, 14), -0.08

    grasp = SimpleNamespace(step=grasp_step)
    common = dict(grasp=grasp, target="sock", marker=None, marker_contacts=[], sock_opposed=False)
    action, jaw = infer_cleanup_action(
        session, torch.zeros(1, 61), 0.45, policy_name="native_walking", sock=None, assistance=None, **common
    )
    assert not measured
    assert jaw == 0.45
    np.testing.assert_array_equal(action, np.zeros((1, 14)))
    nodes = torch.tensor([[[0.0, -0.04, 0.10], [0.02, 0.0, 0.20], [0.04, 0.01, 0.20], [0.0, 0.08, 0.09]]])
    velocity = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.2], [0.3, 0.0, 0.4], [0.0, 0.0, 0.0]]])
    sock = SimpleNamespace(
        data=SimpleNamespace(nodal_pos_w=SimpleNamespace(torch=nodes), nodal_vel_w=SimpleNamespace(torch=velocity))
    )
    assistance = SimpleNamespace(active=True, ids=torch.tensor([1, 2]))
    _, jaw = infer_cleanup_action(
        session, torch.zeros(1, 61), 0.45, policy_name="alpha_ground_pick", sock=sock, assistance=assistance, **common
    )
    assert measured[0][0] == "sock"
    torch.testing.assert_close(measured[0][1], torch.tensor([[0.03, 0.005, 0.20]]))
    torch.testing.assert_close(measured[0][2], torch.tensor([[0.20, 0.0, 0.30]]))
    assert measured[0][3] == (True, True)
    assert jaw == pytest.approx(0.18)
