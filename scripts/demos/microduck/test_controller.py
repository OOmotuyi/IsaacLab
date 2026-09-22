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
