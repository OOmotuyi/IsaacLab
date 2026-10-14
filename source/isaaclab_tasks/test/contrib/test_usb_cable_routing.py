# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric success contracts for USB routing."""

import numpy as np
import torch

from isaaclab_tasks.contrib.usb_cable_routing.assets import ANCHOR, CABLE_LENGTH, START
from isaaclab_tasks.contrib.usb_cable_routing.mdp import cartesian_progress, guide_crossings, initial_centerline


def _assert_flexible_cable_contacts_and_rigid_connector(env):
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_tasks.contrib.usb_cable_routing.assets import SEGMENT_LENGTH

    model = NewtonManager.get_model()
    cable_shapes = {i for i, name in enumerate(model.shape_label) if "mesh_edge_capsule_" in name}
    gripper_shapes = {i for i, name in enumerate(model.shape_label) if "/Robotiq_2F_85/" in name}
    assert not any(
        (int(a) in cable_shapes and int(b) in gripper_shapes) or (int(b) in cable_shapes and int(a) in gripper_shapes)
        for a, b in model.shape_collision_filter_pairs_array()
    )
    ids = [i for i, name in enumerate(model.shape_label) if name.endswith("/blue_strain_relief")]
    assert len(ids) == env.num_envs
    rear = wp.to_torch(model.shape_transform)[ids, 2] - wp.to_torch(model.shape_scale)[ids, 2]
    torch.testing.assert_close(rear, torch.full_like(rear, SEGMENT_LENGTH / 2), atol=1e-7, rtol=0)


def test_guide_requires_cable_through_both_faces_inside_channel():
    guides = torch.tensor([[[0.30, 0.0, 0.0]]]).expand(5, -1, -1)
    points = torch.tensor(
        [
            [[0.25, 0, 0.02], [0.35, 0, 0.02]],
            [[0.25, 0, 0.08], [0.35, 0, 0.08]],
            [[0.25, 0.04, 0.02], [0.35, 0.04, 0.02]],
            [[0.25, 0, 0.02], [0.30, 0, 0.02]],
            [[0.25, -0.04, 0.02], [0.35, 0.04, 0.02]],
        ]
    )
    assert guide_crossings(points, guides).squeeze().tolist() == [True, False, False, False, True]


def test_initial_cable_is_slack_with_correct_endpoints_and_length():
    points = initial_centerline()
    np.testing.assert_allclose(points[0], ANCHOR, atol=1e-7)
    np.testing.assert_allclose(points[-1], START, atol=1e-7)
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    assert abs(length - CABLE_LENGTH) < 0.001
    assert np.linalg.norm(points[-1] - points[0]) < length * 0.95
    assert points[:, 2].min() > 0.005


def test_reset_and_progress_commands_match_episode_state():
    """Reset removes target history, and confirmed routing progress advances the goal."""
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.contrib.usb_cable_routing.env_cfg import UsbRoutingEnvCfg

    cfg = UsbRoutingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.routing.randomize = False
    cfg.commands.routing.curriculum = False
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-UR10e", cfg=cfg).unwrapped
        try:
            first, _ = env.reset()
            _assert_flexible_cable_contacts_and_rigid_connector(env)
            expected = first["policy"].clone()
            for _ in range(5):
                env.step(torch.full((1, 3), 0.1, device=env.device))
            second, _ = env.reset()
            torch.testing.assert_close(second["policy"], expected, atol=2e-5, rtol=1e-5)
            # A confirmed earlier crossing must advance shaping even for an alternative path.
            routing = env.command_manager.get_term("routing")
            routing.completed[:] = 1
            assert not routing.lowering.any()
            lift_height = routing.target()[:, 2].clone()
            env.step(torch.zeros(1, 3, device=env.device))
            assert (routing.target()[:, 2] < lift_height).all()

        finally:
            env.close()


def test_distance_progress_rewards_movement_and_not_stationary_goal_changes():
    previous = torch.tensor([[0.0, 0.0, 0.0]]).repeat(4, 1)
    current = torch.tensor([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-3.0, 2.0, 1.0]])
    torch.testing.assert_close(cartesian_progress(previous, current, targets), torch.tensor([0.25, -0.25, 0.0, 0.0]))


def test_zigzag_requires_bending_and_fits_the_existing_cable_length():
    from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import GUIDES

    guides = torch.tensor([GUIDES])
    route = torch.tensor(
        [
            [
                ANCHOR,
                (0.195, -0.15, 0.022),
                (0.30, -0.11, 0.025),
                (0.44, -0.19, 0.025),
                (0.56, -0.11, 0.025),
                (0.68, -0.15, 0.025),
            ]
        ]
    )
    assert guide_crossings(route, guides).all()
    assert (route[:, 1:] - route[:, :-1]).norm(dim=-1).sum() < CABLE_LENGTH
    straight = torch.tensor([[ANCHOR, (0.78, -0.15, 0.022)]])
    assert not guide_crossings(straight, guides).any()


def test_zigzag_configuration_is_independent_of_straight_task():
    from isaaclab_tasks.contrib.usb_cable_routing.assets import GUIDES as STRAIGHT_GUIDES
    from isaaclab_tasks.contrib.usb_cable_routing.env_cfg import UsbRoutingEnvCfg
    from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import GUIDES, UsbZigzagEnvCfg

    before = UsbRoutingEnvCfg()
    bent = UsbZigzagEnvCfg()
    after = UsbRoutingEnvCfg()
    for i, (straight_position, zigzag_position) in enumerate(zip(STRAIGHT_GUIDES, GUIDES), 1):
        assert getattr(before.scene, f"guide{i}").init_state.pos == straight_position
        assert getattr(after.scene, f"guide{i}").init_state.pos == straight_position
        assert getattr(bent.scene, f"guide{i}").init_state.pos == zigzag_position
    assert before.observations.policy.state.func == after.observations.policy.state.func
    assert bent.observations.policy.state.func != before.observations.policy.state.func
    assert before.episode_length_s == after.episode_length_s == 12.0
    assert before.commands.routing.class_type == after.commands.routing.class_type


def test_zigzag_reports_terminal_cable_before_scene_reset():
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import UsbZigzagEnvCfg

    cfg = UsbZigzagEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.routing.randomize = False
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.guide_count = 3
    cfg.episode_length_s = 8 * cfg.sim.dt * cfg.decimation
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-Zigzag-UR10e", cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            _assert_flexible_cable_contacts_and_rigid_connector(env)
            assert obs["policy"].shape == (1, 169)
            command = env.command_manager.get_term("routing")
            start = command.tip().clone()
            for _ in range(8):
                _, _, _, truncated, _ = env.step(torch.tensor([[0.0, 0.3, 0.0]], device=env.device))
            assert truncated.all()
            assert torch.isfinite(command.last_arm_metrics).all()
            assert command.last_arm_metrics[0, 1] > 0
            assert command.last_arm_metrics[0, 3] > 0
            # The terminal cable follows the commanded motion; the scene has already reset.
            assert command.last_points[0, -1, 1] > start[0, 1] + 0.003
            torch.testing.assert_close(command.tip(), start, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(
                guide_crossings(command.last_points, command.guide_positions), command.last_seated
            )
        finally:
            env.close()


def test_zigzag_curriculum_ignores_previous_stage_episodes():
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import UsbZigzagEnvCfg

    cfg = UsbZigzagEnvCfg()
    cfg.scene.num_envs = 1
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-Zigzag-UR10e", cfg=cfg).unwrapped
        try:
            env.reset()
            command = env.command_manager.get_term("routing")
            command.stage = 2
            command._episodes = command._successes = cfg.commands.routing.curriculum_episodes - 1
            env.termination_manager.terminated[:] = True
            command.success_dwell[:] = cfg.commands.routing.hold_steps
            command.completed[:] = 1
            command.reset(torch.tensor([0], device=env.device))
            assert command.stage == 2
            assert command._episodes == cfg.commands.routing.curriculum_episodes - 1
            assert command.guide_count.item() == 2
            # A subsequent success actually earned on stage two may promote stage three.
            command.success_dwell[:] = cfg.commands.routing.hold_steps
            command.completed[:] = 2
            command.reset(torch.tensor([0], device=env.device))
            assert command.stage == 3
            assert command.guide_count.item() == 3
        finally:
            env.close()


def test_zigzag_penalizes_only_lost_previously_completed_active_guides():
    from isaaclab_tasks.contrib.usb_cable_routing_zigzag.mdp import lost_crossings

    seated = torch.tensor([[False, False, False], [False, True, False], [True, False, False], [False, False, True]])
    completed = torch.tensor([0, 2, 1, 3])
    count = torch.tensor([3, 3, 1, 3])
    torch.testing.assert_close(lost_crossings(seated, completed, count), torch.tensor([0.0, 1.0, 0.0, 2.0]))


def test_scurve_crossings_follow_translated_rotated_passage_frames():
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import guide_crossings as scurve_crossings

    # A 90-degree passage is crossed along world y, not world x. The guide is elevated.
    guides = torch.tensor([[[0.4, -0.1, 0.05]]]).repeat(6, 1, 1)
    yaws = torch.full((6, 1), torch.pi / 2)
    points = torch.tensor(
        [
            [[0.4, -0.14, 0.070], [0.4, -0.06, 0.070]],
            [[0.36, -0.1, 0.070], [0.44, -0.1, 0.070]],
            [[0.408, -0.14, 0.070], [0.408, -0.06, 0.070]],
            [[0.4, -0.14, 0.060], [0.4, -0.06, 0.060]],
            [[0.4, -0.14, 0.10], [0.4, -0.06, 0.10]],
            [[0.4, -0.14, 0.085], [0.4, -0.06, 0.085]],
        ]
    )
    assert scurve_crossings(points, guides, yaws).flatten().tolist() == [True, False, False, False, False, False]


def test_scurve_layout_has_slack_and_does_not_modify_previous_tasks():
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import GUIDE_YAWS, GUIDES
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import UsbSCurveEnvCfg
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import guide_crossings as scurve_crossings
    from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import UsbZigzagEnvCfg

    before = UsbZigzagEnvCfg()
    cfg = UsbSCurveEnvCfg()
    after = UsbZigzagEnvCfg()
    for i in range(1, 4):
        assert getattr(before.scene, f"guide{i}").to_dict() == getattr(after.scene, f"guide{i}").to_dict()
    assert before.scene.cable.spawn.func == after.scene.cable.spawn.func != cfg.scene.cable.spawn.func
    assert before.scene.cable.spawn.physics_material.to_dict() == after.scene.cable.spawn.physics_material.to_dict()
    assert before.sim.physics.default_shape_cfg.mu == after.sim.physics.default_shape_cfg.mu == 0.4
    assert before.observations.policy.state.func == after.observations.policy.state.func
    # Approach each guide along its passage tangent; this also budgets room for bends.
    waypoints = [ANCHOR]
    for (x, y, _), yaw in zip(GUIDES, GUIDE_YAWS):
        for direction in (-1, 1):
            waypoints.append((x + direction * 0.015 * np.cos(yaw), y + direction * 0.015 * np.sin(yaw), 0.014))
    waypoints.append(cfg.commands.routing.seat_targets[-1])
    route = torch.tensor([waypoints], dtype=torch.float32)
    assert scurve_crossings(route, torch.tensor([GUIDES]), torch.tensor([GUIDE_YAWS])).all()
    assert (route[:, 1:] - route[:, :-1]).norm(dim=-1).sum() < CABLE_LENGTH
    offsets = torch.tensor(GUIDES)[:, :2] - torch.tensor(ANCHOR[:2])
    torch.testing.assert_close(offsets, torch.tensor([[0.12, 0], [0.26, 0.10], [0.40, 0]]))


def test_scurve_native_reset_preserves_upright_guide_yaw_and_action_contract():
    import gymnasium as gym
    import warp as wp

    from isaaclab.app import launch_simulation
    from isaaclab.utils.math import quat_apply

    from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import GUIDE_YAWS, GUIDES
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import UsbSCurveEnvCfg

    cfg = UsbSCurveEnvCfg()
    cfg.scene.num_envs = 2
    cfg.commands.routing.guide_count = 3
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.randomize = False
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-SCurve-UR10e", cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            _assert_flexible_cable_contacts_and_rigid_connector(env)
            assert obs["policy"].shape == (2, 175)
            assert env.action_manager.total_action_dim == 3
            # Select the wrist branch that stays above the hand when it lowers toward the table.
            robot = env.scene["robot"]
            wrist_ids, _ = robot.find_bodies(["wrist_1_link", "wrist_2_link"], preserve_order=True)
            wrist_height = robot.data.body_link_pos_w.torch[:, wrist_ids, 2]
            assert (wrist_height[:, 0] > wrist_height[:, 1] + 0.05).all()
            arm = env.action_manager.get_term("arm")
            arm.process_actions(torch.full((2, 3), 100.0, device=env.device))
            arm.apply_actions()
            targets = robot.actuators.target_command.position.torch[:, arm._joint_ids]
            current = robot.data.joint_pos.torch[:, arm._joint_ids]
            assert ((targets - current).abs() <= arm.cfg.maximum_joint_speed * env.physics_dt + 1e-6).all()
            env.reset()
            from isaaclab_newton.physics import NewtonManager

            command = env.command_manager.get_term("routing")
            model = NewtonManager.get_model()
            masses = wp.to_torch(model.body_mass)[command.cable.body_ids]
            assert masses.shape == (2, 40)
            expected_masses = torch.full_like(masses, 0.021 / 40)
            expected_masses[:, -1] += 0.004
            torch.testing.assert_close(masses, expected_masses, atol=1e-8, rtol=1e-5)
            torch.testing.assert_close(masses.sum(dim=1), torch.full_like(masses[:, 0], 0.025))
            inertia = wp.to_torch(model.body_inertia)[command.cable.body_ids]
            assert (torch.linalg.eigvalsh(inertia) > 0).all()
            radius, length = 0.002, 0.015
            area = np.pi * radius**2
            second_moment = np.pi * radius**4 / 4
            expected_gains = torch.tensor(
                [
                    150e6 * area / length,
                    150e6 * area / length,
                    8e6 * second_moment / length,
                    2e6 * (2 * second_moment) / length,
                ],
                device=env.device,
                dtype=torch.float32,
            )
            actual_gains = wp.to_torch(model.joint_target_ke)[command.dof_ids].reshape(2, 39, 4)
            torch.testing.assert_close(actual_gains, expected_gains.expand_as(actual_gains), rtol=1e-5, atol=1e-7)
            for indices in command.cable_shapes:
                mu = wp.to_torch(model.shape_material_mu)[indices]
                torch.testing.assert_close(mu, torch.full_like(mu, 0.40))
            torch.manual_seed(43)
            for _ in range(12):
                obs, reward, terminated, truncated, _ = env.step((torch.rand(2, 3, device=env.device) - 0.5) * 0.1)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert not (terminated | truncated).any()
            c = env.command_manager.get_term("routing")
            assert wp.to_torch(c.cable.grasp_active).bool().all()
            assert (c.stretch() < 0.10).all()
            for i, (position, yaw) in enumerate(zip(GUIDES, GUIDE_YAWS), 1):
                pose = env.scene[f"guide{i}"].data.root_state_w.torch
                torch.testing.assert_close(
                    pose[:, :3] - env.scene.env_origins,
                    torch.tensor(position, device=env.device).expand(2, 3),
                    atol=1e-5,
                    rtol=0,
                )
                up = torch.tensor((0.0, 0.0, 1.0), device=env.device).expand(2, 3)
                torch.testing.assert_close(quat_apply(pose[:, 3:7], up), up)
                passage = torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(2, 3)
                actual = quat_apply(pose[:, 3:7], passage)
                expected = torch.tensor((np.cos(yaw), np.sin(yaw), 0.0), device=env.device, dtype=actual.dtype).expand(
                    2, 3
                )
                torch.testing.assert_close(actual, expected)
            # A measured sharp fold fails the task, and its terminal radius survives reset.
            pose = c.cable.data.segment_pose_w.torch.clone()
            pose[:, 20, :3] = pose[:, 18, :3]
            c.cable.write_segment_pose_to_sim_index(segment_pose=pose)
            c._last_step = -1
            assert env.termination_manager.compute().all()
            assert env.termination_manager.get_term("overbent").all()
            assert (c.minimum_bend_radius < 0.015).all()
            env.reset()
            assert (c.last_minimum_bend_radius < 0.015).all()
            assert (c.minimum_bend_radius >= 0.015).all()
            # Randomization stays centered on the new nominal contact/stiffness values.
            c.cfg.randomize = True
            env.reset()
            assert ((c.material[:, 1] >= 0.36) & (c.material[:, 1] <= 0.44)).all()
            gains = wp.to_torch(model.joint_target_ke)[c.dof_ids].reshape(2, 39, 4)
            torch.testing.assert_close(
                gains, expected_gains[None, None] * c.material[:, 0, None, None].expand(2, 39, 1), rtol=1e-5, atol=1e-7
            )
            for i, indices in enumerate(c.cable_shapes):
                mu = wp.to_torch(model.shape_material_mu)[indices]
                torch.testing.assert_close(mu, c.material[i, 1].expand_as(mu))
            c.cfg.randomize = False
            env.reset()
            torch.testing.assert_close(c.material[:, 1], torch.full_like(c.material[:, 1], 0.40))
        finally:
            env.close()


def test_scurve_bend_radius_matches_circles_and_rejects_folds():
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import cable_bend_radii

    # Equal 15 mm chords on circles straddling the 15 mm limit.
    radii = torch.tensor([0.014, 0.015, 0.020], dtype=torch.float64)
    angle = 2 * torch.asin(0.015 / (2 * radii))
    theta = angle[:, None] * torch.arange(4, dtype=torch.float64)[None]
    points = torch.stack((radii[:, None] * theta.cos(), radii[:, None] * theta.sin(), torch.zeros_like(theta)), dim=-1)
    torch.testing.assert_close(cable_bend_radii(points), radii[:, None].expand(3, 2))
    straight = torch.tensor([[[0.0, 0, 0], [0.015, 0, 0], [0.030, 0, 0]]])
    assert torch.isinf(cable_bend_radii(straight)).all()
    straight[:, -1] = straight[:, 0]
    torch.testing.assert_close(cable_bend_radii(straight), torch.tensor([[0.0075]]))
    straight[:, 1] = straight[:, 0]
    assert (cable_bend_radii(straight) == 0).all()


def test_insert_geometry_accepts_usb_c_flip_and_rejects_false_seating():
    from isaaclab.utils.math import quat_from_angle_axis, quat_mul

    from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import SOCKET_FACE, SOCKET_QUAT
    from isaaclab_tasks.contrib.usb_cable_routing_insert.mdp import insertion_errors, insertion_seated

    socket = torch.tensor(SOCKET_FACE).repeat(7, 1)
    q = torch.tensor(SOCKET_QUAT).repeat(7, 1)
    front = socket.clone()
    front[:, 0] += 0.006
    plug_q = q.clone()
    plug_q[1] = quat_mul(q[1:2], torch.tensor([[0.0, 0.0, 1.0, 0.0]]))[0]
    front[2, 0] = socket[2, 0] + 0.001  # Shallow.
    front[3, 0] = socket[3, 0] + 0.015  # Behind the back stop.
    front[4, 1] += 0.001  # Outside horizontal clearance.
    front[5, 2] += 0.001  # Outside vertical clearance.
    plug_q[6] = quat_mul(q[6:7], quat_from_angle_axis(torch.tensor([0.1]), torch.tensor([[1.0, 0, 0]])))[0]
    local, angle = insertion_errors(front, plug_q, socket, q)
    assert insertion_seated(local, angle).tolist() == [True, True, False, False, False, False, False]


def test_insert_native_six_axis_grasp_socket_and_reset():
    import gymnasium as gym
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.app import launch_simulation
    from isaaclab.utils.math import quat_apply

    from isaaclab_tasks.contrib.usb_cable_routing.assets import SEGMENT_LENGTH
    from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import PLUG_FRONT_OFFSET
    from isaaclab_tasks.contrib.usb_cable_routing_insert.env_cfg import UsbInsertEnvCfg
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import UsbSCurveEnvCfg

    previous = UsbSCurveEnvCfg().to_dict()
    cfg = UsbInsertEnvCfg()
    assert UsbSCurveEnvCfg().to_dict() == previous
    cfg.scene.num_envs = 2
    cfg.commands.routing.randomize = False
    cfg.seed = 43
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-SCurve-Insert-UR10e", cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            _assert_flexible_cable_contacts_and_rigid_connector(env)
            assert env.action_manager.total_action_dim == 6
            assert obs["policy"].shape == (2, 195)
            assert torch.isfinite(obs["policy"]).all()
            c = env.command_manager.get_term("routing")
            initial = c.plug_pose()[1].clone()
            model = NewtonManager.get_model()
            # A pre-grasp must not disable contact between the flexible cable and the gripper.
            cable_shapes = {i for i, name in enumerate(model.shape_label) if "mesh_edge_capsule_" in name}
            gripper_shapes = {i for i, name in enumerate(model.shape_label) if "/Robotiq_2F_85/" in name}
            assert not any(
                (int(a) in cable_shapes and int(b) in gripper_shapes)
                or (int(b) in cable_shapes and int(a) in gripper_shapes)
                for a, b in model.shape_collision_filter_pairs_array()
            )
            poses = wp.to_torch(NewtonManager.get_state_0().body_q)
            wrist_q = poses[wp.to_torch(c.cable._wrist_ids).long(), 3:]
            axis = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).expand(2, -1)
            dot = (quat_apply(wrist_q, axis) * quat_apply(c.plug_pose()[1], axis)).sum(-1)
            assert (dot.abs() < 1e-4).all()  # Side approach keeps the cable clear of the palm.
            relief_ids = [i for i, name in enumerate(model.shape_label) if name.endswith("/blue_strain_relief")]
            relief_pose = wp.to_torch(model.shape_transform)[relief_ids]
            relief_size = wp.to_torch(model.shape_scale)[relief_ids]
            # The flexible cable ends at the back of the rigid strain relief, outside the housing.
            torch.testing.assert_close(
                relief_pose[:, 2] - relief_size[:, 2],
                torch.full_like(relief_pose[:, 2], SEGMENT_LENGTH / 2),
                atol=1e-7,
                rtol=0,
            )
            torch.testing.assert_close(wp.to_torch(model.shape_body)[relief_ids].long(), c.cable.body_ids[:, -1])
            arm = env.action_manager.get_term("arm")
            before = arm.target_quat.clone()
            rotation = torch.zeros(2, 6, device=env.device)
            rotation[:, 5] = 0.2
            arm.process_actions(rotation)
            torch.testing.assert_close(arm.target_quat, before)
            c.completed[:] = c.guide_count
            c.route_mastered[:] = True
            arm.process_actions(rotation)
            torch.testing.assert_close(arm.target_quat, before)
            torch.testing.assert_close(c.target(), c.seat_targets[-1].expand(2, 3))
            c.ready_to_align[:] = True
            arm.process_actions(rotation)
            assert (arm.target_quat - before).abs().max() > 0.001
            c.ready_to_approach[:] = True
            torch.testing.assert_close(c.target()[:, 1:], c.socket_position[:, 1:])
            env.reset()
            # Exercise the optional free-wrist contract independently of routing curriculum.
            arm.cfg.lock_wrist_while_routing = False
            for step in range(45):
                action = torch.zeros(2, 6, device=env.device)
                if 15 <= step < 30:
                    action[:, 5] = 0.15
                if step >= 30:
                    action[:, :3] = (torch.rand(2, 3, device=env.device) - 0.5) * 0.03
                obs, reward, done, timeout, _ = env.step(action)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert not (done | timeout).any(), env.termination_manager.get_active_iterable_terms(0)
            assert wp.to_torch(c.cable.grasp_active).bool().all()
            assert (1.0 - (initial * c.plug_pose()[1]).sum(-1).abs() > 1e-5).all()
            masses = wp.to_torch(NewtonManager.get_model().body_mass)[c.cable.body_ids].sum(-1)
            torch.testing.assert_close(masses, torch.full_like(masses, 0.025))
            assert not c.success_dwell.any()
            assert env.scene["socket"].data.root_state_w.torch.shape[0] == 2
            env.reset()
            torch.testing.assert_close(c.plug_pose()[1], initial, atol=2e-5, rtol=1e-5)
            assert not c.inserting.any() and not c.entered.any()
            assert not c.ready_to_align.any() and not c.ready_to_approach.any()
            # Preliminary curriculum completion must never masquerade as full insertion success.
            from isaaclab_tasks.contrib.usb_cable_routing.mdp import routing_success
            from isaaclab_tasks.contrib.usb_cable_routing_insert.mdp import curriculum_routing_complete

            c._last_step = env.common_step_counter
            c.cfg.curriculum = True
            c.guide_count[:] = 1
            c.route_mastered[:] = True
            assert curriculum_routing_complete(env).all()
            assert not routing_success(env).any()
            c.guide_count[:] = 3
            assert not curriculum_routing_complete(env).any()
            c.cfg.curriculum = False
            c.guide_count[:] = 1
            assert not curriculum_routing_complete(env).any()
            env.reset()
            # A seated plug must not count as success when the guides have not been routed.
            poses = c.cable.data.segment_pose_w.torch.clone()
            poses[:, -1, :3] = c.socket_position + env.scene.env_origins
            poses[:, -1, 0] += 0.006 - PLUG_FRONT_OFFSET
            poses[:, -1, 3:] = c.socket_quat
            c.cable.write_segment_pose_to_sim_index(segment_pose=poses)
            c._last_step = -1
            c.refresh()
            assert c.wrong_entry.all() and not c.success_dwell.any()
        finally:
            env.close()


def test_cable_guide_approach_measures_segments_and_clamps_endpoints():
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import cable_guide_distance

    points = torch.tensor([[[0.0, 0.0, 0.02], [0.1, 0.0, 0.02]]]).repeat(3, 1, 1)
    goals = torch.tensor([[0.05, 0.03, 0.02], [0.13, 0.04, 0.02], [0.05, 0.0, 0.07]])
    torch.testing.assert_close(cable_guide_distance(points, goals), torch.tensor([0.03, 0.05, 0.05]))
    shift = torch.tensor([1.0, -2.0, 0.1])
    torch.testing.assert_close(cable_guide_distance(points + shift, goals + shift), cable_guide_distance(points, goals))


def test_insert_residual_actor_preserves_route_bounds_corrections_and_exports(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from isaaclab_tasks.contrib.usb_cable_routing_insert.models import ResidualInsertionModel

    route_obs = TensorDict({"policy": torch.zeros(2, 175)}, batch_size=[2])
    routing = MLPModel(route_obs, {"actor": ["policy"]}, "actor", 3, [128, 128, 64], "elu", True)
    with torch.no_grad():
        for parameter in routing.parameters():
            parameter.zero_()
        routing.mlp[-1].bias.copy_(torch.tensor([0.2, -0.1, 0.05]))
    checkpoint = tmp_path / "routing.pt"
    torch.save({"actor_state_dict": routing.state_dict()}, checkpoint)
    state = torch.zeros(2, 195)
    state[:, 151] = 1
    state[1, 150] = 1
    state[:, 144:147] = torch.tensor([0.64, -0.23, 0.065])
    state[:, 174] = -0.008
    state[:, 177] = 1.0  # Reversible, aligned socket-frame quaternion (xyzw).
    state[:, 187:189] = 1.0
    obs = TensorDict({"policy": state}, batch_size=[2])
    model = ResidualInsertionModel(obs, {"actor": ["policy"]}, "actor", 6, routing_checkpoint=str(checkpoint))
    model.eval()
    expected_route = torch.tensor([0.2, -0.1, 0.05, 0.0, 0.0, 0.0])
    torch.testing.assert_close(model(obs)[0], expected_route)
    torch.testing.assert_close(model(obs)[1], torch.zeros(6))
    with torch.no_grad():
        model.mlp.residual[-1].bias.fill_(20.0)
    actions = model(obs)
    torch.testing.assert_close(actions[0], expected_route)
    torch.testing.assert_close(actions[1], torch.tensor([0.005, 0.005, 0.005, 0.003, 0.003, 0.003]))
    exported = torch.jit.script(model.as_jit())
    torch.testing.assert_close(exported(state), actions)
    assert not any(parameter.requires_grad for parameter in model.mlp.routing.parameters())

    # Guide counts are exposed as fractions of three, including the routing-policy input.
    with torch.no_grad():
        for parameter in model.mlp.routing.parameters():
            parameter.zero_()
        model.mlp.routing[0].weight[0, 147] = 1.0
        for layer in (2, 4, 6):
            model.mlp.routing[layer].weight[0, 0] = 1.0
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import SCurveCommandCfg

    state[0, 144] = 0.50
    state[0, 150] = 1 / 3
    state[0, 153] = 1.0
    target_x = SCurveCommandCfg().seat_targets[1][0]
    torch.testing.assert_close(model(obs)[0, 0], torch.tensor((target_x - 0.50) / 1.01), atol=1e-7, rtol=1e-5)
    state[0, 150] = 2 / 3
    state[0, 153] = 0.0
    target_x = SCurveCommandCfg().lift_targets[2][0]
    torch.testing.assert_close(model(obs)[0, 0], torch.tensor((target_x - 0.50) / 1.01), atol=1e-7, rtol=1e-5)

    sampled = ResidualInsertionModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        6,
        routing_checkpoint=str(checkpoint),
        distribution_cfg={
            "class_name": "isaaclab_tasks.contrib.usb_cable_routing_insert.models:InsertionGaussianDistribution",
            "init_std": 0.02,
        },
    ).eval()
    sampled_actions = sampled(obs, stochastic_output=True)
    torch.testing.assert_close(sampled.output_std, torch.tensor([[0.02] * 6, [0.002] * 6]))
    assert torch.isfinite(sampled.get_output_log_prob(sampled_actions)).all()
    torch.testing.assert_close(torch.jit.script(sampled.as_jit())(state), sampled(obs))


def test_insert_symmetry_keeps_gripper_above_socket():
    from isaaclab.utils.math import quat_inv, quat_mul

    from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import SOCKET_QUAT
    from isaaclab_tasks.contrib.usb_cable_routing_insert.mdp import select_socket_orientation

    # Opposite side grasps require opposite USB-C symmetries for a downward approach.
    socket = torch.tensor(SOCKET_QUAT).repeat(2, 1)
    wrist = torch.tensor([[-(2**-0.5), 0, 0, 2**-0.5], [2**-0.5, 0, 0, 2**-0.5]])
    plug = torch.tensor([[0.0, 0, 0, 1.0]]).repeat(2, 1)
    relative = quat_mul(quat_inv(socket), plug)
    selected = select_socket_orientation(relative, wrist, socket)
    torch.testing.assert_close(selected, torch.tensor([[0.0, 0, 1.0, 0], [0.0, 0, 0, 1.0]]))


def test_demonstration_normalization_stays_fixed_during_ppo_refinement():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from isaaclab_tasks.contrib.usb_cable_routing.models import FixedNormalizationModel

    observations = TensorDict({"policy": torch.arange(24, dtype=torch.float32).reshape(6, 4)}, batch_size=[6])
    args = (observations, {"actor": ["policy"]}, "actor", 3, [8], "elu", True)
    original = MLPModel(*args)
    original.update_normalization(observations)
    fixed = FixedNormalizationModel(*args)
    fixed.load_state_dict(original.state_dict())
    expected = fixed(observations).detach().clone()
    shifted = TensorDict({"policy": observations["policy"] + 100}, batch_size=[6])
    fixed.train()
    fixed.update_normalization(shifted)
    torch.testing.assert_close(fixed(observations), expected)
    original.update_normalization(shifted)
    assert not torch.allclose(original(observations), expected)
    # Standard playback and TorchScript accept the same checkpoint representation.
    original.load_state_dict(fixed.state_dict())
    torch.testing.assert_close(original(observations), expected)
    torch.testing.assert_close(torch.jit.script(fixed.as_jit())(observations["policy"]), expected)


def _contact_observation_env(insertion):
    from types import SimpleNamespace

    def forbidden_truth_read(*args, **kwargs):
        raise AssertionError("A deployed controller must not read the cable evaluator.")

    origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.3]])
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    poses = torch.zeros(2, 40, 7)
    poses[..., 6] = 1.0
    poses[:, -1, :3] = origins + torch.tensor([0.5, -0.1, 0.1])
    cable = SimpleNamespace(data=SimpleNamespace(segment_pose_w=SimpleNamespace(torch=poses)))
    robot = SimpleNamespace(
        joint_names=[f"arm_{index}" for index in range(6)] + ["finger_joint"],
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.arange(14, dtype=torch.float32).reshape(2, 7)),
            joint_vel=SimpleNamespace(torch=torch.zeros(2, 7)),
            root_pos_w=SimpleNamespace(torch=origins + torch.tensor([0.1, -0.85, 0.0])),
        ),
    )
    command = SimpleNamespace(
        robot=robot,
        cable=cable,
        arm_joint_ids=list(range(6)),
        tool_quat=identity.clone(),
        guide_positions=torch.tensor([[[0.3, -0.15, 0.0], [0.44, -0.05, 0.0], [0.58, -0.15, 0.0]]]).repeat(2, 1, 1),
        guide_yaws=torch.zeros(2, 3),
        guide_count=torch.full((2,), 3),
        points=forbidden_truth_read,
        tip=forbidden_truth_read,
        stretch=forbidden_truth_read,
        refresh=forbidden_truth_read,
    )
    arm = SimpleNamespace(
        _compute_frame_pose=lambda: (torch.tensor([[0.4, 0.7, 0.15]]).repeat(2, 1), identity),
        _target_valid=torch.tensor([True, False]),
        processed_actions=torch.tensor([[0.45, 0.7, 0.15]]).repeat(2, 1),
    )
    if insertion:
        command.socket_position = torch.tensor([[0.68, -0.15, 0.065]]).repeat(2, 1)
        command.socket_quat = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(2, 1)
        arm.target_quat = identity.clone()
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command),
        action_manager=SimpleNamespace(get_term=lambda _: arm, action=torch.zeros(2, 6 if insertion else 3)),
        scene=SimpleNamespace(env_origins=origins),
        max_episode_length_s=60.0,
        episode_length_buf=torch.tensor([3, 7]),
        step_dt=1 / 30,
    )


def test_contact_controller_observation_excludes_cable_shape_and_evaluator_state():
    from isaaclab_tasks.contrib.usb_cable_routing_contact.observations import (
        CONTROLLER_FIELDS,
        INSERTION_FIELDS,
        contact_policy_observation,
        controller_measurements,
    )

    for insertion, dimension in ((False, 55), (True, 65)):
        env = _contact_observation_env(insertion)
        command = env.command_manager.get_term("routing")
        initial = contact_policy_observation(env).clone()
        measurements = controller_measurements(env)
        fields = CONTROLLER_FIELDS + (INSERTION_FIELDS if insertion else ())
        assert set(measurements) == set(fields)
        assert initial.shape == (2, dimension)
        torch.testing.assert_close(initial, torch.cat([measurements[name].flatten(1) for name in fields], dim=1))
        torch.testing.assert_close(measurements["tcp_position_m"], torch.tensor([[0.5, -0.15, 0.15]]).repeat(2, 1))
        torch.testing.assert_close(
            measurements["commanded_tcp_position_m"], torch.tensor([[0.55, -0.15, 0.15], [0.5, -0.15, 0.15]])
        )
        torch.testing.assert_close(measurements["connector_position_m"], torch.tensor([[0.5, -0.1, 0.1]]).repeat(2, 1))

        # Hidden geometry, material and success diagnostics may change without reaching either controller.
        command.cable.data.segment_pose_w.torch[:, :-1, :3] += 10.0
        for name in (
            "completed",
            "seated",
            "lowering",
            "route_mastered",
            "ready_to_align",
            "ready_to_approach",
            "inserting",
            "entered",
            "minimum_bend_radius",
            "material",
            "success_dwell",
        ):
            setattr(command, name, torch.ones(2))
        for name in ("grasp_active", "grasp_acquired", "position_slip", "tension", "socket_contact_force"):
            setattr(command.cable, name, torch.ones(2))
        torch.testing.assert_close(contact_policy_observation(env), initial, atol=0, rtol=0)

        # The explicitly allowed connector pose estimate remains observable.
        command.cable.data.segment_pose_w.torch[:, -1, 0] += 0.01
        changed = contact_policy_observation(env)
        assert (changed != initial).sum() == env.episode_length_buf.numel()
        torch.testing.assert_close(
            controller_measurements(env)["connector_position_m"], torch.tensor([[0.51, -0.1, 0.1]]).repeat(2, 1)
        )


def test_contact_insertion_action_does_not_read_hidden_routing_phase():
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.usb_cable_routing_contact.actions import ContactInsertionAction

    def forbidden_command_read(*args, **kwargs):
        raise AssertionError("The contact servo must not consult ground-truth routing progress.")

    captured = []
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    collisions = torch.ones(2)
    arm = SimpleNamespace(
        device="cpu",
        cfg=SimpleNamespace(scale=0.004, rotation_scale=0.03),
        _raw_actions=torch.zeros(2, 6),
        _processed_actions=torch.tensor([[0.42, 0.0, 0.2]]).repeat(2, 1),
        _target_valid=torch.tensor([False, True]),
        target_quat=identity.clone(),
        _compute_frame_pose=lambda: (torch.tensor([[0.5, 0.0, 0.2]]).repeat(2, 1), identity),
        _asset=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=torch.zeros(2, 3)))),
        _ik_controller=SimpleNamespace(set_command=lambda value: captured.append(value.clone())),
    )

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    arm._env = SimpleNamespace(
        scene=Scene(cable=SimpleNamespace(collisions=collisions)),
        command_manager=SimpleNamespace(get_term=forbidden_command_read),
    )
    ContactInsertionAction.process_actions(arm, torch.tensor([[2.0, -2.0, 0.5, 0.0, 0.0, 2.0]]).repeat(2, 1))
    expected_position = torch.tensor([[0.504, -0.004, 0.202], [0.424, -0.004, 0.202]])
    expected_quat = torch.tensor([[0.0, 0.0, np.sin(0.015), np.cos(0.015)]], dtype=torch.float32).repeat(2, 1)
    torch.testing.assert_close(captured[0], torch.cat((expected_position, expected_quat), dim=1))
    assert arm._target_valid.all()
    assert not collisions.any()


def test_contact_guide_targets_follow_pose_and_partial_randomization_stays_local():
    from types import SimpleNamespace

    from isaaclab_tasks.contrib.usb_cable_routing_contact.env_cfg import ContactSCurveCommandCfg
    from isaaclab_tasks.contrib.usb_cable_routing_contact.mdp import randomize_guide_poses, transformed_guide_targets
    from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import GUIDE_YAWS, GUIDES

    cfg = ContactSCurveCommandCfg()
    nominal = torch.tensor(GUIDES)
    targets = torch.tensor(cfg.seat_targets)
    translated = nominal[None] + torch.tensor([0.005, -0.002, 0.0])
    rotated_yaw = torch.tensor(GUIDE_YAWS)[None] + torch.pi / 2
    offset = targets - nominal
    expected = translated + torch.stack((-offset[:, 1], offset[:, 0], offset[:, 2]), dim=-1)
    torch.testing.assert_close(transformed_guide_targets(targets, translated, rotated_yaw), expected)

    command = SimpleNamespace(
        cfg=cfg,
        guide_positions=nominal.repeat(4, 1, 1),
        guide_yaws=torch.tensor(GUIDE_YAWS).repeat(4, 1),
        seat_targets=targets,
        lift_targets=torch.tensor(cfg.lift_targets),
        contact_seat_targets=targets.repeat(4, 1, 1),
        contact_lift_targets=torch.tensor(cfg.lift_targets).repeat(4, 1, 1),
    )
    env = SimpleNamespace(device="cpu", num_envs=4, command_manager=SimpleNamespace(get_term=lambda _: command))
    names = ("guide_positions", "guide_yaws", "contact_seat_targets", "contact_lift_targets")
    previous = {name: getattr(command, name).clone() for name in names}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        randomize_guide_poses(env, torch.tensor([1, 3]))
    for name in names:
        torch.testing.assert_close(getattr(command, name)[[0, 2]], previous[name][[0, 2]], atol=0, rtol=0)
    assert not torch.equal(command.guide_positions[[1, 3]], previous["guide_positions"][[1, 3]])
    assert (command.guide_positions[..., :2] - nominal[..., :2]).abs().max() <= cfg.guide_xy_jitter
    assert (command.guide_yaws - torch.tensor(GUIDE_YAWS)).abs().max() <= cfg.guide_yaw_jitter
    torch.testing.assert_close(command.guide_positions[..., 2], torch.zeros(4, 3))


def test_contact_configs_use_restricted_observations_and_generic_recurrent_actors():
    from isaaclab.utils.string import string_to_callable

    from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

    from isaaclab_tasks.contrib.usb_cable_routing_contact.actions import ContactInsertionAction
    from isaaclab_tasks.contrib.usb_cable_routing_contact.env_cfg import (
        UsbContactInsertEnvCfg,
        UsbContactInsertPPORunnerCfg,
        UsbContactSCurveEnvCfg,
        UsbContactSCurvePPORunnerCfg,
    )
    from isaaclab_tasks.contrib.usb_cable_routing_contact.observations import contact_policy_observation

    for env_type, runner_type in (
        (UsbContactSCurveEnvCfg, UsbContactSCurvePPORunnerCfg),
        (UsbContactInsertEnvCfg, UsbContactInsertPPORunnerCfg),
    ):
        cfg = env_type()
        runner = runner_type()
        assert string_to_callable(cfg.observations.policy.state.func) is contact_policy_observation
        assert cfg.observations.policy.last_action is None  # Already present in the shared measurement contract.
        assert isinstance(runner.actor, RslRlRNNModelCfg)
        assert runner.actor.class_name == RslRlRNNModelCfg().class_name
        assert runner.actor.rnn_type == "gru"
        if env_type is UsbContactInsertEnvCfg:
            assert string_to_callable(cfg.actions.arm.class_type) is ContactInsertionAction
            assert not cfg.actions.arm.lock_wrist_while_routing
