# UR10e–Robotiq cable routing and USB-C insertion

## Robot posture revision

Current posture-corrected policy: **30/100** fresh randomized successes. Reliability remains limited. Updated videos, policies, measurements and exact commands are in the [posture results report](../../../logs/usb_posture_revision/final/REPORT.md).

The robot base is now at `(0.10, -0.85, 0.0)` m. The tabletop is 1.1 × 1.6 m to support it. Reset selects a measured IK branch with the wrist above the hand; the cable, guide centers and socket retain their existing positions. Routing still uses three Cartesian actions and a fixed grasp orientation. Joint targets are limited to a 2 rad/s equivalent increment and to joint position limits; measured joint velocity can differ from this command bound. The TCP target stays at least 45 mm above the tabletop. Rewards discourage wrist folding, excess roll and wrist heights below 140 mm; wrist heights below 90 mm or invalid joint states terminate the episode.

S-curve seating targets are `(0.60, -0.15, 0.075)`, `(0.60, 0.10, 0.060)` and `(0.64, -0.23, 0.065)` m. The insertion controller chooses between the two USB-C symmetries using rotation distance and gripper approach direction, favoring the palm above the socket. Its episode limit is 60 s. Evaluation records terminal minimum wrist height, maximum joint speed, wrist roll and joint-limit margin. Wrist height measures the `wrist_1_link` frame origin, not the lowest robot surface. Registered collision steps count contact candidates and do not measure penetration; the separate native clearance audit computes contact separation.

Checkpoints trained with the earlier base position require adaptation even though observation dimensions remain 175/195. Historical results below describe their original source snapshots. Posture-refinement artifacts, evaluations and playback commands are saved under `logs/usb_posture_revision/`; the original results remain under `logs/usb_parallel_training/final/`.

Task: `IsaacContrib-Route-USB-SCurve-Insert-UR10e`.

This independent task extends the [S-curve environment](../usb_cable_routing_scurve/README.md). The robot must route the cable through the red, yellow and green guides in order, then align the free USB-C plug with one socket beyond the third guide and seat it. All four cable tasks now share the corrected rigid strain-relief attachment and side grasp. Flexible cable–gripper contacts remain enabled; only the rigid plug contact is substituted by a breakable compliant grasp. The shared cable uses 40 segments, a 21 g body plus 4 g plug, 150/8/2 MPa effective moduli and 0.40 active friction. Static friction 0.50 and restitution 0.05 are calibration targets. Earlier straight/zigzag checkpoints are historical results with different mechanics and have not been revalidated against this version.

## Assets and layout

The existing UR10e with Robotiq 2F-85, table, anchor and colored guides are reused. Guide centers relative to the anchor remain `(120, 0)`, `(260, 100)` and `(400, 0)` mm, with passage yaws +10°, 0° and −10°. The upright guides keep the inward 90° lips, 10 mm cradle height and 10 mm top opening.

The new socket entrance is at world `(0.680, -0.150, 0.065)` m: 100 mm beyond the third guide along x and 65 mm above the tabletop. Insertion follows world +x. The raised receptacle has a mounting base, post, rounded metal shell, central insulating tongue, visible gold contacts and green target marker. The clear opening is 8.8 × 3.0 mm with its wide axis horizontal. A faceted primitive collision shell leaves the opening hollow; it does not use a convex hull that would seal the port.

The blue plug retains its 10 × 6 × 18 mm housing and approximately 8.4 × 2.6 × 7 mm mating shell. The housing, strain relief and shell form one rigid compound with the final cable segment. The cable centerline terminates at the rear of the strain relief, at local z = +7.5 mm on that segment; the housing origin is at +22.5 mm and the mating nose at +38.5 mm. Bending joints lie outside the housing, so the cable cannot bend through its side. Connector mass, center of mass and inertia are accumulated at these authored offsets. Its cavity is opened to receive the socket tongue; the removed collision/visual tongue retains its lumped mass allocation. Decorative contacts do not collide. Nominal insertion depth is 6 mm. These are approximate mechanical assets informed by the [USB Type-C specification](https://www.usb.org/sites/default/files/USB%20Type-C%20Spec%20R2.0%20-%20August%202019.pdf), not certified connector CAD. Electrical connection, retention springs and latching are not simulated.

Reusable generated files:

- [`usb_c_socket.usda`](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing_insert/assets/usb_c_socket.usda): fixed receptacle and mount.
- [`usb_c_plug.usda`](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing_insert/assets/usb_c_plug.usda): standalone 4 g plug. The running task adds the same geometry to the final cable segment.
- [`assets.py`](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing_insert/assets.py): procedural generator and runtime connector hook.

The held cable endpoint begins at `(0.580, 0.080, 0.3275)` m with an upward tangent. The wrist approaches the upright plug from the side along world +y. Its approach axis is perpendicular to the plug axis; the cable hangs below the fingers, clear of the palm. The initial slack curve stays in front of the arm. The grasp is on the housing, 21 mm behind the plug nose. An endpoint IK check verifies a clear arm configuration at insertion; it does not prove a collision-free motion between reset and insertion. A smooth geometric path through all three guides to the seated plug uses less than 600 mm of cable, with the latest measured length recorded in `logs/usb_parallel_training/socket_clearance_final.json` against the 600 mm length budget. Full physical routing and insertion have since been demonstrated; randomized reliability is reported below.

## Physics and task contract

The [S-curve cable parameters](../usb_cable_routing_scurve/README.md#cable-physical-properties) are inherited: 600 mm length, 4 mm diameter, 40 Newton VBD elastic segments, 21 g flexible body plus 4 g connector, and effective stretch/bend/twist moduli of 150/8/2 MPa. Active VBD friction is 0.40. Static friction 0.50 and restitution 0.05 remain calibration targets. The 15 mm minimum bend radius is checked as a task failure limit, rather than imposed as a hard solver constraint. All mechanical parameters remain estimates for video calibration.

The UR10e uses MJWarp; the cable and socket use the coupled Newton VBD setup. VBD runs 12 iterations with 8 substeps per 1/120 s physics step. The detailed receptacle uses a 4096-contact per-body buffer. The pre-grasp remains a breakable compliant attachment, with the housing held by a 400 N/m translational spring and 0.03 N·s/m damping. It breaks beyond 6 N or 15 mm slip. Added rotational retention uses implicit PD stabilization with nominal gains 0.15 N·m/rad and 0.0001 N·m·s/rad; angular slip above 1.2 rad or torque above 0.04 N·m releases it. Every flexible cable segment has collision enabled against the gripper, including the final three segments. Only rigid connector–gripper contacts are replaced by the compliant pre-grasp. This is not a friction-only grasp model.

The policy has six actions: three Cartesian translations and three axis-angle increments at 30 Hz. Translation is bounded at 10 mm per axis before routing and 1 mm afterward; rotational components are bounded at 0.025 rad. The observation contains 195 simulator-state values, including cable and guide state, plug/socket relative pose, commanded wrist orientation, insertion phase, grasp force/torque and previous action. No camera or physical force sensor drives the policy.

All three guides are active during evaluation. Training advances through one and two guides using a separate routing-stage completion flag, then requires full insertion with three guides. Preliminary stage completion never counts as final insertion success. Rotation is held fixed until routing is complete and the cable endpoint has reached the lowering target. Explicit settled/aligned flags then separate wrist alignment from socket approach. The actor receives both flags; previous 193-input insertion checkpoints require migration and refinement. Insertion rewards extend the inherited ordered-routing rewards with measured angular-error reduction, alignment and insertion progress, plus penalties for premature entry and excessive grasp force. The parent collision, drop, stretch, bend-radius and skipped-guide penalties remain active. Nominal training randomization retains stiffness scaling 0.85–1.15, friction 0.36–0.44 and 5 mm starting-centerline jitter; preview disables randomization.

Success requires all three cable crossings to remain seated, prior aligned entry from in front of the port, nose depth 5.5–6.8 mm, horizontal error at most 0.15 mm, vertical error at most 0.12 mm and orientation error at most 2°. USB-C's 180° reversible orientation is accepted. The grasp must remain attached with force below 2 N, joint-gap ratio below 0.10 and sampled bend radius at least 15 mm for 12 control steps (0.4 s). Entering the port before routing terminates the episode. Episodes have a 48 s limit to allow a slow, compliant final approach.

## Exact commands

Run from `/home/oomotuyi/IsaacLab-fork`:

```bash
export OMNI_KIT_ACCEPT_EULA=yes
export PXR_WORK_THREAD_LIMIT=1

# Regenerate the new connector assets when needed.
uv run python -c 'from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import generate_assets; generate_assets()'

# Open the new environment without training.
uv run python scripts/demos/usb_cable_routing_insert/preview.py

# Save OVRTX scene views from a short, zero-action preview.
PXR_WORK_THREAD_LIMIT=1 uv run --extra ovrtx python scripts/demos/usb_cable_routing_insert/preview.py \
  --viz newton_rtx --headless --steps 40 \
  --output_dir logs/usb_cable_routing_insert/ovrtx_preview

# Record six seconds of zero-action physics with OVRTX.
PXR_WORK_THREAD_LIMIT=1 uv run --extra ovrtx python scripts/demos/usb_cable_routing_insert/preview.py \
  --viz newton_rtx --headless --steps 215 --video \
  --output_dir logs/usb_cable_routing_insert/ovrtx_video

# Replay the saved hybrid routing/insertion policy and record it with OVRTX.
PXR_WORK_THREAD_LIMIT=1 uv run --extra ovrtx python scripts/demos/usb_cable_routing/evaluate.py \
  --task IsaacContrib-Route-USB-SCurve-Insert-UR10e \
  --agent rsl_rl_residual_cfg_entry_point \
  --checkpoint logs/usb_posture_revision/final/insert/checkpoint.pt \
  --episodes 1 --num_envs 1 --seed 660001 --viz newton_rtx --headless \
  --video logs/usb_ovrtx_trial/policy/demonstration.mp4 \
  --output logs/usb_ovrtx_trial/policy/evaluation.json

# Record a six-second stationary physics preview, screenshots and measurements.
uv run python scripts/demos/usb_cable_routing_insert/preview.py \
  --headless --steps 215 --video \
  --output_dir logs/usb_cable_routing_insert/gripper_preview

# Check cable length, a 120-step cable/robot contact rollout, and endpoint clearance.
uv run python scripts/demos/usb_cable_routing_insert/check_reachability.py \
  --output logs/usb_cable_routing_insert/gripper_clearance.json

# Physically mate an isolated plug with the socket; also test a misaligned approach.
uv run python scripts/demos/usb_cable_routing_insert/probe_socket.py

# Run all existing and new cable-task tests.
uv run --extra test python -m pytest \
  source/isaaclab_tasks/test/contrib/test_usb_cable_routing.py -q
```

The preview defaults to `--viz newton_gl`. Selecting `newton_rtx` uses the experimental
OVRTX viewer in path-tracing mode; the headless capture command changes visualization
only and does not load or train a policy.
The commands serialize USD asset loading because this local setup encountered native
allocation errors with parallel loading in both viewers. The first RTX image can take
several minutes while the driver compiles shaders; subsequent launches reuse its cache.

The task has a separate PPO configuration and checkpoint namespace, `usb_cable_routing_insert`. Before the robot posture revision, parallel PPO training alongside S-curve routing completed. The selected hybrid controller achieved 60/100 full routing-and-insertion successes on new randomized configurations; it is not reliably solved and successful trials still registered penalized robot–fixture contacts. Saved policies, a successful full-task video, all trial outcomes and exact playback commands are in [the results report](../../../logs/usb_parallel_training/final/REPORT.md). The successful physical demonstrations are scripted data, not learned-policy evaluation. Collect and fit them with:

```bash
uv run python scripts/demos/usb_cable_routing_insert/collect_demonstrations.py \
  --routing_actions logs/usb_parallel_training/stochastic_route_probe.pt \
  --output logs/usb_parallel_training/insertion_demonstrations.pt --seed 511
uv run python scripts/demos/usb_cable_routing_scurve/initialize_from_demonstrations.py \
  --checkpoint logs/usb_parallel_training/insertion_phase_template.pt \
  --demonstrations logs/usb_parallel_training/insertion_demonstrations.pt \
  --output logs/usb_parallel_training/insertion_initialized.pt \
  --epochs 600 --action_std 0.03 --ignore_previous_actions
```

An independent training command is:

```bash
uv run isaaclab train --task IsaacContrib-Route-USB-SCurve-Insert-UR10e \
  --rl_library rsl_rl --num_envs 128 --max_iterations 500 --seed 43 \
  --run_name usb_c_insertion env.commands.routing.guide_count=1 env.commands.routing.curriculum=true
```

Three-action routing checkpoints cannot be loaded directly into the six-action actor. Use the transfer utility or the residual recipe below to reuse them.

## Validation artifacts

[Scene overview](../../../logs/usb_cable_routing_insert/gripper_preview/overview.png) · [Attachment close-up](../../../logs/usb_cable_routing_insert/gripper_preview/attachment.png) · [Port close-up](../../../logs/usb_cable_routing_insert/gripper_preview/port.png) · [Preview video](../../../logs/usb_cable_routing_insert/gripper_preview/preview_0000.mp4) · [Scene measurements](../../../logs/usb_cable_routing_insert/gripper_preview/scene.json) · [Reachability and cable length](../../../logs/usb_cable_routing_insert/gripper_clearance.json) · [Physical mating results](../../../logs/usb_cable_routing_insert/socket_probe.json)

The preview holds the starting grasp with zero action. The mating probe drives the identical plug geometry into the actual socket in VBD, independently of the robot and routed cable. The checker steps the initial grasp for 120 control steps and measures cable–robot penetration from the enabled collision geometry; a 0.2 mm penetration tolerance is the acceptance limit. It separately checks the insertion endpoint with kinematics, without traversing a routing trajectory. These checks validate the scene, action contract and mating geometry; they are not a successful full routing-and-insertion rollout or a trained-policy evaluation.

Earlier grasp validation: all 15 then-existing cable-task tests passed. The posture revision expanded suite has 19 passing tests. The regression rejects the original exclusions between gripper shapes and the final cable segments, and checks that the reset wrist and plug axes are perpendicular. The 120-step contact probe detected zero cable–robot penetration. The measured wrist/plug axis dot product after settling was 0.0246 (approximately 88.6 degrees). Reset and insertion-endpoint clearance checks passed. The updated 215-step preview had zero resets or dropped grasps, a minimum sampled bend radius of 18.97 mm, peak grasp force of 3.805 N, and maximum joint-gap ratio of 0.00633. Cable-plus-connector mass remained 25.0 g. The housing/strain-relief rigid-attachment regression remains covered. The same attachment and flexible-cable contact regression now passes for the straight, zigzag and S-curve routing tasks. All 15 cable tests passed after the common refactor.

The socket was lowered and moved farther beyond the third guide after full-motion probes exposed excessive curvature behind the plug during alignment. This preserves the requested cable parameters and bend limit. The historical images above show the earlier 85 mm-high socket at x=0.660 m; consult the latest clearance report and captures for the current fixture. The routing actor initializes the shared translation features of the insertion actor; socket features and rotation outputs then train separately. Initialization provenance and exact checkpoint hashes are saved beside each initialized checkpoint.

## Residual PPO recipe

The optional `rsl_rl_residual_cfg_entry_point` agent embeds a frozen, trained S-curve actor. It then lowers the routed cable, aligns the wrist and approaches the socket with measured pose feedback. PPO learns bounded corrections around that feedback. This is a hybrid controller, not an end-to-end learned insertion policy. The same complete insertion success checks apply. No cable poses are prescribed during control.

The deterministic residual mean is limited to 50 micrometers of target translation and 0.00075 radians of target rotation per control step; within 5 mm of lateral alignment near the port, these limits decrease to 5 micrometers and 0.000075 radians. Gaussian action exploration decreases tenfold after routing. PPO log probabilities use the same state-conditioned distribution as sampling. The routing network, its normalization, feedback parameters and residual network are included in the saved policy and TorchScript export. The recipe assumes the task's default 0.010/0.001 m translation and 0.025 rad rotation action scales.

```bash
uv run isaaclab train --task IsaacContrib-Route-USB-SCurve-Insert-UR10e \
  --rl_library rsl_rl --agent rsl_rl_residual_cfg_entry_point \
  --num_envs 128 --max_iterations 200 --seed 43 --run_name residual_insertion \
  agent.actor.routing_checkpoint=logs/usb_posture_revision/routing_posture_init.pt \
  env.commands.routing.guide_count=3 env.commands.routing.curriculum=false
```

Use `--agent rsl_rl_residual_cfg_entry_point` with the common evaluation script for these checkpoints; the direct recipe remains `rsl_rl_cfg_entry_point`. The source routing file is needed to initialize training, but subsequent checkpoint loading and exported inference include its weights.
