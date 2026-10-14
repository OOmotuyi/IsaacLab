# UR10e–Robotiq S-curve cable routing

## Robot posture revision

Current posture-corrected policy: **28/100** fresh randomized successes. Reliability remains limited. Updated videos, policies, measurements and exact commands are in the [posture results report](../../../logs/usb_posture_revision/final/REPORT.md).

The robot base is now at `(0.10, -0.85, 0.0)` m. The tabletop is 1.1 × 1.6 m to support it. Reset selects a measured IK branch with the wrist above the hand; the cable, guide centers and socket retain their existing positions. Routing still uses three Cartesian actions and a fixed grasp orientation. Joint targets are limited to a 2 rad/s equivalent increment and to joint position limits; measured joint velocity can differ from this command bound. The TCP target stays at least 45 mm above the tabletop. Rewards discourage wrist folding, excess roll and wrist heights below 140 mm; wrist heights below 90 mm or invalid joint states terminate the episode.

S-curve seating targets are `(0.60, -0.15, 0.075)`, `(0.60, 0.10, 0.060)` and `(0.64, -0.23, 0.065)` m. The insertion controller chooses between the two USB-C symmetries using rotation distance and gripper approach direction, favoring the palm above the socket. Its episode limit is 60 s. Evaluation records terminal minimum wrist height, maximum joint speed, wrist roll and joint-limit margin. Wrist height measures the `wrist_1_link` frame origin, not the lowest robot surface. Registered collision steps count contact candidates and do not measure penetration; the separate native clearance audit computes contact separation.

The posture-refinement actor was initialized from 12 successful physical trajectories (2,920 state/action samples) and refined with PPO. Use `agent.actor.class_name=isaaclab_tasks.contrib.usb_cable_routing.models:FixedNormalizationModel` when continuing this initialization: it preserves the demonstrated observation scaling while learning policy weights. Standard actor normalization adapted too rapidly and a development checkpoint regressed to 0/16. The conservative continuation uses action standard deviation 0.03, fixed learning rate 0.00003 and entropy weight 0.0001. Playback uses the standard MLP checkpoint format.

Checkpoints trained with the earlier base position require adaptation even though observation dimensions remain 175/195. Historical results below describe their original source snapshots. Posture-refinement artifacts, evaluations and playback commands are saved under `logs/usb_posture_revision/`; the original results remain under `logs/usb_parallel_training/final/`.

Task: `IsaacContrib-Route-USB-SCurve-UR10e`. This is a separate environment. Before the robot posture revision, the selected trained PPO actor completed 46/100 new randomized configurations; it is not reliably solved. A successful simulator video, checkpoint, TorchScript policy, full evaluation and exact playback commands are in [the results report](../../../logs/usb_parallel_training/final/REPORT.md).

All four cable tasks now share the corrected rigid strain-relief attachment and side grasp. Flexible cable–gripper contacts remain enabled; only the rigid plug contact is substituted by a breakable compliant grasp. The shared cable uses 40 segments, a 21 g body plus 4 g plug, 150/8/2 MPa effective moduli and 0.40 active friction. Static friction 0.50 and restitution 0.05 are calibration targets. Earlier straight/zigzag checkpoints are historical results with different mechanics and have not been revalidated against this version.

| Guide / color | Offset from fixed endpoint [mm] | Passage yaw [deg] | Cradle height [mm] |
|---|---|---|---|
| 1 / red | (120, 0) | +10 | 10 |
| 2 / yellow | (260, 100) | 0 | 10 |
| 3 / green | (400, 0) | −10 | 10 |

All guides remain upright and open upward. Yaw rotates the passage direction around world +z. Neighboring centers are 140 mm apart along x and 172.0 mm apart in the tabletop plane. The fixed endpoint is `(0.18, -0.15, 0.022)` meters in the environment, and the table surface is z=0.

The red, yellow, and green U-guides (in routing order) have an 18 mm clear channel, 20 mm passage length, and 40 mm wall-top height. Each upright ends in a fixed horizontal lip turning inward at exactly 90 degrees. Each lip projects 4 mm into the channel and is 4 mm thick, leaving a 10 mm entry opening for the 4 mm cable. The lips are physical box colliders; success requires the cable to sit below their 36 mm-high undersides. This is narrower than the earlier task's 42 mm channel. The guide floor is 10 mm above the table; the 4 mm cable center rests approximately 12 mm above it. The requested center path through all three guides and onward to `(0.64, -0.23, 0.065)`, including a 30 mm straight approach through each guide along its yaw direction, is about 0.593 m, leaving about 7 mm within the 600 mm cable for smooth bends and slack. The held-end targets were revised using successful Newton rollouts. Six complete routes subsequently replayed successfully with sampled bend radii above 19 mm.

The appearance retains the blue cable and molded housing from `cable_route.png`, with the free connector now approximating the requested USB-C type. Its housing is 10 × 6 × 18 mm; the hollow metal shell is approximately 8.4 × 2.6 × 7 mm, with faceted rounded ends and a dark tongue. Dimensions are estimates, not measurements from the photograph. The housing is pre-grasped by the existing UR10e/Robotiq 2F-85. Newton MJWarp/VBD coupling and the breakable compliant pre-grasp are reused. The [shared physical assumptions](../usb_cable_routing/README.md) apply except for the S-curve-specific cable parameters below.

Crossing and tool-incursion checks use each guide's rotated coordinate frame. Crossing tests account for the cable radius and both guide faces. Observations include each guide's position and yaw sine/cosine (175 values total). The independent PPO configuration uses a 1→2→3 guide curriculum. Training artifacts and exact commands are recorded under `logs/usb_parallel_training/`; interim checkpoints do not establish task success. The scene preview holds the reset pose with zero Cartesian increments.

## Preview commands

Run from the repository root:

```bash
export OMNI_KIT_ACCEPT_EULA=yes
export PXR_WORK_THREAD_LIMIT=1

# Generate this task's guide USD if setting up a fresh checkout.
uv run python -c 'from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import generate_assets; generate_assets()'

# Open the actual environment with all three guides, without training.
uv run python scripts/demos/usb_cable_routing_scurve/preview.py

# Save an untrained six-second physics preview and five camera views.
uv run python scripts/demos/usb_cable_routing_scurve/preview.py \
  --headless --steps 215 --video \
  --output_dir logs/usb_cable_routing_scurve/physics_preview

# Focused geometry and native simulation checks.
uv run --extra test python -m pytest \
  source/isaaclab_tasks/test/contrib/test_usb_cable_routing.py -q
```

The updated physics capture and measurements are saved in `logs/usb_cable_routing_scurve/physics_preview/`. An interactive run without `--output_dir` writes to `logs/usb_cable_routing_scurve/preview/`. The preview window runs until closed; it does not load a policy or launch an optimizer. Training was authorized after scene review; preview commands still run without an optimizer.

The middle guide was shifted an additional 50 mm sideways to `(260, 100)` mm, and the final holding target was brought closer to preserve cable slack. The guide layouts remain independent; attachment and cable material fixes are shared. Preview artifacts below predate the common side-grasp update.

[Overview](../../../logs/usb_cable_routing_scurve/physics_preview/overview.png) · [Top view](../../../logs/usb_cable_routing_scurve/physics_preview/top.png) · [Inward 90-degree lips](../../../logs/usb_cable_routing_scurve/physics_preview/lips.png) · [Cable tip](../../../logs/usb_cable_routing_scurve/physics_preview/tip.png) · [Preview video](../../../logs/usb_cable_routing_scurve/physics_preview/preview_0000.mp4) · [Measured geometry](../../../logs/usb_cable_routing_scurve/physics_preview/scene.json)

Geometry code now uses `LIP_OVERHANG` and `LIP_UNDERSIDE`; the former `FLARE_OFFSET` and `FLARE_START` symbols remain as deprecated aliases.

## Cable physical properties

The nominal estimates are centralized in [`parameters.py`](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing/parameters.py). These values are intended for calibration from measured video; no video-based fit has been performed.

| Property | Value |
|---|---|
| Flexible length / outer diameter | 600 mm / 4 mm |
| Discretization | 40 Newton VBD elastic rod segments of 15 mm |
| Flexible body mass | 21 g; 0.525 g per segment, including the fixed first segment |
| Free USB-C connector mass | 4 g, added to the final segment |
| Total simulated cable assembly mass | 25 g; excludes the fixed table fixture |
| Effective bulk density | 2785.21 kg/m³; derived from the cylindrical body volume |
| Stretch / bend / twist moduli | 150 / 8 / 2 MPa |
| Shear modulus | Unspecified; installed legacy material conversion inherits the 150 MPa stretch modulus |
| Active VBD Coulomb friction | 0.40 |
| Static friction calibration target | 0.50; not separately applied by installed VBD |
| Restitution calibration target | 0.05; not applied by installed VBD |
| Minimum allowed bend radius | 15 mm, enforced as a task failure limit |
| Solver | Newton VBD, 12 iterations, 8 substeps per 1/120 s physics step |

The builder normalizes each segment's mass and inertia before adding connector shapes. Connector component densities are scaled together to add exactly 4 g, including their center-of-mass and inertia contributions. They are effective mass allocations rather than measured plastic/metal densities. The final body therefore has 4.525 g, and the other 39 bodies each have 0.525 g.

The cable moduli are converted by the installed USD rod importer using cross-sectional area `A = πr²`, second moment `I = πr⁴/4`, polar moment `J = 2I`, and joint dual length `L = 15 mm`: stretch/shear stiffness `EA/L = 125663.71 N/m`, bend stiffness `E_b I/L = 0.00670206 N·m/rad`, and twist stiffness `G_t J/L = 0.00335103 N·m/rad`. The native test checks these values in the built model, together with positive inertia and the mass budget in two parallel environments.

The installed VBD contact implementation uses one Coulomb coefficient for sticking and sliding. As agreed, 0.40 is active; 0.50 static friction and 0.05 restitution are recorded as calibration targets rather than claimed as active physics. Normal contact stiffness remains 20000 N/m and damping 20 N·s/m; this damping is not a calibrated restitution model.

VBD does not provide a hard angular limit for these rod joints. The task estimates each local radius as `R = mean_adjacent_chord_length / (2 sin(turn_angle/2))`, exact for equal chords sampled from a circle. At each 30 Hz control step, a radius below 15 mm (with 0.1 µm numerical tolerance) terminates and penalizes the episode and prevents success. Curvature is also penalized as the radius approaches the limit. This detects invalid configurations; it does not physically prevent a tighter bend within a solver step. The preview records the minimum sampled radius, and terminal measurements survive reset.

The first cable segment is fixed. The robot's breakable pre-grasp uses a 400 N/m spring and 0.03 N·s/m damping; it releases above 6 N or 15 mm displacement. These limits describe grasp retention, not the cable's physical breaking strength.

Training configuration randomization scales the elastic stiffnesses by 0.85–1.15, samples cable-shape friction over 0.36–0.44, and perturbs the initial centerline by up to 5 mm away from its fixed endpoints. Mass and the 15 mm bend limit remain fixed. Randomization is disabled in the scene preview. The straight and zigzag tasks now use these common mass and material estimates too; their historical checkpoints require reevaluation.

For calibration, use a known length scale and frame timing in the video. Compare centerline sag and transverse deflection under known loads to tune bend modulus; use axial displacement under known tension for stretch modulus, and visible orientation markers under known torque for twist modulus. Sliding trials can tune the active friction coefficient. Record stick/slip and rebound observations against the separate static-friction and restitution targets; fitting those independently requires additional solver support. Masses and geometry should stay fixed during stiffness fitting. Individual copper strands, jacket layers, plastic deformation, and fatigue are not modeled.

Historical preview validation before the shared side-grasp update: all 13 then-existing cable-task tests passed. The updated preview ran 215 control steps (7.17 simulated seconds) without an automatic reset or dropped grasp. The minimum sampled bend radius was 19.46 mm, the maximum sampled grasp force was 1.025 N, and the maximum cable joint-gap ratio was 0.00111. The built model measured 21.0 g flexible mass plus 4.0 g connector mass. This is a stationary pre-grasp preview, not a trained routing rollout or success-rate evaluation.

## Historical training method before the posture revision

PPO uses full simulator state, 128 parallel environments, a 128–128–64 actor/critic, 32-step rollouts, clipping 0.2, five optimization epochs and initial learning rate 0.0003. Guide-count curriculum advances at 80% mastery. Shaping measures the cable approaching the active guide opening, with a smaller free-end waypoint term. Crossing success still checks both guide faces, order, retention, grasp, stretch and the 15 mm bend limit. The final-guide lift/seating targets are `(0.64, -0.23, 0.13)` and `(0.64, -0.23, 0.065)` m.

Sparse exploration stalled, so six physically replayed successful trajectories initialized the actor with behavior cloning; a whole trajectory was held out from that fit. PPO refinement then continued with the configured physical randomization. Final policy evaluation uses the learned actor alone. The first development check of the initialized actor achieved 3/8 full routes; this is not the requested final 100-configuration evaluation.

The reproducible initialization utility is `scripts/demos/usb_cable_routing_scurve/initialize_from_demonstrations.py`. Exact experiment commands, source snapshots and intermediate results are under `logs/usb_parallel_training/`.
