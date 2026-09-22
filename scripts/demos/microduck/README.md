# MicroDuck policies in Isaac Lab + Newton

Play all ten released MicroDuck ONNX policies with Isaac Lab articulations and
Newton's GPU MJWarp solver. This extends the original walking demo to standing
recovery, sit/stand, ground picking, both kicks, a forward roll, and skating.

An experimental native walking training task is now available. See the
[training migration status and commands](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/README.md).
The demonstrations below still use the released upstream checkpoints.

The `microduck` extra installs ONNX Runtime for policy inference. Generated USDs,
trained checkpoints, videos, and experiment results live under the ignored `logs/`
directory and are not distributed with this code. The dated result paths below
identify local experiments; supply your own checkpoint paths when reproducing them.
For offline RTX rendering, also enable the `ovrtx` extra.

From the Isaac Lab repository root:

```bash
# Prepare all four robot models, ten checkpoints, and motor parameters.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra importers python scripts/demos/microduck/prepare.py

# Show all ten behaviors consecutively.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/showcase.py --real_time

# Select a behavior and repeat its demonstration.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/play.py \
    --behavior roulade --viz newton_gl --real_time --repeat --duration 0
```

Close the viewer or press Ctrl+C to exit an individual playback. The showcase
starts a separate scene for each behavior. Its default is ten simulated seconds
per behavior. Pass `--behaviors NAME ...` to show a subset.

## Behaviors

Use these exact names with `--behavior`:

| Name | Demonstration | Robot model |
| --- | --- | --- |
| `alpha_walking` | Original walking policy | Walk |
| `velstand` | New walking policy; holds still at zero command | All collisions |
| `alpha_stand` | Hold standing, or recover with `--start_pose prone` | Ground contact |
| `alpha_sitstand` | Stand for 2 s, sit for 4 s, then stand again | Ground contact |
| `alpha_ground_pick` | Reach mouth toward ground, then return upright | Ground contact |
| `ball_kick_left` | Kick a physical ball with the left foot | Ground contact + ball |
| `ball_kick_right` | Kick a physical ball with the right foot | Ground contact + ball |
| `roulade` | Forward roll, then recover to standing | Ground contact |
| `roller` | Skate using four passive wheels | Rollers |
| `roller_crouch` | Crouch on rollers, then return upright | Rollers |

One-shot behaviors settle for 1 s before activation. Ground picking runs for
2.8 s with a 4 s phase clock; roller crouching runs for 3.5 s with a 5 s clock.
Kicks run for 0.5 s. The forward roll runs for **2 s**, matching the upstream
`infer_policy.py` simulator default: the hardware manifest's 1 s window hands
back too early for a complete Newton roll. The standing policy takes over
after ground picking, kicks, and rolls; the roller policy takes over after
roller crouching. Previous raw actions carry across these transitions.

Without `--repeat`, a one-shot runs once and then holds. With `--repeat`, the
robot and ball explicitly reset after each demonstration cycle. Resets are
printed and counted in the metrics; they are not a hidden fall-recovery mechanism.
The showcase uses a prone start for `alpha_stand` to demonstrate getting up.

The following camera keeps the robot visible. Pass `--free_camera` to navigate
the viewer yourself. `--velocity VX VY WZ` sets walking velocity
[m/s, m/s, rad/s]; the default is `0.3 0 0`.
For roller policies the upstream command means **push/brake, unused lateral,
heading error [rad]**, rather than a conventional velocity twist.
One-shots ignore the walking command. `--policy PATH` overrides the selected
behavior's checkpoint while retaining that behavior's model and command contract.

## Ground-pick object interaction

The released ground-pick checkpoint crouches and returns upright. To run that
motion against a visible, freely simulated target:

```bash
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/play.py \
    --behavior alpha_ground_pick --pickup_object --viz newton_gl \
    --real_time --repeat --duration 0
```

The orange block measures 25 × 25 × 30 mm and weighs 10 g. It starts on the floor
79 mm ahead and 6 mm sideways from the initial base, at the mouth's measured
approach point. It has gravity, collisions, and friction, and resets only at
episode boundaries. No attachment or grasp assistance is applied.

This is an object-interaction probe, not a demonstrated grasp policy. The
released model has no jaw-opening action; its soft mouth meshes are visual-only.
Adding a block does not reproduce the real robot's compliant mouth.
With `--metrics PATH` and `--trace PATH`, playback records object position,
mouth distance, and maximum center rise. A bounce or tip can raise the center
without a grasp. Add `--offscreen` when recording with `--viz newton_gl` to
render without opening a window.

## Experimental marker pickup and bin drop

The official robot has a separate mouth servo. Its released policies produce
14 actions and exclude that servo; the runtime applies mouth commands separately.
See the [official policy contract](https://github.com/pollen-robotics/microduck/blob/344925c9f8fa031f85428a305b1e8ec2eaae29c1/docs/design/robotd-design.md#22-one-observation-builder)
and [mouth controller](https://github.com/pollen-robotics/microduck/blob/344925c9f8fa031f85428a305b1e8ec2eaae29c1/duck-control/src/model.rs).

This prototype adds a physical mouth hinge to a separate robot asset and combines
the released ground-pick, standing, and walking policies with a scripted jaw and
task sequencer. **It is not a newly trained pick-and-place policy.**
The marker moves through contact and gravity; it is never attached or teleported.

```bash
# Create marker.usda, bin.usda, and the separate robot_marker USD directory.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra importers python scripts/demos/microduck/marker_assets.py

# Watch one pickup, carry, and release attempt.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/marker_pick.py \
    --viz newton_gl --real_time

# Record and measure the entire attempt.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra video python scripts/demos/microduck/marker_pick.py \
    --viz newton_gl --offscreen \
    --record logs/microduck/marker_pick/validated/marker_to_bin.mp4 \
    --output_dir logs/microduck/marker_pick/validated
```

Assets default to `logs/microduck/marker_pick/assets_lip/`. The marker is
100 mm long, 18 mm in collision diameter, and 12 g. Its starting support is
30 mm high, between the feet. The bin has a 180 × 160 mm interior, a 3 mm floor,
and 45 mm walls; its five separate colliders leave the interior open.

The mouth range follows the official controller: −5° closed to +30° open.
The hinge location, jaw mass split, rigid rubber-pad proxies, retaining lip, and
drive gains are experimental approximations, not a validated model of the real
compliant mouth. Total robot mass and its closed-pose center of mass are preserved.
The fourteen existing policy joints retain BAM control at 200 Hz and inference at
50 Hz; the separate mouth drive is limited to 0.05 N·m. The crouch phase pauses
for one second while the jaw closes.

`trace.jsonl` records marker position/orientation, mouth angle, penetrating
contacts, controller phase, and bin containment. `contact_model.json` records
resolved solver contact settings. `metrics.json` reports three distinct results:
a sustained lift with opposing jaw contacts, release above the bin, and final
settlement. Lift requires 50 mm of center rise and at least 0.3 s of opposed
contact. Settlement requires the entire oriented cylinder inside the bin's
horizontal bounds, its bottom near the floor, speed below 0.05 m/s, and a
0.5 s dwell. These checks do not infer success from the animation or reward.

The fixed scene has completed pickup, transport, and release in Newton.
This is a contact-physics demonstration; randomized reliability, floor-level
pickup, and a newly trained mouth/pick-and-place policy remain unvalidated.

## Record and validate

```bash
# Record every behavior; save per-policy MP4, metrics, and state traces.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra video python scripts/demos/microduck/showcase.py --record

# Headless rollout.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/play.py \
    --behavior ball_kick_left --duration 10 \
    --metrics logs/microduck/kick.json --trace logs/microduck/kick.jsonl

# Repeat a complete roll and recovery twice.
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/play.py \
    --behavior roulade --repeat --duration 14

uv run --extra microduck --extra test python -m pytest scripts/demos/microduck/test_controller.py -q
```

Recordings default to `logs/microduck/showcase/`; open its `index.html` for a video gallery. Metrics include active-policy
durations, root position and orientation, wheel speed, and ball speed/displacement.
`completed` means the rollout loop finished; inspect behavior-specific metrics
and recordings to judge task success. State traces record at 50 Hz.

Validated locally with Isaac Lab 3.0, Newton 1.6.0, MuJoCo Warp 3.12, and an
RTX PRO 6000: walking, VelStand locomotion, prone recovery, sit/stand, ground-pick
handoff, both physical-ball kicks, a full forward roll with standing recovery,
skating, and roller crouching. The tests cover command timing, raw action history
under roller scaling, motor/friction equations, and damping persistence after
Newton's first physics step. The damping regression fails with the original
direct solver write and passes with the persistent Isaac Lab property write.

This ports **pretrained policy playback**. The upstream rewards, PPO training,
terrain curricula, randomization, and 33 registered training variants are not
ported. Locomotion undertracks the requested command and heading drifts; roller
speed also differs from CPU MuJoCo. These demonstrations are not a robustness
evaluation or exact trajectory-parity claim. The converter's current
self-collision setting remains disabled; imported body geometry still contacts
the floor and ball. The original walking-only fall guard remains enabled for
`alpha_walking`; recovery and acrobatic policies intentionally permit low or
inverted poses.

## Physics and policy contract

- 50 Hz inference; 200 Hz physics and motor control.
- 61 inputs: body angular velocity (3), projected gravity (3), joint offsets
  (14), joint velocities (14), previous raw actions (14), and commands (13).
  Head/body command slots remain zero; twist slots carry behavior-specific commands.
- Joint names map explicitly between ONNX, Isaac Lab, and MJWarp. The four
  passive wheels are excluded from servo observations and actions.
- STAND2 reference pose; joint target scale 1.0 rad for feet and 0.8 rad for
  roller policies. The pinned release manifest's roller scale takes precedence
  over the generic ONNX metadata value of 1.0.
- BAM M6 XL330 voltage controller, back-EMF, reflected armature, viscous and
  load-dependent dry friction, and the Stribeck effect. Supply is 7.4 V,
  firmware gain 200, voltage sag gain 0.1, and no current limiter.
- Viscous friction is written through Isaac Lab's persistent passive-damping API.
  A direct MJWarp write is overwritten by deferred model synchronization.
- Dry friction uses MJWarp's constraints. Private solver access for bias and
  constraint forces is isolated in `controller.py`.
- Rollers retain 0.0001 kg·m² wheel armature and 0.003 N·m bearing resistance.
  Tire collision hulls retain the source geometry: the default 64-vertex
  reduction made these 15 mm-radius tires polygonal and impeded rolling.
- The ball has radius 0.035 m, mass 0.015 kg, and hollow-sphere diagonal inertia
  1.225e-5 kg·m². It is placed 0.09 m ahead and ±0.042 m sideways in the
  robot's heading frame when a kick begins.
- The ONNX graph retains its observation normalizer.

The importer creates out-of-scope physics-material targets inside geometry
prototypes. Preparation removes those targets and binds the shared material at
the robot root. Root placement is written explicitly to avoid adding the MJCF
body offset twice. Importer warnings about absent PhysX drive gains are expected:
playback uses explicit BAM efforts and zero-gain implicit actuator channels.
On this machine the viewer falls back to CPU copies when CUDA/OpenGL
interoperability is unavailable; simulation still runs on CUDA.

The base environment supplies ONNX Runtime. For a minimal environment without
it, add `--with onnxruntime`. Conversion uses the maintained Isaac Lab importer
with asset transformation and multi-physics payload generation enabled.

## Sources and licensing

Downloads and generated assets stay in the ignored `logs/microduck/assets/`
directory. `sources.json` records the pinned revisions:

- [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl/tree/cb70b792312d559a4da09064d92009079671815f):
  robot models and reference playback. Code is Apache-2.0; the upstream README
  identifies the 3D models as Creative Commons BY-SA-NC.
- [MicroDuck policies](https://huggingface.co/pollen-robotics/microduck-policies/tree/1b56c396825c052a4e26e95cf2b8d8298af9e9b4):
  all ten released ONNX checkpoints and deployment manifest, Apache-2.0.
- [BAM](https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967):
  lockfile-pinned motor fit and equations, Apache-2.0.

Robot meshes and policy weights are downloaded rather than vendored.

## NVIDIA Endeavor atrium

The standalone Endeavor scene follows the black faceted central structure,
diagonal braces, white roof trusses, skylights, and open ground floor in
[Connie Zhou's published Endeavor atrium photograph](https://interiordesign.net/projects/nvidia-campus-by-gensler-2018-best-of-year-winner-for-large-tech-office/).
It is an approximate reconstruction from photographs, not a surveyed model or
an official NVIDIA asset. Dimensions are estimates; no AI-generated imagery or
photograph textures are used in the USD.

```bash
# Create the environment-only asset; existing output is never overwritten.
uv run --extra microduck python scripts/demos/microduck/endeavor.py build

# Run the existing alpha_walking.onnx and robot_walk.usda in Newton MJWarp.
uv run --extra microduck python scripts/demos/microduck/endeavor.py record --duration 8

# Render the environment and the recorded motion with path-traced RTX lighting.
uv run --extra microduck --extra ovrtx python scripts/demos/microduck/endeavor.py render_scene
uv run --extra microduck --extra ovrtx --extra video python scripts/demos/microduck/endeavor.py render
```

Outputs are in `logs/microduck/endeavor/`. The self-contained `endeavor.usda`
contains only architecture, materials, lights, and an overview camera. It uses
meters and Z-up, with a continuous collision floor at `z=0`. The validated
walking area is `x=[-12,12]`, `y=[-12,8]` meters. Architecture outside that area
is visual-only; this is a flat-floor walking demonstration, not an obstacle
navigation environment. Floor friction matches the existing playback demo.

`walking.usda` references the existing robot and stores its measured link poses
at 50 Hz. Newton runs at 200 Hz with the existing BAM controller and ONNX policy;
offline rendering does not change physics timing. The PNGs and
`microduck_endeavor.mp4` are RTX renders of those recorded poses. The default
video is 1920 × 1080 at 25 FPS, with 64 path-tracing samples per pixel.
`metrics.json` records completion, minimum root height, start/end positions,
and input checksums. Recording aborts if the robot
falls or leaves the clear walking area, without silently resetting it.

Use a new `--output_dir` for another build or recording. `--policy` accepts an
existing ONNX export with the same 61-observation, 14-action walking contract.
The robot, its referenced mesh files, and policy weights remain unchanged.
The animated USD depends on the existing robot asset directory; only the
environment USD is self-contained.


## Compare native and pretrained walking

Record synchronized videos of an Isaac Lab-trained checkpoint and the released
MuJoCo-trained walking policy, both running in Isaac Lab + Newton:

```bash
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra video python scripts/demos/microduck/compare.py \
    --checkpoint logs/rsl_rl/microduck_velocity/2026-09-18_16-52-41_native_walking_v2/model_2999.pt \
    --velocity 0.3 0 0 --duration 20
```

Open `logs/microduck/comparison/index.html` to play both clips together.
The page shows frame-synchronized commanded and actual forward/sideways velocity
[m/s], turn rate [rad/s], and ground speed [m/s and km/h]. Pause or scrub the
videos or click a plot to inspect a moment. Playback defaults to real time;
the playback-rate selector does not change the measured physical speeds.
The summary compares full-rollout means and tracking RMSE, including startup.
Ground speed is the per-frame magnitude of world-horizontal root-link velocity;
it includes sideways sway, so it can exceed mean forward velocity.
Both recorders save these samples and summary measurements in their JSON files.
Existing recordings without telemetry must be re-recorded to use this page.
Both policies receive the same command but retain their corresponding playback
setups; this is a visual comparison, not a controlled training benchmark.
The transferred policy stays nearly stationary at 0.15 m/s in the current setup;
the 0.30 m/s command produces walking.

For a live view of the native checkpoint with a following camera, use
`evaluate.py --checkpoint PATH --metrics PATH --num_envs 1 --duration 600 --viz newton_gl --real_time`.

For repeatable checkpoint comparisons, `evaluate.py --seed 123` fixes the evaluation
seed. Add `--randomized` to retain training sensor noise, motor/inertial/friction
randomization and pushes. Commands remain fixed and the curriculum is disabled
in both evaluation modes. Metrics record the seed and whether randomization was enabled.

Add --offscreen to evaluate.py when recording with --viz newton_gl to render
without opening an interactive window.

## Child bedroom and deformable sock cleanup prototype

Build the room, open bin, and hollow textile sock:

```bash
uv run --extra microduck python scripts/demos/microduck/room_assets.py
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra video python scripts/demos/microduck/room_cleanup.py \
    --mode cleanup --duration 65 --visualizer newton_gl --offscreen \
    --output_dir logs/microduck/room_cleanup/recording \
    --record logs/microduck/room_cleanup/recording/cleanup.mp4
```

This uses the previously prepared marker/mouth assets under
`logs/microduck/marker_pick/assets_lip/`, the released posture policies under
`logs/microduck/assets/`, and the native walking ONNX export from
`logs/rsl_rl/microduck_velocity/2026-09-18_16-52-41_native_walking_v2/exported/`.
The default is the existing native walking v2 export. Use `--walking_policy`
to select another compatible export. An export of the existing optimized
checkpoint is also available under the 2026-09-19 optimization run; its maximum
absolute error against the PyTorch actor was below 1e-6 on 32 inputs.
The command is a development rollout: check `metrics.json` for actual completion.
Separate marker placement and sock placement have been validated locally,
including an upright robot at the end of the isolated sock run. The continuous contact-only
two-object cleanup remains unvalidated because cloth grasp retention is unreliable.
Use `--first_object sock` to collect the cloth before returning for the marker.

The 2.8 × 2.4 m cutaway room contains a bed, chair, marker, sock, and bin.
The marker starts on a 3 cm support; the sock rests across two supports outside
the beak. `--sock_shape folded` retains an experimental rolled variant.
Floor pickup and perception-driven discovery of scattered objects have not been
validated. The sock is a 12 cm hollow triangular cloth mesh, with an open cuff,
a closed toe, and no rigid interior. Default contact-only mode has no grasp attachment. Its material and the
prototype mouth geometry are estimates, not measurements of real hardware.

Isaac Lab couples Newton MJWarp for the robot/marker to Newton VBD for cloth.
The cloth solver uses full-surface rigid contact, self-contact, and signed-distance
fields cooked for the robot meshes. Rigid/rigid pairs are handled by MJWarp.
Physics advances at 200 Hz with four solver substeps; policy inference runs at
50 Hz. The native walking export receives the three-step motor delay used for
its deterministic evaluation. `--coupling_relaxation` controls cloth-force feedback
(default 1.0). A value of zero is a diagnostic approximation with no cloth force
feedback to the robot; it does not attach the cloth to the jaw.

The jaw, sock carrying head pose, and pickup/navigation sequence are scripted,
with released pickup posture policies and Isaac Lab-trained walking as low-level
skills. Walking remains active through release, with the held head pose supplied
as a command. The native policy controls all 14 body joints during marker
transport, empty navigation, and release. Sock transport holds the four head
joint targets while the native policy controls the legs. The experimental
`--align_beak` option adds measured head-yaw correction, with a CPU/GPU frame
check; it remains disabled by default because it reduced pickup reliability.
The 52 cm wide bin has a 6 mm loading lip for foot clearance.
The sock route backs away from its supports before moving sideways into a clear
carrying lane. Cloth lifting begins after sustained upper/lower beak contact;
upper contact
may be on the pad or the front of the original upper shell.
After alignment, a separate standing phase prevents repeated switches between
walking and pickup preparation. This is **not a new
trained cleanup policy** or an RL training environment. In default contact-only mode,
objects are not attached or teleported during the rollout. Completion requires measured lift, release,
and settlement of each entire object inside the bin, with the robot upright
at the end. Wall contact is allowed within a 1 mm numerical tolerance. `--mode marker`,
`--mode sock`, and `--mode navigate` isolate development phases.

Each run writes the composed `scene.usda`, `trace.jsonl`, `metrics.json`, and
`sock_nodes.npz` (measured cloth vertex positions at 10 Hz). Rendered runs also
write `room_overview.jpg`. Create a local video review page with:

```bash
uv run --extra microduck python scripts/demos/microduck/cleanup_report.py logs/microduck/room_cleanup/recording
```

Asset-only USDs are in
`logs/microduck/room_cleanup/assets/`; the composed robot scene references the
existing prepared robot assets.

The cloth contact design follows [Newton collision documentation](https://newton-physics.github.io/newton/latest/concepts/collisions.html) and the maintained Franka cloth example in this repository.

### Explicitly assisted cloth-grasp demo

Add `--assisted_grasp` to the room cleanup command to hold a small sock patch
in the beak frame while the jaw is closed. This is an assisted demonstration,
not a trained cleanup policy. Contact-only mode remains the default.
The sock-first return route backs along its carrying lane before moving across
to the marker, with a standing-policy handoff before backward walking.
Marker-first assisted transport remains experimental after a
numerical failure in one long rollout.

The attachment engages only during sock pickup, with the jaw commanded to a
0.18 rad opening, measured jaw angle below 0.25 rad, jaw speed below 0.25 rad/s,
and a cloth node within 14 mm of the pinch point or beak tip. At most two nearby
nodes are captured without an initial jump, then translated together into the jaw aperture over
0.35 s while preserving their separation and clearing the lower lip. The remaining nodes continue through VBD and can fold around the jaws.

Beak/cloth collisions remain enabled throughout pickup, carry, and release.
The assisted beak uses a VBD proxy with 1000 times its normal mass and no cloth
contact-force feedback to the robot, avoiding conflict with the imposed cloth
attachment. Other robot/cloth contact feedback remains enabled. This is an
explicit approximation; neither the attachment force nor the beak/cloth
reaction force loads the robot. Opening the jaw immediately restores dynamic
motion and the patch's measured release velocity. The marker is never attached
and its rigid contacts are unchanged.

In assisted mode the native walking policy controls all 14 body joints during
transport with neutral head commands and a 0.32 m/s command limit. The attachment
follows the measured head pose, so the carrying head joints do not need to be frozen.

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck --no-sync python scripts/demos/microduck/room_cleanup.py \
  --mode cleanup --first_object sock --assisted_grasp --duration 70 \
  --visualizer newton_gl --offscreen \
  --output_dir logs/microduck/room_cleanup/assisted_cleanup \
  --record logs/microduck/room_cleanup/assisted_cleanup/cleanup.mp4
uv run --extra microduck --no-sync python scripts/demos/microduck/cleanup_report.py \
  logs/microduck/room_cleanup/assisted_cleanup
```

The recording has a persistent **ASSISTED DEMO** banner and an attachment-active
indicator. Metrics include attachment events, selected node IDs, engagement
distance, total active time, and measured node tracking error. Physical contact
flags remain separate. Completion still requires both objects to settle fully
inside the bin and the robot to remain upright.

The earlier [assisted_cleanup_10 recording](../../../logs/microduck/room_cleanup/assisted_cleanup_10/index.html)
completed both placements but held the cloth in front of the beak with its
collisions disabled. It has a visible grasp gap and mesh penetration and does
not validate grasp clearance. The current implementation seats the patch inside
the aperture and retains collisions. Contact-only completion of both objects
in one run remains unvalidated.

The corrected [assisted_clearance_04 recording](../../../logs/microduck/room_cleanup/assisted_clearance_04/index.html)
completed both placements in one 46.725 s rollout, ending upright. Two nodes were
attached from 1.91 s to 14.44 s; all nodes were dynamic after release. Recording
and simulation took 528.0 s of wall time. The
[before/after close-up](../../../logs/microduck/room_cleanup/assisted_clearance_04/grasp_comparison.jpg)
shows the cloth between the jaws.

The [clearance audit](../../../logs/microduck/room_cleanup/assisted_clearance_04/beak_clearance.json)
sampled cloth vertices and triangle centroids at 10 Hz during attachment,
including seating, against the source beak meshes and pad/lip boxes. The largest
sampled overlap was 0.994 mm; no sample exceeded 1 mm. This is a sampled check,
not a proof of zero intersection. The
[validation summary](../../../logs/microduck/room_cleanup/assisted_clearance_04/validation.json)
also checks final containment, robot posture, release, and source hashes.
This validates the fixed scene and starting arrangement, not a success rate
across randomized trials.


## Fine-tuned cleanup body skills

The room runner accepts `--standing_policy /path/to/stand.onnx` and
`--pickup_policy /path/to/pickup.onnx` for the new Isaac Lab + Newton PPO exports.
See the [native skill training workflow](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/README.md#standing-and-pickup-ppo-fine-tuning).
These train standing and reaching body motion. The room sequence, jaw control and
optional sock attachment remain scripted. Policy paths, hashes and training metadata
are saved in each run's `metrics.json`.

Use `--sock_close_time` to adapt the scripted jaw-closing time [s] after pickup
begins to an overridden body actor. The default remains 0.7 s. This parameter
does not change the assisted attachment gates or physical collision settings.

## Textured bedroom and lidless wire basket

`room_assets.py` now builds a tapered black wire basket based on the local
`bin.png` reference, without a lid. Its rim is 0.130 m above the floor, its
outer rim diameter is 0.191 m, and its solid bottom is 0.004 m thick.
The crossing side wires, six intermediate horizontal rings, rim, and bottom have
actual collision geometry. The horizontal wire gaps are smaller than the marker's
diameter, preventing it from escaping through the weave. Dimensions in
`bin_geometry.json` also drive the cleanup controller's containment checks and
release clearance. Older room directories without that file retain the original
rectangular tray behavior. The isolated marker demo's bin is unchanged.

The basket approach aligns the held marker laterally before forward transport.
A scripted 0.4 s neck/head transition extends the beak by approximately 14 mm
without tilting it. The sock must be centered farther over the opening before release;
the marker uses a conservative cylindrical hull so empty corners of its bounding
box cannot prevent a valid placement. Both require 6 mm lateral and 8 mm vertical
clearance at release. These adjustments use the existing body actors; no new
policy training was performed for the basket.

For this wire-basket setup, the VBD view uses marker contact stiffness 1000 N/m
and damping 0.5 N*s/m. Its material arrays are copied before changing the marker
entries, so MJWarp retains the marker's rigid-grasp `solref` settings. The imported
rigid settings resolved to 10000 N/m and 200 N*s/m in the VBD view and caused
unstable marker impacts on the sock. Recorded-state impact tests retained the
marker after the correction. Two-way marker/cloth feedback remains enabled.

The [validated basket cleanup](../../../logs/microduck/bedroom_basket_2026-09-21_031222/validation_08/index.html)
placed both objects in 44.985 s of simulated time and ended upright. The run took
9.74 minutes of wall time. Release clearance above the rim was 36.8 mm for the
sock and 34.8 mm for the marker. The recorded-state and source checks passed;
this validates the fixed scene, not a randomized success rate. The sampled beak
clearance audit found at most 0.79 mm overlap, and sampled robot/basket contacts
had at most 1.26 mm solver penetration. The sock remained explicitly assisted.

The textured assets are under
`logs/microduck/bedroom_basket_2026-09-21_031222/assets/`. They include generated
oak and cotton albedo textures, USD Preview Surface materials, rounded furniture
and bedding, curtains, and daylight/fill lights. Furniture retains its existing
collision proxies. To rebuild this room, retain its `textures/` directory and run:

```bash
uv run --extra microduck python scripts/demos/microduck/room_assets.py --photorealistic \
  --output_dir logs/microduck/bedroom_basket_2026-09-21_031222/assets
```

Capture real policy-driven rigid-body poses and every cloth vertex at 50 Hz,
then render them offline with RTX. This keeps renderer performance separate
from Newton physics and control timing:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck python scripts/demos/microduck/room_cleanup.py \
  --mode cleanup --first_object sock --assisted_grasp --sock_close_time 0.55 \
  --duration 70 --visualizer none --record_states \
  --room_dir logs/microduck/bedroom_basket_2026-09-21_031222/assets \
  --standing_policy logs/microduck/cleanup_native_training_2026-09-21_021025/exports/alpha_stand_conservative_50.onnx \
  --pickup_policy logs/microduck/cleanup_native_training_2026-09-21_021025/exports/alpha_ground_pick_conservative_50.onnx \
  --output_dir logs/microduck/bedroom_basket_2026-09-21_031222/recording

OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck --extra ovrtx --extra video python scripts/demos/microduck/room_render.py \
  --run_dir logs/microduck/bedroom_basket_2026-09-21_031222/recording \
  --output logs/microduck/bedroom_basket_2026-09-21_031222/recording/cleanup.mp4
```

`--scene_only --overview --room_dir <assets> --output <preview.png>` renders the
room without a rollout. `--run_dir <recording> --time <seconds> --output <frame.png>`
renders a measured simulation frame. Add `--closeup` for a tighter view of the
beak, objects, and basket. Animation time codes use seconds explicitly.
The movie retains an assisted-grasp label: standing, pickup and walking use the
existing Isaac Lab-trained body actors; jaw timing and task sequencing are
scripted, and two sock nodes are temporarily attached at the beak. Rendering
adds no grasp support. Check `metrics.json` for actual placement success,
`release_rim_clearance_m`, and `maximum_robot_bin_penetration_m`.

To capture the same measured run with Newton's native GUI (requires a working
`DISPLAY`), use:

```bash
uv run --extra microduck --extra video python scripts/demos/microduck/room_newton_replay.py \
  --run_dir logs/microduck/bedroom_basket_2026-09-21_031222/validation_08 \
  --output logs/microduck/bedroom_basket_2026-09-21_031222/validation_08/cleanup_newton_gui.mp4
```

This records the actual `ViewerGL` controls, recorded task phase, measured speed,
and sock-assistance status. It replays measured robot/marker poses and all 401
cloth vertices; it does not run new physics or policy inference. The RTX and
Newton GUI videos use the same 45-second recording at 1920×1080 and 25 fps.
The adjacent JSON records input hashes, body mappings, and sampled replay errors.
`--frames 3` provides a short capture check. Native GUI statistics describe the
replay model; the cloth is drawn from recorded vertices as a dynamic mesh, so it
does not appear in the model's particle or triangle counts.
