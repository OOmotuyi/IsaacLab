# USB routing with physical contact grasp

Development tasks:

- `IsaacContrib-Route-USB-SCurve-Contact-UR10e`
- `IsaacContrib-Route-USB-SCurve-Insert-Contact-UR10e`

These are separate from the frozen assisted-grasp benchmark in
`logs/usb_controller_comparison/20260922_014003`. Its checkpoints, sources and
results are preserved. Restricted-observation development checkpoints are available,
but no full-task contact-grasp solution or completed benchmark is available yet.

## Mechanical model

The Robotiq fingers must hold the connector through collision contact and friction.
The contact cable class registers no grasp spring or torque attachment. The
connector remains rigidly connected to the terminal cable segment, and the first
segment remains anchored to the table. Opening the fingers must permit release.
A scripted pre-held reset excludes grasp acquisition from the routing task.

The cable retains the existing 600 mm length, 4 mm diameter, 40 Newton VBD segments,
21 g flexible body and 4 g connector. The effective stretch/bend/twist moduli are
150/8/2 MPa, subject to the existing small material randomization. VBD contact
friction uses 0.40 nominal; separate static friction 0.50 and restitution 0.05
remain calibration targets. The 15 mm minimum bend radius is an evaluator/task
limit, not a hard constitutive constraint.

Guide centers vary independently by up to 10 mm in both horizontal axes and yaw
by up to 5 degrees. Openings remain upward, cradle heights fixed, and guide order
is preserved. Layouts are rejected if the guide-center polyline leaves less than
15 mm nominal cable slack. Insertion also checks the final inserted connector-tail
position. This geometric check does not prove dynamic feasibility or satisfaction
of the bend limit throughout a motion.

Two standalone development probes passed four seconds of stationary contact
holding at the task's 30 Hz control rate, followed by deliberate opening and drop
within two control steps. Maximum measured slip was 1.086 mm (routing) and 1.107 mm
(insertion). These probes used a 0.5 s initialization support, removed before the
hold; no retaining force or torque callback was registered during the hold.

The successful calibration used VBD compliant ALM, robot-only MJWarp gravity
compensation, a 0.80 rad closing target, 10 N m auxiliary finger effort limits and
0.002 s Robotiq mimic constraint time constants. Contact pad deformation and
actuator parameters remain simulation approximations, not hardware calibration.
The new task configuration incorporates those settings and measured, named
six-joint preload maps. The maps are applied once at reset; actuator targets and
physical contacts then determine the motion. Reset performs no supported settling
steps, and a selected environment reset does not step the others.

Separate support-free probes passed the same four-second hold, a one-second
10 mm horizontal / 8 mm vertical TCP move, and another one-second hold before
opening. The insertion probe also commanded a three-degree rotation. Maximum
slip was 2.965 mm / 3.046 mm; measured pad indentation was about 0.49 mm / 0.47 mm.
They retained actual housing/pad collision overlap throughout the held frames.
Fresh registered-default probes also passed with clean exits: routing peak slip
2.891 mm, insertion 2.658 mm, and release detected within two control steps.
Reproducible runners, commands and 1080p videos are archived under
`logs/usb_contact_probe/registered_defaults_scurve_022/` and
`logs/usb_contact_probe/registered_defaults_insert_024/`.

Both two-environment indexed-reset checks passed. Resetting environment 1 left
environment 0's physical state, contact telemetry, commands and controller inputs
unchanged, without advancing its clock. After further hold and a 2 mm move, both
grasps remained active. Randomized physical guide transforms matched their targets
to 2.98e-8 m. The checks read Newton `body_q`, including compacted VBD entry data;
its guide `root_state_w` reporting uses stale joint coordinates after VBD stepping
and must not be used to diagnose collision geometry. Evidence is in
`logs/usb_contact_probe/indexed_reset_{scurve,insert}_003/`.

Development records and 1920 by 1080 videos:

- `logs/usb_contact_probe/preheld_canonical_scurve_016/canonical/`
- `logs/usb_contact_probe/preheld_canonical_insert_017/canonical/`
- `logs/usb_contact_probe/preloaded_dynamic_scurve_018/canonical/`
- `logs/usb_contact_probe/preloaded_dynamic_insert_019/canonical/`

These are grasp tests, not successful routing or insertion episodes. The mechanics
and reset checks permit training to begin; they establish no routing success rate.

## Controller information

The new tasks remove full cable geometry from both controller interfaces. The
shared `controller_measurements()` function in the contact task package supplies:

- Six arm joint positions and velocities and one gripper drive position/velocity.
- Tool pose and commanded tool pose.
- Connector rigid-body pose, guide positions/yaws, and the requested guide count.
- Remaining episode time and previous action.
- Socket pose for the insertion task.

The connector pose uses the terminal rigid segment origin, not the housing center
or metal nose; its transform to those points is known asset geometry.

The routing vector has 55 values; insertion has 65. Both PPO actor and critic use
the same group. The PPO recipe uses a recurrent network because cable deformation
and routing state are no longer directly observed. The existing 175/195-value
policies and programmed insertion residual head are incompatible and must not be
silently reused.

**Connector, guide and socket poses currently come from ideal simulator pose
estimates. There is no implemented camera estimator, pose noise, occlusion model
or sensor latency in this interface yet.** Robot readings likewise emulate ideal
encoders. Removing cable geometry does not make this a demonstrated real-sensor
pipeline.

Controller inputs exclude all interior cable points, exact bend radius/stretch,
randomized material values, exact guide completion/retention, dwell counters,
attachment/contact flags, contact-pair forces and task-generated waypoints.
Insertion actions use constant translation and rotation scales; evaluator flags
cannot unlock rotation or switch gains. The direct servo accepts tool-frame goals
only and uses the same measurements. Ground truth remains in private diagnostics,
training rewards and task evaluation. It must not be passed to a model during an
evaluation episode.

The contact force diagnostics are solver measurements, not physical wrist or
tactile sensors. In particular the socket force is a pair-filtered **net resultant
at the final solver substep**, so opposing contacts can cancel and intermediate
peaks are not represented. A 2 N resultant criterion does not establish a 2 N
maximum at every individual contact. Real wrist sensing must include combined
loads, calibration errors and sensor limits.

## Validation commands

From the repository root, the mechanical probe records exact commands, reset
geometry, joint traces, pad loads and slip. Each invocation is one development
probe with no internal retries; it is not a scored routing evaluation.

```bash
DISPLAY=:1 uv run python scripts/demos/usb_cable_routing_contact/probe.py \
  --variant scurve --output_dir logs/usb_contact_probe/manual_hold_release \
  --steps 240 --open_after 120 --images --video
```

This uses the registered preload map and 0.80 rad closing target. An explicit
`--gripper_initial_position` disables that map for scalar-angle calibration. Use a
new output directory for each probe. Once mechanics and controller interfaces pass,
the requested next benchmark is ten fresh matched layouts per task and controller
(40 native episodes total), with varied guide poses and no scored retries.

## Remaining real-sensor work

Replace ideal connector/fixture pose sources with calibrated image-based estimates.
An overhead view can check visible cable occupancy in each guide; a close view can
estimate connector/socket alignment. Full 3D cable reconstruction is not required.
Unseen spans must stay uncertain instead of being filled from simulator state.
Grasp/slip detection should use available gripper telemetry and vision, with a
specified force sensor if required. Give both controllers the same observations,
calibration, update rates and latency; account for Astra reasoning time instead of
silently freezing physical dynamics. Freeze and retrain the resulting interface
before evaluating it as a sensor-based system.

The development direct server uses the shared measurement vector and still pauses
physics between commands. It labels that timing limitation in every observation.
Optional `--images` mode is for inspection; it adds an RGB channel available only
to Astra and must not be described as an identical-input comparison. By default,
camera requests are rejected. Private evaluator records are excluded from the
protocol, but filesystem/tool access must also be restricted for an evaluation;
this server alone is not an operating-system sandbox.

```bash
DISPLAY=:1 uv run python scripts/demos/usb_cable_routing_contact/server.py \
  --variant scurve --run_dir logs/usb_contact_astra/development_001 \
  --seed 920001 --no_randomization
```

## Focused checks

```bash
uv run --extra test python -m pytest \
  source/isaaclab_tasks/test/contrib/test_usb_cable_routing.py \
  -k contact -q
uv run python -m unittest discover \
  -s scripts/demos/usb_cable_routing_contact -p test_server.py -v
uv run --no-project --offline --with ruff ruff check \
  source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing_contact \
  scripts/demos/usb_cable_routing_contact
```

The focused observation/action/config tests and direct-interface tests passed.
Both untrained recurrent actor/critic pairs also passed construction and forward
checks using the training runner's configuration normalization. These checks
validate software contracts, not routing success or grasp stability.

## Fresh recurrent policy evaluation

`evaluate_rl.py` accepts a newly trained generic recurrent checkpoint or its
TorchScript export, plus the saved training `params/agent.yaml`. It resets recurrent
memory before every case, applies the same action clipping as training, and sends
only the 55/65 shared inputs to the actor. Every declared case gets one fresh
simulator process and one native episode. Infrastructure errors stop the run;
completed or interrupted directories cannot be silently overwritten or retried.

```bash
DISPLAY=:1 uv run python scripts/demos/usb_cable_routing_contact/evaluate_rl.py \
  --variant scurve --manifest <ten-case-manifest.json> \
  --checkpoint <contact-training-run/model_N.pt> \
  --training_config <contact-training-run/params/agent.yaml> \
  --output_dir <new-evaluation-directory>
```

Use `--variant insert` for insertion. Use `--policy <exported-policy.pt>` instead of
`--checkpoint` for a recurrent export, and `--initial_reference_dir <paired-results>`
to require exact matched initial physical state and observations. These are command
templates, not claims that a trained contact policy or completed benchmark exists.
The evaluator and direct-server CPU checks passed thirteen tests, including recurrent
checkpoint/export parity, memory reset, legacy-input rejection, no retry after a
worker failure, recording the terminal measurement before automatic reset, and
strict paired-state checks before publishing direct-controller observations.

Native evaluator integration checks also completed with clean exits using the
two-update smoke checkpoints. Routing ran to its 24 s timeout; insertion ended at
12.833 s on the arm-safety limit after one guide. Both retained the physical grasp.
Recorded observations/actions had matching initial-state hashes and the expected
N+1/N shapes: 721 by 55 / 720 by 3, and 386 by 65 / 385 by 6. Commands and evidence
are under `logs/usb_contact_evaluator_smoke/`. These are software checks on
minimally trained policies, not held-out success-rate measurements.

## Training status

Two-iteration, four-environment PPO integration runs passed for both tasks with
native exit code zero. Each saved a finite recurrent `model_1.pt` checkpoint that
loads through the new evaluator, with reproducible memory reset and correct
55/65 input and 3/6 action shapes. Exact commands, paths and source hashes are in
`logs/usb_contact_training/smoke_001.json`.

The separate 32-environment, 200-iteration development training runs completed
with native exit code zero, saving `model_199.pt` and verified
`exported/policy_199.pt` artifacts. Both remained in the one-guide stage; neither
solved the full task. They use the success-driven one-to-two-to-three curriculum.
Exact commands, paths, artifact hashes and completion status are in
`logs/usb_contact_training/development_001.json`. Their logged training outcomes
are not held-out success rates. Further training and direct-controller development
are required before freezing the requested ten-case comparison.
The requested ten fresh matched cases per task have not been evaluated yet.
