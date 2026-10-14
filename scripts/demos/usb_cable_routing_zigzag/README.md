# UR10e–Robotiq USB zigzag routing

All four cable tasks now share the corrected rigid strain-relief attachment and side grasp. Flexible cable–gripper contacts remain enabled; only the rigid plug contact is substituted by a breakable compliant grasp. The shared cable uses 40 segments, a 21 g body plus 4 g plug, 150/8/2 MPa effective moduli and 0.40 active friction. Static friction 0.50 and restitution 0.05 are calibration targets. Earlier straight/zigzag checkpoints are historical results with different mechanics and have not been revalidated against this version.

Task: **`IsaacContrib-Route-USB-Zigzag-UR10e`**. This separate manager-based task reuses the straight-routing task's combined UR10e/Robotiq 2F-85 USD, 0.6 m × 4 mm cable, USB connectors, table, Newton MJWarp/VBD coupling, and breakable pre-grasp. Its policy, logs, and evaluation use a separate namespace.

The three guide centers are `(0.30, -0.11)`, `(0.44, -0.19)`, and `(0.56, -0.11)` meters in table coordinates. Adjacent guides alternate laterally by 80 mm. Each channel is 42 mm wide, so a straight cable cannot satisfy all three crossings. The guides keep their original orientation along the table's x axis. The center-to-center route from the anchor through these guides to `(0.68, -0.15)` is approximately 0.56 m.

The cable must pass through both faces of each guide in order and remain seated in all active guides for 12 control steps. Each guide is credited after five consecutive crossing steps. Guide-specific clearance and seating targets supply distance-progress reward; they do not prescribe cable motion or PPO actions. Cable/guide contact and cable bending remain physical. Sharp changes above 0.65 rad between adjacent rod segments receive an additional penalty, alongside collision, stretch, drop, skip, and action-change penalties. Losing an already-completed crossing costs 8 reward units per simulated second. Dropping, skipping, or severe stretching costs 200 units at termination; a guide bonus is approximately 33.3 units, so repeatedly collecting an early bonus and failing is not profitable.

The policy observes robot state, the full cable centerline, the current shaping target, guide progress, grasp state, previous action, and all three guide positions. Control remains three Cartesian increments at 30 Hz with fixed gripper orientation. The curriculum increases the number of physical guides from one to two to three after at least 128 episodes with at least 80% success. Only episodes started at the current stage count toward its promotion. Cable stiffness, friction, and initial centerline perturbations use the same ranges as the straight task.

Saved trained result: **98/100** held-out successes. See [results and limits](../../../logs/usb_cable_routing_zigzag/RESULTS.md) and the [learned-policy video](../../../logs/usb_cable_routing_zigzag/demonstration_0000.mp4).

## Commands

Run from the Isaac Lab repository root:

```bash
export OMNI_KIT_ACCEPT_EULA=yes
export PXR_WORK_THREAD_LIMIT=1

# Validate physical feasibility with a separately identified scripted probe.
uv run python scripts/demos/usb_cable_routing_zigzag/evaluate.py \
  --probe scripted --guides 3 --episodes 10 --num_envs 10 --seed 31415 \
  --output logs/usb_cable_routing_zigzag/physics_validation.json

# Train a new PPO policy; this has a different observation size from the straight task.
uv run isaaclab train --task IsaacContrib-Route-USB-Zigzag-UR10e \
  --rl_library rsl_rl --num_envs 256 --max_iterations 800 --seed 43 \
  --run_name robotiq_zigzag_stage_aware

# Evaluate a frozen checkpoint on 100 new three-guide configurations.
uv run python scripts/demos/usb_cable_routing_zigzag/evaluate.py \
  --checkpoint logs/usb_cable_routing_zigzag/policy/model.pt \
  --guides 3 --episodes 100 --num_envs 100 --seed 910001 \
  --export_policy logs/usb_cable_routing_zigzag/policy \
  --output logs/usb_cable_routing_zigzag/evaluation.json

# Record a separate learned-policy demonstration.
uv run python scripts/demos/usb_cable_routing_zigzag/evaluate.py \
  --checkpoint logs/usb_cable_routing_zigzag/policy/model.pt \
  --guides 3 --episodes 1 --num_envs 1 --seed 910102 \
  --video logs/usb_cable_routing_zigzag/demonstration.mp4 \
  --output logs/usb_cable_routing_zigzag/demonstration.json

# Play the saved policy interactively with all three guides.
uv run isaaclab play --task IsaacContrib-Route-USB-Zigzag-UR10e \
  --rl_library rsl_rl --num_envs 1 --seed 910102 \
  --checkpoint logs/usb_cable_routing_zigzag/policy/model.pt \
  --viz newton_gl --real-time \
  env.commands.routing.guide_count=3 env.commands.routing.curriculum=False \
  env.sim.physics.load_visual_shapes=True \
  'env.sim.default_visualizer_cfg.eye=[1.25,-1.25,1.05]' \
  'env.sim.default_visualizer_cfg.lookat=[0.35,0.04,0.35]'
```

Training checkpoints are written under `logs/rsl_rl/usb_cable_routing_zigzag/`. Select a checkpoint using validation configurations before copying it to `logs/usb_cable_routing_zigzag/policy/model.pt` for the held-out test. A resumed run needs the guide count explicitly set to its earned curriculum stage because rolling curriculum counters are runtime state. Native video recording adds `_0000` to the requested filename.

Evaluation JSON records each assigned case, randomizations, initial cable centerline, guide positions, terminal outcomes, retained crossings, sampled peak grasp force and rod-joint gaps, contact steps, checkpoint/source hashes, versions, and exact command. Fixed episode quotas avoid selecting only fast-finishing episodes. Scripted probe results are explicitly labeled and are not learned-policy results.

The cable's effective material properties and missing gripper-link inertials are estimates. The breakable compliant attachment is a pre-grasp approximation, not a calibrated friction-only grasp. See the [shared physical assumptions](../usb_cable_routing/README.md) for details. The original straight-task result does not establish performance on this zigzag task.

Run the focused checks with:

```bash
uv run --extra test python -m pytest \
  source/isaaclab_tasks/test/contrib/test_usb_cable_routing.py -q
```

The development run earned two guides with 110/129 current-stage successes. After validating the stronger failure and retention penalties, training continued from its checkpoint 25 with:

```bash
uv run isaaclab train --task IsaacContrib-Route-USB-Zigzag-UR10e \
  --rl_library rsl_rl --num_envs 256 --max_iterations 400 --seed 43 \
  --run_name robotiq_zigzag_retention \
  --checkpoint logs/rsl_rl/usb_cable_routing_zigzag/2026-09-19_22-54-56_robotiq_zigzag_stage_aware/model_25.pt \
  env.commands.routing.guide_count=2
```
