# UR10e–Robotiq USB cable routing

All four cable tasks now share the corrected rigid strain-relief attachment and side grasp. Flexible cable–gripper contacts remain enabled; only the rigid plug contact is substituted by a breakable compliant grasp. The shared cable uses 40 segments, a 21 g body plus 4 g plug, 150/8/2 MPa effective moduli and 0.40 active friction. Static friction 0.50 and restitution 0.05 are calibration targets. Earlier straight/zigzag checkpoints are historical results with different mechanics and have not been revalidated against this version.

Registered task: `IsaacContrib-Route-USB-UR10e`. This manager-based Isaac Lab task uses the installed **combined UR10e USD with its `Gripper=Robotiq_2f_85` variant**, Newton MJWarp for the robot, Newton VBD elastic rods for the cable, and RSL-RL PPO for training. The open orange guides and USB anchor were modeled from the project-root `cable_routing.png` reference.

Historical run with earlier mechanics and base position: **96/100** held-out three-guide successes. See [results and limitations](../../../logs/usb_cable_routing/RESULTS.md) and the [learned-policy video](../../../logs/usb_cable_routing/demonstration_0000.mp4).

The shared robot base now sits at `(0.10, -0.85, 0.0)` m with a wrist-above-hand reset branch, bounded joint target increments and posture checks. The straight and zigzag policies have not been retrained for this base position. See the [S-curve posture revision](../usb_cable_routing_scurve/README.md#robot-posture-revision).

## Run

Run these commands from the Isaac Lab repository root, using its existing uv environment:

```bash
export OMNI_KIT_ACCEPT_EULA=yes
export PXR_WORK_THREAD_LIMIT=1

# Check the physical routing with an explicitly scripted controller.
uv run python scripts/demos/usb_cable_routing/evaluate.py \
  --probe scripted --guides 3 --episodes 10 --num_envs 10 --seed 31337 \
  --output logs/usb_cable_routing/physics_validation.json

# Train PPO from scratch; curriculum starts with one physical guide.
uv run isaaclab train --task IsaacContrib-Route-USB-UR10e \
  --rl_library rsl_rl --num_envs 256 --max_iterations 1500 --seed 42 \
  --run_name robotiq_usb

# Evaluate the saved policy on 100 new randomized configurations, always with three guides.
uv run python scripts/demos/usb_cable_routing/evaluate.py \
  --checkpoint logs/usb_cable_routing/policy/model.pt \
  --guides 3 --episodes 100 --num_envs 100 --seed 900001 \
  --export_policy logs/usb_cable_routing/policy \
  --output logs/usb_cable_routing/evaluation.json

# Record a separate learned-policy demonstration directly from the simulation framebuffer.
uv run python scripts/demos/usb_cable_routing/evaluate.py \
  --checkpoint logs/usb_cable_routing/policy/model.pt \
  --guides 3 --episodes 1 --num_envs 1 --seed 900101 \
  --video logs/usb_cable_routing/demonstration.mp4 \
  --output logs/usb_cable_routing/demonstration.json
```

Training writes checkpoints every 25 iterations under `logs/rsl_rl/usb_cable_routing/`. Copy the checkpoint selected for final evaluation to `logs/usb_cable_routing/policy/model.pt`; the final results report identifies that source checkpoint. Native recording appends `_0000` to the requested video filename. Add `--viz newton_gl` to the evaluation command to open a live window. Lower `--num_envs` if GPU memory is limited. `--no_graph` disables CUDA graphs for debugging.

The recorded development run advanced 1 → 2 guides after 111/128 successes and 2 → 3 after 105/128. A regression check then corrected the Cartesian-target observation at reset. Training continued from checkpoint 100 at its already-earned three-guide stage:

```bash
uv run isaaclab train --task IsaacContrib-Route-USB-UR10e \
  --rl_library rsl_rl --num_envs 64 --max_iterations 150 --seed 42 \
  --run_name robotiq_usb_reset_fix \
  --checkpoint logs/rsl_rl/usb_cable_routing/2026-09-19_15-15-20_robotiq_usb/model_100.pt \
  env.commands.routing.guide_count=3
```

A subsequent phase correction allowed confirmed guide crossings to advance the shaping target even when the policy used a different path from the scripted lift. Continuations used these commands:

```bash
uv run isaaclab train --task IsaacContrib-Route-USB-UR10e \
  --rl_library rsl_rl --num_envs 64 --max_iterations 125 --seed 42 \
  --run_name robotiq_usb_progress \
  --checkpoint logs/rsl_rl/usb_cable_routing/2026-09-19_15-33-17_robotiq_usb_reset_fix/model_125.pt \
  env.commands.routing.guide_count=3

uv run isaaclab train --task IsaacContrib-Route-USB-UR10e \
  --rl_library rsl_rl --num_envs 256 --max_iterations 125 --seed 42 \
  --run_name robotiq_usb_256 \
  --checkpoint logs/rsl_rl/usb_cable_routing/2026-09-19_15-43-06_robotiq_usb_progress/model_150.pt \
  env.commands.routing.guide_count=3
```

The final continuation replaced stationary proximity shaping with distance-progress shaping, which rewards movement toward the target and gives zero reward for hovering:

```bash
uv run isaaclab train --task IsaacContrib-Route-USB-UR10e \
  --rl_library rsl_rl --num_envs 256 --max_iterations 150 --seed 42 \
  --run_name robotiq_usb_distance_progress \
  --checkpoint logs/rsl_rl/usb_cable_routing/2026-09-19_15-48-45_robotiq_usb_256/model_175.pt \
  env.commands.routing.guide_count=3
```

`--max_iterations` on a resumed run is the number of additional updates. The final results report identifies the selected checkpoint; use the from-scratch command above with the final source to train a new policy without this development interruption.

The evaluator explicitly distinguishes `mode: ppo` from `mode: scripted`, `zero`, or `random`. Scripted probe performance is a physics validation and is not evidence of learned-policy performance. Evaluation JSON includes every outcome, terminal guide occupancy, peak force and joint gap, collision steps, randomized parameters, initial centerlines, checkpoint hash, source hashes, versions, seed, and command. The held-out seed is different from both training and physics-validation seeds. Each environment has a fixed episode quota, so the evaluator waits for every assigned configuration instead of preferentially counting fast-finishing episodes. The reported confidence interval is a Wilson 95% binomial interval.

## Task and physical assumptions

- The flexible cable section is 0.6 m long and 4 mm in diameter, represented by 40 coupled 15 mm elastic rod segments. A static USB-A housing and strain relief fix the first segment to the table. A USB-C housing and metal shell add collision geometry and mass to the free-end rod.
- Three open guides have 42 mm internal width and walls reaching 54 mm above the table. Their centers are at x = 0.30, 0.44, and 0.58 m. The second guide has a 10 mm lateral offset. The asset is an interpretation of the reference image, not measured fixture CAD.
- The Robotiq 2F-85 starts closed around the free end. Its six finger joints use consistent closed-chain positions at a 0.730 rad drive angle. The measured grasp frame lies 143 mm along `wrist_3_link`'s local z axis; that axis points along world +y, perpendicular to the upright plug axis. Reset solves the six arm joints with differential IK and rejects position error above 3 mm.
- The USD omits mass properties on three Robotiq links. Task-local estimates fill those omissions, giving approximately 0.903 kg total gripper mass. Existing authored inertials on other links are preserved. Missing-link COMs and inertias use the asset bounds and are not manufacturer-calibrated.
- Grasp retention uses a **breakable compliant attachment**, with equal/opposite forces on wrist and cable, 400 N/m stiffness, 0.03 N·s/m damping, a 6 N force limit, and a 15 mm slip limit. Flexible cable–gripper contact is enabled; only contact between the rigid plug compound and gripper is replaced by the compliant grasp. This is a pre-grasp routing task, not a validated friction-only Robotiq grasp simulation. The fingers remain articulated and actuated.
- The cable uses effective density 2785.21 kg/m³, axial modulus 150 MPa, bending modulus 8 MPa, and torsional modulus 2 MPa, with normalized body/connector masses of 21/4 g. It approximates the jacket and conductors as one elastic rod; it does not model individual copper strands, plasticity, electrical behavior, or calibrated real-cable hysteresis. These parameters require physical measurements before hardware transfer.
- Per episode, rod stiffness is scaled uniformly by 0.85–1.15, cable friction is sampled from 0.36–0.44, and the slack starting centerline receives up to 5 mm horizontal perturbation away from its fixed endpoints. The native Newton model is updated and the coupled solvers are notified.

The policy outputs three bounded Cartesian increments at 30 Hz. Differential IK maintains a fixed gripper orientation and closes the low-level arm position loop. The arm uses gravity compensation. Distance reduction toward a lift/lower target provides progress shaping, with no stationary hover reward; switching to lowering uses measured cable clearance above the first guide or a confirmed earlier guide crossing. This target does not move the cable or supply policy actions. The separate scripted probe follows it only for physical validation.

A guide is credited only when the **cable centerline crosses both guide faces inside the open channel**, retained for five control steps. The robot must visit guides in order and retain all active crossings for 12 steps to succeed. Rewards include ordered guide bonuses, final success, and distance-progress shaping; penalties cover robot/fixture contacts, excessive rod-joint extension, dropped grasps, skipped guides, and action changes. Fixed-base mounting contact is allowed; intended cable/guide contact is not a robot collision. Drops, severe stretch, skipped guides, and the 12 s time limit terminate an episode.

The curriculum advances from one to two to three physical guides after at least 128 completed episodes with at least 80% success. Inactive guides are below the workspace. PPO checkpoints save the policy/optimizer, while this task's rolling curriculum counters are runtime state; resuming a training process starts the configured stage again. Evaluation freezes the stage at three and never uses the training curriculum.

## Components and checks

The robot configuration comes from `isaaclab_assets.robots.universal_robots.UR10e_ROBOTIQ_2F_85_CFG`. Cable construction, rod state access, differential IK, manager-based RL, native video recording, and the coupled MJWarp/VBD arrangement reuse installed Isaac Lab components. The maintained Franka cable-lifting example was the coupling reference.

```bash
uv run --extra test python -m pytest \
  source/isaaclab_tasks/test/contrib/test_usb_cable_routing.py -q
```

The task source and procedural USD fixtures are under `source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing/`. To regenerate the fixture USD files after editing their dimensions:

```bash
uv run python -c 'from isaaclab_tasks.contrib.usb_cable_routing.assets import generate_assets; generate_assets()'
```

Useful primary references: [Newton ModelBuilder](https://newton-physics.github.io/newton/latest/api/_generated/newton.ModelBuilder.html), [Robotiq 2F grippers](https://robotiq.com/products/adaptive-grippers), [UR10e specifications](https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-usr-man/complianceUR10e/H_g5_sections/appendix_g5/tech_spec_sheet.htm), and the installed `docs/source/concepts/deformables.rst` and `core/lift/config/franka_soft/franka_cable_env_cfg.py`.
