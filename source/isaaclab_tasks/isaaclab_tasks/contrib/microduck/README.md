# MicroDuck training migration

The experimental native task is **IsaacContrib-MicroDuck-Velocity-Flat**.
It uses Isaac Lab's ManagerBasedRLEnv, Newton MJWarp physics, and RSL-RL PPO.
It trains from a randomly initialized network. It does not load an upstream ONNX policy.

Prepare assets from the repository root:

```bash
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck --extra importers python scripts/demos/microduck/prepare.py
```

Train and view a resulting checkpoint:

```bash
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck isaaclab train --rl_library rsl_rl     --task IsaacContrib-MicroDuck-Velocity-Flat --num_envs 1024 --viz none

OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck isaaclab play --rl_library rsl_rl     --task IsaacContrib-MicroDuck-Velocity-Flat --num_envs 1     --checkpoint /absolute/path/to/model_1000.pt --viz newton_gl
```

Assets default to `logs/microduck/assets` in this checkout.
Set `MICRODUCK_ASSET_DIR` to use another prepared asset directory.
Training logs and checkpoints are under `logs/rsl_rl/microduck_velocity`.

## Implemented

- Fourteen actions in upstream servo order; 61 actor observations and 76 critic observations.
- 200 Hz physics and 50 Hz policy evaluation.
- Vectorized BAM M6 servos, per-robot supply/sag, 3–6 physics-step action delay,
  independently reset motor histories, and randomized friction.
- Source walking/head commands, velocity/upright/posture/head objectives,
  foot air-time, clearance, slip and swing-height objectives, joint-limit and smoothness costs.
- Source walking collision groups, self-contact costs and whole-body angular momentum.
- Independent MuJoCo reference check of angular momentum after inertial randomization.
- Source step schedules for smoothness, standing fraction, head command ranges and head bias.
- Trunk mass/inertia, link center-of-mass, armature, foot friction, encoder and IMU
  randomization; observation latency and additive pushes.
- Native RSL-RL checkpoint, logging, resume and playback integration.

## Remaining migration work

This is an initial walking training environment, **not the completed upstream suite**.
Exact upstream recovery/velstand, sit/stand, ground picking, kicking, roulade, skating,
swizzle, roller crouch/slope/recovery, spin, rough terrain and backlash training variants remain to be ported.
The ten original ONNX demos remain pretrained-policy transfers. Local standing and pickup
fine-tuning tasks are described below; they do not complete the upstream task-suite migration.

The flat task computes foot clearance relative to its ground plane.
Contact timing uses the native force sensor; upstream contact detection uses contact counts.
Foot friction currently samples each foot independently, while upstream shares the sample.
These remaining sensor/randomization differences need to be resolved for strict parity.
The critic places linear velocity and contact features after the policy features,
so upstream critic checkpoints are not directly interchangeable.

Short PPO runs establish execution and checkpoint creation, not locomotion quality.
The first longer run (v1) learned to crouch with negligible command tracking and was stopped.
Its checkpoints and video are failed experiments, not validated walking policies.
The v2 exploratory run includes collision, reward-frame, inertial-randomization and motor-property
synchronization corrections. Fixed-command evaluation showed stable nominal walking but
substantial velocity undertracking.

Evaluate a native checkpoint and optionally record a following-camera video:

```bash
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/evaluate.py \
    --checkpoint /absolute/path/to/model_1000.pt \
    --metrics logs/microduck/evaluation.json --num_envs 64 --duration 20

OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck python scripts/demos/microduck/evaluate.py \
    --checkpoint /absolute/path/to/model_1000.pt \
    --metrics logs/microduck/video-evaluation.json --num_envs 1 --duration 20 \
    --viz newton_gl --record logs/microduck/native-policy.mp4
```

## Provenance

Task definitions and robot contract:
[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck/tree/cb70b792312d559a4da09064d92009079671815f).

Motor equations:
[Rhoban/bam](https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967), Apache-2.0.
The training action uses that revision's GPU training friction equation; the
standalone demonstration retains its CPU playback friction equation.
These differ in the quadratic load term. The preparation script preserves downloaded asset licenses.

The environment-browser preview shows the robot in the Newton GL visualizer, taken from the transferred-policy demo.


## Controlled walking continuation

A PPO checkpoint does not restore the environment curriculum clock. To continue
the mature schedule after 3,000 iterations of 24 control steps, pass a 72,000-step
offset. The action-rate multiplier defaults to 1.0, preserving the source
schedule. Set it to 0.4 to test a weaker penalty while retaining the other
curricula, physics, observations, and PPO settings:

```bash
OMNI_KIT_ACCEPT_EULA=yes uv run --extra microduck isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-MicroDuck-Velocity-Flat --num_envs 1024 \
    --checkpoint /absolute/path/to/model_2999.pt --max_iterations 1000 \
    --run_name action_rate_040 --seed 42 --viz none \
    env.curriculum.walking.params.step_offset=72000 \
    env.curriculum.walking.params.action_rate_scale=0.4
```

Use scale 1.0 with a separate run name for the matched control. The iteration
limit is the number of additional updates, not the final checkpoint number.
These runs restart simulator states and random streams while restoring the
actor, critic, observation normalizers, and optimizer. Compare checkpoints using
identical fixed-command evaluations, including standing and turning.


## Walking optimization results

The local optimization study is saved in
`logs/microduck/optimization_v1/index.html`, with exact commands in
`experiment.json` and measured rollouts in `evaluations/`.
The original v2 checkpoint averaged 0.099 m/s at a 0.30 m/s forward command
over 64 robots and 20 seconds. The selected weaker-action-penalty checkpoint,
`2026-09-19_12-59-11_optimization_v1_action_rate_040/model_3750.pt`,
averaged 0.123 m/s, reduced yaw tracking RMSE from 0.170 to 0.107 rad/s,
and completed 122/128 randomized rollouts without a fall reset versus 119/128
originally. These are results from one training seed per variant.

Separate stronger-velocity-reward runs reached higher forward speeds but
introduced low-speed dead zones and unreliable backward/lateral tracking.
A subsequent angular-reward fine-tune reduced unwanted turning but did not
remove those limitations. These faster checkpoints remain experimental.
The 0.30 m/s tracking target is not yet achieved; the other task families
listed above remain unmigrated.


## Standing and pickup PPO fine-tuning

`IsaacContrib-MicroDuck-Stand` and `IsaacContrib-MicroDuck-Ground-Pick`
use manager-based Isaac Lab environments with Newton MJWarp and the prototype
15-joint cleanup robot. Fourteen body actions are learned; the mouth is scripted.
These are local cleanup skill adaptations, not exact reproductions of the upstream
reward functions or an end-to-end cleanup task.

The actor starts from the released MuJoCo ONNX weights. Fixed observation
normalization is folded into its first layer and checked against ONNX Runtime.
PPO then updates the actor with a fresh critic and optimizer. Rewards measure
beak position, phase-conditioned trunk orientation and height, stationary balance,
joint limits and smoothness, with a source-action prior to retain the existing skill.
Pickup targets come from measured source-policy motion in a cleanup trace.
The task has no object, cloth, or bin: grasp success is tested separately in the room.
Servo friction and zero action delay match the deployed standing/pickup controller.
Battery, motor friction, initial pose, foot friction and observations are randomized.

Prepare the existing marker/room assets as described in the demo README, then:

```bash
uv run --extra microduck python scripts/demos/microduck/prepare_skill_reference.py \
    --trace logs/microduck/room_cleanup/assisted_clearance_04/trace.jsonl

uv run --extra microduck python scripts/demos/microduck/skill_checkpoint.py \
    --source logs/microduck/assets/alpha_ground_pick.onnx \
    --output /absolute/results/warmstart/pickup.pt

OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-MicroDuck-Ground-Pick --num_envs 256 --max_iterations 250 \
    --checkpoint /absolute/results/warmstart/pickup.pt --viz none --seed 42 \
    agent.experiment_name=/absolute/results/training/pickup
```

Use the Stand task with `alpha_stand.onnx` for standing. The default reference is
`logs/microduck/assets/pickup_reference.json`; set `MICRODUCK_SKILL_REFERENCE`
to preserve a separate experiment reference.

Evaluate source and trained actors under identical seeds, then export a selected checkpoint:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck python scripts/demos/microduck/evaluate_skill.py \
    --skill pickup --checkpoint /absolute/results/training/pickup/RUN/model_249.pt \
    --output /absolute/results/evaluation/pickup.json --num_envs 32 --viz none

uv run --extra microduck python scripts/demos/microduck/skill_checkpoint.py \
    --source logs/microduck/assets/alpha_ground_pick.onnx \
    --checkpoint /absolute/results/training/pickup/RUN/model_249.pt \
    --output /absolute/results/exports/alpha_ground_pick.onnx
```

The exporter verifies changed actor weights, optimizer steps, and ONNX parity;
it records checkpoint hashes and source provenance. The room runner accepts
`--standing_policy`, `--pickup_policy`, and `--walking_policy`. Its metrics record
which skill exports were used. Navigation targets, skill switching, head hold,
jaw motion and sock assistance remain scripted.


## Standing and reaching from scratch

`IsaacContrib-MicroDuck-Stand-Scratch` and
`IsaacContrib-MicroDuck-Ground-Pick-Scratch` use random actor/critic weights,
RSL-RL PPO, and Newton MJWarp. Neither task loads a teacher, rewards imitation,
or reads a recorded policy trajectory. They retain the physical robot and BAM
motor parameters. The actor has 61 observations and 14 body-joint actions; the
jaw remains scripted. There are no objects in these skill environments.

The standing goal is a stationary upright posture. Reaching follows hand-authored
beak/trunk waypoints, lowering the beak to 0.065 m before returning upright.
Depth grows between control steps 2,400 and 36,000 (PPO updates 100–1,500 with
24 steps per update). The third phase-command component exposes the depth
fraction to the actor; playback uses full depth. These geometric objectives
have not yet been validated as successful object pickups.

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck --extra rsl-rl isaaclab train \
    --rl_library rsl_rl --task IsaacContrib-MicroDuck-Stand-Scratch \
    --num_envs 512 --max_iterations 3000 --seed 42 \
    agent.experiment_name=/absolute/results/training/stand

OMNI_KIT_ACCEPT_EULA=YES uv run --extra microduck --extra rsl-rl isaaclab train \
    --rl_library rsl_rl --task IsaacContrib-MicroDuck-Ground-Pick-Scratch \
    --num_envs 512 --max_iterations 3000 --seed 43 \
    agent.experiment_name=/absolute/results/training/reach
```

Omit `--checkpoint` for a new run. Use the standard `isaaclab play` and
`isaaclab export` commands with the matching Scratch task to retain observation
normalization. `skill_checkpoint.py --from_scratch --skill stand|pickup` also
exports the normalized actor with provenance for the room runner, which supplies
the full-depth command for scratch reaching. `evaluate_skill.py` remains specific
to MuJoCo-initialized tasks. Scratch actors require measured cleanup validation;
geometric reaching training alone does not establish object-grasping success.

## Object-conditioned marker grasp training

`IsaacContrib-MicroDuck-Marker-Grasp-Scratch` adds the physical marker and its
two 30 mm end supports to a manager-based Newton MJWarp task. Its actor receives 65
observations and outputs 14 body offsets plus a jaw target. Object-relative
position/velocity, marker-relative heading, jaw state, and measured grasp phase replace the old phase
clock. Contact-only marker success requires opposing jaw contacts, actual lift
to at least 0.20 m for a ground pickup, object speed below 0.12 m/s, and a
0.30 s dwell. The task remains experimental; registering it or producing a
checkpoint does not establish complete bedroom cleanup success. The manager-based
environment filters support-floor contacts while preserving marker/robot contacts
with those immovable fixtures. Newton uses eight substeps per 5 ms physics update.
Invalid robot or object states terminate as failures with a finite termination
penalty and no shaping reward; they never count toward curriculum advancement.

The staged curriculum keeps the marker on its bedroom support while lowering a
virtual reaching goal from standing height toward the marker. Reaching goals
advance in 5 mm steps only after sustained position and aperture alignment
at the end of non-failed episodes. The final approach enters the mouth
aperture, then contact gates the lift goal. The goal rises continuously at 40 mm/s
from the measured object height, capped at a 10 mm lead so it cannot run away from
a stationary object. Final approach uses a 10 mm distance scale; the broader
virtual-reaching curriculum retains 50 mm. During physical approach, insertion,
and holding, centering reward follows the measured object to avoid reward jumps at
phase changes. Lift shaping rewards vertical progress from the measured grasp
height, including recovery if the marker slipped down before contact settled.
Success still requires the original absolute bin-clearance height. Final grasp
orientation uses a sharper alignment score and insertion requires a raw aperture
score above 0.995; the virtual-reaching curriculum retains its 0.9 threshold.
After opposed contact persists for 0.10 s and motion relative to the head stays
below 3 mm/s for 0.10 s, the tracker records the measured object center in head
coordinates. This fixed grasp reference exposes later slip and keeps the upward
lift goal correct when the marker settles at the lip, away from the nominal open-mouth
center. It changes observations only and does not attach or reposition the object.
Lost opposed contact for 0.16 s returns the controller to approach, including after
an earlier successful lift. An aperture-orientation reward prevents
backward head folding or sideways alignment from earning the same reaching reward.
The `grasp_proprio_object_v3` contract replaces the unused object-kind input with
marker-relative heading. Older `v2` exports must be retrained with this observation
and orientation objective before using the updated bedroom playback. The mouth rotates downward early in the approach and points down for floor pickup.
A soft neck-position objective selects the continuous downward-reaching branch
found by independent robot-model kinematic checks. This is a hand-designed reward,
not an imitation loss or a recorded policy-motion reference. Analytic trunk pitch/height guidance,
base anchoring, and whole-body balance support exploration. No source-policy
motion references are used. The supports leave the center of the marker clear for
the lower jaw. An optional elevated-object curriculum lowers the real marker after
sustained grasp-and-lift success; elevated resets move it forward enough to clear
the standing robot's open jaw. Final evaluation uses the original 39.5 mm object
height and 68 mm forward placement.

The current training workflow continues the standing actor previously trained
from random weights in Isaac Lab, with new object inputs, a jaw output, and a
random critic. Initial exploration uses 0.025 rad for body joints and 0.03 rad
for neck/head pitch. This is **continued Isaac-only training**, not a freshly randomized
grasp actor. It loads no MuJoCo policy weights or teacher actions. To train every
parameter from random initialization instead, omit `--checkpoint` below.
A near-object diagnostic can also initialize a random network with a geometric
output bias using `grasp_checkpoint.py --initialize_geometric RECORD.json`.
The record provides `output_bias_rad` (fourteen servo offsets), `geometric_reset_pose`,
and an optional seed. This loads no actor or critic weights. Configure the robot's
reset pose separately to match the record. Success from this reset does not prove
a complete approach from standing.

```bash
uv run --extra microduck python scripts/demos/microduck/grasp_checkpoint.py \
    --initialize_from /absolute/path/to/isaac_scratch_stand/model_2999.pt \
    --output /absolute/results/initialization/model_0.pt

uv run --extra microduck isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-MicroDuck-Marker-Grasp-Scratch --num_envs 256 \
    --checkpoint /absolute/results/initialization/model_0.pt --max_iterations 1200

uv run --extra microduck python scripts/demos/microduck/evaluate_grasp.py \
    --checkpoint /absolute/path/to/trained/model_1199.pt \
    --output /absolute/results/evaluation.json --object_height 0.0395 --duration 8 --viz none
```

Evaluation defaults to the actual pickup goal, with the curriculum disabled.
`--reach_height` evaluates a particular intermediate curriculum level explicitly.
The report records contact, lift, failure, and pose traces separately.
`--training_reset` explicitly evaluates a custom near-object robot reset from the
saved training configuration; this diagnostic does not establish an approach
from standing. A stable success requires the object to remain held at the end.
Export
requires an `experiment.json` recording the run command, initialization identity,
and source hashes; it verifies Isaac-only lineage and nonzero actor updates.

The bedroom runtime accepts the resulting actor through `--grasp_policy` and
shares its grasp phases and observation construction with the training task.
Navigation, transport coordination, and release remain scripted. Sock assistance
remains an explicit two-node cloth attachment. Using a marker-trained actor for
the sock is a transfer experiment until validated on the deformable object.

The marker-grasp environment preview uses Newton GL and shows the physical task
setup during an intermediate reaching evaluation; it is not a pickup-success image.
