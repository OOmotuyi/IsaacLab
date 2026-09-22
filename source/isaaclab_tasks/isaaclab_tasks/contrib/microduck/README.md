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
