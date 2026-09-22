Added
^^^^^

* Added the experimental ``IsaacContrib-MicroDuck-Velocity-Flat`` manager-based
  training task with Newton MJWarp, batched BAM motor dynamics, contact-based
  gait rewards, observation and motor randomization, and an RSL-RL PPO configuration.
  Documented the remaining task families and physics-parity work.

* Added curriculum step offsets and action-rate penalty scaling for controlled
  MicroDuck checkpoint continuation experiments.

* Added MuJoCo-initialized standing and phase-conditioned pickup PPO tasks in Newton,
  verified actor conversion, evaluation and export utilities, and cleanup policy overrides.
  Kept jaw motion, cleanup sequencing and optional sock assistance explicitly scripted.

* Added a bedroom cleanup demonstration with a rigid marker, deformable sock,
  reachable wire basket, and coupled MJWarp/VBD simulation. Added an optional
  sub-solver argument to the MicroDuck controller and explicitly labeled
  assisted sock grasping, recorded-state RTX rendering, and native Newton GUI replay.
