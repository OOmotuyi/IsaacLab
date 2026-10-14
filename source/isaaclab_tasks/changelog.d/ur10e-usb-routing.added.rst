* Added the contributed ``IsaacContrib-Route-USB-UR10e`` task using the combined
  UR10e/Robotiq 2F-85 USD, Newton elastic cable physics, ordered open-guide routing,
  a success-driven guide curriculum, PPO configuration, and evaluation/video tooling.
* Shared the corrected rigid USB-C attachment, side grasp, cable/gripper contacts and
  25 g cable assembly parameters across the straight, zigzag, S-curve and insertion tasks.
  Earlier straight/zigzag checkpoint results used different mechanics and require reevaluation.
* Extended held-out policy evaluation to S-curve routing and USB-C insertion, including
  terminal bend-radius and insertion-depth measurements.
* Repositioned the shared robot base and selected a wrist-above-hand reset branch.
  Added bounded IK joint targets, posture rewards and terminations, and terminal
  wrist/joint measurements. Earlier cable-task policies required adaptation to
  the changed robot state and action constraints.
* Revised S-curve shaping targets for clearance and chose the USB-C mating symmetry
  using wrist approach direction and rotation distance. Extended insertion episodes
  to 60 seconds for the bounded approach.
* Added optional fixed observation normalization for PPO refinement of demonstration-
  initialized actors, preserving standard checkpoint and TorchScript inference formats.
