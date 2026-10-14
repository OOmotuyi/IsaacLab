* Added the independent ``IsaacContrib-Route-USB-SCurve-Insert-UR10e`` task, extending ordered
  three-guide cable routing with six-axis USB-C alignment and insertion into a physical
  receptacle. Added reusable plug/socket USD assets, insertion rewards and success checks,
  and simulation preview, mating-contact, and endpoint-clearance validation scripts.

* Corrected the insertion task's rigid connector attachment so the flexible cable ended
  at the rear of the strain relief and its bending joints remained outside the housing.
  Preserved the robot grasp pose and cable mass while updating insertion target offsets.

* Restored flexible cable collisions with the Robotiq gripper and changed the insertion
  task to grasp the upright plug from the side. Added a cable/robot contact rollout
  check alongside the existing endpoint-clearance validation.

* Added staged wrist control, routing-only preliminary curriculum completion and angular-progress shaping.
  Moved the socket to provide more cable-bend clearance and increased its VBD contact buffer.
* Added reproducible PPO parameter transfer and physical-demonstration actor initialization utilities.

* Added an optional residual PPO recipe with an embedded, frozen routing actor and bounded
  corrections around measured insertion feedback. Exported policies included both networks.
* Added explicit settling and alignment flags, expanding insertion observations from 193 to
  195 values. Earlier insertion checkpoints required observation migration and refinement;
  the direct and residual agent recipes remained separately selectable.
