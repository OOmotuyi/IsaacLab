* Added separate UR10e USB routing and insertion development tasks with physical
  Robotiq contact grasping and randomized guide poses. Removed full cable geometry
  and evaluator progress flags from their shared controller observations and removed
  privileged insertion phase gates. Existing assisted-grasp tasks remained unchanged.
  The new observation dimensions required newly trained policies; connector and
  fixture poses still used ideal simulated estimates pending camera integration.
