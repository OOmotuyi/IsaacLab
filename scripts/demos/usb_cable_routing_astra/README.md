# Direct Astra control of the Newton cable tasks

Astra chooses each short robot movement from the current simulation state. The bridge executes that one request using Cartesian inverse kinematics, returns measured results, and pauses physics while waiting for the next request. It does not load an RL checkpoint, call an actor, select task stages, or follow a stored waypoint policy. This is full-state model control, not camera-only control or a trained replacement policy.

The two environments retain their cable physics, guide ordering, randomization, failure conditions, and success tests. The robot starts with the cable grasped. Astra must select movements, orientation changes, holds, and any recovery or reset explicitly. Resetting starts a new attempt and must remain visible in the recorded results.

## Launch a fresh direct-control session

Run these commands from the repository root, in separate terminals. Each server requires a new run directory. This workstation uses `DISPLAY=:1`; use the active desktop display on another machine. The Newton GL windows are hidden during model control, but an OpenGL display is still required for observation images. The NVIDIA offload variables select this workstation's NVIDIA GPU for OpenGL rendering.

```bash
DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  OMNI_KIT_ACCEPT_EULA=yes PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  uv run python scripts/demos/usb_cable_routing_astra/server.py \
  --variant scurve --seed 650001 --run_dir logs/usb_astra_direct/new_session/scurve
```

```bash
DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  OMNI_KIT_ACCEPT_EULA=yes PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  uv run python scripts/demos/usb_cable_routing_astra/server.py \
  --variant insert --seed 660001 --run_dir logs/usb_astra_direct/new_session/insert
```

Wait for `ASTRA_BRIDGE_READY` before submitting commands. Start one model controller per server so commands cannot race between independent decision makers.

In the demonstrated runs, the Astra models are Codex session agents. They read full-state observations, reason in the session, and invoke `client.py` to submit each selected movement. The filesystem protocol transports their decisions to Newton; no separate OpenAI API credentials or standalone API inference loop is required. Starting `server.py` alone only starts the paused simulator: an active Astra session must still choose and send requests. The command log records those explicit decisions and their rationales.

The completed demonstrations and all failed or interrupted attempts are listed in [the run report](../../../logs/usb_astra_direct/20260921_172702/REPORT.md).

## Client and file protocol

Start `server.py` for one task, following its `--help`, and use the same run directory for the client. Read the latest observation without advancing simulation:

```bash
uv run --no-project python scripts/demos/usb_cable_routing_astra/client.py \
  --run_dir logs/usb_cable_routing_astra/example --observe
```

Submit one explicitly selected operation as JSON:

```bash
uv run --no-project python scripts/demos/usb_cable_routing_astra/client.py \
  --run_dir logs/usb_cable_routing_astra/example \
  --request '{"op":"observe"}'
```

The client is standard-library-only. It atomically publishes `commands/<uuid>.json`, waits for `responses/<uuid>.json`, and prints the complete response. The server publishes the latest state in `observation.json`. Request IDs are printed to stderr. `--timeout_s` defaults to 120 seconds and limits waiting, not simulation execution.

A timeout does **not** cancel a submitted movement. Resume waiting with the same JSON and `--request_id <printed-uuid>`. The permanent `client_requests/<uuid>.json` record prevents the client from publishing the command twice, even if the server already consumed it. A reused UUID with different JSON is rejected. Do not create a new request ID to retry an operation whose result is still unknown. If a client was interrupted between recording and publishing a request, inspect its command and response files before restarting the server; the client deliberately avoids an ambiguous resubmission.

| Operation | Required fields | Optional fields |
| --- | --- | --- |
| `observe` | `op` | `camera` |
| `move` | `op`, `rationale`, `frame`, `target_position_m`, `steps` | `target_quat_xyzw`, `speed_m_s`, `angular_speed_rad_s`, `stop_on_stage_change`, `camera` |
| `hold` | `op`, `rationale`, `steps` | — |
| `reset` | `op`, `rationale` | `seed` |
| `close` | `op` | — |

An `observe` or `move` request can select `"camera":"routing"` or `"camera":"socket"`; the socket view is a close-up for insertion. The `--observe` shortcut only reads the latest saved observation and does not change its camera.

Every `move`, `hold`, and `reset` requires Astra's explicit reasoning in `rationale`. Positions are absolute environment-frame coordinates in meters, with the tabletop at z = 0. `frame` chooses the tracked point: `tcp`, cable `tip`, or connector `plug_front`. An optional quaternion is xyzw and specifies the orientation of that requested frame; the client requires unit norm within 0.001. A cable tip or plug quaternion is not the wrist quaternion.

`steps` is 1–90 control steps, at 30 Hz. A movement can end before reaching its target; inspect the returned measured position before choosing another action. Linear speed defaults to 0.10 m/s and is limited to 0.15 m/s; angular speed defaults to 0.30 rad/s and is limited to 0.50 rad/s. `stop_on_stage_change` defaults to true so Astra can observe newly completed guide stages. A hold maintains the current target while physics evolves. These operations do not automatically choose the next task stage.

## Geometry and measured state

The 600 mm cable has a 4 mm diameter, 40 Newton VBD segments, a 21 g body, and a 4 g rigid connector. The anchor is at `(0.180, -0.150, 0.022)` m. Material values remain estimates: stretch/bend/twist moduli 150/8/2 MPa, dynamic friction 0.40, and the existing small stiffness, friction, and initial-position randomizations. Static friction 0.50 and restitution 0.05 remain calibration targets; the 15 mm minimum bend radius is a task limit.

| Guide, in required order | Center [m] | Vertical-axis yaw |
| --- | --- | --- |
| Red | `(0.300, -0.150, 0)` | +10 degrees |
| Yellow | `(0.440, -0.050, 0)` | 0 degrees |
| Green | `(0.580, -0.150, 0)` | -10 degrees |

Each guide is upright, has a 10 mm cradle height, and retains its inward right-angle lips. The passage is 20 mm long and 18 mm wide; the lip opening is 10 mm wide. Seating checks the cable centerline at both passage faces: lateral offset below 7 mm and height from 11.5 to 34 mm. A guide advances after five consecutive valid steps. Routing succeeds after all three guides remain seated together for twelve steps (0.4 s). A later guide seated before the current guide sets a sticky skipping failure. Completed-guide count alone is insufficient: previous guides must still retain the cable.

Measured cable centerline, cable endpoint, TCP, and connector nose are distinct. The TCP is wrist-3 local `(0, 0, 0.143)` m. The connector nose is approximately 21 mm beyond the grasp point along the plug axis; the compliant grasp makes its measured pose preferable to a fixed-offset assumption.

The effective TCP workspace for both current tasks is x `[0.17, 0.80]`, y `[-0.38, 0.22]`, z `[0.045, 0.35]` m. The base routing action uses y >= -0.25 m; the S-curve task's inherited `ZigzagRoutingAction` expands this to -0.38 m. Bounds apply to the TCP, not directly to a requested cable-tip or plug-nose point. Existing Cartesian actions and hard joint limits remain in force, including the 2 rad/s bound on commanded joint increments. The S-curve task fixes wrist orientation; the insertion task permits rotation after its existing alignment gate opens.

## Insertion-specific constraints

The USB-C socket face is at `(0.680, -0.150, 0.065)` m. Insertion is along world +X; the socket's wide axis is +Y and narrow axis is +Z. Its xyzw quaternion is `(0.5, 0.5, 0.5, 0.5)`. USB-C also permits a 180-degree flip about its insertion axis; Astra must choose an orientation compatible with robot and tabletop clearance.

The existing task unlocks wrist rotation only after routing is retained for twelve steps and the cable endpoint is within 4 mm of the task's alignment staging position `(0.640, -0.230, 0.065)` m. The bridge exposes the state; it does not move to that position automatically. `ready_to_approach` requires orientation error below 0.025 rad.

Entry must begin with the nose 0.5–12 mm before the face, lateral errors below 0.5 mm, and angular error below 4 degrees. The nose then has to cross the entrance while the route is retained. Final acceptance requires nose depth 5.5–6.8 mm, wide-axis error at most 0.15 mm, vertical error at most 0.12 mm, angular error at most 2 degrees, grasp force below 2 N, and all three guides retained for the final success dwell. A rendered overlap alone is not success.

## Limits, outcomes, and provenance

A bend radius below 15 mm, skipped guide, dropped cable, or unsafe arm posture terminates the task. Wrist-1 must stay at least 90 mm above the tabletop. Stretch is the maximum neighboring rod-end gap divided by the 15 mm segment length: routing acceptance requires less than 0.10, and more than 0.30 terminates after the initial five steps. The breakable grasp fails above 15 mm positional error or 6 N force, or above 1.2 rad angular error or 0.04 Nm torque. The original horizons are 24 simulated seconds for S-curve routing and 60 seconds for routing plus insertion. Time spent awaiting Astra's next command does not consume simulated time.

Retain the run directory as the reproducible record: commands with Astra rationales, responses, observations, the server's state traces and outcome artifacts, and any recorded video. These establish which bounded actions Astra selected and whether the environment's success condition was reached. Report unsuccessful attempts and resets separately; one successful demonstration does not establish a randomized success rate. The client never synthesizes success or treats the absence of an error as task completion.

## Render a recorded successful episode

Rendering reads the measured per-step body poses; it does not advance physics or make new decisions. It verifies the recorded success flag, absence of an RL policy, body ordering, and trace/pose counts before encoding two matching 1920 × 1080 videos at 30 fps. The clean version omits the Newton UI; the GUI version includes it. The camera switches to a socket close-up during insertion approach, and the final successful frame is held for two seconds.

The verified S-curve demonstration is `logs/usb_astra_direct/20260921_172702/scurve/episode_001` (182 control steps, 6.0667 simulated seconds). Recreate its videos with:

```bash
DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  OMNI_KIT_ACCEPT_EULA=yes PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  uv run python scripts/demos/usb_cable_routing_astra/render.py \
  --episode_dir logs/usb_astra_direct/20260921_172702/scurve/episode_001
```

The verified routing-plus-insertion demonstration is `logs/usb_astra_direct/20260921_172702/insert/episode_003` (976 control steps, 32.5333 simulated seconds). It succeeded after three failed primary attempts. A separate parallel attempt on seed 660002 was stopped once this success was confirmed. Recreate its videos with:

```bash
DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  OMNI_KIT_ACCEPT_EULA=yes PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  uv run python scripts/demos/usb_cable_routing_astra/render.py \
  --episode_dir logs/usb_astra_direct/20260921_172702/insert/episode_003
```

The renderer writes `astra_<variant>_clean_1080p.mp4`, `astra_<variant>_newton_gui_1080p.mp4`, first/final PNG frames, and `render_report.json` beside the episode artifacts. The report includes the source trajectory SHA-256 and maximum body-pose replay error. An unfinished or unsuccessful episode is not a successful demonstration.

Replaying saved commands executes previously selected decisions; it is **not a new Astra run**. Rendering saved poses is a recorded-state replay, not a new physics evaluation. A fresh Astra evaluation requires a fresh environment attempt and new observation-driven decisions from the model, with its own outcomes and provenance.

Run the isolated client protocol checks without launching Newton:

```bash
uv run --no-project python scripts/demos/usb_cable_routing_astra/test_client.py
```

## Matched fresh-configuration evaluation

The comparison uses a frozen manifest with 100 cases per task, one native episode per controller per case, and no policy retries. Existing RL exports are fixed. Each Astra case uses a new GPT-6 Astra agent context and the same frozen prompt, with only its case identifiers changed. The current benchmark assets are under `logs/usb_controller_comparison/20260922_014003/`; the manifest and `astra_controller_prompt.md` were written before scored episodes.

The two controllers keep their existing observation interfaces: Astra receives full measured simulator state and optional images, while RL consumes its trained observation vector. This compares the existing controllers on matched physical configurations; it does not isolate the effect of model type under identical observation inputs. The insertion RL baseline remains a hybrid of learned routing, pose feedback, and PPO residual corrections.

`benchmark_state.py` records and checks actual initial body and cable poses/velocities, robot joints and actuator targets, cable stiffness and friction, grasp state, guide/socket geometry, task settings, and the native policy observation. The comparison requires exact matches. Every case starts in a fresh simulator process and explicitly reseeds after policy loading. This avoids history-dependent wrist-orientation caches observed during repeated insertion resets. No RSL wrapper introduces another reset. A failed match quarantines the Astra case before any action; infrastructure errors are distinct from native task failures.

Run the fixed RL exports on the manifest, using new output directories:

```bash
export DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export OMNI_KIT_ACCEPT_EULA=yes PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1

uv run python scripts/demos/usb_cable_routing_astra/evaluate_rl.py \
  --manifest logs/usb_controller_comparison/20260922_014003/manifest.json \
  --variant scurve --output_dir logs/usb_controller_comparison/20260922_014003/rl/scurve

uv run python scripts/demos/usb_cable_routing_astra/evaluate_rl.py \
  --manifest logs/usb_controller_comparison/20260922_014003/manifest.json \
  --variant insert --output_dir logs/usb_controller_comparison/20260922_014003/rl/insert
```

After the paired RL case has an `initial_state.json`, initialize an Astra worker on that case:

```bash
uv run python scripts/demos/usb_cable_routing_astra/server.py \
  --variant scurve --seed 850001 --no_images \
  --run_dir logs/usb_controller_comparison/20260922_014003/astra/scurve_000 \
  --evaluation_manifest logs/usb_controller_comparison/20260922_014003/manifest.json \
  --reference_dir logs/usb_controller_comparison/20260922_014003/rl/scurve
```

An active, fresh Astra session must then follow the frozen prompt and control exactly the assigned case through `client.py`. Benchmark workers reject every reset. After native termination, the coordinator closes the worker and starts a fresh process and Astra context for the next assigned case. `--no_images` suppresses automatic snapshots, while an explicit `observe` or `move` camera request still renders an image. Simulation time pauses between model commands. Starting the server alone does not invoke Astra.

The comparison records every outcome, the initial-state match, explicit model decisions, measured trajectories, fixed source/checkpoint/prompt hashes, and fresh-agent assignments. Unscored smoke cases use separate seeds and directories. Rates are reported only with their completed/expected denominators; a full comparison requires all 100 paired cases per task.

Generate the audited comparison report (the completeness check fails until all 400 episodes are accounted for):

```bash
uv run python scripts/demos/usb_cable_routing_astra/benchmark_report.py \
  --run_dir logs/usb_controller_comparison/20260922_014003 --require_complete
```
