# Tactile Fuse Relay

Tactile Fuse Relay is a self-contained MuJoCo disaster-response task. A 15-actuator, five-finger hand isolates a failed high-voltage fuse, disposes of it, grasps a replacement, performs wrist reorientation, rejects a seeded slip with closed-loop feedback, inserts the fuse, and presses a safety latch. The run exports its complete sensor/action trajectory and a fixed-seed controller ablation.

## Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 submissions/tactile-fuse-relay/run.py
```

The default command deterministically creates a 60-second `artifacts/demo.mp4` and four JSON evidence files. For CI or a fast local check:

```bash
python3 submissions/tactile-fuse-relay/run.py --quick
python3 submissions/tactile-fuse-relay/validate.py
```

Use `--no-video` when only physics/controller validation is required. Rendering selects EGL automatically and falls back to a schematic evidence frame if offscreen OpenGL is unavailable.

## What the robot actually uses

- **MuJoCo physics:** free fuse bodies, collision geometry, friction, gravity, 14 hand joints, a latch joint, 15 position actuators, and equality constraints used as a reproducible high-friction grasp.
- **Sensors:** four frame-position streams, five independent fingertip touch sensors, latch touch/depth, wrist position, and replacement-fuse acceleration.
- **Control:** a deterministic nine-stage task plan plus 50 Hz bounded residual control driven by measured palm/object error. The five fingers retain independent joint targets.
- **Recovery:** a deterministic lateral slip is introduced during transport. The controller observes the resulting pose error and corrects the object target. A 24-seed analytic ablation compares the same bounded residual law to an open-loop baseline.
- **Data collection:** every 20 ms sample records task state, sensor position, target, visual-servo error, residual action, five touch forces, active contacts, wrist angle, and latch depth.

The grasp constraint is disclosed rather than hidden: finger closure creates the enveloping grasp, then an equality weld models the high static friction required for repeatable transport. It is released for placement, leaving the fuse as a normal free body.

## Task sequence

1. Inspect the two fuse bays and empty socket.
2. Align and close all five fingers around the failed fuse.
3. Lift and discard it into the isolated reject bin.
4. Align and grasp the replacement fuse.
5. Rotate the wrist 90 degrees while maintaining the grasp.
6. Detect and correct an injected lateral slip.
7. Insert and release the replacement at the socket.
8. Press the safety latch.
9. Verify tolerances and export the labeled episode.

## Generated evidence

| File | Evidence |
|---|---|
| `artifacts/demo.mp4` | Robot, task stages, progress, active contacts and residual magnitude |
| `artifacts/trajectory.json` | 50 Hz observations, actions, tactile forces, stages and errors |
| `artifacts/report.json` | Success, final tolerances, model scale and controller metrics |
| `artifacts/evaluation.json` | 24 fixed-seed open-loop vs residual-controller trials |
| `artifacts/policy_card.json` | Exact policy observations, actions and recovery behavior |

## Rubric map

| Criterion | Direct evidence |
|---|---|
| Runnability | One command, no downloads or learned weights, quick/headless modes, validator |
| MuJoCo depth | Articulated MJCF, free bodies, contacts, 15 actuators, 13 named sensors, constraints |
| Task design | Clear safety-critical, nine-stage repair with measurable terminal conditions |
| Control | Task planner, visual servo, tactile state, bounded residual actions, seeded ablation |
| Dexterity | Five fingers / ten independently actuated finger joints, grasp and wrist reorientation |
| Engineering | Config separated from model/controller, structured artifacts and deterministic seed |
| Presentation | Generated 60-second video with readable live evidence overlays |
| Innovation | A safety interlock repair benchmark combining dexterity, recovery and dataset export |

## Limitations and next steps

The high-level planner is scripted for judging reproducibility, and the post-contact transport uses a disclosed grasp constraint. A learned policy could replace the residual law, while randomized fuse geometries and cap-style keyed insertion would make the benchmark harder.

Before submitting, replace the placeholders in `registration.json` and put the identical UUID in your pull-request description.
