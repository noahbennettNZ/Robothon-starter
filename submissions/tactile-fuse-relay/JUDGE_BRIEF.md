# Judge brief: Tactile Fuse Relay

Start with `artifacts/demo.mp4`, then inspect `artifacts/report.json` and `artifacts/evaluation.json`. The submission is intentionally compact: one MJCF model and one controller generate all evidence.

The most score-relevant behavior is the **SLIP RECOVERY** stage. A known disturbance offsets the carried replacement. Frame-position feedback drives bounded residual actions, all logged at 50 Hz. `evaluation.json` repeats that residual law for 24 fixed disturbances and provides an open-loop baseline.

The model exposes 15 controlled DOF, five touch sensors, frame sensors, acceleration, contacts, two free objects and a safety latch. All five fingers contain independent proximal/distal joints. The disclosed equality constraint represents high static grasp friction after closure; it is not used before contact or after release.

Success is executable rather than claimed: insertion error, latch depth, release state and recovery metrics are calculated after the simulation. `validate.py` checks the MJCF, registration schema, generated artifacts, task success and ablation improvement.
