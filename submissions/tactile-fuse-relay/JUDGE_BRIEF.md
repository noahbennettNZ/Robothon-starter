# Judge brief: Tactile Fuse Relay

Start with `artifacts/demo.mp4`, then inspect `artifacts/report.json` and `artifacts/evaluation.json`. The submission is intentionally compact: one MJCF model and one controller generate all evidence.

The most score-relevant behavior is the **SLIP RECOVERY** stage. A known disturbance offsets the carried replacement. Frame-position feedback drives bounded residual actions, all logged at 50 Hz. `evaluation.json` contains 24 paired MuJoCo physics rollouts under fixed disturbances, not an analytic proxy.

The model exposes 15 controlled DOF, five touch sensors, frame sensors, acceleration, collision filtering, two free objects, an open reject tray and a spring-return safety latch. The five opposed radial fingers contain independent proximal/distal joints and independent force limiting. The disclosed equality constraint represents high static grasp friction; code gates it on measured contact and disables it before physical placement.

Success is executable rather than claimed: eight terminal checks cover both grasps, measured contacts, reject containment, recovery, physical release, insertion tolerance and latch depth. `validate.py` also checks MJCF/config consistency, 50 Hz trajectory integrity, complete actions, all stages, video presence and rollout evidence.
