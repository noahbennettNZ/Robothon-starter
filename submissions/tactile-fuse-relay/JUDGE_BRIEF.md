# Judge brief: Tactile Fuse Relay

Start with `artifacts/demo.mp4`, then inspect `artifacts/report.json` and `artifacts/evaluation.json`. The submission is intentionally compact: one MJCF model and one controller generate all evidence.

The most score-relevant behavior is the **SLIP RECOVERY** stage. A known disturbance changes the measured palm/object grasp frame. Frame-position feedback drives bounded robot XYZ residual actions at 50 Hz; the object is never commanded directly. `evaluation.json` contains 24 paired MuJoCo robot-actuator rollouts under identical fixed-seed disturbances, not an analytic proxy.

The model exposes 15 controlled DOF, 14 sensors / 27 channels, collision filtering, two keyed free objects, an open reject tray and a spring-return safety latch. The five opposed radial fingers contain independent proximal/distal joints, tactile unloading and ten XML hard force limits. The disclosed equality state represents high static grasp friction; it is coupled to the actuated palm, captures the measured contact transform, is gated on real contact and is disabled before physical placement. The model contains zero mocap bodies.

Success is executable rather than claimed: twelve terminal checks cover both grasps, measured contacts, strict tray containment, injected/recovered slip, physical release, XY/depth/key-yaw insertion, force safety and the conditional latch interlock. `validate.py` independently recomputes terminal geometry from the trajectory, checks hard actuator limits, rejects mocap manipulation, decodes three video frames, verifies ordered 50 Hz evidence and requires a 5x median ablation improvement.
