## 🤖 ThermaSort Battery Cell Triage

**UUID**: `f0aa00b3-8361-4e1f-b0ac-8f6bd01cd00a`

### Key Innovations
- **Autonomous Thermal Triage**: Four-cell inspection and unsafe-cell classification against a configurable 55 °C threshold
- **Contact-Gated Pickup**: Touch sensing verifies tool contact before the electromagnetic gripper is enabled
- **Force-Limited Compliant Grasp**: Bounded spring-damper magnetic attachment handles a free-body cell without direct pose commands
- **Minimum-Jerk Gantry Control**: Smooth 200 Hz closed-loop motion through inspection, pickup, transport, and release

### Tasks (5/5 passed)
1. Thermal Anomaly Identification
2. Contact Verification Before Pickup
3. Minimum Lift Achievement
4. Quarantine Placement
5. Magnetic Force Limit Compliance

### Performance
- Flagged Cell: `cell_2` at 72.4 °C
- Pickup Contact Force: 0.872 N
- Peak Magnetic Force: 7.602 N
- Lift Height: 160.8 mm
- Final Quarantine XY Error: 0.0 mm
- Control Frequency: 200 Hz

### Files
- controller.py - Autonomous controller, evaluation, and video generation
- scene.xml - MuJoCo workcell and robot model
- config.json - Motion, gripper, and success thresholds
- demo.mp4 - 31.8-second simulation video
- evaluation_report.json - Measured results and terminal checks
- trajectory.json - Recorded simulation trajectory
- registration.json - UUID and participant registration
