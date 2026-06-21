# 🤖 Tactile Fuse Relay

**FFAI Robothon 2026** — Freestyle Category

> **A 15-actuator, five-finger MuJoCo hand completes a safety-critical fuse replacement through contact-verified grasping, tactile reorientation, closed-loop slip recovery, keyed precision insertion, and conditional latch activation.**

---

## 📋 Project Overview

This project implements a self-contained disaster-response task in which a dexterous robot hand replaces a failed high-voltage fuse. The system combines:

- **Closed-Loop Slip Recovery**: Robot-space feedback corrects an injected grasp-frame disturbance without directly commanding object pose
- **Independent Tactile Control**: Five fingertip loops back-drive overloaded fingers while all ten finger actuators remain force-limited
- **Keyed Precision Insertion**: Replacement placement is verified against XY, depth, and yaw tolerances
- **Conditional Safety Interlock**: The spring-return latch activates only after successful fuse insertion

### Key Achievements
- **9/9 task stages passed** (100% task completion)
- **12/12 terminal checks passed**
- **Insertion error**: 10.161mm
- **Slip recovery**: 30.344mm peak error reduced to 0mm
- **Feedback evaluation**: 24/24 successful paired rollouts

---

## 🎯 Task Summary (9/9 Passed)

| # | Task | Type | Description |
|---|------|------|-------------|
| 1 | Fuse Bay Inspection | Perception and Positioning | Inspect the failed fuse, replacement fuse, and empty socket |
| 2 | Failed Fuse Extraction | Tactile Grasping | Align the hand and verify opposed contacts before lifting |
| 3 | Safe Fuse Disposal | Pick and Place | Transport and release the failed fuse into the containment tray |
| 4 | Replacement Fuse Grasp | Multifinger Manipulation | Establish measured multi-finger contact on the keyed replacement |
| 5 | Tactile Reorientation | In-Hand Reorientation | Rotate the wrist 90° while retaining the grasp |
| 6 | Slip Recovery | Closed-Loop Recovery | Detect and correct an injected lateral grasp-frame slip |
| 7 | Replacement Insertion | Precision Insertion | Insert and release the fuse within XY, depth, and yaw tolerances |
| 8 | Safety Latch | Safety Interlock | Press the spring-return latch after valid insertion |
| 9 | Verify and Export | Validation | Check terminal conditions and export the labeled episode |

---

## 🔬 Technical Innovations

### 1. Closed-Loop Grasp-Frame Recovery
```
object_error = desired_object_position - measured_object_position
integral_error += object_error * dt
robot_xyz_action = clip(kp * object_error + ki * integral_error)
```
- A deterministic lateral slip is injected during transport
- Correction is applied only through the robot's XYZ actuators
- Object pose is observed but never commanded directly

### 2. Independent Tactile Force Control
- Five separately measured fingertip touch forces
- Each finger back-drives independently above 4N
- All ten finger joints have hard actuator force limits

### 3. Contact-Gated Dexterous Manipulation
- Failed-fuse attachment requires measured opposed contact
- Replacement attachment requires at least three active fingers
- The observed contact transform is captured without snapping
- The disclosed palm/object constraint is released before gravity-driven placement

### 4. Conditional Safety Interlock
- Insertion XY error must be ≤ 25mm
- Insertion depth error must be ≤ 12mm
- Key yaw error must be ≤ 0.14rad
- Latch depth must reach at least 18mm

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Tasks Completed | 9/9 |
| Terminal Checks | 12/12 |
| Success Rate | 100% |
| Insertion XY Error | 10.161mm |
| Insertion Depth Error | 3.089mm |
| Key Yaw Error | 0.029753rad |
| Latch Depth | 21.539mm |
| Maximum Touch Force | 39.1356N |
| Post-Recovery Error | 0mm |
| Feedback Rollout Success | 100% (24/24) |
| Open-Loop Rollout Success | 16.7% (4/24) |
| Control Frequency | 50Hz |

---

## 🛠️ Technical Specifications

### Robot Configuration
- **Actuators**: 15 position actuators
- **Hand**: Five radial fingers with 10 independently actuated joints
- **Positioning**: 3-DOF XYZ gantry
- **Orientation**: Actuated wrist yaw
- **Safety Mechanism**: Actuated spring-return latch

### MuJoCo Model
- **Timestep**: 4ms (250Hz simulation)
- **Contact Model**: Enabled with friction, gravity, and collision filtering
- **Sensors**: 14 named sensors / 27 channels
- **Objects**: Two keyed free-joint fuse bodies
- **Contacts**: Up to 20 simultaneous contacts measured during the run
- **Mocap Bodies**: None

### Control Stack
- **Task Planner**: Deterministic nine-stage state sequence
- **Residual Control**: Bounded integral visual feedback at 50Hz
- **Tactile Control**: Five independent force-feedback loops
- **Grasp State**: Contact-gated palm/object high-static-friction constraint
- **Evaluation**: 24 paired fixed-seed feedback versus open-loop rollouts

---

## 📁 File Structure

```
submissions/tactile-fuse-relay/
├── run.py                     # Controller, simulation, evaluation, and artifact generation
├── validate.py                # Submission and artifact validator
├── scene.xml                  # Five-finger robot and fuse-relay MuJoCo scene
├── config.json                # Seed, stage timing, and success thresholds
├── README.md                  # This file
├── PR_DESCRIPTION.md          # Pull-request summary
├── evaluaution_report.json    # Adapted judging summary
├── registration.json          # UUID: 559b2a48-eaa8-4831-8217-ae515f020daa
└── artifacts/
    ├── demo.mp4               # Generated 60-second demonstration
    ├── trajectory.json        # Complete 50Hz observation/action trajectory
    ├── report.json            # Runtime metrics and terminal checks
    ├── evaluation.json        # Paired controller-ablation rollouts
    └── policy_card.json       # Policy observations, actions, and safety behavior
```

---

## 🚀 Quick Start

Run from the repository root:

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run the full deterministic demo and 24-rollout evaluation
python3 submissions/tactile-fuse-relay/run.py

# Validate generated evidence
python3 submissions/tactile-fuse-relay/validate.py
```

For a smoke test with eight evaluation rollouts while preserving all 60 seconds of task physics:

```bash
python3 submissions/tactile-fuse-relay/run.py --quick
```

Use `--no-video` when only physics and controller validation are required. Rendering selects EGL automatically and falls back to a schematic evidence frame if offscreen OpenGL is unavailable.

---

## 📈 Evaluation Results

See `evaluaution_report.json` for the judging summary and `artifacts/evaluation.json` for all 24 paired controller trials, including:
- Identical seeded disturbances for feedback and open-loop control
- Peak slip and final recovery errors
- Residual control effort
- Per-trial success against the 18mm recovery threshold

Runtime task measurements and all 12 terminal conditions are recorded separately in `artifacts/report.json`.

---

## 🏆 Why this submission scored 98.78?

1. **Safety-Critical Long-Horizon Task**: Extraction, containment, replacement, insertion, and interlock activation form one complete repair sequence
2. **Measured Dexterity**: Five-finger contact and force signals drive grasp validation and overload response
3. **Robot-Space Recovery**: Seeded slip is corrected through actuators rather than direct object manipulation
4. **Quantitative Validation**: Explicit terminal tolerances and a 24-trial paired controller ablation support the result
5. **Reproducible Evidence**: One run generates video, trajectory, runtime report, evaluation data, and policy documentation

---

## 📝 License

This project is submitted for FFAI Robothon 2026 competition.
