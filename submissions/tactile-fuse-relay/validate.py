#!/usr/bin/env python3
"""Validate MJCF, metadata, controller evidence, and terminal task results."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
errors: list[str] = []


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.relative_to(ROOT)}: {exc}")
        return None


config = load_json(ROOT / "config.json") or {}
registration = load_json(ROOT / "registration.json") or {}
try:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
except Exception as exc:
    errors.append(f"MJCF failed to compile: {exc}")
    model = None

if model is not None:
    if model.nu != 15:
        errors.append(f"expected 15 independently controlled actuators, got {model.nu}")
    if model.nsensor < 14 or model.nsensordata < 27:
        errors.append(f"expected >=14 sensors / >=27 channels, got {model.nsensor} / {model.nsensordata}")
    if model.nmocap:
        errors.append("object manipulation must not use directly commanded mocap bodies")
    expected_dt = 1 / config.get("simulation_hz", 0) if config.get("simulation_hz") else None
    if expected_dt is None or not np.isclose(model.opt.timestep, expected_dt):
        errors.append("config simulation_hz does not match the MJCF timestep")
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)}
    required = {f"{finger}_force" for finger in ("thumb", "index", "middle", "ring", "little")}
    if not required.issubset(names):
        errors.append(f"missing fingertip sensors: {sorted(required - names)}")
    finger_actuators = range(4, 14)
    if not all(model.actuator_forcelimited[i] for i in finger_actuators):
        errors.append("all ten finger actuators must have hard force limits")
    if any(np.max(np.abs(model.actuator_forcerange[i])) > 3.01 for i in finger_actuators):
        errors.append("finger actuator force range exceeds configured safety bound")
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if not all(model.eq_obj1id[i] == palm_id for i in range(model.neq)):
        errors.append("grasp constraints must be robot-palm coupled")

if not all(registration.get(key) for key in ("uuid", "participant_name", "project_name")):
    errors.append("registration.json has blank or missing required fields")

report = load_json(ARTIFACTS / "report.json")
evaluation = load_json(ARTIFACTS / "evaluation.json")
trajectory = load_json(ARTIFACTS / "trajectory.json")
policy = load_json(ARTIFACTS / "policy_card.json")

if report:
    failed_checks = [name for name, passed in report.get("terminal_checks", {}).items() if not passed]
    if not report.get("success") or failed_checks:
        errors.append(f"task is not successful; failed checks: {failed_checks or ['missing terminal checks']}")
    thresholds = config.get("success_thresholds", {})
    if report.get("insertion_error_m", float("inf")) >= thresholds.get("insertion_error_m", 0):
        errors.append("reported insertion error exceeds configured threshold")
    if report.get("latch_depth_m", 0) <= thresholds.get("latch_depth_m", float("inf")):
        errors.append("reported latch depth does not exceed configured threshold")
    if report.get("maximum_engaged_fingers", 0) < thresholds.get("minimum_active_fingers", 99):
        errors.append("insufficient measured multi-finger contact")
    if report.get("maximum_touch_force_n", float("inf")) >= thresholds.get("maximum_touch_force_n", 0):
        errors.append("reported touch force exceeds the safety threshold")
    if report.get("maximum_finger_actuator_force", float("inf")) > 3.01:
        errors.append("reported finger actuator force exceeds the XML hard limit")
    if report.get("insertion_depth_error_m", float("inf")) >= thresholds.get("insertion_depth_error_m", 0):
        errors.append("reported insertion depth exceeds configured threshold")
    if report.get("key_yaw_error_rad", float("inf")) >= thresholds.get("key_yaw_error_rad", 0):
        errors.append("reported keyed orientation exceeds configured threshold")
    required_checks = {
        "failed_fuse_physically_contained", "replacement_multifinger_contacts",
        "slip_injected_and_recovered", "insertion_xy_tolerance",
        "insertion_depth_tolerance", "key_orientation_tolerance",
        "touch_force_safe", "safety_latch_interlock",
    }
    if not required_checks.issubset(report.get("terminal_checks", {})):
        errors.append("report is missing strengthened physical terminal checks")

if evaluation:
    if not evaluation.get("method", "").startswith("paired MuJoCo robot-actuator"):
        errors.append("evaluation is not identified as robot-actuator MuJoCo rollouts")
    if evaluation.get("rollouts", 0) < 8:
        errors.append("evaluation needs at least 8 physics rollouts")
    if evaluation.get("residual_success_rate", 0) <= evaluation.get("baseline_success_rate", 1):
        errors.append("residual controller did not improve over open loop")
    if evaluation.get("residual_success_rate", 0) < .90:
        errors.append("residual controller success rate is below 90%")
    if evaluation.get("median_residual_error_m", float("inf")) * 5 >= evaluation.get("median_baseline_error_m", 0):
        errors.append("residual controller does not reduce median error by at least 5x")
    if len(evaluation.get("trials", [])) != evaluation.get("rollouts"):
        errors.append("evaluation rollout count does not match trial records")

if trajectory:
    required_samples = int(config.get("control_hz", 50) * 50)
    if len(trajectory) < required_samples:
        errors.append(f"trajectory is too short: {len(trajectory)} samples")
    times = np.array([row.get("time_s", -1) for row in trajectory])
    if len(times) > 1 and (np.any(np.diff(times) <= 0) or not np.isclose(np.median(np.diff(times)), .02, atol=.001)):
        errors.append("trajectory is not monotonic 50 Hz data")
    sample = trajectory[0]
    for field in ("touch_forces_n", "residual_action_xyz", "action", "replacement_fuse_xyz",
                  "replacement_fuse_quat", "replacement_yaw_rad", "object_tracking_error_m",
                  "grasp_lock_active"):
        if field not in sample:
            errors.append(f"trajectory does not record required field: {field}")
    stages = list(dict.fromkeys(row.get("stage") for row in trajectory))
    expected_stages = [stage["name"] for stage in config.get("stages", [])]
    if stages != expected_stages:
        errors.append("trajectory stage order/coverage does not match config")
    trajectory_peak_force = max(max(row["touch_forces_n"].values()) for row in trajectory)
    if trajectory_peak_force >= config.get("success_thresholds", {}).get("maximum_touch_force_n", 0):
        errors.append("trajectory contains an unsafe fingertip force")
    if report:
        last = trajectory[-1]
        socket = np.asarray(report.get("socket_xyz", [np.nan]*3))
        spare = np.asarray(last["replacement_fuse_xyz"])
        failed = np.asarray(last["failed_fuse_xyz"])
        xy_error = float(np.linalg.norm(spare[:2] - socket[:2]))
        depth_error = float(abs(spare[2] - socket[2]))
        yaw_error = abs((last["replacement_yaw_rad"] - math.pi/2 + math.pi) % (2*math.pi) - math.pi)
        if not np.isclose(xy_error, report.get("insertion_error_m", np.nan), atol=2e-4):
            errors.append("report insertion error does not match trajectory")
        if not np.isclose(depth_error, report.get("insertion_depth_error_m", np.nan), atol=2e-4):
            errors.append("report depth error does not match trajectory")
        if not np.isclose(yaw_error, report.get("key_yaw_error_rad", np.nan), atol=2e-4):
            errors.append("report yaw error does not match trajectory")
        if not (-.025 <= failed[0] <= .125 and .185 <= failed[1] <= .315 and 0 <= failed[2] <= .14):
            errors.append("trajectory terminal failed fuse is outside containment tray")
        if last.get("latch_depth_m", 0) <= config["success_thresholds"]["latch_depth_m"]:
            errors.append("trajectory terminal latch is not engaged")

video = ARTIFACTS / "demo.mp4"
if not video.exists() or video.stat().st_size < 100_000:
    errors.append("demo.mp4 is missing or implausibly small")
else:
    try:
        reader = imageio.get_reader(video)
        metadata = reader.get_meta_data()
        if metadata.get("size") != (960, 540) or not np.isclose(metadata.get("duration", 0), 60, atol=.2):
            errors.append("demo.mp4 must be a 60-second 960x540 presentation")
        for frame_index in (0, 750, 1499):
            frame = reader.get_data(frame_index)
            if frame.std() < 5:
                errors.append(f"demo.mp4 frame {frame_index} is visually blank")
        reader.close()
    except Exception as exc:
        errors.append(f"demo.mp4 cannot be decoded: {exc}")
if not policy or "tactile_control" not in policy:
    errors.append("policy card does not document tactile control")
elif "never commanded directly" not in policy.get("recovery", ""):
    errors.append("policy card does not disclose actuator-only object recovery")

if errors:
    print("VALIDATION FAILED")
    print("\n".join(f"- {error}" for error in errors))
    sys.exit(1)

print(
    "VALIDATION PASSED: "
    f"{model.nu} actuators, {model.nsensor} sensors, {len(trajectory)} control samples, "
    f"{evaluation['rollouts']} paired actuator rollouts, all {len(report['terminal_checks'])} terminal checks passed"
)
