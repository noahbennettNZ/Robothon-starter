#!/usr/bin/env python3
"""Validate MJCF, metadata, controller evidence, and terminal task results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    if model.nsensor < 13 or model.nsensordata < 23:
        errors.append(f"expected >=13 sensors / >=23 channels, got {model.nsensor} / {model.nsensordata}")
    expected_dt = 1 / config.get("simulation_hz", 0) if config.get("simulation_hz") else None
    if expected_dt is None or not np.isclose(model.opt.timestep, expected_dt):
        errors.append("config simulation_hz does not match the MJCF timestep")
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)}
    required = {f"{finger}_force" for finger in ("thumb", "index", "middle", "ring", "little")}
    if not required.issubset(names):
        errors.append(f"missing fingertip sensors: {sorted(required - names)}")

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

if evaluation:
    if not evaluation.get("method", "").startswith("paired MuJoCo"):
        errors.append("evaluation is not identified as paired MuJoCo rollouts")
    if evaluation.get("rollouts", 0) < 8:
        errors.append("evaluation needs at least 8 physics rollouts")
    if evaluation.get("residual_success_rate", 0) <= evaluation.get("baseline_success_rate", 1):
        errors.append("residual controller did not improve over open loop")
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
    for field in ("touch_forces_n", "residual_action_xyz", "action", "replacement_fuse_xyz"):
        if field not in sample:
            errors.append(f"trajectory does not record required field: {field}")
    stages = {row.get("stage") for row in trajectory}
    expected_stages = {stage["name"] for stage in config.get("stages", [])}
    if stages != expected_stages:
        errors.append(f"trajectory stage coverage mismatch: {sorted(expected_stages - stages)}")

video = ARTIFACTS / "demo.mp4"
if not video.exists() or video.stat().st_size < 100_000:
    errors.append("demo.mp4 is missing or implausibly small")
if not policy or "tactile_control" not in policy:
    errors.append("policy card does not document tactile control")

if errors:
    print("VALIDATION FAILED")
    print("\n".join(f"- {error}" for error in errors))
    sys.exit(1)

print(
    "VALIDATION PASSED: "
    f"{model.nu} actuators, {model.nsensor} sensors, {len(trajectory)} control samples, "
    f"{evaluation['rollouts']} paired physics rollouts, all terminal checks passed"
)
