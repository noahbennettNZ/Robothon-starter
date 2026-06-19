#!/usr/bin/env python3
"""Validate model structure, metadata and generated task evidence."""
import json
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent
errors = []

try:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    if model.nu < 15: errors.append(f"expected >=15 actuators, got {model.nu}")
    if model.nsensor < 13: errors.append(f"expected >=13 sensors, got {model.nsensor}")
except Exception as exc:
    errors.append(f"MJCF failed to compile: {exc}")

registration = json.loads((ROOT / "registration.json").read_text())
if not all(k in registration for k in ("uuid", "participant_name", "project_name")):
    errors.append("registration.json is missing required fields")

for name in ("report.json", "evaluation.json", "trajectory.json", "policy_card.json"):
    path = ROOT / "artifacts" / name
    if not path.exists():
        errors.append(f"missing {path.relative_to(ROOT)}; run run.py --quick first")

report_path = ROOT / "artifacts" / "report.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    if not report.get("success"): errors.append("task report does not show success")
    if report.get("actuated_dof", 0) < 15: errors.append("report actuator count is too low")

eval_path = ROOT / "artifacts" / "evaluation.json"
if eval_path.exists():
    evaluation = json.loads(eval_path.read_text())
    if evaluation["residual_success_rate"] <= evaluation["baseline_success_rate"]:
        errors.append("residual controller did not improve over baseline")

if errors:
    print("VALIDATION FAILED")
    print("\n".join(f"- {e}" for e in errors))
    sys.exit(1)
print(f"VALIDATION PASSED: {model.nu} actuators, {model.nsensor} sensors, successful run and improved residual policy")
