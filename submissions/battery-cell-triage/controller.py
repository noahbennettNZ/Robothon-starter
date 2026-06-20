#!/usr/bin/env python3
"""Autonomous controller for the ThermaSort battery-cell triage task."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent


class ThermaSortController:
    """Scan cells and move the hottest one with a compliant magnetic tool."""

    def __init__(self, frame_callback: Optional[Callable[[mujoco.MjData], None]] = None):
        self.config = json.loads((ROOT / "config.json").read_text())
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.frame_callback = frame_callback

        self.cell_names = self.config["cell_names"]
        self.cell_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.cell_names
        }
        self.tool_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tool"
        )
        self.touch_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "tool_contact"
        )
        self.touch_adr = self.model.sensor_adr[self.touch_sensor_id]

        magnet = self.config["magnetic_gripper"]
        self.magnet_kp = float(magnet["spring_n_per_m"])
        self.magnet_kd = float(magnet["damping_ns_per_m"])
        self.magnet_max_force = float(magnet["max_force_n"])
        self.magnet_offset = np.asarray(magnet["cell_offset_m"], dtype=float)

        self.magnet_cell: Optional[str] = None
        self.peak_magnet_force = 0.0
        self.peak_touch_force = 0.0
        self.step_count = 0
        self.trajectory: list[dict] = []
        self.events: list[dict] = []

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = (0.0, 0.0, 0.08)
        self.magnet_cell = None
        self.peak_magnet_force = 0.0
        self.peak_touch_force = 0.0
        self.step_count = 0
        self.trajectory.clear()
        self.events.clear()
        self.step(180)

    def _apply_magnet(self) -> None:
        if self.magnet_cell is None:
            return

        body_id = self.cell_ids[self.magnet_cell]
        target = self.data.xpos[self.tool_id] + self.magnet_offset
        position_error = target - self.data.xpos[body_id]
        tool_velocity = self.data.cvel[self.tool_id, 3:6]
        cell_velocity = self.data.cvel[body_id, 3:6]
        force = self.magnet_kp * position_error + self.magnet_kd * (
            tool_velocity - cell_velocity
        )
        magnitude = float(np.linalg.norm(force))
        if magnitude > self.magnet_max_force:
            force *= self.magnet_max_force / magnitude
            magnitude = self.magnet_max_force
        self.data.xfrc_applied[body_id, :3] = force
        self.peak_magnet_force = max(self.peak_magnet_force, magnitude)

    def _sample_trajectory(self) -> None:
        if self.step_count % 10:
            return
        sample = {
            "time_s": round(float(self.data.time), 4),
            "gantry_qpos": np.round(self.data.qpos[:3], 6).tolist(),
            "tool_xyz": np.round(self.data.xpos[self.tool_id], 6).tolist(),
            "magnet_active": self.magnet_cell is not None,
        }
        if self.magnet_cell:
            sample["held_cell"] = self.magnet_cell
            sample["held_cell_xyz"] = np.round(
                self.data.xpos[self.cell_ids[self.magnet_cell]], 6
            ).tolist()
        self.trajectory.append(sample)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.data.xfrc_applied[:] = 0.0
            self._apply_magnet()
            mujoco.mj_step(self.model, self.data)
            touch = float(self.data.sensordata[self.touch_adr])
            self.peak_touch_force = max(self.peak_touch_force, touch)
            self.step_count += 1
            self._sample_trajectory()
            if self.frame_callback:
                self.frame_callback(self.data)

    def move_to(self, x: float, y: float, depth: float, label: str) -> float:
        """Move the gantry with minimum-jerk command interpolation."""
        start = self.data.ctrl[:3].copy()
        goal = np.array([x, y, depth], dtype=float)
        distance = float(np.linalg.norm(goal - start))
        interpolation_steps = max(80, int(500 * distance))

        for index in range(interpolation_steps):
            t = (index + 1) / interpolation_steps
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            self.data.ctrl[:3] = start + blend * (goal - start)
            self.step()

        tolerance = float(self.config["motion"]["position_tolerance_m"])
        max_steps = int(self.config["motion"]["max_steps"])
        error = math.inf
        for _ in range(max_steps):
            error = float(np.linalg.norm(self.data.qpos[:3] - goal))
            if error <= tolerance and np.linalg.norm(self.data.qvel[:3]) < 0.02:
                break
            self.step()

        self.events.append(
            {
                "event": label,
                "time_s": round(float(self.data.time), 3),
                "target": np.round(goal, 5).tolist(),
                "position_error_m": round(error, 6),
            }
        )
        return error

    def read_temperature(self, cell_name: str) -> float:
        """Read the MJCF thermal-channel value attached to a cell body."""
        body_id = self.cell_ids[cell_name]
        return float(self.model.body_user[body_id, 0])

    def run_task(self) -> dict:
        self.reset()
        motion = self.config["motion"]
        safe_depth = float(motion["transport_depth_m"])
        scan_depth = float(motion["scan_depth_m"])
        pickup_depth = float(motion["pickup_depth_m"])
        quarantine_xy = np.asarray(self.config["quarantine_xy_m"], dtype=float)

        print("ThermaSort: scanning battery cells")
        readings: dict[str, float] = {}
        initial_positions = {
            name: self.data.xpos[body_id].copy()
            for name, body_id in self.cell_ids.items()
        }

        for name in self.cell_names:
            cell_xy = self.data.xpos[self.cell_ids[name], :2].copy()
            self.move_to(cell_xy[0], cell_xy[1], safe_depth, f"approach_{name}")
            self.move_to(cell_xy[0], cell_xy[1], scan_depth, f"scan_{name}")
            readings[name] = self.read_temperature(name)
            print(f"  {name}: {readings[name]:.1f} C")

        hottest = max(readings, key=readings.get)
        limit = float(self.config["thermal_limit_c"])
        flagged = readings[hottest] > limit
        self.events.append(
            {
                "event": "classification",
                "hottest_cell": hottest,
                "temperature_c": readings[hottest],
                "limit_c": limit,
                "flagged": flagged,
            }
        )
        print(f"Flagged {hottest} at {readings[hottest]:.1f} C")

        hot_id = self.cell_ids[hottest]
        hot_xy = self.data.xpos[hot_id, :2].copy()
        self.move_to(hot_xy[0], hot_xy[1], safe_depth, "return_to_hot_cell")
        self.move_to(hot_xy[0], hot_xy[1], pickup_depth, "contact_pickup")
        self.step(80)
        pickup_touch = float(self.data.sensordata[self.touch_adr])

        self.magnet_cell = hottest
        self.events.append(
            {
                "event": "magnet_engaged",
                "cell": hottest,
                "touch_force_n": round(pickup_touch, 5),
            }
        )
        self.step(100)
        self.move_to(hot_xy[0], hot_xy[1], safe_depth, "lift")
        lifted_z = float(self.data.xpos[hot_id, 2])
        lift_height = lifted_z - float(initial_positions[hottest][2])

        self.move_to(
            quarantine_xy[0], quarantine_xy[1], safe_depth, "transport_to_quarantine"
        )
        self.move_to(quarantine_xy[0], quarantine_xy[1], 0.335, "lower_into_tray")
        self.magnet_cell = None
        self.events.append({"event": "magnet_released", "cell": hottest})
        self.step(240)
        self.move_to(quarantine_xy[0], quarantine_xy[1], safe_depth, "retreat")
        self.step(120)

        final_position = self.data.xpos[hot_id].copy()
        quarantine_error = float(np.linalg.norm(final_position[:2] - quarantine_xy))
        thresholds = self.config["success_thresholds"]
        checks = {
            "thermal_anomaly_identified": bool(flagged and hottest == "cell_2"),
            "contact_verified_before_pickup": pickup_touch > 0.01,
            "minimum_lift_achieved": lift_height
            >= float(thresholds["minimum_lift_m"]),
            "cell_inside_quarantine": quarantine_error
            <= float(thresholds["quarantine_xy_error_m"]),
            "magnetic_force_within_limit": self.peak_magnet_force
            <= float(thresholds["maximum_peak_force_n"]) + 1e-6,
        }
        success = all(checks.values())

        report = {
            "project": "ThermaSort Battery Cell Triage",
            "uuid": "559b2a48-eaa8-4831-8217-ae515f020daa",
            "success": success,
            "simulation_time_s": round(float(self.data.time), 3),
            "control_frequency_hz": self.config["control_frequency_hz"],
            "thermal_readings_c": readings,
            "flagged_cell": hottest,
            "pickup_touch_force_n": round(pickup_touch, 5),
            "peak_touch_force_n": round(self.peak_touch_force, 5),
            "peak_magnetic_force_n": round(self.peak_magnet_force, 5),
            "lift_height_m": round(lift_height, 6),
            "quarantine_xy_error_m": round(quarantine_error, 6),
            "final_cell_xyz": np.round(final_position, 6).tolist(),
            "checks": checks,
            "events": self.events,
        }
        print("PASS" if success else "FAIL", json.dumps(checks, indent=2))
        return report


def save_artifacts(report: dict, trajectory: list[dict]) -> None:
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (artifact_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "project": report["project"],
                "sample_interval_steps": 10,
                "samples": trajectory,
            },
            indent=2,
        )
        + "\n"
    )


def evaluate() -> int:
    controller = ThermaSortController()
    report = controller.run_task()
    save_artifacts(report, controller.trajectory)
    return 0 if report["success"] else 1


def record_video(output: Path) -> int:
    import imageio.v2 as imageio

    renderer_holder: dict[str, object] = {}
    frame_index = 0

    def capture(data: mujoco.MjData) -> None:
        nonlocal frame_index
        frame_index += 1
        if frame_index % 10:
            return
        renderer = renderer_holder["renderer"]
        camera = renderer_holder["camera"]
        writer = renderer_holder["writer"]
        renderer.update_scene(data, camera=camera)
        writer.append_data(renderer.render())

    controller = ThermaSortController(frame_callback=capture)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=30, codec="libx264", quality=8)
    renderer = mujoco.Renderer(controller.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.02, 0.27]
    camera.distance = 1.25
    camera.azimuth = 135
    camera.elevation = -28
    renderer_holder.update(renderer=renderer, camera=camera, writer=writer)
    try:
        report = controller.run_task()
        save_artifacts(report, controller.trajectory)
    finally:
        writer.close()
        renderer.close()
    print(f"Wrote {output}")
    return 0 if report["success"] else 1


def run_viewer() -> int:
    import mujoco.viewer

    controller: Optional[ThermaSortController] = None
    viewer = None
    try:
        controller = ThermaSortController()
        viewer = mujoco.viewer.launch_passive(controller.model, controller.data)

        def sync(_: mujoco.MjData) -> None:
            if viewer and viewer.is_running():
                viewer.sync()
                time.sleep(controller.model.opt.timestep)

        controller.frame_callback = sync
        report = controller.run_task()
        save_artifacts(report, controller.trajectory)
        return 0 if report["success"] else 1
    finally:
        if viewer:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--record", type=Path, metavar="OUTPUT_MP4")
    mode.add_argument("--viewer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.evaluate:
        raise SystemExit(evaluate())
    if args.record:
        raise SystemExit(record_video(args.record))
    raise SystemExit(run_viewer())
