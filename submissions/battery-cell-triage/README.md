# ThermaSort Battery Cell Triage

**FFAI Robothon 2026 — Freestyle Category**

Registration UUID: `f0aa00b3-8361-4e1f-b0ac-8f6bd01cd00a`

ThermaSort is an autonomous MuJoCo workcell that inspects four EV battery cells,
identifies a thermal outlier, verifies tool contact, and relocates the unsafe cell
to a guarded quarantine tray. The project was built as an original submission by
NoahBennett.

## Robot platform

The robot is a purpose-built Cartesian gantry with three actuated slide joints and
a compliant electromagnetic pickup tool. Four free-body battery cells interact
with the workbench, tool, and quarantine tray through MuJoCo contact dynamics.

The magnetic attachment is modeled as a bounded spring-damper force applied to the
selected cell. The controller never writes the cell pose. Tool contact is measured
by a MuJoCo touch sensor before the magnet is enabled.

## Autonomous task

1. Visit all four battery-cell inspection stations.
2. Read each cell's simulated thermal-channel value from its MJCF body metadata.
3. Compare the readings with the configured 55 °C safety threshold.
4. Return to the hottest cell and descend until contact is measured.
5. Enable the compliant magnetic tool, lift the cell, and transport it safely.
6. Lower the cell into the guarded tray, release it, and retreat.
7. Export the measured trajectory, events, forces, lift height, and final error.

## Technical highlights

- 200 Hz closed-loop MuJoCo simulation
- Minimum-jerk gantry command interpolation
- Three position-actuated prismatic joints
- Contact-gated pickup using a touch sensor
- Force-limited spring-damper electromagnetic grasp model
- Free-body cell dynamics; no direct object-pose commands
- Automated pass/fail checks and JSON artifact export
- Headless video generation from the same controller used for evaluation

## Files

```text
battery-cell-triage/
├── README.md
├── registration.json
├── requirements.txt
├── config.json
├── scene.xml
├── controller.py
├── run.sh
└── artifacts/
    ├── demo.mp4
    ├── evaluation_report.json
    └── trajectory.json
```

## Run

From this directory:

```bash
python3 -m pip install -r requirements.txt
chmod +x run.sh
./run.sh evaluate
```

Other modes:

```bash
./run.sh record   # regenerate artifacts/demo.mp4 and evaluation artifacts
./run.sh viewer   # run the autonomous task in the interactive MuJoCo viewer
```

## Evaluation

`./run.sh evaluate` writes `artifacts/evaluation_report.json` and
`artifacts/trajectory.json`. The report only contains values measured from the
submitted simulation; no score or success rate is manually asserted.

Latest deterministic evaluation:

| Metric | Measured value |
|---|---:|
| Terminal checks | 5/5 passed |
| Flagged cell | `cell_2` at 72.4 °C |
| Pickup contact force | 0.872 N |
| Peak magnetic force | 7.602 N |
| Lift height | 160.8 mm |
| Final quarantine XY error | 0.0 mm |
| Simulation time | 47.755 s |

The included `artifacts/demo.mp4` is a 31.8-second, 640×480 rendering generated
by `./run.sh record` from the same autonomous episode.

The terminal checks cover:

- correct thermal-anomaly classification;
- contact before pickup;
- minimum lift height;
- final quarantine placement error; and
- compliance with the magnetic force limit.
