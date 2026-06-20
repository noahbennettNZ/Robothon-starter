#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "${1:-evaluate}" in
  evaluate)
    python3 controller.py --evaluate
    ;;
  record)
    MUJOCO_GL="${MUJOCO_GL:-egl}" python3 controller.py --record artifacts/demo.mp4
    ;;
  viewer)
    python3 controller.py --viewer
    ;;
  *)
    echo "Usage: ./run.sh [evaluate|record|viewer]" >&2
    exit 2
    ;;
esac
