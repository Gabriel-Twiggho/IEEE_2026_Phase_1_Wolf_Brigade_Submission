# Wolf Brigade — IEEE SMCS SAR Competition 2026 Phase 1

This repository contains Wolf Brigade's Python submission for the
[2026 IEEE SMCS Search and Rescue Competition — Phase 1](https://github.com/IEEE-SMCS/2026-ieee-smcs-competition-phase-1).

The evaluation system clones this repository into
`controllers/proposed_solution`. All commands below assume that placement and
should be run from that directory.

## Solution overview

The solution has two connected stages:

1. **Flyover information extraction**
   - Estimates the UAV path from IMU data and the origin mat's ArUco markers.
   - Detects victims with a bundled Ultralytics YOLO segmentation model,
     projects detections into the competition reference frame, and fuses
     repeated observations.
   - Segments walls with a bundled SegFormer model and projects the results
     into the required binary `600 x 600` map.
2. **Two-robot autonomous mission**
   - Uses `proposed_solution.py` as the Webots controller entrypoint.
   - Seeds planning from the extracted wall map, victim estimates, and UAV
     path.
   - Balances victim assignments between both ROSbots and plans routes with
     map-aware, UAV-corridor-weighted A*.
   - Combines wheel/compass odometry, lidar, RGB-D and infrared sensing for
     live mapping, replanning, terrain handling, collision avoidance and local
     recovery.
   - Confirms victims with a second bundled YOLO segmentation model plus depth
     measurements, then sends an exactly-once supervisor report.

All model weights are included under `models/`; no model download or external
API is required at runtime.

## Main files

| Path | Purpose |
| --- | --- |
| `proposed_solution.py` | Required Webots controller entrypoint |
| `mission_extraction.py` | One-command flyover preprocessing entrypoint |
| `requirements.txt` | Exact pip dependency versions |
| `extraction/` | UAV path, victim-location and wall-map extraction |
| `coordinator/` | Two-robot assignment, route planning and coordination |
| `robot/` | Sensing, mapping, navigation, safety and victim reporting |
| `models/` | Bundled ground-victim, drone-victim and wall models |
| `parameters.py` | Human-facing ground-controller defaults |
| `extraction/config/` | Flyover pipeline configuration |

## Environment requirements

- Webots **R2025a**
- Python **3.10.12** (Python 3.10 or newer is required)
- Git LFS for cloning the official competition repository and its recordings
- A CUDA-capable NVIDIA GPU is recommended for real-time model inference;
  CPU execution is supported but slower

The Webots `controller` module is supplied by Webots and must not be installed
from PyPI.

## Environment setup

Create the virtual environment inside `controllers/proposed_solution`.

### Linux or macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Webots, open `Tools > Preferences > General` and set **Python command** to
the absolute venv interpreter path:

- Linux/macOS: `<submission-directory>/.venv/bin/python`
- Windows: `<submission-directory>\.venv\Scripts\python.exe`

## Required flyover preprocessing command

Run the following from `controllers/proposed_solution` before starting the
corresponding world:

```bash
python mission_extraction.py --video /recordings/small_world
```

Here, `/recordings/` is a portable shortcut for the official competition
repository's `recordings` directory; it is not an absolute filesystem path.
The command finds `small_world_flyover.mp4` and automatically uses the sibling
`small_world_flyover.csv` IMU file. The following forms are also accepted:

- `--video small_world`
- `--video recordings/small_world`
- `--video /recordings/small_world_flyover.mp4`
- `--video /an/explicit/path/small_world_flyover.mp4`

Use `--imu /an/explicit/path/file.csv` only when the IMU file does not share
the resolved video's base name. Wait for the `MISSION EXTRACTION COMPLETE`
message, then reload/open the Webots world so both robots read the new mission
files.

Add `--debug` only when diagnostic images and detailed detection files are
needed:

```bash
python mission_extraction.py --video /recordings/small_world --debug
```

When running this submission outside the official repository layout, point it
at a recordings directory once:

```bash
export SAR_RECORDINGS_DIR=/path/to/2026-ieee-smcs-competition-phase-1/recordings
```

On Windows PowerShell:

```powershell
$env:SAR_RECORDINGS_DIR = "C:\path\to\2026-ieee-smcs-competition-phase-1\recordings"
```

The preprocessing command writes:

| Output | Consumer |
| --- | --- |
| `sim_logs/victim_location_estimates.csv` | Official extraction scorer and robot coordinator |
| `sim_logs/map_estimate.png` | Official extraction scorer and robot path planner |
| `sim_logs/drone_path.csv` | UAV-corridor route planner |
| `sim_logs/map_estimate_info.json` | Map geometry loader |

## Running the autonomous mission

1. Complete flyover preprocessing for the selected world.
2. Confirm that Webots is configured to use the venv interpreter.
3. Open the selected official `.wbt` world in Webots.
4. Start the simulation.

Webots automatically launches `proposed_solution.py` for both ROSbots. The
default `normal` mode enables autonomous coordination, navigation, victim
confirmation and reporting. Do not run `proposed_solution.py` directly from a
normal terminal because the Webots-provided `controller` module and simulated
devices are only available in a Webots controller process.

For a CPU-only ground-controller run, set the following environment variable
before launching Webots:

```bash
export VICTIM_MODEL_DEVICE=cpu
```

On Windows PowerShell:

```powershell
$env:VICTIM_MODEL_DEVICE = "cpu"
```

The flyover extraction configuration defaults to CUDA but automatically falls
back to CPU if CUDA is unavailable.
