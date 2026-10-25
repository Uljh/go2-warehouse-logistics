# Running the demos

## 1. Check the repository

```powershell
python scripts/verify_release.py
python scripts/run_demo.py dual_delivery --dry-run
```

The first command checks the packaged files. The dry run checks the launch configuration, writes `outputs/.../launch_probe.json` and exits before starting Isaac Lab. It requires PowerShell 7 on PATH; use `--powershell PATH` to select another installation.

## 2. Set up the environment

The demos were recorded on Windows with an NVIDIA RTX 4090:

| Component | Recorded version |
|---|---|
| Isaac Sim | 6.0.1.0 |
| Isaac Lab | 6.1.17 |
| PyTorch | 2.10.0+cu128 |
| RSL-RL | 5.0.1 |
| NumPy / SciPy | 2.3.1 / 1.17.0 |
| OSQP / Clarabel | 1.0.5 / 0.11.1 |

Install Isaac Sim and Isaac Lab using the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then install the repository dependencies in the Isaac Lab Python environment:

```powershell
C:/IsaacLab/isaaclab.bat -p -m pip install -r requirements-tools.txt
```

## 3. Prepare the assets

Robot meshes, USD assets and simulator binaries are obtained separately under their upstream terms. Download the Z1 description at the version used by the project:

```powershell
git clone --no-checkout https://github.com/unitreerobotics/unitree_ros.git assets/real_arm_z1/upstream/unitree_ros
git -C assets/real_arm_z1/upstream/unitree_ros checkout 8ebecd883a842a4631fcfa98933fae586aeb6709
```

The importer expects `robots/z1_description` below that checkout. The Unitree license is included in [third_party/](../third_party/).

Use the Isaac Python environment (`pxr` required) to cache the NVIDIA robot and warehouse assets. These downloads can be large:

```powershell
python scripts/cache_nvidia_assets.py ground
python scripts/cache_nvidia_assets.py sky
python scripts/cache_nvidia_assets.py warehouse
python scripts/cache_nvidia_assets.py go2
```

Use `--proxy URL` if needed. The downloader preserves asset paths and writes a source/hash manifest.

Import the floating-base Z1:

```powershell
$env:PYTHONPATH = (Get-Location).Path
C:/IsaacLab/isaaclab.bat -p go2_logistics/import_z1_isaac.py
```

Check `artifacts/real_arm_z1/usd/import_validation.json` for the mass, inertia and joint-limit checks, including `fixed_base=false`.

## 4. Launch a demo

```powershell
python scripts/run_demo.py dual_delivery --isaaclab C:/IsaacLab --isaacsim C:/isaacsim --headless
python scripts/run_demo.py far_shelf --isaaclab C:/IsaacLab --isaacsim C:/isaacsim --headless
python scripts/run_demo.py dynamic_navigation --isaaclab C:/IsaacLab --isaacsim C:/isaacsim --headless
```

Run one command at a time. Omit `--headless` to show the simulator.

The launcher prepares a separate workspace, restores the selected demo's source profile and links the asset directories. Each profile supplies its recorded seed, gait weights and controller settings.

**Setup status:** all three launch chains have been checked in dry-run mode. The complete asset-download, import and simulation sequence has not yet been rerun from a fresh clone. The videos and results come from the original runs; GPU physics can vary between environments.

## 5. Review a run

Delivery runs write `fleet_result.json`, per-step measurements, RGB-D observations and videos. The task checks cover parcel placement, arm withdrawal and return.

To inspect the saved data included in this repository:

```powershell
python scripts/verify_release.py --measurements --media --models
```

The measurement checks use Python's standard library; video and model checks also need OpenCV and PyTorch. See [results](RESULTS.md) for the experiment data and [repository checks](VALIDATION.md) for what has been verified.
