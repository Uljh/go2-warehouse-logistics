# Reproducing the demos

## 1. Inspect the publication first

```powershell
python scripts/verify_release.py
python scripts/run_demo.py dual_delivery --dry-run
```

The dry run executes the recorded wrapper chain, captures actual arguments/environment in `outputs/.../launch_probe.json`, then deliberately stops before calling Isaac Lab. It never creates a physical success result. PowerShell7 must be on PATH, or pass `--powershell PATH`.

## 2. Environment

Recorded stack: Windows, NVIDIA RTX4090, Isaac Sim6.0.1.0, Isaac Lab6.1.17, PyTorch2.10.0+cu128, RSL-RL5.0.1, NumPy2.3.1, SciPy1.17.0, OSQP1.0.5, Clarabel0.11.1. These versions describe the original recordings; other combinations are not validated. Simulator installation is separate from `requirements-tools.txt`.

Install Isaac Sim and Isaac Lab from their [official instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Use the configured Isaac Lab Python environment for this repository's dependencies:

```powershell
C:/IsaacLab/isaaclab.bat -p -m pip install -r requirements-tools.txt
```

## 3. Original vendor assets

This repository includes neither simulator binaries nor vendor mesh/USD bundles. Obtain them under their upstream terms. Pin the Z1 description:

```powershell
git clone --no-checkout https://github.com/unitreerobotics/unitree_ros.git assets/real_arm_z1/upstream/unitree_ros
git -C assets/real_arm_z1/upstream/unitree_ros checkout 8ebecd883a842a4631fcfa98933fae586aeb6709
```

The source expects `robots/z1_description` below that checkout. The original URDF/gripper inertial and joint-limit values are preserved; the required Unitree license is retained in `third_party/`.

Cache official Isaac6 robot and warehouse dependencies with the Isaac Python environment (`pxr` required). These commands download unchanged published assets and may transfer substantial data:

```powershell
python scripts/cache_nvidia_assets.py ground
python scripts/cache_nvidia_assets.py sky
python scripts/cache_nvidia_assets.py warehouse
python scripts/cache_nvidia_assets.py go2
```

An explicit `--proxy URL` is supported when needed. The downloader preserves upstream relative paths and writes source/hash manifests. Network/asset-layout changes are surfaced as errors rather than silently substituting a robot model.

Import and audit the original floating-base Z1 with the official URDF importer:

```powershell
$env:PYTHONPATH = (Get-Location).Path
C:/IsaacLab/isaaclab.bat -p go2_logistics/import_z1_isaac.py
```

Check `artifacts/real_arm_z1/usd/import_validation.json`: it must pass the original mass/inertia/joint-limit checks and have `fixed_base=false`. This is an asset check, not a grasp or locomotion pass.

## 4. Run one profile

```powershell
python scripts/run_demo.py dual_delivery --isaaclab C:/IsaacLab --isaacsim C:/isaacsim --headless
python scripts/run_demo.py far_shelf --isaaclab C:/IsaacLab --isaacsim C:/isaacsim --headless
python scripts/run_demo.py dynamic_navigation --isaaclab C:/IsaacLab --isaacsim C:/isaacsim --headless
```

Run separately. Omitting `--headless` shows the simulator. Each run uses a fresh workspace/output directory, reconstructs its captured source profile, and links prepared asset directories. The original recipe contains its recorded seed, gait files and control switches; the launcher only adapts local host paths and provides a dry-run boundary.

Current source snapshot coverage is60/62/66 captured files for dual/far/dynamic. Additional shared dependencies absent from those historical manifests are identified in `repro/shared_dependency_provenance.json`; recovered files are not presented as historically frozen. Fresh-run GPU physics can diverge even with the same seed.

**Publication validation:** the three wrapper chains, captured source identities, source syntax, model interfaces, measurement projections and videos were checked. The portable fresh-checkout asset-download/import/full-physics sequence was not rerun during publication. The recordings prove the original accepted runs, not this new environment setup.

## 5. Review the output

Delivery runs write `fleet_result.json`, per-step traces, RGB-D observation evidence and videos. Success requires actual parcel placement, arm retreat and the original return checks. A timeout or guard stop remains a failure.

The `evidence/<demo>/auditors/` directory preserves the original review-source versions. These expect a complete native run directory. The compact public projections are independently checked with:

```powershell
python scripts/verify_release.py --measurements --media --models
```

No result should be inferred solely from a launch message, a policy filename or a selected actor counter.
