# Checking the repository

The source files, measurements and videos were checked against the original local experiment archive on 2026-10-25.

| Check | Result |
|---|---|
| Captured source files restored per demo | Dual 60/60; far 62/62; dynamic 66/66 |
| Controller Python preserved during host-path adaptation | Dual 54; far 56; dynamic 60 files |
| Full-recording hashes | All three match the original files |
| Complete 200 Hz measurement projections | Dual 72,814; far 49,431; dynamic 7,434 steps, with both robots present |
| Body and gripper-force limits | Rechecked across all published measurement frames |
| Video decode | Showreel 261; dual highlights 506/full 1,820; far excerpt 150/full 1,235; dynamic cameras 185/full 185 frames |
| Checkpoint hashes | Both match the original weights |
| Actor structure | 48 inputs, 128/128/128 hidden units, 12 outputs; all actor tensors finite |
| Launch configuration | All three dry runs completed |
| PPO adaptation sources | Two five-file snapshots match their recorded hashes |
| Python syntax | 109 runtime, overlay, tool and training sources parsed |

To run the checks locally:

```powershell
python scripts/verify_release.py --measurements --media --models
```

These commands inspect the saved files. Environment setup and simulation launch status are covered in the [setup guide](REPRODUCING.md). Shared dependencies recovered outside the original source snapshots are listed in [shared_dependency_provenance.json](../repro/shared_dependency_provenance.json).
