# Captured locomotion-adaptation source

The two folders preserve the actual implementation files recorded when the published gait checkpoints were trained:

| Snapshot | Weight role | Captured files |
|---|---|---|
| `carry/` | General mounted gait used for the accepted delivery recordings | Training script, adaptation environment, mounted asset helpers and original host launcher |
| `empty_return/` | Empty/stowed return specialist and dynamic-navigation actor | The corresponding earlier versions of those five files |

Every captured file is unchanged, with its original SHA-256 in `source_manifest.json`. The full run settings, source checkpoint hash, measured arm-pose bank and training scope are in `weights/<role>/manifest.json`.

This is PPO adaptation of pretrained locomotion for a mounted original-mass arm. The recorded scope uses empty-gripper home/stow and measured extended poses; it is not learned contact grasping or an end-to-end warehouse policy. Historical training evaluation scores are not used as current reliability claims.

The archived PowerShell launchers contain historical host paths and are retained as provenance. Retraining requires adapting host paths, obtaining the source checkpoint and matching the recorded simulator/environment versions. No retraining is launched by this publication, and this archive is not presented as a newly validated training recipe.

RSL-RL and robot-description terms remain applicable; see `../THIRD_PARTY.md`.
