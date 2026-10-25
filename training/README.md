# Locomotion training sources

These folders contain the adaptation code captured when the published gait checkpoints were trained.

| Folder | Checkpoint role | Contents |
|---|---|---|
| [carry/](carry/) | General mounted gait used in the delivery demos | Training script, adaptation environment, mounted-asset helpers and host launcher |
| [empty_return/](empty_return/) | Empty/stowed return gait, also used for dynamic navigation | The corresponding earlier versions of those five files |

Training uses **RSL-RL / PPO** to adapt a pretrained Go2 policy to the mounted Z1 arm. Both runs use an empty gripper with home/stow and measured extended arm poses. Grasping and arm motion are controlled separately in the warehouse runtime.

Each folder's `source_manifest.json` records the original file hashes. The settings, source checkpoint hash and arm-pose bank are in the [carry manifest](../weights/carry/manifest.json) and [return manifest](../weights/empty_return/manifest.json).

For retraining, update the archived host paths, obtain the source checkpoint and match the environment versions in the manifests. The archived launchers retain their original machine paths.

See the [model card](../docs/MODEL_CARD.md) for the policy interface and [THIRD_PARTY.md](../THIRD_PARTY.md) for upstream terms.
