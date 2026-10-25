# Locomotion model card

| File | Role | SHA-256 |
|---|---|---|
| `weights/carry/mounted_gait.pt` | General mounted-robot locomotion used for the accepted delivery recordings | `80e6088b17bd3c0ee4194c7ece34d2a21ed25c925d13b80f26807a0da266613d` |
| `weights/empty_return/mounted_gait.pt` | Empty/stowed return specialist; also the base actor in the dynamic navigation recording | `47234403cf76658274023344985e78f58df776b95c9649e9f033426cf0b1177e` |

These are RSL-RL PyTorch checkpoint dictionaries containing `actor_state_dict`, `critic_state_dict` and training metadata, not TorchScript exports. They were adapted from pretrained Go2 locomotion using RSL-RL/PPO in the project. The actor has48 input values, hidden widths128/128/128 and12 leg-joint outputs. Nominal inference is50Hz; physical/contact control is200Hz in the Z1 warehouse.

The runtime loads the checkpoint through RSL-RL's `OnPolicyRunner`; a return specialist is loaded from `actor_state_dict` into the matching actor architecture. The publication model check uses `torch.load(..., weights_only=True)` to verify identity, tensor dimensions and finiteness; it does not claim a new inference or rollout test.

The name `carry` identifies its runtime role. The training scope in the supplied manifests is mounted-arm home/stow plus measured extended poses with an empty gripper; this is not contact-grasp training or an end-to-end learned manipulation policy.

Training manifests are supplied beside each weight with original machine/user path prefixes redacted. Historical evaluation numbers are provenance, not a newly verified success-rate estimate. Current portfolio claims come from the three named physical recordings and their original independent audits.

The actual captured adaptation implementations are preserved in [training/](../training), including each run's five-file source manifest. They are research source snapshots, not a freshly validated one-command training pipeline.

The return actor is gated by empty payload state, phase and measured arm stow, with a finite blend. Support/recovery controllers may still have final action authority; actor selection and physical task success are audited separately.

Models were developed and recorded in simulation on a known environment. No hardware transfer, onboard perception, random warehouse robustness or larger-fleet performance is established. Original simulator, robot description and RSL-RL terms remain applicable; see `THIRD_PARTY.md`.
