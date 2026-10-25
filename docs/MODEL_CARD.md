# Locomotion models

The project uses two Go2 locomotion checkpoints, adapted from a pretrained policy with RSL-RL / PPO.

| Checkpoint | Runtime role | Training settings |
|---|---|---|
| [carry/mounted_gait.pt](../weights/carry/mounted_gait.pt) | General mounted-robot gait used in the delivery demos | [manifest](../weights/carry/manifest.json) |
| [empty_return/mounted_gait.pt](../weights/empty_return/mounted_gait.pt) | Empty/stowed return gait; also used in the dynamic-navigation demo | [manifest](../weights/empty_return/manifest.json) |

## Policy interface

The actor takes **48 inputs**, has three hidden layers of **128 units** each and produces **12 leg-joint actions**. It runs at **50 Hz**, alongside the warehouse's **200 Hz** physics and contact-control loop.

The files are RSL-RL PyTorch checkpoint dictionaries containing `actor_state_dict`, `critic_state_dict` and training metadata. The main policy is loaded through `OnPolicyRunner`; the return specialist loads `actor_state_dict` into the matching actor architecture.

## Training and use

Both adaptations trained mounted-robot locomotion with an empty gripper, using home/stow and measured extended arm poses. The `carry` name describes the checkpoint's role in the delivery runtime. Arm motion and contact grasping are handled by separate controllers.

The return specialist is selected according to payload state, task phase and measured arm stow, then blended in gradually. Support and recovery controllers can also adjust the final leg targets.

The [training directory](../training/) contains the captured adaptation sources. Settings, source checkpoint hashes and arm-pose banks are recorded in the manifests beside the weights.

These models were developed in simulation; hardware transfer remains future work. See [THIRD_PARTY.md](../THIRD_PARTY.md) for upstream sources and terms.
