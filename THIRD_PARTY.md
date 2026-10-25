# Third-party provenance

| Component | Use | Upstream terms / source |
|---|---|---|
| Isaac Sim / PhysX | Rigid-body physics, rendering, contact sensors, asset import | [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/), separately installed; vendor asset terms remain applicable |
| Isaac Lab | Robot task configuration and policy runtime | [Isaac Lab](https://github.com/isaac-sim/IsaacLab), upstream BSD-3-Clause and dependency notices |
| Unitree Go2 assets | Original robot USD supplied by the simulator asset distribution | [NVIDIA robot assets](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html); downloaded separately |
| Unitree Z1 description | Original URDF, inertial data, joint limits and meshes | [unitree_ros](https://github.com/unitreerobotics/unitree_ros), pinned commit `8ebecd883a842a4631fcfa98933fae586aeb6709`; [BSD-3-Clause notice](third_party/UNITREE_ROS_LICENSE.txt) |
| RSL-RL | PPO and policy export used for locomotion adaptation | [RSL-RL](https://github.com/leggedrobotics/rsl_rl), [upstream license](third_party/RSL_RL_LICENSE.txt) |
| PyTorch / NumPy / SciPy / OSQP / Clarabel | Learning, numerical kinematics and constrained support allocation | Each upstream project's license applies |

Original Unitree mass/inertia/limit constants reproduced by `z1_asset.py` retain the Unitree notice above. Meshes, generated USD assets and simulator binaries are not included in this repository. The source and weights describe a simulation research prototype and are not a hardware control deployment package.

This initial publication makes project source visible for review; no project-wide open-source license has been selected. Third-party licenses are retained, and no additional rights over vendor assets are asserted.
