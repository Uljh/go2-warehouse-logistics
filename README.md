# Go2 Warehouse Logistics

We started this project to bring together a quadruped's mobility and a robot arm's ability to handle objects: a warehouse system that can pick up parcels, carry them and put them on the right shelves.

We've built a **Go2 + Z1 mobile manipulation system in Isaac Sim**, connecting parcel identification, task assignment, grasping, transport, shelf placement and return. The three demos below show two-robot delivery, delivery to a distant shelf and unloaded navigation around a moving obstacle.

[中文说明](README.zh-CN.md) · [Watch the demos](https://uljh.github.io/go2-warehouse-logistics/docs/) · [Setup guide](docs/REPRODUCING.md) · [Architecture](docs/ARCHITECTURE.md) · [Experimental results](docs/RESULTS.md)

[![Go2 warehouse delivery](docs/media/dual_delivery/poster.jpg)](https://uljh.github.io/go2-warehouse-logistics/docs/media/showreel_1x.mp4)

**[Watch the 52-second showreel](https://uljh.github.io/go2-warehouse-logistics/docs/media/showreel_1x.mp4)** — original-speed simulation clips from all three demos.

## Demos

| Two robots, two deliveries | Delivery to shelf 23 | Dynamic obstacle avoidance |
|:---:|:---:|:---:|
| [![Dual delivery](docs/media/dual_delivery/preview.gif)](https://uljh.github.io/go2-warehouse-logistics/docs/media/dual_delivery/demo_1x.mp4) | [![Far shelf](docs/media/far_shelf/preview.gif)](https://uljh.github.io/go2-warehouse-logistics/docs/media/far_shelf/demo_1x.mp4) | [![Dynamic navigation](docs/media/dynamic_navigation/preview.gif)](https://uljh.github.io/go2-warehouse-logistics/docs/media/dynamic_navigation/demo_1x.mp4) |
| Both robots identify and grasp parcels, deliver to shelves 1 and 23, then return | One robot carries a parcel to the far shelf, places it, retracts the arm and returns | Two unloaded robots yield to a moving cart using range measurements, then continue |
| [364 s full recording](https://uljh.github.io/go2-warehouse-logistics/docs/media/dual_delivery/full_run_1x.mp4) | [247 s full recording](https://uljh.github.io/go2-warehouse-logistics/docs/media/far_shelf/full_run_1x.mp4) | [37 s full recording](https://uljh.github.io/go2-warehouse-logistics/docs/media/dynamic_navigation/full_run_1x.mp4) |

## How it works

The system runs in **Isaac Sim / PhysX**. Parcels are independent dynamic bodies, and the Z1 uses its original approximately 5.221 kg mass and motor limits. Grasping and carrying rely on gripper contact.

**RGB-D images** provide parcel codes and grasp positions. A shared scheduler assigns orders and reserves receiving zones, corridors and shelf areas. Range measurements update a local occupancy map for **A\*** path planning and yielding around moving obstacles.

Go2 walks using a pretrained locomotion policy adapted with **RSL-RL / PPO**. The Z1 uses inverse kinematics and gripper-force feedback, with constrained support control during stops and braking. The gait policy runs at 50 Hz; physics and contact control run at 200 Hz.

See the [architecture](docs/ARCHITECTURE.md) for module connections and code entry points, and the [model card](docs/MODEL_CARD.md) and [training sources](training/) for locomotion details.

## View and run

```bash
git clone https://github.com/Uljh/go2-warehouse-logistics.git
cd go2-warehouse-logistics
python -m http.server 8000
```

Open **http://localhost:8000/docs/** to watch the demos.

Running the simulation requires **Isaac Sim 6.0.1 / Isaac Lab 6.1.17** and the robot and warehouse assets. Follow the [setup guide](docs/REPRODUCING.md) for installation and launch commands. This repository includes the controllers, model weights, training source snapshots and demo configurations.

## What's next

Our goal is a warehouse logistics system with a configurable fleet of robot dogs that can work independently in parallel. Parallel operation and long-duration tasks remain the main challenges, and we'll continue to update this repository.

## Credits

Built with NVIDIA Isaac Sim / Isaac Lab, PhysX, Unitree robot descriptions, PyTorch and RSL-RL. See [THIRD_PARTY.md](THIRD_PARTY.md) for sources and terms. A project-wide open-source license has not yet been selected.
