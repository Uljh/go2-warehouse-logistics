# Go2 多机械狗仓库物流

我们想把四足机器人的移动能力和机械臂的操作能力结合起来，做一套能自主取件、搬运和归位的仓库物流系统。这是这个项目的出发点。

目前，我们在 **Isaac Sim** 中搭建了由 **Go2 + Z1 机械臂**组成的仓库搬运系统，把识别快递编号、分配任务、抓取、运输、按编号放架和返回待命点的流程串了起来。下面的三个 Demo 分别展示双狗配送、远端货架配送和空载动态绕障。

[English](README.md) · [在线 Demo](https://uljh.github.io/go2-warehouse-logistics/docs/) · [运行说明](docs/REPRODUCING.md) · [系统架构](docs/ARCHITECTURE.md) · [实验数据](docs/RESULTS.md)

[![双狗仓库配送](docs/media/dual_delivery/poster.jpg)](https://uljh.github.io/go2-warehouse-logistics/docs/media/showreel_1x.mp4)

**[观看 52 秒演示短片](https://uljh.github.io/go2-warehouse-logistics/docs/media/showreel_1x.mp4)**，由三个 Demo 的原速仿真片段剪辑而成。

## Demo

| Demo | 展示内容 | 完整录像 |
|---|---|---|
| [双狗双件配送](https://uljh.github.io/go2-warehouse-logistics/docs/media/dual_delivery/demo_1x.mp4) | 两只机械狗识别并抓取快递，分别送到 1、23 号货架，完成放置和返回 | [364 秒](https://uljh.github.io/go2-warehouse-logistics/docs/media/dual_delivery/full_run_1x.mp4) |
| [远端货架配送](https://uljh.github.io/go2-warehouse-logistics/docs/media/far_shelf/demo_1x.mp4) | 夹持快递前往远端 23 号货架，完成放置、撤臂和返回 | [247 秒](https://uljh.github.io/go2-warehouse-logistics/docs/media/far_shelf/full_run_1x.mp4) |
| [动态绕障](https://uljh.github.io/go2-warehouse-logistics/docs/media/dynamic_navigation/demo_1x.mp4) | 两只空载机械狗根据测距信息避让移动箱体，再继续行进并返位 | [37 秒](https://uljh.github.io/go2-warehouse-logistics/docs/media/dynamic_navigation/full_run_1x.mp4) |

## 怎么做的

系统运行在 **Isaac Sim / PhysX** 中。快递是独立的动态刚体，Z1 保留原有约 5.221 kg 的质量和电机限制，通过夹爪接触完成抓取和搬运。

感知端用 **RGB-D 图像**读取快递编号和抓取位置；任务层分配订单，并预约取件区、通道和货架区域，协调两只机械狗的行动。导航层根据测距信息更新局部占据地图，用 **A\*** 规划路径，并在遇到移动障碍时退让。

Go2 的行走使用经过 **RSL-RL / PPO** 适配的预训练步态策略；Z1 通过逆运动学和夹爪力反馈完成操作，静止和制动阶段配合受约束的支撑控制。步态推理以 50 Hz 运行，物理和接触控制以 200 Hz 运行。

模块关系和代码入口见[系统架构](docs/ARCHITECTURE.md)，步态模型与训练代码见[模型说明](docs/MODEL_CARD.md)和 [training/](training/)。

## 查看与运行

```powershell
git clone https://github.com/Uljh/go2-warehouse-logistics.git
cd go2-warehouse-logistics
python -m http.server 8000
```

打开 **http://localhost:8000/docs/** 即可观看 Demo。

运行仿真需要安装 **Isaac Sim 6.0.1 / Isaac Lab 6.1.17**，并准备机器人和仓库资产，具体步骤见[运行说明](docs/REPRODUCING.md)。仓库中提供了控制代码、模型权重、训练源码快照和各 Demo 的运行配置。

## 后续计划

最终目标是**可配置数量的多机械狗、可并行独立工作的仓库物流搬运系统**。目前在多狗并行、长程工作方面还存在一定挑战，后续仓库会持续更新。

## 致谢

项目基于 NVIDIA Isaac Sim / Isaac Lab、PhysX、Unitree 机器人描述、PyTorch 和 RSL-RL 构建。第三方来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)；本项目尚未指定统一的开源许可证。
