# Experimental results

Three simulation runs are included, each with a full recording, measured results and its original experiment report.

| Demo | Duration | Delivered | Placement error | Return-position error |
|---|---:|---:|---:|---:|
| [Two-robot delivery](../evidence/dual_delivery/acceptance.json) | 364.070 s | 2 | 2.12 / 5.42 mm | 19.87 / 19.97 mm |
| [Far-shelf delivery](../evidence/far_shelf/acceptance.json) | 247.155 s | 1 | 5.42 mm | 19.65 mm for the active robot |
| [Dynamic navigation](../evidence/dynamic_navigation/acceptance.json) | 37.170 s | Unloaded | — | 24.86 / 32.07 mm |

In the two-robot run, parcels are delivered to shelves 1 and 23. Navigation and arm tasks overlap, while the robots hold their parcels at different times. The far-shelf run completes one delivery to shelf 23.

The dynamic-navigation run records a maximum cart contact force of **0 N**. It uses a scripted moving cart and PhysX ranges with **1 cm synthetic noise, 5 mm quantization and 3% dropout**. Localization uses simulator pose.

These measurements describe the three recorded runs. A repeated-trial success rate has not yet been measured.

## Physical setup

The Z1 retains its approximately **5.221 kg** mass, original joint/motor limits and **20 N** gripper-force guard. Parcels are independent dynamic bodies, grasped through contact. The runs use no parcel binding, runtime pose resets or rescue forces.

## Further work

The goal is a configurable fleet that can work independently in parallel. The current shared-world runtime has two robots; multi-robot parallel carrying and longer continuous delivery runs are the main areas of work. Onboard localization, continuous replenishment, dropped-parcel recovery and picking from clutter are also planned.

## Experiment files

Each `evidence/<demo>/` folder contains:

- `acceptance.json`: the original experiment verdict and metrics.
- `recorded_configuration.json` and `original_source_manifest.json`: run settings and captured source hashes.
- `measurements_200hz.jsonl.gz`: robot, parcel, contact and task-phase measurements at the original 200 Hz rate.
- `projection_manifest.json`: data coverage, included fields and hashes.
- `auditors/`: the review scripts used with the original full run directory.

The public measurement files omit large joint/action arrays and raw image/range streams. Local machine/user path prefixes have been redacted from the reports. Original file hashes are retained.

Video sources, edit ranges and frame counts are listed in [media_manifest.json](media/media_manifest.json). Full recordings are sampled at 5 fps; their final sampled frames precede the end of the physical traces by less than one frame interval.

For data checks, see [VALIDATION.md](VALIDATION.md).
