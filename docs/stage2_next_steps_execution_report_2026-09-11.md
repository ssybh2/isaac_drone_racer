# Stage 2 next-step execution report

Date: 2026-09-11

Source plan: `docs/stage2_next_steps_review_2026-09-11.md`

## Decision

Checkpoints A-D were executed. Checkpoint E was not started because no tested
model reached the required offline gate-pose target of translation p95 below
0.10-0.15 m and rotation p95 below 5 degrees. The best tested model reaches
0.266 m and 15.34 degrees on the untouched seed-3 test set. Full-track or
single-gate closed-loop integration is therefore not authorized by the plan.

## Checkpoint A: physical RGB and extrinsic validation

The diagnostic generated 32 rendered overlays covering all seven gates,
2.67-7.89 m range, 57.9-159.9 px projected width, lateral/vertical offsets,
yaw/pitch/roll, and image-edge cases.

- Local generated path: `outputs/stage2a_overlay/`
- Tracked copy: `artifacts/stage2_next_steps_20260911/physical_validation/`
- Four-corner result: pass. `LB`, `RB`, `RT`, and `LT` consistently land on the
  physical inner opening corners in the contact sheet and individually checked
  near, far, rolled, and edge samples.
- Systematic pixel offset: none visible in the 32-image manual inspection.
- Configured-vs-Isaac `T_bc` translation error: mean `1.87e-7 m`, maximum
  `4.20e-7 m`.
- Configured-vs-Isaac `T_bc` rotation error: mean `1.10e-5 deg`, maximum
  `1.93e-5 deg`.

`perception/stage2_calibration.py` is now the authoritative source for the
pinhole camera model, OpenCV/ROS optical convention, camera mount parameters,
`T_bc`, gate calibration path, and ordered gate geometry. Dataset collection,
Stage2A evaluation, Stage2B evaluation, Stage2 camera configuration, and the
Stage2 runtime use that source. No duplicate hard-coded Stage2 `T_bc` remains
in those paths.

## Checkpoint B: independent and binned evaluation

Three seeds have distinct roles:

- train seed 1: expanded 20,480-frame set (Checkpoint C);
- validation seed 2: 2,048 frames, used for checkpoint selection;
- test seed 3: 2,048 frames, never used for optimization or selection.

The validation/test manifests are tracked under
`artifacts/stage2_next_steps_20260911/params/`. Both use the original reference
distribution and gate 0 so the comparison with the published current model is
controlled. The expanded training distribution covers every gate.

### Published current model on untouched seed-3 test data

Detector/PnP metrics by distance:

| Distance (m) | Frames | Complete GT | Complete detection | Corner mean / RMSE / p95 (px) | PnP success |
|---|---:|---:|---:|---:|---:|
| 2.5-4.0 | 570 | 358 | 97.21% | 1.825 / 5.244 / 3.096 | 97.21% |
| 4.0-5.5 | 611 | 605 | 100.00% | 1.593 / 4.310 / 2.947 | 99.83% |
| 5.5-7.0 | 564 | 564 | 100.00% | 1.656 / 4.563 / 2.944 | 99.65% |
| 7.0-8.0 | 277 | 277 | 100.00% | 1.994 / 4.527 / 4.201 | 99.64% |

Pose metrics by distance (mean / p95):

| Distance (m) | Gate translation (m) | Gate rotation (deg) | Body translation (m) | Body rotation (deg) |
|---|---:|---:|---:|---:|
| 2.5-4.0 | 0.064 / 0.205 | 4.77 / 12.79 | 0.297 / 0.888 | 4.77 / 12.79 |
| 4.0-5.5 | 0.094 / 0.255 | 6.64 / 15.98 | 0.579 / 1.398 | 6.64 / 15.98 |
| 5.5-7.0 | 0.140 / 0.379 | 9.65 / 21.00 | 1.085 / 2.446 | 9.65 / 21.00 |
| 7.0-8.0 | 0.207 / 0.487 | 14.06 / 28.50 | 1.839 / 3.739 | 14.06 / 28.50 |

Detector/PnP metrics by projected gate width:

| Width (px) | Frames | Complete GT | Complete detection | Corner mean / RMSE / p95 (px) | PnP success |
|---|---:|---:|---:|---:|---:|
| <64 | 249 | 249 | 100.00% | 2.034 / 4.726 / 4.252 | 99.60% |
| 64-96 | 904 | 904 | 100.00% | 1.660 / 4.750 / 2.955 | 99.67% |
| 96-128 | 478 | 456 | 99.78% | 1.562 / 3.859 / 2.893 | 99.78% |
| >=128 | 417 | 196 | 95.41% | 1.966 / 5.419 / 3.674 | 95.41% |

Pose metrics by projected gate width (mean / p95):

| Width (px) | Gate translation (m) | Gate rotation (deg) | Body translation (m) | Body rotation (deg) |
|---|---:|---:|---:|---:|
| <64 | 0.212 / 0.483 | 14.26 / 28.92 | 1.872 / 3.786 | 14.26 / 28.92 |
| 64-96 | 0.128 / 0.345 | 8.87 / 20.81 | 0.954 / 2.366 | 8.87 / 20.81 |
| 96-128 | 0.080 / 0.240 | 5.63 / 15.06 | 0.430 / 1.157 | 5.63 / 15.06 |
| >=128 | 0.061 / 0.201 | 4.73 / 10.48 | 0.274 / 0.675 | 4.73 / 10.48 |

The dominant failure region is unambiguous: far gates whose apparent width is
below 64 px. The high RMSE despite a lower p95 also reveals a small number of
large keypoint outliers. At the opposite extreme, the largest/nearest gates
contain most partial and edge cases, which explains their lower complete
detection rate.

## Checkpoint C: expanded data

The new training set contains 20,480 frames (about 1.9 GiB locally):

- all gate indices 0-6, sampled uniformly;
- approach distance 2.5-8.0 m;
- lateral offset +/-1.8 m and vertical offset +/-1.2 m;
- yaw +/-0.35 rad, pitch +/-0.25 rad, roll +/-0.30 rad;
- dome intensity 1,800-4,200 and RGB tint 0.65-1.0;
- image gain 0.65-1.35, contrast 0.8-1.2, noise sigma 0-6 levels;
- Gaussian blur probability 15% and directional motion blur probability 10%;
- 11,843 complete and 8,637 partial samples; no fully out-of-frame samples.

Every label records the realized pose, light, and image randomization values.
The labels remain exact because photometric transforms do not move pixels.
Background diversity comes from all seven track locations and other gates in
view. Gate material randomization and physically meaningful camera-calibration
perturbations were investigated but not enabled in this pass.

Visibility remains explicitly versioned as
`geometric_in_front_and_in_frame`. Renderer-aware depth/segmentation occlusion
was not added because the calibrated keypoints lie exactly on the inner
silhouette boundary, where a naive single-pixel depth comparison would produce
unstable labels. This work does not claim renderer-aware occlusion supervision.

## Checkpoint D: training and detector comparison

The shared OpenCV IPPE PnP backend was unchanged. Both models were trained on
the 20,480-frame set and selected using all 2,048 seed-2 validation frames.

Untouched seed-3 aggregate comparison:

| Metric | Published compact | Retrained compact | Torchvision Keypoint R-CNN |
|---|---:|---:|---:|
| Parameters | 527,976 | 527,976 | 59,083,869 |
| Complete detection | 99.45% | 99.06% | 99.94% |
| PnP success | 99.22% | 99.06% | 99.72% |
| Corner mean / RMSE / p95 (px) | 1.720 / 4.619 / 3.239 | 2.028 / 4.726 / 4.071 | 1.530 / 7.047 / 2.437 |
| Gate translation mean / p95 (m) | 0.120 / 0.365 | 0.126 / 0.332 | 0.096 / 0.266 |
| Gate rotation mean / p95 (deg) | 8.36 / 21.89 | 9.21 / 21.95 | 6.74 / 15.34 |
| Body translation mean / p95 (m) | 0.877 / 2.578 | 0.964 / 2.581 | 0.701 / 1.893 |
| Body rotation mean / p95 (deg) | 8.36 / 21.89 | 9.21 / 21.95 | 6.74 / 15.34 |
| RTX 4090 latency mean / p95 (ms) | 1.48 / 1.61 | 1.47 / 1.54 | 11.00 / 14.47 |
| Peak CUDA allocation (MiB) | 14.97 | 14.97 | 336.70 |
| Checkpoint size (MiB) | 2.03 | 6.08 | 225.72 |

The retrained compact model improves far-range translation p95 but regresses
some image-space and angular metrics. Keypoint R-CNN materially improves most
downstream pose metrics and completeness, but has rare large outliers, is much
larger/slower, was trained from scratch for only three epochs, and still misses
the closed-loop target. This is an engineering comparison, not a general claim
about Keypoint R-CNN.

## Commands and tests

The key commands were:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. ./.conda-env/bin/python \
  scripts/perception/visualize_stage2a_overlay.py --headless --enable_cameras \
  --output outputs/stage2a_overlay --num_samples 32 --width 256 --height 256 --seed 11

OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. ./.conda-env/bin/python \
  scripts/perception/collect_stage2_dataset.py --headless --enable_cameras \
  --output logs/stage2/datasets/2026-09-11_multigate_dr_seed1_train_20480 \
  --num_envs 64 --batches 320 --width 256 --height 256 --seed 1 \
  --min_distance 2.5 --max_distance 8.0 --lateral_range 1.8 \
  --vertical_range 1.2 --yaw_range 0.35 --pitch_range 0.25 --roll_range 0.30 \
  --random_gates --randomize_images --randomize_lighting

PYTHONPATH=. ./.conda-env/bin/python scripts/perception/train_gate_detector.py \
  --dataset logs/stage2/datasets/2026-09-11_multigate_dr_seed1_train_20480 \
  --validation_dataset logs/stage2/datasets/2026-09-11_oracle_gate_256_seed2_validation \
  --output logs/stage2/models/2026-09-11_gate_keypoint_multigate_dr_seed1/best_detector.pt \
  --epochs 30 --batch_size 128 --input_size 128 --width 32 --seed 1 --device cuda

PYTHONPATH=. ./.conda-env/bin/python scripts/perception/train_torchvision_keypoint_detector.py \
  --dataset logs/stage2/datasets/2026-09-11_multigate_dr_seed1_train_20480 \
  --validation_dataset logs/stage2/datasets/2026-09-11_oracle_gate_256_seed2_validation \
  --output logs/stage2/models/2026-09-11_torchvision_keypointrcnn_seed1/best_detector.pt \
  --epochs 3 --batch_size 16 --image_size 128 --seed 1 --device cuda
```

Final verification includes `pytest -q tests/perception`, Python compilation,
`git diff --check`, the 32-frame Isaac overlay diagnostic, two independent
2,048-frame dataset collections, the 20,480-frame randomized collection, both
training runs, and unified seed-3 evaluation.
