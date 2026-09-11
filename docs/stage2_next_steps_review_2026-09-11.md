# Stage 2 next-step review and execution plan

Date: 2026-09-11

Branch: `feature/stage2a-perfect-pnp-framework`

Reviewed head before this document: `f72a2f8cb1ff1211db53cc29aa304cbe2e7cee6b`

## 1. Current status

Stage2A has progressed from architecture-only code to a numerically validated oracle projection and PnP pipeline. The repository now contains measured gate opening keypoints, a camera-enabled Stage2 environment, a shared OpenCV IPPE pose-recovery backend, dataset collection/evaluation scripts, and a first Stage2B learned ordered-corner detector baseline.

The current measured gate opening is approximately `1.524 m x 1.524 m`, with calibrated corners stored in `assets/gate/gate_keypoints.json` in the gate actor frame. The gate actor origin is not the opening center; the opening center is approximately `(-0.0661411, 0, 1.0668) m` in actor coordinates. `GatePoseRecovery` correctly targets the calibrated opening centroid rather than the actor origin.

The current Stage2A equivalence artifact reports 1,835 complete samples out of 2,048 frames and near-floating-point closure through oracle projection -> OpenCV IPPE PnP -> recovered camera/body pose:

- mean reprojection error: `1.36e-13 px`
- mean gate translation error: `1.25e-14 m`
- mean gate rotation error: `4.85e-7 deg`
- mean body translation error: `1.29e-7 m`
- mean body rotation error: `8.34e-6 deg`

This strongly validates the mathematics, transform directions, and current dataset/PnP consistency.

The first Stage2B detector baseline is also trained and published through Git LFS. On the current held-out split it reports approximately:

- complete detection rate on complete labels: `99.2%`
- corner RMSE at original 256 x 256 resolution: `3.57 px`
- mean / p95 gate translation error: `0.132 / 0.375 m`
- mean / p95 gate rotation error: `9.14 / 23.15 deg`
- mean / p95 recovered body translation error: `0.996 / 2.534 m`

The detector is therefore a useful offline baseline but is not accurate enough for direct closed-loop racing.

## 2. Important distinction: mathematical closure is not yet physical RGB validation

The excellent Stage2A numerical closure does **not by itself prove** that the four calibrated 3D gate keypoints land exactly on the physical opening corners in rendered RGB.

A self-consistent but wrong geometry could still produce nearly zero closure error if the same geometry is used both to generate oracle pixels and to solve PnP.

Therefore the next highest-priority validation is a physical RGB overlay test.

---

# Priority 0 - Required before any further detector optimization

## 3. Add a Stage2A RGB oracle-corner overlay tool

Create a script such as:

```text
scripts/perception/visualize_stage2a_overlay.py
```

It should:

1. launch a small Stage2 camera-enabled environment;
2. sample representative relative gate/camera poses;
3. read the real rendered RGB image;
4. read `T_wg`, `T_wc`, the runtime intrinsic matrix `K`, and the calibrated `GateGeometry`;
5. compute `T_cg_truth = inverse(T_wc) @ T_wg`;
6. project the four calibrated opening corners into the image using the exact Stage2 camera model;
7. draw and label the ordered points `LB`, `RB`, `RT`, `LT`;
8. connect them with a polygon;
9. save the rendered overlays under an ignored directory such as:

```text
outputs/stage2a_overlay/
```

Generate at least 20-50 varied examples covering:

- near / medium / far distance;
- left / right offset;
- up / down offset;
- yaw / pitch / roll variation;
- near-image-edge cases.

### Acceptance criterion

A human inspection must confirm that all four oracle points consistently lie on the **actual physical gate opening corners** in RGB.

If they do not, do **not** tune PnP or the learned detector. Instead fix one of:

- gate keypoint geometry;
- gate actor-frame interpretation;
- camera optical convention;
- runtime camera pose;
- camera intrinsics;
- corner ordering.

Do not use learned offsets or hard-coded pixel corrections.

---

# Priority 1 - Remove calibration duplication and ambiguity

## 4. Create one authoritative Stage2 camera/extrinsic configuration source

The current code repeats the camera-body transform in multiple evaluation scripts. Consolidate it into one reusable source.

For example, add a module such as:

```text
perception/stage2_calibration.py
```

or extend an existing calibration module so every Stage2 component obtains the same values from one place.

The authoritative configuration should include at minimum:

- camera model (`pinhole` for the current reference path);
- camera-body transform `T_bc`;
- optical-frame convention;
- gate keypoint calibration path;
- corner order.

Avoid duplicating this in:

- `scripts/perception/evaluate_stage2a_dataset.py`;
- `scripts/perception/evaluate_gate_detector.py`;
- Stage2 runtime integration code;
- future closed-loop code.

### Specific issue to resolve

Current documentation/configuration semantics are inconsistent: one location still describes the mount as inherited/placeholder-like while training documentation treats it as the current calibrated reference mount.

Resolve this explicitly:

- if `T_bc` has been validated against simulator truth, document it as a validated reference extrinsic;
- if it has not, keep it marked as provisional and add an explicit runtime extrinsic check.

### Acceptance criterion

Changing the Stage2 camera-body calibration in one location must update all Stage2A/Stage2B evaluation and runtime users.

---

# Priority 2 - Add runtime extrinsic verification

## 5. Verify `T_bc` directly from Isaac truth

For representative frames, calculate:

```text
T_bc_truth = inverse(T_wb_truth) @ T_wc_truth
```

Compare this with the configured `T_bc`.

Record:

- translation error in meters;
- rotation error in degrees.

Add a small diagnostic or test report.

### Acceptance criterion

The configured extrinsic should match Isaac truth to numerical precision for the current simulated reference camera. If not, fix frame conventions/configuration before continuing.

---

# Priority 3 - Improve evaluation quality before more training

## 6. Add distance-binned detector and PnP metrics

The current aggregate metrics hide where the pose error is coming from. Add evaluation grouped by distance, for example:

```text
2.5-4.0 m
4.0-5.5 m
5.5-7.0 m
7.0-8.0 m
```

For each bin report:

- number of samples;
- complete detection rate;
- corner mean / RMSE / p95 error;
- PnP success rate;
- gate translation mean / p95;
- gate rotation mean / p95;
- recovered body translation mean / p95;
- recovered body rotation mean / p95.

Also consider binning by projected gate size in pixels, which may be more informative than physical distance alone.

### Goal

Determine whether the large pose errors are dominated by far-range angular resolution, oblique views, visibility classification, or another failure mode.

Do not assume that more training epochs solve this until the error distribution is understood.

---

# Priority 4 - Create a genuinely independent test set

## 7. Do not rely only on a random split of one generated dataset

The current baseline trains and validates from one 2,048-frame dataset using a deterministic shuffled split. Keep this split for development, but create separate datasets for final evaluation.

Suggested minimum structure:

```text
train:       seed 1
validation:  seed 2
test:        seed 3
```

Preferably also vary generation distributions between train and test.

The independent test set should never be used for model selection.

### Acceptance criterion

Report final Stage2B metrics on a separately generated test dataset that was not used for optimization or checkpoint selection.

---

# Priority 5 - Scale up and diversify the synthetic dataset

## 8. Increase data volume

The current 2,048-frame dataset is sufficient for a first baseline but too small for robust gate perception.

Target the next experiment at approximately:

```text
20,000-50,000 frames
```

This is a recommended engineering range, not a hard requirement.

## 9. Expand pose and scene diversity

Do not collect only around gate 0. Add support for:

- random gate index;
- multiple track locations;
- wider yaw/pitch/roll distribution;
- different lateral and vertical offsets;
- more near-edge and partial-visibility cases;
- multiple apparent gate scales;
- multiple gates in view where possible.

## 10. Add domain randomization

At minimum investigate randomization for:

- lighting intensity/direction;
- brightness/contrast/exposure;
- image noise;
- blur and motion blur;
- gate material/color/texture;
- background appearance;
- modest camera-calibration perturbations where physically meaningful.

Keep geometric ground-truth labels exact.

---

# Priority 6 - Improve visibility and occlusion supervision

## 11. Current visibility labels are only geometric in-frame visibility

The current dataset path marks corners visible when they are in front of the camera and within image bounds. That is not the same as true renderer visibility.

Future dataset versions should use depth, segmentation, or another renderer-aware test so a corner can be marked invisible when physically occluded.

Do not silently mix these definitions. Version the dataset schema if the visibility semantics change.

---

# Priority 7 - Detector development

## 12. Keep the current `GateKeypointNet` as a baseline

Do not delete the current compact detector. It is valuable as a small, fast reference model and already has reproducible metrics/checkpoint artifacts.

## 13. Compare against at least one mature keypoint framework

After the evaluation/data pipeline above is improved, compare the current detector against a mature pose/keypoint implementation such as YOLO Pose or another maintained keypoint framework.

Use the same independent test set and compare:

- corner error;
- downstream PnP translation/rotation error;
- detection completeness;
- inference latency;
- GPU memory;
- model size.

Do not select a model based only on image-space corner RMSE. The downstream PnP pose error matters more for this project.

## 14. Investigate pose-aware training objectives only after baseline analysis

If image-space keypoint improvements do not translate into sufficient PnP accuracy, consider an additional differentiable geometric/pose-aware loss or distance-dependent weighting. Do not replace the explicit PnP backend with direct 6-DoF neural regression at this stage.

---

# Priority 8 - Do not connect Stage2B directly to full racing yet

## 15. Current Stage2B accuracy is not sufficient for full-track closed loop

Current baseline errors are approximately:

```text
mean body position error ~= 1.0 m
p95 body position error  ~= 2.53 m
mean rotation error      ~= 9.14 deg
p95 rotation error       ~= 23.15 deg
```

Do not replace the Stage1 oracle gate-relative observation in full racing with these estimates yet.

A reasonable next engineering target before initial controlled closed-loop testing is approximately:

```text
gate translation p95 < 0.10-0.15 m
gate rotation p95    < 5 deg
```

These are project targets, not universal OpenCV requirements.

---

# Priority 9 - Closed-loop integration sequence

When offline accuracy improves sufficiently, proceed incrementally:

1. static single-gate perception playback;
2. single-gate low-speed closed-loop approach;
3. single-gate pass-through;
4. repeated gate approach from randomized starts;
5. two-gate sequence;
6. short multi-gate segment;
7. full track.

At every stage log both perception metrics and flight outcome metrics.

Do not jump directly from the current offline detector to full-track racing.

---

# Priority 10 - Architecture constraints that must remain unchanged

Preserve the current separation:

```text
Stage2A:
Isaac truth -> perfect ordered pixels -> CornerObservation -> shared PnP -> pose

Stage2B:
RGB -> detector -> CornerObservation -> the same shared PnP -> pose
```

Maintain the transform convention:

```text
T_ab maps frame b -> frame a
p_a = R_ab @ p_b + t_ab
T_ab @ T_bc = T_ac
```

Keep these relations:

```text
T_gc = inverse(T_cg)
T_bg = T_bc @ T_cg
T_wc_est = T_wg @ T_gc
T_wb_est = T_wg @ inverse(T_bg)
```

Keep `target_pos_b` defined using the calibrated **opening center**, not the gate actor origin.

Do not:

- rewrite OpenCV PnP;
- introduce a second inconsistent pose backend;
- use gate COM position with actor-frame geometry;
- reintroduce guessed gate dimensions;
- use learned offsets to hide calibration errors;
- directly regress 6-DoF pose from RGB as a replacement for the geometry pipeline;
- merge PR #1 until the next validation stage is complete;
- run `git clean -fdx` because ignored local training artifacts may exist.

---

# Recommended Codex execution order

Use the following order and finish each checkpoint before moving to the next:

## Checkpoint A - Physical Stage2A validation

- implement RGB overlay tool;
- generate 20-50 overlay examples;
- verify corner alignment manually;
- verify `T_bc` from Isaac truth;
- centralize calibration values.

Stop here and report if any overlay/extrinsic mismatch exists.

## Checkpoint B - Evaluation improvements

- add distance/projected-size binned metrics;
- create independent validation/test datasets;
- rerun current detector without changing architecture;
- identify dominant failure regions.

## Checkpoint C - Data improvements

- scale dataset to 20k-50k frames;
- randomize gates, viewpoints, background/lighting/image effects;
- preserve exact oracle labels;
- add renderer-aware occlusion supervision if practical.

## Checkpoint D - Detector comparison

- retrain current `GateKeypointNet`;
- evaluate on untouched test set;
- compare with a mature keypoint model;
- compare downstream PnP pose errors, not only pixel errors.

## Checkpoint E - Controlled closed loop

Only after acceptable offline pose accuracy:

- integrate detector output into the shared `Stage2BPerceptionPipeline`;
- start with one static gate and low speed;
- expand gradually toward multiple gates/full track.

---

# Required report after the next development pass

Codex should report all of the following clearly:

## Physical validation

- path to generated overlay images;
- whether all 4 projected corners align with physical RGB opening corners;
- any systematic pixel offsets;
- measured configured-vs-truth `T_bc` translation/rotation error.

## Calibration cleanup

- single authoritative source of `T_bc` and gate calibration;
- all call sites migrated to it;
- no duplicate hard-coded Stage2 extrinsic remaining.

## Evaluation

- distance-binned detector/PnP metrics;
- independent validation and test dataset manifests;
- current-model test-set performance.

## Data

- number of frames;
- randomization ranges;
- gate indices represented;
- complete/partial/occluded sample counts.

## Detector

- architecture/checkpoint;
- corner error;
- PnP success rate;
- gate pose error;
- body pose error;
- inference latency and hardware.

## Tests

Run at minimum:

```bash
pytest -q tests/perception
```

Also run any Isaac Stage2 integration/overlay diagnostics added during this work.

---

# Final decision rule

Do not optimize for a good-looking detector metric alone.

The project objective is:

```text
rendered RGB
-> physically correct ordered gate corners
-> stable shared PnP
-> sufficiently accurate gate/body-relative pose
-> safe closed-loop gate traversal
```

The next immediate milestone is therefore **physical RGB corner validation + calibration cleanup**, not more epochs and not full-track closed-loop training.
