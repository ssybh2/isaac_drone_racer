# Stage 2 perception architecture: oracle geometry first, learned detector second

This document is the implementation contract for Stage2A and Stage2B. The
important design decision is that **Stage2A and Stage2B differ only in how the
four image corners are produced**. PnP, frame transforms, pose recovery and
metrics are shared.

## 1. Correct mental model

Stage2A is not a learned perception model. It is a geometric reference/calibration
pipeline:

```text
Isaac gate actor pose ─┐
Isaac optical camera pose ─┼─> T_cg truth ─> calibrated 3D gate keypoints
Isaac camera K ───────────┘                         |
                                                     v
                                           exact 2D projection
                                                     |
                                                     v
                                            CornerObservation
                                                     |
                                                     v
                                        shared PnP / pose recovery
                                                     |
                   ┌─────────────────────────────────┼──────────────────────────┐
                   v                                 v                          v
                 T_cg                              T_bg                  target_pos_b
                   |                                 |
                   v                                 v
           camera pose in gate                body pose in gate/world
```

The Stage2A residual against Isaac truth is a **validation/calibration signal**.
A persistent Stage2A error should first be treated as a wrong gate geometry,
camera model, intrinsics, extrinsic, corner order or frame convention. Do not
train a network to hide a deterministic transform bug.

Stage2B changes only the upstream corner source:

```text
Isaac RGB -> GateCornerDetector -> CornerObservation -> SAME pose backend
```

During simulator training/evaluation, the Stage2A oracle projection can still be
computed in parallel as a label/metric, but it is not fed to the detector or
policy.

## 2. Frame convention

Every transform follows:

```text
T_ab : frame b -> frame a
p_a = R_ab p_b + t_ab
T_ab @ T_bc = T_ac
```

Frames:

- `W`: Isaac world.
- `B`: drone body/root frame.
- `C`: camera optical frame used by OpenCV/ROS: +X right, +Y down, +Z forward.
- `G`: **actual gate actor frame** used by the gate USD / RigidObjectCollection.

PnP returns `T_cg`.

With calibrated camera mount `T_bc`:

```text
T_bg = T_bc @ T_cg
target_pos_b = translation(T_bg)
```

PnP also gives camera relative to gate:

```text
T_gc = inverse(T_cg)
```

When the active gate map pose `T_wg` is known:

```text
T_wc_est = T_wg @ T_gc
T_wb_est = T_wg @ inverse(T_bg)
```

These are compared with Isaac `T_wc_truth` and `T_wb_truth` for Stage2A
equivalence tests.

## 3. Gate geometry: never guess 1.0 m x 1.0 m

PnP object points must be the four physical opening corners expressed in the
same **gate actor frame** whose pose is read from Isaac.

The old Stage2A placeholder used a hard-coded 1 m x 1 m opening. That is removed.
`assets/gate/gate_keypoints.json` is now the intended calibration artifact and
must contain:

```json
{
  "frame": "gate_actor",
  "corner_order": ["left_bottom", "right_bottom", "right_top", "left_top"],
  "object_points_gate_actor_m": [
    [0.0, 0.75, -0.75],
    [0.0, -0.75, -0.75],
    [0.0, -0.75, 0.75],
    [0.0, 0.75, 0.75]
  ]
}
```

The numbers above are an **example schema only**, not the measured gate asset.

Use `scripts/perception/inspect_gate_usd.py` inside the Isaac/pxr environment to
inspect the USD actor bounds and prim hierarchy. The full mesh bounding box is
diagnostic only; the four opening corners still need to be identified from the
asset geometry. Save those corners explicitly. Do not use a rigid-body COM pose
with actor-frame keypoints.

## 4. Camera model and extrinsics

The base repository currently configures a fisheye camera, while the reference
PnP path uses a pinhole/OpenCV calibration. Those models must not be silently
mixed.

For the first Stage2A calibration milestone, use one of these explicit choices:

1. configure a pinhole camera and use its `intrinsic_matrices`, or
2. keep fisheye rendering but undistort the detected/oracle pixels into a
   calibrated pinhole image before PnP.

`CameraCalibration` intentionally rejects an unknown/non-pinhole model. This is
a guardrail, not a limitation hidden inside the solver.

The Isaac adapter reads the camera optical pose using the ROS/OpenCV convention.
The configured/static mount `T_bc` is independently checked against simulator
truth:

```text
T_bc_truth = inverse(T_wb_truth) @ T_wc_truth
```

If this error is non-zero, fix the mount/convention before tuning PnP.

## 5. Stage2A modules

Core modules:

- `rigid_transform.py`: explicit named SE(3) transforms.
- `gate_geometry.py`: calibrated 3D gate opening keypoints.
- `camera_model.py`: camera K, image size, distortion and projection.
- `isaac_adapter.py`: the only simulator-specific truth boundary.
- `perfect_gate_corner_sensor.py`: perfect four-pixel oracle source.
- `planar_pnp.py`: reference OpenCV IPPE backend.
- `pose_recovery.py`: shared `T_cg -> T_bg -> world/body` transform chain.
- `stage2a_pipeline.py`: truth projection, recovery and equivalence metrics.

The OpenCV backend is for correctness/calibration. It is not intended to run a
Python CPU loop over thousands of environments. `PnPBackend` is an interface so
a batched Torch/GPU implementation can later replace it without changing the
detector or policy API.

### Required Stage2A acceptance tests

A Stage2A setup is not calibrated until all of these are near numerical noise
for synthetic cases and acceptably small in Isaac:

- oracle projected pixels vs reprojected PnP pixels;
- `T_cg_pnp` vs `T_cg_truth`;
- `T_wc_est` vs Isaac optical camera truth;
- `T_wb_est` vs Isaac drone truth;
- configured `T_bc` vs simulator-derived `T_bc_truth`.

Also test multiple distances, yaw/pitch/roll offsets and off-centre image
positions. A single head-on gate is insufficient because planar PnP can expose
pose ambiguities.

## 6. Stage2B modules and training contract

`GateCornerDetector` is intentionally a small protocol:

```text
RGB -> four ordered pixels + visibility + confidence
```

The detector can later be a heatmap network, direct keypoint regressor, YOLO
pose model, transformer, or another architecture. It must not own PnP or camera
extrinsics.

`Stage2DatasetWriter` stores:

- RGB image;
- ordered oracle corners;
- visibility/confidence;
- exact camera K/model;
- gate index;
- `T_wg`, `T_wc`, and `T_wb` truth.

This lets Stage2B be scored at two levels:

1. **detector error**: corner RMSE / per-corner error / missing-corner rate;
2. **navigation error**: PnP pose error and recovered drone/body pose error.

Train the detector supervised against the Stage2A oracle labels. Domain
randomization belongs in dataset generation (lighting, textures, backgrounds,
motion blur, exposure, gate pose/distance, occlusion), while deterministic
geometry/extrinsic calibration stays outside the learned model.

## 7. Recommended implementation milestones

### Stage2A.0 — asset and camera calibration
- inspect `gate.usd`;
- write exact `gate_keypoints.json`;
- choose pinhole or explicit fisheye-undistort path;
- verify camera optical convention and `T_bc`.

### Stage2A.1 — pure geometry equivalence
- run synthetic unit tests without Isaac;
- verify projection -> PnP -> inverse transform closes.

### Stage2A.2 — Isaac truth equivalence
- read active gate actor pose, camera pose/K and body pose;
- overlay oracle pixels on rendered frames;
- log Stage2A error metrics across randomized gate/drone poses.

### Stage2A.3 — policy observation replacement
- replace oracle `target_pos_b` with `GatePoseSolution.target_pos_b`;
- keep simulator truth only in diagnostics/evaluation;
- add a batched PnP backend before large parallel training.

### Stage2B.0 — labeled dataset generation
- use the camera-enabled `Isaac-Drone-Racer-Stage2-Data-v0` config for calibration/export;
- export RGB + oracle labels from many randomized Isaac environments;
- version camera/gate calibration with the dataset.

### Stage2B.1 — detector training
- train four ordered corners plus confidence/visibility;
- validate pixel error independently of PnP.

### Stage2B.2 — detector + shared PnP
- swap `PerfectGateCornerSensor` for `GateCornerDetector`;
- keep the exact same `GatePoseRecovery`.

### Stage2B.3 — closed-loop validation
- compare detector-derived body pose with Isaac truth only as an evaluation
  signal;
- measure racing performance separately from perception metrics.

## 8. Local debug items intentionally left explicit

The framework does not invent values for:

- the exact four gate opening coordinates in `gate.usd`;
- the final camera projection model/distortion;
- the measured/static camera-to-body transform;
- detector architecture and loss weights;
- acceptable pixel/pose thresholds.

Those are calibration/tuning tasks. The code structure makes each one an
explicit input so local debugging cannot silently change the geometry contract.
