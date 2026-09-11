# Stage2A implementation plan

The canonical Stage2A/Stage2B architecture is documented in
[`stage2_perception_architecture.md`](stage2_perception_architecture.md).

Stage2A is now treated as a geometric reference pipeline:

```text
Isaac actor/link gate pose + optical camera pose/K
    -> exact gate-corner pixels
    -> shared CornerObservation
    -> planar PnP
    -> T_cg
    -> calibrated T_bc
    -> T_bg / target_pos_b
    -> optional T_wb estimate
    -> compare with Isaac truth
```

The remaining local calibration tasks are deliberately explicit:

1. extract/measure the four opening corners from `assets/gate/gate.usd` in the
   **gate actor frame** and save `assets/gate/gate_keypoints.json`;
2. use a consistent camera model (pinhole, or explicit fisheye undistortion);
3. validate the camera optical convention and `T_bc`;
4. run oracle-vs-PnP equivalence across many randomized poses;
5. only then replace the policy's oracle `target_pos_b`.

Stage2B plugs a learned RGB corner detector into the same
`CornerObservation -> GatePoseRecovery` backend and uses Stage2A projections as
training/evaluation labels.
