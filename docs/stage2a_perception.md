# Stage2A perfect-corner perception

Stage2A is the **oracle-corner reference path** for the Stage 2 perception
stack. It is not a learned correction model.

```text
Isaac truth geometry/poses
        |
        v
exact 2D gate corners  ---- Stage2A ends its oracle access here
        |
        v
CornerObservation
        |
        +---------------- same boundary used by Stage2B detector output
        |
        v
shared planar PnP -> T_cg -> camera/body transform -> target_pos_b
```

The simulator body/camera/gate poses are retained only for calibration metrics:
PnP pose error, camera pose error, body pose error and camera extrinsic error.

Important constraints:

- gate 3D keypoints must be calibrated in the gate **actor/link frame**, not
  guessed from a 1 m square and not paired with the COM pose;
- image pixels, camera intrinsics and PnP must use the same optical convention
  and distortion model;
- the current repository's fisheye camera cannot be silently treated as an
  ideal pinhole camera;
- OpenCV IPPE is a correctness backend. Parallel RL should later provide a
  batched implementation through the same `PnPBackend` interface.

See [`stage2_perception_architecture.md`](stage2_perception_architecture.md) for
the complete Stage2A/Stage2B design and calibration milestones. See
[`stage2_training.md`](stage2_training.md) for the calibrated repository values,
dataset commands, detector training, and current validation results.
