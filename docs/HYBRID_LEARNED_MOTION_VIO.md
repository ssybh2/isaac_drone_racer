# Hybrid Learned Motion-Constrained VIO

This branch adds a learned short-horizon motion constraint between raw OpenVINS and the existing Swift mapped-gate correction.

The runtime chain is:

```text
Isaac / vehicle IMU + collective thrust
                 |
                 v
           Learned TCN
        dp + uncertainty
                 |
                 v
raw OpenVINS -> learned drift EKF -> learned-corrected VIO -> Swift gate fusion
     |                    |                       |                |
     +---- raw output     +---- local drift      +---- VIO        +---- final
```

The learned model never receives simulator truth at inference. Truth is used only to create supervised relative-displacement labels during training. OpenVINS attitude is preserved; the learned filter corrects translation and velocity. Gate PnP remains the intermittent global/absolute correction layer.

## 1. Collect training traces

Collect many complete trajectories rather than slicing one long trajectory into random train/test rows. Use different profiles and repeat them with different amplitudes/frequencies/seeds or simulator randomization.

```bash
cd ~/isaac_projects/isaac_drone_racer
source /opt/ros/humble/setup.bash
source ~/openvins_ws/install/setup.bash

./.conda-env/bin/python scripts/estimation/collect_learned_motion_data.py \
  --headless \
  --profile hover \
  --steps 5000 \
  --output artifacts/learned_motion/traces/hover_01.csv

./.conda-env/bin/python scripts/estimation/collect_learned_motion_data.py \
  --headless \
  --profile translate_x \
  --steps 5000 \
  --translation_m 2.0 \
  --output artifacts/learned_motion/traces/translate_x_01.csv

./.conda-env/bin/python scripts/estimation/collect_learned_motion_data.py \
  --headless \
  --profile lissajous \
  --steps 5000 \
  --amplitude_m 0.8 \
  --frequency_hz 0.10 \
  --output artifacts/learned_motion/traces/lissajous_01.csv

./.conda-env/bin/python scripts/estimation/collect_learned_motion_data.py \
  --headless \
  --profile circle \
  --steps 5000 \
  --amplitude_m 0.8 \
  --frequency_hz 0.10 \
  --output artifacts/learned_motion/traces/circle_01.csv
```

Each CSV contains truth pose/velocity/quaternion, body gyro/acceleration, applied collective thrust, body thrust vector and motor action. The learned dataset code converts body gyro/thrust to world coordinates using the truth attitude and labels each window with truth relative displacement.

Recommended first corpus: at least several independent files per motion regime. Validation and test splits are performed by complete CSV file so samples from the same trajectory cannot leak across splits.

## 2. Train the learned displacement model

The first model follows the UZH learned-inertial-odometry idea: 100 Hz gyro + thrust over a 0.5 s window. This implementation predicts both displacement and diagonal uncertainty:

```text
input  : [gyro_w xyz, thrust_w xyz] x 50 samples
output : [dp_w xyz, log_var xyz]
```

Train with:

```bash
./.conda-env/bin/python scripts/estimation/train_learned_motion.py \
  artifacts/learned_motion/traces/*.csv \
  --output artifacts/learned_motion/model.pt \
  --window_time_s 0.5 \
  --sample_rate_hz 100 \
  --stride_time_s 0.01 \
  --epochs 200 \
  --batch_size 128 \
  --device cuda
```

The trainer stores model weights and the window/sample-rate metadata in the checkpoint. It also writes `model.pt.json` with validation/test relative-displacement errors.

## 3. Run the hybrid estimator

Start OpenVINS in one terminal using the same simulator camera/IMU calibration used by the current branch. Then run the dedicated hybrid evaluation task:

```bash
./.conda-env/bin/python scripts/estimation/run_hybrid_openvins_evaluation.py \
  --headless \
  --checkpoint artifacts/learned_motion/model.pt \
  --profile translate_x \
  --steps 4000 \
  --motion_start_s 4 \
  --translation_m 1.5 \
  --translation_duration_s 8 \
  --output_dir artifacts/hybrid_openvins_evaluation/translate_x
```

The task ID is `Isaac-Drone-Racer-Swift-Hybrid-OpenVINS-v0`. It exposes:

- `openvins_raw_vio_estimate`: world-aligned raw OpenVINS;
- `openvins_learned_vio_estimate`: learned translation/velocity-corrected OpenVINS;
- `openvins_vio_estimate`: compatibility/downstream estimate, equal to the learned-corrected state when the model is enabled;
- `swift_fused_estimate`: existing mapped-gate fusion result when a detector is enabled.

The evaluation script writes raw-versus-learned position/velocity error statistics and a per-step CSV. Do not judge the model only from its training loss; the main acceptance test is whether held-out trajectory drift is lower than raw OpenVINS over time.

## 4. Enable mapped-gate absolute correction

To evaluate the full stack, provide the existing Swift gate detector checkpoint as well:

```bash
./.conda-env/bin/python scripts/estimation/run_hybrid_openvins_evaluation.py \
  --headless \
  --checkpoint artifacts/learned_motion/model.pt \
  --detector_checkpoint /path/to/gate_corner_detector.pt \
  --visibility_checkpoint /path/to/visibility_model.pt \
  --profile lissajous \
  --steps 5000 \
  --output_dir artifacts/hybrid_openvins_evaluation/full_stack
```

The final ordering is intentional: the learned model supplies a continuous local motion constraint, while gate PnP supplies intermittent global position information.

## 5. Learned drift measurement

For a learned window `[t_i,t_j]`:

```text
dp_vio = p_vio(t_j) - p_vio(t_i)
dp_nn  = learned physical displacement
z      = dp_vio - dp_nn
```

`z` observes drift accumulated during that interval. The drift EKF uses state

```text
x = [p_d_anchor, p_d_current, v_d_current]
```

and measurement Jacobian

```text
H = [-I, I, 0].
```

The update uses learned covariance, a configurable NIS/Mahalanobis gate and Joseph-form covariance update. Accepted or rejected windows both advance the window anchor so a bad network output is not replayed indefinitely.

## 6. What to compare

For every held-out trajectory compare at least:

- raw OpenVINS position RMSE and error at 1/5/10/20 s;
- learned-corrected position RMSE and the same time checkpoints;
- raw versus learned velocity RMSE;
- learned update acceptance/rejection rate and NIS distribution;
- correction magnitude versus true OpenVINS drift;
- if detector is enabled, final gate-fused position RMSE and rejected gate measurements.

A useful model must improve held-out trajectories, not only trajectories used for training. Keep hover, translation, richer 3-D motion and racing trajectories as separate evaluation regimes.

## 7. Important limitations of this first implementation

The first model follows the public UZH formulation closely enough for a clean baseline: gyro and thrust are represented in the world frame. At runtime body measurements are timestamp-aligned with raw OpenVINS attitude before rotation. If the remaining dominant failure is attitude error, the next ablation should predict displacement in the window-start body frame to reduce dependence on VIO attitude.

The learned layer does not repair an arbitrary sensor convention/calibration error inside OpenVINS. It is a probabilistic secondary motion constraint. Large out-of-distribution learned measurements are intended to be rejected by the innovation gate rather than trusted unconditionally.
