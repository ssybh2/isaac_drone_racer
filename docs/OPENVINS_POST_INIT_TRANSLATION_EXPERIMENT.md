# OpenVINS post-initialization translation isolation

## Why this experiment exists

The existing `strict_static_translate_x` artifact is not a clean test of
post-initialization translational tracking.  That run requested translation at
`t=4 s` for `8 s`, so the commanded ramp nominally finished at `t=12 s`.
OpenVINS was first observed at approximately `t=12.41 s`.  In other words, the
estimator became available only after the planned translation ramp was already
complete.

That run is still useful evidence of divergence, but it cannot distinguish
"translation after a valid static initialization is broken" from "the previous
motion affected initialization/recovery".

The fault-isolation runner now supports `--motion_start_after_init_s`.  When the
option is present, the vehicle holds the initial XY target until an OpenVINS
estimate has actually been observed.  The requested delay is then measured from
that observation time.  The summary records both the observed initialization
time and resolved motion-start time.

## Experiment P1 — strict-static initialization, then X translation

Restart OpenVINS before the run.

Terminal 1:

```bash
cd ~/isaac_projects/isaac_drone_racer
source /opt/ros/humble/setup.bash
source ~/openvins_ws/install/setup.bash
ros2 launch ov_msckf subscribe.launch.py \
  config_path:=$(pwd)/config/openvins/swift_sim/estimator_config_debug_strict_static.yaml \
  use_stereo:=false max_cameras:=1 \
  2>&1 | tee artifacts/openvins_fault_isolation/post_init_translate_x/openvins.log
```

Terminal 2:

```bash
cd ~/isaac_projects/isaac_drone_racer
source /opt/ros/humble/setup.bash
source ~/openvins_ws/install/setup.bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
./.conda-env/bin/python scripts/estimation/run_openvins_fault_isolation.py \
  --headless --steps 4000 \
  --profile translate_x \
  --motion_start_after_init_s 1.0 \
  --translation_m 1.5 \
  --translation_duration_s 8.0 \
  --output_dir artifacts/openvins_fault_isolation/post_init_translate_x
```

Expected timing contract:

- before OpenVINS is observed: target XY stays at the reset position;
- after OpenVINS is observed: wait exactly 1 s;
- then ramp the X target by 1.5 m over 8 s;
- do not reset/teleport Isaac while the external estimator is running.

After the run, verify the JSON summary first.  `motion_schedule.resolved_start_s`
should equal `openvins_initialized_time_s + 1.0` within one simulation step, and
`motion_schedule.starts_after_openvins_observed` should be `true`.

## Experiment P0 — matched hover control

Restart OpenVINS again, then run the same strict-static configuration without
translation:

```bash
./.conda-env/bin/python scripts/estimation/run_openvins_fault_isolation.py \
  --headless --steps 4000 \
  --profile hover \
  --output_dir artifacts/openvins_fault_isolation/post_init_hover_control
```

Compare P0 and P1 at the same elapsed times after initialization.  The important
quantities are velocity-error growth, position-error growth and orientation
error.  If P1 degrades sharply only after the commanded translation while P0
remains comparatively stable, focus next on accelerometer propagation,
camera/IMU time alignment and visual translation updates.  If both runs drift
at nearly the same rate before P1 motion starts, the primary fault is already
present in static propagation/initialization and the commanded translation is
not the trigger.

## What not to conclude from the old translate-X artifact

Do not use the old run by itself as proof that deliberate translational parallax
failed to help OpenVINS.  Its commanded ramp and estimator-availability windows
did not overlap in the intended way.  Keep the artifact for regression history,
but use P0/P1 as the next controlled A/B pair.
