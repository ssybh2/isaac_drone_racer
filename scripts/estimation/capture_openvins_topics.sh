#!/usr/bin/env bash
set -euo pipefail

# Capture exactly what OpenVINS receives/produces during a diagnostic run.
# Usage: source ROS/OpenVINS first, then run this script after Isaac starts.

OUT_DIR="${1:-artifacts/openvins_fault_isolation/ros_capture}"
DURATION_S="${2:-30}"
mkdir -p "${OUT_DIR}"

printf '[capture] output: %s\n' "${OUT_DIR}"
printf '[capture] duration: %ss\n' "${DURATION_S}"

(timeout "${DURATION_S}s" ros2 topic hz /swift/imu \
  >"${OUT_DIR}/imu_hz.txt" 2>&1 || true) &
PID_IMU_HZ=$!
(timeout "${DURATION_S}s" ros2 topic hz /swift/camera/image_raw \
  >"${OUT_DIR}/camera_hz.txt" 2>&1 || true) &
PID_CAM_HZ=$!
(timeout "${DURATION_S}s" ros2 topic hz /ov_msckf/odomimu \
  >"${OUT_DIR}/odom_hz.txt" 2>&1 || true) &
PID_ODOM_HZ=$!

# Record IMU and estimator output at their native ROS rates. Camera pixels are
# intentionally omitted to keep the debug bag small; camera rate is measured
# separately above.
BAG_DIR="${OUT_DIR}/imu_odom_bag"
rm -rf "${BAG_DIR}"
(timeout "${DURATION_S}s" ros2 bag record \
  -o "${BAG_DIR}" \
  /swift/imu /ov_msckf/odomimu \
  >"${OUT_DIR}/rosbag.log" 2>&1 || true) &
PID_BAG=$!

# Also retain one human-readable sample from each numerical stream.
(timeout "${DURATION_S}s" ros2 topic echo /swift/imu --once \
  >"${OUT_DIR}/imu_once.yaml" 2>&1 || true) &
PID_IMU_ONE=$!
(timeout "${DURATION_S}s" ros2 topic echo /ov_msckf/odomimu --once \
  >"${OUT_DIR}/odom_once.yaml" 2>&1 || true) &
PID_ODOM_ONE=$!

wait "${PID_IMU_HZ}" "${PID_CAM_HZ}" "${PID_ODOM_HZ}" "${PID_BAG}" "${PID_IMU_ONE}" "${PID_ODOM_ONE}" || true

printf '\n[capture] complete\n'
for file in imu_hz.txt camera_hz.txt odom_hz.txt imu_once.yaml odom_once.yaml rosbag.log; do
  printf '%-18s ' "${file}"
  if [[ -s "${OUT_DIR}/${file}" ]]; then
    printf 'OK (%s bytes)\n' "$(wc -c <"${OUT_DIR}/${file}")"
  else
    printf 'EMPTY\n'
  fi
done
