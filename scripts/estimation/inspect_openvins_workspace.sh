#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-${HOME}/openvins_ws}"
SRC_DIR="${WORKSPACE}/src"

fail() {
  printf '[openvins-inspect] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -d "${SRC_DIR}" ]] || fail "workspace source directory not found: ${SRC_DIR}"

mapfile -t PACKAGE_FILES < <(find "${SRC_DIR}" -type f -path '*/ov_msckf/package.xml' -print | sort)
[[ ${#PACKAGE_FILES[@]} -gt 0 ]] || fail "could not find ov_msckf/package.xml under ${SRC_DIR}"
[[ ${#PACKAGE_FILES[@]} -eq 1 ]] || fail "found multiple ov_msckf packages; inspect manually before patching: ${PACKAGE_FILES[*]}"

PACKAGE_DIR="$(dirname "${PACKAGE_FILES[0]}")"
GIT_ROOT="$(git -C "${PACKAGE_DIR}" rev-parse --show-toplevel 2>/dev/null)" \
  || fail "ov_msckf source is not inside a Git worktree: ${PACKAGE_DIR}"

for rel in \
  ov_msckf/src/core/VioManager.h \
  ov_msckf/src/ros/ROS2Visualizer.h \
  ov_msckf/src/ros/ROS2Visualizer.cpp; do
  [[ -f "${GIT_ROOT}/${rel}" ]] \
    || fail "expected OpenVINS source file missing: ${GIT_ROOT}/${rel}"
done

grep -q 'initialize_with_gt' "${GIT_ROOT}/ov_msckf/src/core/VioManager.h" \
  || fail "VioManager::initialize_with_gt API not found; refusing to prepare a patch"

printf 'OPENVINS_WORKSPACE=%s\n' "${WORKSPACE}"
printf 'OPENVINS_PACKAGE_DIR=%s\n' "${PACKAGE_DIR}"
printf 'OPENVINS_GIT_ROOT=%s\n' "${GIT_ROOT}"
printf 'OPENVINS_HEAD=%s\n' "$(git -C "${GIT_ROOT}" rev-parse HEAD)"
printf 'OPENVINS_BRANCH=%s\n' "$(git -C "${GIT_ROOT}" branch --show-current || true)"
printf 'OPENVINS_DESCRIBE=%s\n' "$(git -C "${GIT_ROOT}" describe --always --dirty --tags 2>/dev/null || true)"

printf '\n=== remotes ===\n'
git -C "${GIT_ROOT}" remote -v || true
printf '\n=== status ===\n'
git -C "${GIT_ROOT}" status --short --branch
printf '\n=== initialize_with_gt ===\n'
grep -n 'initialize_with_gt' "${GIT_ROOT}/ov_msckf/src/core/VioManager.h"

if command -v ros2 >/dev/null 2>&1; then
  printf '\n=== installed ov_msckf prefix ===\n'
  ros2 pkg prefix ov_msckf 2>/dev/null \
    || printf 'ov_msckf not visible in current ROS environment\n'
fi
