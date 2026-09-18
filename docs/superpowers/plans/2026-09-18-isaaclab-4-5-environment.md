# Isaac Lab 4.5 Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the checked-out `feature/learned-inertial-racing` branch runnable with its supported Isaac Sim and Isaac Lab versions, then prove that a GPU-backed drone-racing environment can start and step.

**Architecture:** Reuse the repository-local Conda prefix so the project remains isolated from the system Python. Keep Isaac Lab and this repository installed in editable mode, validate package provenance and task registration, and finish with a finite headless simulation smoke test that creates, resets, steps, and closes the environment.

**Tech Stack:** Ubuntu 22.04, NVIDIA RTX 4090, Python 3.10, Isaac Sim 4.5.0, Isaac Lab v2.1.0, PyTorch 2.5.1, Gymnasium, skrl

**Spec:** `README.md` and `docs/LEARNED_INERTIAL_RACING.md`

## Global Constraints

- Use Isaac Sim `4.5.0` as declared in `README.md` and `setup.py`.
- Use the Isaac Lab `v2.1.0` checkout required by `README.md`.
- Use Python `3.10` as declared in `README.md` and `setup.py`.
- Preserve all existing untracked `.venv-pure/` and `artifacts/` contents.
- Do not upgrade the environment to the installed Isaac Sim 5.1 / Isaac Lab 2.3 candidates.

---

### Task 1: Audit the Existing Local Environment

**Files:**
- Inspect: `.conda-env/conda-meta/history`
- Inspect: `README.md`
- Inspect: `setup.py`
- Inspect: `/home/donglei/isaac_projects/IsaacLab-v2.1.0`

**Interfaces:**
- Consumes: Repository requirements and local GPU/driver state.
- Produces: An exact version/provenance inventory used by Tasks 2 and 3.

- [ ] **Step 1: Confirm repository and branch state**

Run:

```bash
git status --short --branch
git branch --show-current
git log -1 --oneline
```

Expected: branch `feature/learned-inertial-racing`; user artifacts remain untouched.

- [ ] **Step 2: Confirm host compatibility**

Run:

```bash
cat /etc/os-release
nvidia-smi
```

Expected: Ubuntu 22.04 and an available NVIDIA GPU.

- [ ] **Step 3: Inventory the isolated Python packages**

Run:

```bash
./.conda-env/bin/python --version
./.conda-env/bin/python -m pip show isaacsim isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl skrl isaac_drone_racer
```

Expected: Python 3.10, Isaac Sim 4.5.0, Isaac Lab packages sourced from the v2.1.0 checkout, and this repository installed editable.

### Task 2: Refresh Editable Installations

**Files:**
- Modify outside Git: `.conda-env/lib/python3.10/site-packages/` package metadata only
- Source: `/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab`
- Source: `/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab_assets`
- Source: `/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab_mimic`
- Source: `/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab_rl`
- Source: `/home/donglei/isaac_projects/IsaacLab-v2.1.0/source/isaaclab_tasks`
- Source: `setup.py`

**Interfaces:**
- Consumes: The verified `.conda-env` Python interpreter and Isaac Lab v2.1.0 checkout.
- Produces: Editable imports that follow the current source trees without copying code.

- [ ] **Step 1: Refresh the project editable install without dependency upgrades**

Run:

```bash
./.conda-env/bin/python -m pip install --no-deps -e .
```

Expected: `isaac_drone_racer==0.1.0` is installed from the current repository.

- [ ] **Step 2: Verify exact package provenance**

Run:

```bash
./.conda-env/bin/python -m pip show isaacsim isaaclab isaac_drone_racer
```

Expected: Isaac Sim is 4.5.0, Isaac Lab points to `IsaacLab-v2.1.0`, and the project points to the current checkout.

- [ ] **Step 3: Verify branch-specific modules compile**

Run:

```bash
./.conda-env/bin/python -m compileall -q estimation tasks scripts
```

Expected: exit status 0.

### Task 3: Run a Finite Isaac Sim Smoke Test

**Files:**
- Inspect at runtime: `tasks/drone_racer/__init__.py`
- Inspect at runtime: `tasks/drone_racer/drone_racer_env_cfg.py`

**Interfaces:**
- Consumes: The package installations produced by Task 2 and task ID `Isaac-Drone-Racer-v0`.
- Produces: Evidence that Isaac Sim launches on the GPU and the environment completes reset plus one simulation step.

- [ ] **Step 1: Launch Isaac Sim headlessly using the repository interpreter**

Run a finite Python program which creates `AppLauncher(headless=True)`, imports `tasks`, constructs `Isaac-Drone-Racer-v0` with one environment, resets it, applies one zero action, and closes both the environment and application.

Expected: Isaac Sim startup completes without an import, extension, CUDA, asset, or task-registration error.

- [ ] **Step 2: Confirm the environment stepped**

Expected terminal marker:

```text
ISAAC_DRONE_RACER_SMOKE_OK
```

- [ ] **Step 3: Recheck the working tree**

Run:

```bash
git status --short
```

Expected: no user artifacts were removed or modified; only this plan is a deliberate tracked-tree addition.

### Task 4: Hand Off Repeatable Start Commands

**Files:**
- No additional file changes.

**Interfaces:**
- Consumes: The verified repository-local interpreter and registered task IDs.
- Produces: Copy/paste commands for GUI smoke testing, policy training, policy playback, learned-inertial data collection, and troubleshooting.

- [ ] **Step 1: Provide the preferred activation-free commands**

Use `./.conda-env/bin/python` in every command so shell-level Conda activation is optional.

- [ ] **Step 2: Provide optional Conda activation**

Run:

```bash
source /home/donglei/.local/share/miniconda3/etc/profile.d/conda.sh
conda activate /home/donglei/isaac_projects/isaac_drone_racer/.conda-env
```

Expected: `python` resolves to the repository-local Python 3.10 interpreter.

- [ ] **Step 3: Document the task-specific limitation**

Explain that `Isaac-Drone-Racer-Learned-Inertial-v0` requires learned-motion and gate-detector checkpoint paths to run the complete learned-inertial estimator; the base environment can be launched immediately.
