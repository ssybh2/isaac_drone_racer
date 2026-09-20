from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_actor_observation_module_has_no_runtime_root_truth_access():
    source = _text("tasks/drone_racer/mdp/learned_inertial_observations.py")
    forbidden = (
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_state_w",
    )
    for token in forbidden:
        assert token not in source


def test_estimated_state_command_owns_actor_gate_progression():
    source = _text("tasks/drone_racer/mdp/commands.py")
    start = source.index("class EstimatedStateGateTargetingCommand")
    end = source.index(
        "@configclass\nclass GateTargetingCommandCfg",
        start,
    )
    block = source[start:end]

    assert "self.next_gate_idx[self._mission_gate_passed] += 1" in block
    assert "self.next_gate_idx[self._gt_gate_passed]" not in block
    assert "self._gt_next_gate_idx[self._gt_gate_passed] += 1" in block
    assert "gt_gate_indices = self._gt_next_gate_idx.to(dtype=torch.long)" in block
    assert "Ground truth is deliberately kept on a separate reward/evaluation" in block


def test_frozen_rl_profile_uses_validated_direct_reprojection_stack():
    source = _text(
        "tasks/drone_racer/drone_racer_learned_inertial_env_cfg.py"
    )
    start = source.index("class DroneRacerLearnedInertialRLCfg")
    block = source[start:]

    expected = (
        'model_v6_2_body_delta_velocity_balanced.pt',
        'self.learned_update_rate_hz = 20.0',
        'self.learned_fusion_rate_hz = 2.0',
        'self.gate_measurement_model = "direct_reprojection"',
        'self.gate_reprojection_sigma_px = 0.85',
        'self.gate_reprojection_min_visible_corners = 2',
        'self.gate_debug_gt_diagnostics = False',
    )
    for token in expected:
        assert token in block


def test_frozen_rl_task_is_registered():
    source = _text("tasks/drone_racer/__init__.py")
    assert "Isaac-Drone-Racer-Learned-Inertial-RL-v0" in source
    assert "DroneRacerLearnedInertialRLCfg" in source



def test_scripted_controller_action_has_no_simulator_truth_access():
    source = _text("scripts/rl/smoke_estimated_state_closed_loop.py")
    start = source.index("def _controller_action")
    end = source.index("def _diagnostic_truth_error", start)
    block = source[start:end]

    forbidden = (
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_state_w",
    )
    for token in forbidden:
        assert token not in block
