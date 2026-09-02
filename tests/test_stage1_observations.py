import importlib.util
from pathlib import Path

import torch

from estimation.state_estimate import SourceStatus, StateEstimate


MODULE_PATH = Path(__file__).parents[1] / "tasks/drone_racer/mdp/stage1_observations.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage1_observations_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_estimate() -> StateEstimate:
    status = SourceStatus(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([True]))
    return StateEstimate(
        publish_timestamp_s=torch.tensor([0.0]),
        position_w_v=torch.zeros(1, 3),
        orientation_w_v=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        position_w_b=torch.tensor([[1.0, 2.0, 3.0]]),
        orientation_w_b=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        linear_velocity_w_b=torch.tensor([[4.0, 5.0, 6.0]]),
        angular_velocity_b=torch.tensor([[7.0, 8.0, 9.0]]),
        vio_status=status,
        imu_status=status,
    )


def test_observation_reads_estimate_without_scene_access():
    module = load_module()
    env = type("EstimatorOnlyEnv", (), {"stage1_state_estimate": state_estimate()})()

    output = module.estimated_drone_state(env)

    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]),
    )


def test_observation_dimension_probe_does_not_require_estimator():
    module = load_module()
    env = type("InitializingEnv", (), {"num_envs": 2, "device": "cpu"})()

    output = module.estimated_drone_state(env)

    assert output.shape == (2, 13)
    torch.testing.assert_close(output, torch.zeros(2, 13))


def test_observation_module_has_no_isaac_ground_truth_access():
    source = MODULE_PATH.read_text()
    assert "env.scene" not in source
    assert "root_pos_w" not in source
    assert "root_quat_w" not in source
    assert "root_lin_vel" not in source
    assert "root_ang_vel" not in source
