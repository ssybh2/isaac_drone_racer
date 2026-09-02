import pytest
import torch

from estimation.state_estimate import SourceStatus, StateEstimate


def make_state_estimate(num_envs: int = 2) -> StateEstimate:
    zeros = torch.zeros(num_envs)
    status = SourceStatus(timestamp_s=zeros.clone(), age_s=zeros.clone(), valid=torch.ones(num_envs, dtype=torch.bool))
    identity = torch.zeros(num_envs, 4)
    identity[:, 0] = 1.0
    return StateEstimate(
        publish_timestamp_s=zeros.clone(),
        position_w_v=torch.zeros(num_envs, 3),
        orientation_w_v=identity.clone(),
        position_w_b=torch.zeros(num_envs, 3),
        orientation_w_b=identity.clone(),
        linear_velocity_w_b=torch.zeros(num_envs, 3),
        angular_velocity_b=torch.zeros(num_envs, 3),
        vio_status=status,
        imu_status=SourceStatus(
            timestamp_s=zeros.clone(), age_s=zeros.clone(), valid=torch.ones(num_envs, dtype=torch.bool)
        ),
    )


def test_state_estimate_accepts_consistent_batched_tensors():
    estimate = make_state_estimate()

    assert estimate.validate() is estimate
    assert estimate.num_envs == 2
    assert estimate.device == torch.device("cpu")


def test_state_estimate_rejects_wrong_position_shape():
    estimate = make_state_estimate()
    estimate.position_w_b = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="position_w_b"):
        estimate.validate()


def test_state_estimate_rejects_non_finite_values():
    estimate = make_state_estimate()
    estimate.position_w_b[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        estimate.validate()


def test_state_estimate_rejects_non_unit_quaternion():
    estimate = make_state_estimate()
    estimate.orientation_w_b[0] = torch.tensor([2.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="unit quaternion"):
        estimate.validate()


def test_source_status_requires_boolean_validity():
    status = SourceStatus(timestamp_s=torch.zeros(2), age_s=torch.zeros(2), valid=torch.ones(2))

    with pytest.raises(ValueError, match="valid"):
        status.validate(num_envs=2, device=torch.device("cpu"))
