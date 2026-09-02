import torch

from estimation.policy_adapter import policy_drone_state
from estimation.state_estimate import SourceStatus, StateEstimate


def estimate() -> StateEstimate:
    status = SourceStatus(torch.tensor([0.1]), torch.tensor([0.0]), torch.tensor([True]))
    return StateEstimate(
        publish_timestamp_s=torch.tensor([0.1]),
        position_w_v=torch.zeros(1, 3),
        orientation_w_v=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        position_w_b=torch.tensor([[1.0, 2.0, 3.0]]),
        orientation_w_b=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        linear_velocity_w_b=torch.tensor([[1.0, 2.0, 3.0]]),
        angular_velocity_b=torch.tensor([[4.0, 5.0, 6.0]]),
        vio_status=status,
        imu_status=status,
    )


def test_policy_adapter_preserves_stage0_13_value_order():
    output = policy_drone_state(estimate())

    assert output.shape == (1, 13)
    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )


def test_policy_adapter_rotates_world_velocity_into_body_frame():
    value = estimate()
    half = 2.0**-0.5
    value.orientation_w_b[:] = torch.tensor([half, 0.0, 0.0, half])
    value.linear_velocity_w_b[:] = torch.tensor([0.0, 2.0, 0.0])

    output = policy_drone_state(value)

    torch.testing.assert_close(output[:, 7:10], torch.tensor([[2.0, 0.0, 0.0]]), atol=1.0e-6, rtol=1.0e-6)
