import math

import torch

from estimation.state_estimate import ImuEstimate, SourceStatus, VioEstimate
from estimation.state_estimate_assembler import StateEstimateAssembler


def status(timestamp: float, count: int = 2) -> SourceStatus:
    return SourceStatus(
        timestamp_s=torch.full((count,), timestamp),
        age_s=torch.zeros(count),
        valid=torch.ones(count, dtype=torch.bool),
    )


def vio() -> VioEstimate:
    return VioEstimate(
        position_v_b=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        orientation_v_b=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        linear_velocity_v_b=torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        status=status(0.08),
    )


def imu() -> ImuEstimate:
    return ImuEstimate(
        angular_velocity_b=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        status=status(0.09),
    )


def test_identity_alignment_preserves_vio_frame_state():
    assembler = StateEstimateAssembler.identity(2, "cpu")

    estimate = assembler.assemble(vio(), imu(), publish_timestamp_s=0.1)

    torch.testing.assert_close(estimate.position_w_b, vio().position_v_b)
    torch.testing.assert_close(estimate.orientation_w_b, vio().orientation_v_b)
    torch.testing.assert_close(estimate.linear_velocity_w_b, vio().linear_velocity_v_b)
    torch.testing.assert_close(estimate.angular_velocity_b, imu().angular_velocity_b)
    torch.testing.assert_close(estimate.vio_status.age_s, torch.full((2,), 0.02))
    torch.testing.assert_close(estimate.imu_status.age_s, torch.full((2,), 0.01))


def test_nonidentity_alignment_applies_t_wv_to_pose_and_velocity():
    half_angle = math.pi / 4.0
    orientation_w_v = torch.tensor(
        [[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]]
    ).repeat(2, 1)
    position_w_v = torch.tensor([[10.0, -2.0, 0.5], [10.0, -2.0, 0.5]])
    assembler = StateEstimateAssembler(position_w_v, orientation_w_v)

    estimate = assembler.assemble(vio(), imu(), publish_timestamp_s=0.1)

    torch.testing.assert_close(
        estimate.position_w_b,
        torch.tensor([[10.0, -1.0, 0.5], [8.0, -2.0, 0.5]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        estimate.linear_velocity_w_b,
        torch.tensor([[0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(estimate.orientation_w_b, orientation_w_v)


def test_alignment_can_be_updated_per_environment():
    assembler = StateEstimateAssembler.identity(2, "cpu")
    position_w_v = torch.tensor([[3.0, 4.0, 5.0]])
    orientation_w_v = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    assembler.set_alignment(torch.tensor([1]), position_w_v, orientation_w_v)
    estimate = assembler.assemble(vio(), imu(), publish_timestamp_s=0.1)

    torch.testing.assert_close(estimate.position_w_v[0], torch.zeros(3))
    torch.testing.assert_close(estimate.position_w_v[1], position_w_v[0])

