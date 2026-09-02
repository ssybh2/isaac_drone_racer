import math

import torch

from estimation.frame_math import (
    compose_transform_w_b,
    quaternion_multiply,
    rotate_world_to_body,
)


def test_t_wb_equals_t_wv_times_t_vb_for_yaw_alignment():
    root_half = math.sqrt(0.5)
    yaw_90 = torch.tensor([[root_half, 0.0, 0.0, root_half]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    position_w_b, orientation_w_b = compose_transform_w_b(
        position_w_v=torch.tensor([[10.0, 0.0, 0.0]]),
        orientation_w_v=yaw_90,
        position_v_b=torch.tensor([[1.0, 0.0, 0.0]]),
        orientation_v_b=identity,
    )

    torch.testing.assert_close(position_w_b, torch.tensor([[10.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(orientation_w_b, yaw_90, atol=1e-6, rtol=0)


def test_world_vector_rotates_into_body_frame():
    root_half = math.sqrt(0.5)
    orientation_w_b = torch.tensor([[root_half, 0.0, 0.0, root_half]])

    vector_b = rotate_world_to_body(orientation_w_b, torch.tensor([[0.0, 2.0, 0.0]]))

    torch.testing.assert_close(vector_b, torch.tensor([[2.0, 0.0, 0.0]]), atol=1e-6, rtol=0)


def test_quaternion_multiply_normalizes_composed_rotation():
    result = quaternion_multiply(
        torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[3.0, 0.0, 0.0, 0.0]]),
    )

    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
