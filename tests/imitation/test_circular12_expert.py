import numpy as np
import torch

from imitation.circular12_expert import (
    Circular12ExpertConfig,
    circular12_expert_action,
)
from imitation.circular12_reference import (
    Circular12Reference,
    Circular12ReferenceConfig,
)


def test_expert_matches_reference_without_tumble_error():
    speed = 14.0
    reference = Circular12Reference(
        Circular12ReferenceConfig(speed_mps=speed)
    )
    sample = reference.sample_phase(-0.4)
    cfg = Circular12ExpertConfig(
        target_speed_mps=speed,
        max_collective_accel_mps2=45.0,
    )

    p = torch.tensor(sample.position_w, dtype=torch.float64).view(1, 3)
    v = torch.tensor(sample.velocity_w, dtype=torch.float64).view(1, 3)
    R = torch.tensor(sample.rotation_wb, dtype=torch.float64).view(1, 3, 3)
    output = circular12_expert_action(p, v, R, cfg)

    assert float(torch.linalg.vector_norm(
        output.attitude_error_rotvec_b, dim=-1
    )[0]) < 1.0e-6
    assert torch.all(output.action <= 1.0)
    assert torch.all(output.action >= -1.0)
    np.testing.assert_allclose(
        output.desired_rotation_wb[0].numpy(),
        sample.rotation_wb,
        atol=1.0e-9,
    )


def test_expert_uses_small_structured_rates_on_reference():
    speed = 17.712658128452922
    reference = Circular12Reference(
        Circular12ReferenceConfig(speed_mps=speed)
    )
    sample = reference.sample_phase(0.2)
    cfg = Circular12ExpertConfig(
        target_speed_mps=speed,
        max_collective_accel_mps2=45.0,
    )
    output = circular12_expert_action(
        torch.tensor(sample.position_w).view(1, 3),
        torch.tensor(sample.velocity_w).view(1, 3),
        torch.tensor(sample.rotation_wb).view(1, 3, 3),
        cfg,
    )
    rates = output.desired_body_rate_b[0]
    assert float(torch.linalg.vector_norm(rates)) < 2.0
    assert abs(float(rates[0])) < 1.0e-6
