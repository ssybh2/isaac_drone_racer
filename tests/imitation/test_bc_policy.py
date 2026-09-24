import torch

from imitation.bc_policy import Circular12BCPolicy


def test_bc_policy_is_bounded_and_has_expected_topology():
    model = Circular12BCPolicy()
    x = torch.randn(7, 31)
    y = model(x)

    assert y.shape == (7, 4)
    assert torch.all(y <= 1.0)
    assert torch.all(y >= -1.0)
    linears = model.linear_layers()
    assert [(m.in_features, m.out_features) for m in linears] == [
        (31, 256),
        (256, 256),
        (256, 256),
        (256, 4),
    ]
