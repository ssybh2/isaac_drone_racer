from types import SimpleNamespace

import pytest
import torch

from utils.training_overrides import apply_post_load_training_overrides


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.log_std_parameter = torch.nn.Parameter(
            torch.log(torch.tensor([1.0, 2.0, 3.0]))
        )


def make_agent():
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=7.5e-4)
    policy.weight.grad = torch.ones_like(policy.weight)
    optimizer.step()
    scheduler = SimpleNamespace(base_lrs=[1.0e-4], _last_lr=[7.5e-4])
    return SimpleNamespace(policy=policy, optimizer=optimizer, scheduler=scheduler)


def test_override_preserves_optimizer_state_and_clamps_only_high_std():
    agent = make_agent()
    optimizer_state_ids = set(agent.optimizer.state)

    metadata = apply_post_load_training_overrides(
        agent, learning_rate=1.0e-4, max_action_std=1.5
    )

    assert set(agent.optimizer.state) == optimizer_state_ids
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert agent.scheduler.base_lrs == [1.0e-4]
    assert agent.scheduler._last_lr == [1.0e-4]
    torch.testing.assert_close(
        agent.policy.log_std_parameter.exp(), torch.tensor([1.0, 1.5, 1.5])
    )
    assert metadata["optimizer_learning_rate_before"] == [7.5e-4]
    assert metadata["policy_action_std_before"] == pytest.approx([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.0}, "learning rate"),
        ({"max_action_std": -1.0}, "standard deviation"),
    ],
)
def test_invalid_overrides_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        apply_post_load_training_overrides(make_agent(), **kwargs)
