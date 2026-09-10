"""Exercise the actual evaluator loop without starting Isaac at module import."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.stage1_metrics import EpisodeRecord, Stage1EpisodeAccumulator


def test_fast_failures_cannot_displace_slow_successful_environments():
    source = Path(__file__).parents[1] / "scripts/rl/evaluate_stage1.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_case")
    namespace = dict(torch=torch, _seed_case=lambda seed: None,
                     EpisodeRecord=EpisodeRecord, Stage1EpisodeAccumulator=Stage1EpisodeAccumulator)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)

    class Environment:
        num_envs = 2
        device = "cpu"
        max_episode_length = 10
        step_dt = 0.01
        command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(num_gates=1))

        def __init__(self):
            self.unwrapped = self
            self.steps = 0

        def set_fake_sensor_profile(self, *args, **kwargs):
            pass

        def reset(self):
            return torch.zeros(2, 20), {}

        def step(self, actions):
            self.steps += 1
            # Slot 0 crashes every step. Slot 1 completes a lap at step 3,
            # then reaches the underlying time limit at step 5.
            self.stage1_evaluation_snapshot = {
                "gate_passed": torch.tensor([False, self.steps == 3]),
                "next_gate_index": torch.zeros(2, dtype=torch.long),
                "collision": torch.tensor([True, False]),
                "flyaway": torch.zeros(2, dtype=torch.bool),
            }
            for key in ("position_error_m", "attitude_error_rad", "velocity_error_mps", "vio_age_s", "imu_age_s"):
                self.stage1_evaluation_snapshot[key] = torch.zeros(2)
            for key in ("vio_valid", "imu_valid", "vio_dropped", "imu_dropped"):
                self.stage1_evaluation_snapshot[key] = torch.zeros(2, dtype=torch.bool)
            return (torch.zeros(2, 20), torch.ones(2), torch.tensor([True, False]),
                    torch.tensor([False, self.steps % 5 == 0]), {})

    agent = SimpleNamespace(act=lambda *args, **kwargs: (torch.zeros(2, 4), {}))
    result = namespace["evaluate_case"](Environment(), agent, "clean", 11, 2)
    assert result.summary()["completion_rate"] == 0.5
    assert sorted(r.duration_s for r in result.records) == [0.01, 0.03]
