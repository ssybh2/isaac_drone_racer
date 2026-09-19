import numpy as np
import pytest

from scripts.estimation.train_learned_motion import trace_balanced_sample_weights


def test_trace_balanced_weights_equalize_total_mass_per_trace():
    weights = trace_balanced_sample_weights([2, 8, 5])

    assert weights.shape == (15,)
    assert np.isclose(np.sum(weights[:2]), 1.0)
    assert np.isclose(np.sum(weights[2:10]), 1.0)
    assert np.isclose(np.sum(weights[10:15]), 1.0)


def test_trace_balanced_weights_reject_non_positive_counts():
    with pytest.raises(ValueError):
        trace_balanced_sample_weights([3, 0, 4])

    with pytest.raises(ValueError):
        trace_balanced_sample_weights([])
