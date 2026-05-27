import numpy as np
import pytest

from funfuncs.statistics import normality_tester


def test_normal_sample_often_passes():
    rng = np.random.default_rng(123)
    result = normality_tester(rng.normal(size=500), plot=False, print_summary=False)
    assert result.shapiro_wilk_p > 0
    assert isinstance(result.is_normal, bool)


def test_mixture_often_fails():
    rng = np.random.default_rng(0)
    mixed = np.concatenate([rng.normal(size=250), rng.normal(loc=5, scale=2, size=250)])
    result = normality_tester(mixed, plot=False, print_summary=False)
    assert result.is_normal is False


def test_too_few_values_raises():
    with pytest.raises(ValueError, match="At least 3"):
        normality_tester([1.0, 2.0], plot=False, print_summary=False)
