import numpy as np
import pytest

from funfuncs.statistics import (
    multivariate_normality_check_high_dim,
    multivariate_power_study_high_dim,
)


def test_multivariate_returns_result():
    rng = np.random.default_rng(123)
    x = rng.multivariate_normal(np.zeros(8), np.eye(8), size=120)
    result = multivariate_normality_check_high_dim(x, print_summary=False)
    assert result.generalized_shapiro_wilk_p >= 0
    assert isinstance(result.is_normal, bool)


def test_rejects_non_matrix():
    with pytest.raises(ValueError, match="2D"):
        multivariate_normality_check_high_dim([1, 2, 3], print_summary=False)


def test_rejects_small_n():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 4))
    with pytest.raises(ValueError, match="between 12 and 5000"):
        multivariate_normality_check_high_dim(x, print_summary=False)


def test_power_study_returns_rates():
    result = multivariate_power_study_high_dim(
        d=6,
        m=30,
        distr="normal",
        choice="Sig1",
        B=3,
        BB=20,
        L=1,
        random_state=0,
    )
    as_dict = result.as_dict()
    assert as_dict["B"] == 3
    assert 0.0 <= as_dict["generalized_shapiro_wilk_power"] <= 1.0
