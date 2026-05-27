"""Statistical utilities."""

from funfuncs.statistics.multivariate_normality import (
    MultivariateNormalityResult,
    MultivariatePowerStudyResult,
    multivariate_normality_check_high_dim,
    multivariate_power_study_high_dim,
    multivariate_normality_tester,
)
from funfuncs.statistics.normality import NormalityResult, normality_tester

__all__ = [
    "NormalityResult",
    "normality_tester",
    "MultivariateNormalityResult",
    "MultivariatePowerStudyResult",
    "multivariate_normality_check_high_dim",
    "multivariate_power_study_high_dim",
    "multivariate_normality_tester",
]
