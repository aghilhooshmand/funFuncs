"""Statistical utilities."""

from funfuncs.statistics.multivariate_normality import (
    MultivariateNormalityResult,
    MultivariatePowerStudyResult,
    multivariate_normality_check_high_dim,
    multivariate_power_study_high_dim,
    multivariate_normality_tester,
)
from funfuncs.statistics.nn_funcs import (
    GetPResult,
    XYStatResult,
    gen,
    getdis,
    getp,
    getpow,
    get_r,
    get_xy_stat,
    g_tests_generalized,
    mstree,
)
from funfuncs.statistics.nn_funcs2 import adapt_thres_cov, matrix_sqrtm, mv_shapiro_test_adapt_thres_mod
from funfuncs.statistics.normality import NormalityResult, normality_tester

__all__ = [
    "NormalityResult",
    "normality_tester",
    "MultivariateNormalityResult",
    "MultivariatePowerStudyResult",
    "multivariate_normality_check_high_dim",
    "multivariate_power_study_high_dim",
    "multivariate_normality_tester",
    "GetPResult",
    "XYStatResult",
    "RunmeResult",
    "adapt_thres_cov",
    "gen",
    "getdis",
    "getp",
    "getpow",
    "get_r",
    "get_xy_stat",
    "g_tests_generalized",
    "matrix_sqrtm",
    "mstree",
    "mv_shapiro_test_adapt_thres_mod",
    "print_runme_summary",
    "run_simulation",
]


def __getattr__(name: str):
    if name in {"RunmeResult", "run_simulation", "print_runme_summary"}:
        from funfuncs.statistics.nn_runme import RunmeResult, print_runme_summary, run_simulation

        return {"RunmeResult": RunmeResult, "run_simulation": run_simulation, "print_runme_summary": print_runme_summary}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
