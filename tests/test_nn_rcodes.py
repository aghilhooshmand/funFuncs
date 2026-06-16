import numpy as np
import pytest

from funfuncs.statistics.nn_funcs import gen, getdis, getp, getpow, get_xy_stat
from funfuncs.statistics.nn_funcs2 import adapt_thres_cov, mv_shapiro_test_adapt_thres_mod
from funfuncs.statistics.nn_runme import run_simulation


def test_adapt_thres_cov_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    cov = adapt_thres_cov(x)
    assert cov.shape == (8, 8)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_getdis_symmetric():
    x = np.array([[0, 0], [3, 4]], dtype=float)
    d = getdis(x)
    assert d.shape == (2, 2)
    assert np.isclose(d[0, 1], 5.0)


def test_getp_returns_pvalues():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 6))
    out = getp(x, l=1, bb=10, rng=rng)
    assert out.yp.shape == (2,)
    assert np.all((out.yp >= 0) & (out.yp <= 1))


def test_mv_shapiro_runs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 5))
    res = mv_shapiro_test_adapt_thres_mod(x)
    assert 0 <= res["p_value"] <= 1


def test_run_simulation_smoke():
    result = run_simulation(
        d=8,
        m=30,
        distr="normal",
        choice="Sig1",
        b=2,
        bb=8,
        l=1,
        random_state=0,
        print_progress=False,
    )
    summary = result.summary()
    assert "NEW" in summary
    assert summary["Fisher"] >= 0
