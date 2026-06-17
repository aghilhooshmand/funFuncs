"""Python port of RCodes/funcs2.R (adaptive-threshold covariance + modified mvSW)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm


def adapt_thres_cov(x: np.ndarray, kk: int = 5) -> np.ndarray:
    """Adaptive threshold covariance estimate (Cai & Liu, 2011), from authors' R code."""
    m, d = x.shape
    i_d = np.eye(d)
    xcov0 = np.cov(x, rowvar=False, ddof=1)

    sm = m * (kk - 1) / kk
    lambdas = np.array([6 * j / 50 * np.sqrt(np.log(max(d, 2))) for j in range(1, 51)])
    dif = np.zeros(50)

    for j, lam in enumerate(lambdas):
        for fi in range(1, kk + 1):
            m1 = int(m / kk * (fi - 1)) + 1
            m2 = int(m / kk * fi)
            sx = np.delete(x, np.arange(m1 - 1, m2), axis=0)
            cx = x[m1 - 1 : m2]
            sxcov = np.cov(sx, rowvar=False, ddof=1)
            cxcov = np.cov(cx, rowvar=False, ddof=1)

            cc1 = (sx * sx).T @ (sx * sx) / sm
            cc2 = ((sx.T @ sx) / sm) * sxcov
            cc3 = sxcov * sxcov
            t1 = cc1 - 2 * cc2 + cc3
            aa = np.sqrt(np.divide(sm * (sxcov * sxcov), t1, out=np.zeros_like(t1), where=t1 > 0))
            masked = sxcov * (np.abs(aa) >= lam)
            dif[j] += np.linalg.norm(masked - cxcov, ord="fro") ** 2

    thr = float(lambdas[dif == dif.min()].max())

    cc1 = (x * x).T @ (x * x) / m
    cc2 = ((x.T @ x) / m) * xcov0
    cc3 = xcov0 * xcov0
    t1 = cc1 - 2 * cc2 + cc3
    aa = np.sqrt(np.divide(m * (xcov0 * xcov0), t1, out=np.zeros_like(t1), where=t1 > 0))
    xcov = xcov0 * (np.abs(aa) >= thr)

    eigvals = np.linalg.eigvalsh(xcov)
    de = max(-float(eigvals.min()), 0.0) + 0.05
    return (xcov + de * i_d) / (1.0 + de)


def _matrix_inv_sqrt_from_cov(cov: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def mv_shapiro_test_adapt_thres_mod(
    x: np.ndarray,
    cov_est: np.ndarray | None = None,
) -> dict[str, float | str]:
    """
    Modified multivariate Shapiro-Wilk test using adaptive-threshold covariance.

    Port of ``mvShapiro.Test.adapt.thres.mod`` in RCodes/funcs2.R.

    If ``cov_est`` is provided it is used instead of calling ``adapt_thres_cov(x)``
    (same matrix, avoids redundant work when the caller already estimated it).
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    n, p = arr.shape
    if n < 12 or n > 5000:
        raise ValueError("Sample size must be between 12 and 5000.")

    centered = arr - arr.mean(axis=0)
    if cov_est is None:
        cov_est = adapt_thres_cov(arr)
    inv_sqrt = _matrix_inv_sqrt_from_cov(cov_est)
    z = centered @ inv_sqrt

    w = np.array([stats.shapiro(z[:, k]).statistic for k in range(p)])
    wast = float(np.mean(w))

    y = np.log(n)
    w1 = np.log(max(1.0 - wast, 1e-12))
    mu = -1.5861 - 0.31082 * y - 0.083751 * y**2 + 0.0038915 * y**3
    s = np.exp(-0.4803 - 0.082676 * y + 0.0030302 * y**2)
    s2 = s**2
    sigma2 = np.log((p - 1 + np.exp(s2)) / p)
    mu1 = mu + s2 / 2 - sigma2 / 2
    p_value = float(stats.norm.sf(w1, loc=mu1, scale=np.sqrt(sigma2)))

    return {
        "statistic": wast,
        "p_value": p_value,
        "method": (
            "Generalized Shapiro-Wilk with adaptive-threshold covariance "
            "(Villasenor-Alva & Gonzalez-Estrada; Cai & Liu 2011)"
        ),
    }


def matrix_sqrtm(a: np.ndarray) -> np.ndarray:
    """Matrix square root (R ``sqrtm``), real symmetric PSD input expected."""
    out = sqrtm(a)
    return np.real_if_close(out, tol=1e5).astype(float)
