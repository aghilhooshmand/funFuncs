"""High-dimensional multivariate normality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MultivariateNormalityResult:
    """Results for high-dimensional multivariate normality checks."""

    n_samples: int
    n_features: int
    adaptive_threshold: float
    generalized_shapiro_wilk_stat: float
    generalized_shapiro_wilk_p: float
    mardia_skewness_stat: float
    mardia_skewness_p: float
    mardia_kurtosis_stat: float
    mardia_kurtosis_p: float
    henze_zirkler_stat: float
    henze_zirkler_p: float
    fisher_marginal_p: float
    is_normal: bool
    generalized_shapiro_wilk_label: str
    mardia_skewness_label: str
    mardia_kurtosis_label: str
    henze_zirkler_label: str
    fisher_marginal_label: str

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "adaptive_threshold": self.adaptive_threshold,
            "generalized_shapiro_wilk_stat": self.generalized_shapiro_wilk_stat,
            "generalized_shapiro_wilk_p": self.generalized_shapiro_wilk_p,
            "mardia_skewness_stat": self.mardia_skewness_stat,
            "mardia_skewness_p": self.mardia_skewness_p,
            "mardia_kurtosis_stat": self.mardia_kurtosis_stat,
            "mardia_kurtosis_p": self.mardia_kurtosis_p,
            "henze_zirkler_stat": self.henze_zirkler_stat,
            "henze_zirkler_p": self.henze_zirkler_p,
            "fisher_marginal_p": self.fisher_marginal_p,
            "is_normal": self.is_normal,
        }


@dataclass(frozen=True)
class MultivariatePowerStudyResult:
    """Rejection-rate summary from simulation study."""

    d: int
    m: int
    B: int
    BB: int
    L: int
    distr: str
    choice: str
    alpha: float
    generalized_shapiro_wilk_power: float
    mardia_skewness_power: float
    mardia_kurtosis_power: float
    mardia_bonferroni_power: float
    henze_zirkler_power: float
    fisher_marginal_power: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "d": self.d,
            "m": self.m,
            "B": self.B,
            "BB": self.BB,
            "L": self.L,
            "distr": self.distr,
            "choice": self.choice,
            "alpha": self.alpha,
            "generalized_shapiro_wilk_power": self.generalized_shapiro_wilk_power,
            "mardia_skewness_power": self.mardia_skewness_power,
            "mardia_kurtosis_power": self.mardia_kurtosis_power,
            "mardia_bonferroni_power": self.mardia_bonferroni_power,
            "henze_zirkler_power": self.henze_zirkler_power,
            "fisher_marginal_power": self.fisher_marginal_power,
        }


def _clean_matrix(data: Any) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("Input data must be a 2D array-like object.")
    return x[~np.isnan(x).any(axis=1)]


def _p_label(p_value: float, alpha: float) -> str:
    return "Normal" if p_value >= alpha else "Not Normal"


def _adaptive_threshold_covariance(x: np.ndarray, kk: int = 5) -> tuple[np.ndarray, float]:
    """Adaptive threshold covariance (in spirit of Cai & Liu), ported from R workflow."""
    m, d = x.shape
    i_d = np.eye(d)
    xcov0 = np.cov(x, rowvar=False)

    sm = m * (kk - 1) / kk
    lambdas = np.array([6 * j / 50 * np.sqrt(np.log(max(d, 2))) for j in range(1, 51)])
    dif = np.zeros_like(lambdas)

    for j, lam in enumerate(lambdas):
        for fi in range(kk):
            m1 = int(m / kk * fi)
            m2 = int(m / kk * (fi + 1))
            idx = np.r_[0:m1, m2:m]
            sx = x[idx, :]
            cx = x[m1:m2, :]
            sxcov = np.cov(sx, rowvar=False)
            cxcov = np.cov(cx, rowvar=False)

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
    xcov = (xcov + de * i_d) / (1.0 + de)
    return xcov, thr


def _matrix_inv_sqrt(a: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(a)
    eps = 1e-12
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _generalized_shapiro_wilk_modified(x: np.ndarray, cov_est: np.ndarray) -> tuple[float, float]:
    """Generalized SW approximation as in the referenced R code."""
    n, p = x.shape
    z = (x - x.mean(axis=0)) @ _matrix_inv_sqrt(cov_est)
    w = np.array([stats.shapiro(z[:, k]).statistic for k in range(p)])
    wast = float(np.mean(w))

    y = np.log(n)
    w1 = np.log(max(1.0 - wast, 1e-12))
    m = -1.5861 - 0.31082 * y - 0.083751 * y**2 + 0.0038915 * y**3
    s = np.exp(-0.4803 - 0.082676 * y + 0.0030302 * y**2)
    s2 = s**2
    sigma2 = np.log((p - 1 + np.exp(s2)) / p)
    mu1 = m + s2 / 2 - sigma2 / 2
    p_value = float(stats.norm.sf(w1, loc=mu1, scale=np.sqrt(sigma2)))
    return wast, p_value


def _mardia_tests(x: np.ndarray, cov_est: np.ndarray) -> tuple[float, float, float, float]:
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s_inv = np.linalg.pinv(cov_est)
    d2 = np.sum((xc @ s_inv) * xc, axis=1)
    a = xc @ s_inv @ xc.T

    b1p = float(np.mean(a**3))
    skew_stat = n * b1p / 6.0
    skew_df = p * (p + 1) * (p + 2) / 6.0
    skew_p = float(stats.chi2.sf(skew_stat, df=skew_df))

    b2p = float(np.mean(d2**2))
    expected = p * (p + 2)
    var = (8.0 * p * (p + 2)) / n
    kurt_z = (b2p - expected) / np.sqrt(var)
    kurt_p = float(2.0 * stats.norm.sf(abs(kurt_z)))
    return skew_stat, skew_p, kurt_z, kurt_p


def _henze_zirkler_test(
    x: np.ndarray,
    cov_est: np.ndarray,
    *,
    reps: int = 150,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Approximate HZ statistic with permutation-based p-value."""
    n, p = x.shape
    if n < 8:
        return np.nan, np.nan

    xc = x - x.mean(axis=0)
    s_inv = np.linalg.pinv(cov_est)
    d2 = np.sum((xc @ s_inv) * xc, axis=1)
    pair = xc @ s_inv @ xc.T
    dm2 = d2[:, None] + d2[None, :] - 2 * pair

    beta = (1.0 / np.sqrt(2.0)) * ((2 * p + 1) / 4.0) ** (1.0 / (p + 4.0)) * n ** (1.0 / (p + 4.0))
    term1 = np.exp(-(beta**2 / 2.0) * dm2).sum() / (n**2)
    term2 = (
        2
        * (1 + beta**2) ** (-p / 2.0)
        * np.exp(-(beta**2 / (2 * (1 + beta**2))) * d2).mean()
    )
    term3 = (1 + 2 * beta**2) ** (-p / 2.0)
    hz = float(n * (term1 - term2 + term3))

    # simple parametric bootstrap for p-value
    rng = np.random.default_rng(42) if rng is None else rng
    hz_null = np.empty(reps, dtype=float)
    for i in range(reps):
        z = rng.multivariate_normal(np.zeros(p), cov_est, size=n)
        hz_null[i] = _henze_zirkler_stat_only(z, cov_est)
    p_value = float((np.sum(hz_null >= hz) + 1) / (reps + 1))
    return hz, p_value


def _henze_zirkler_stat_only(x: np.ndarray, cov_est: np.ndarray) -> float:
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s_inv = np.linalg.pinv(cov_est)
    d2 = np.sum((xc @ s_inv) * xc, axis=1)
    pair = xc @ s_inv @ xc.T
    dm2 = d2[:, None] + d2[None, :] - 2 * pair
    beta = (1.0 / np.sqrt(2.0)) * ((2 * p + 1) / 4.0) ** (1.0 / (p + 4.0)) * n ** (1.0 / (p + 4.0))
    term1 = np.exp(-(beta**2 / 2.0) * dm2).sum() / (n**2)
    term2 = (
        2
        * (1 + beta**2) ** (-p / 2.0)
        * np.exp(-(beta**2 / (2 * (1 + beta**2))) * d2).mean()
    )
    term3 = (1 + 2 * beta**2) ** (-p / 2.0)
    return float(n * (term1 - term2 + term3))


def _fisher_marginal_p(x_whitened: np.ndarray) -> float:
    _, p = x_whitened.shape
    pvals = np.array([stats.shapiro(x_whitened[:, j]).pvalue for j in range(p)], dtype=float)
    pvals = np.clip(pvals, 1e-300, 1.0)
    fisher_stat = float(-2.0 * np.log(pvals).sum())
    return float(stats.chi2.sf(fisher_stat, 2 * p))


def multivariate_normality_check_high_dim(
    data: Any,
    *,
    alpha: float = 0.05,
    print_summary: bool = True,
    hz_bootstrap_reps: int = 150,
    random_state: int | None = None,
) -> MultivariateNormalityResult:
    """
    High-dimensional multivariate normality checker inspired by your R workflow.

    Included diagnostics:
    - adaptive-threshold covariance estimate,
    - modified generalized Shapiro-Wilk (multivariate),
    - Mardia skewness and kurtosis tests,
    - Henze-Zirkler test (bootstrap p-value),
    - Fisher combined p-value from marginal Shapiro tests after whitening.
    """
    x = _clean_matrix(data)
    n, p = x.shape
    if n < 12 or n > 5000:
        raise ValueError("Sample size must be between 12 and 5000.")
    if p < 2:
        raise ValueError("Use univariate tester for single-variable data.")

    cov_est, thr = _adaptive_threshold_covariance(x)
    inv_sqrt = _matrix_inv_sqrt(cov_est)
    x_white = (x - x.mean(axis=0)) @ inv_sqrt

    gsw_stat, gsw_p = _generalized_shapiro_wilk_modified(x, cov_est)
    m_skew_stat, m_skew_p, m_kurt_stat, m_kurt_p = _mardia_tests(x, cov_est)
    hz_rng = np.random.default_rng(random_state) if random_state is not None else None
    hz_stat, hz_p = _henze_zirkler_test(x, cov_est, reps=hz_bootstrap_reps, rng=hz_rng)
    fisher_p = _fisher_marginal_p(x_white)

    gsw_label = _p_label(gsw_p, alpha)
    m_skew_label = _p_label(m_skew_p, alpha)
    m_kurt_label = _p_label(m_kurt_p, alpha)
    hz_label = "Normal" if (np.isnan(hz_p) or hz_p >= alpha) else "Not Normal"
    fisher_label = _p_label(fisher_p, alpha)

    votes = [gsw_label, m_skew_label, m_kurt_label, fisher_label]
    if not np.isnan(hz_p):
        votes.append(hz_label)
    is_normal = all(v == "Normal" for v in votes)

    result = MultivariateNormalityResult(
        n_samples=n,
        n_features=p,
        adaptive_threshold=thr,
        generalized_shapiro_wilk_stat=gsw_stat,
        generalized_shapiro_wilk_p=gsw_p,
        mardia_skewness_stat=m_skew_stat,
        mardia_skewness_p=m_skew_p,
        mardia_kurtosis_stat=m_kurt_stat,
        mardia_kurtosis_p=m_kurt_p,
        henze_zirkler_stat=hz_stat,
        henze_zirkler_p=hz_p,
        fisher_marginal_p=fisher_p,
        is_normal=is_normal,
        generalized_shapiro_wilk_label=gsw_label,
        mardia_skewness_label=m_skew_label,
        mardia_kurtosis_label=m_kurt_label,
        henze_zirkler_label=hz_label,
        fisher_marginal_label=fisher_label,
    )

    if print_summary:
        _print_summary(result)
    return result


# Backward-compatible alias
multivariate_normality_tester = multivariate_normality_check_high_dim


def _cov_from_choice(d: int, choice: str, rng: np.random.Generator) -> np.ndarray:
    if choice == "Sig1":
        return np.eye(d)
    if choice == "Sig11":
        return np.diag(rng.uniform(1.0, 5.0, size=d))
    if choice == "Sig12":
        return np.diag(rng.uniform(1.0, 20.0, size=d))
    if choice == "Sig2":
        idx = np.arange(d)
        return 0.5 ** np.abs(idx[:, None] - idx[None, :])
    if choice == "Sig3":
        s = np.eye(d)
        upper = rng.random((d, d))
        mask = rng.random((d, d)) < 0.02
        s = s + np.triu(mask * upper, 1)
        s = s + s.T - np.eye(d)
        min_eig = float(np.linalg.eigvalsh(s).min())
        de = abs(min_eig) + 0.05
        return (s + de * np.eye(d)) / (1.0 + de)
    raise ValueError("Unsupported choice. Use one of: Sig1, Sig11, Sig12, Sig2, Sig3.")


def _gen_sample(
    d: int,
    m: int,
    distr: str,
    choice: str,
    rng: np.random.Generator,
) -> np.ndarray:
    s1 = _cov_from_choice(d, choice, rng)
    zeros = np.zeros(d)
    if distr == "normal":
        return rng.multivariate_normal(zeros, s1, size=m)
    if distr in {"t05", "t025", "t2", "t4"}:
        factor = {"t05": 0.5, "t025": 0.25, "t2": 2.0, "t4": 4.0}[distr]
        df = max(int(factor * d), 1)
        g = rng.chisquare(df, size=m)
        z = rng.multivariate_normal(zeros, s1, size=m)
        return z / np.sqrt(g[:, None] / df)
    if distr == "mixture":
        ss1 = 1.0 - 1.8 / np.sqrt(d)
        ss2 = 1.0 + 1.8 / np.sqrt(d)
        half = m // 2
        x = np.vstack(
            [
                rng.multivariate_normal(zeros, ss1 * np.eye(d), size=half),
                rng.multivariate_normal(zeros, ss2 * np.eye(d), size=m - half),
            ]
        )
        return x @ np.linalg.cholesky(s1).T
    if distr.startswith("chisquare"):
        df = int(distr.replace("chisquare", ""))
        x0 = rng.chisquare(df=df, size=(m, d)) - df
        return (x0 @ np.linalg.cholesky(s1).T) / np.sqrt(2.0 * df)
    raise ValueError(
        "Unsupported distr. Use: normal, t05, t025, t2, t4, mixture, chisquare3/5/10/20."
    )


def multivariate_power_study_high_dim(
    *,
    d: int = 100,
    m: int = 100,
    distr: str = "normal",
    choice: str = "Sig3",
    B: int = 200,
    BB: int = 150,
    L: int = 1,
    alpha: float = 0.05,
    random_state: int | None = None,
    print_progress: bool = False,
) -> MultivariatePowerStudyResult:
    """
    Simulation/power loop inspired by `runme.R`.

    Parameters mirror the R script:
    - `B`: number of simulation replications,
    - `BB`: inner bootstrap reps for HZ p-value calibration,
    - `L`: repeated p-value estimates per replication (averaged).
    """
    if d < 2 or m < 12 or B < 1 or BB < 10 or L < 1:
        raise ValueError("Require d>=2, m>=12, B>=1, BB>=10, L>=1.")

    rng = np.random.default_rng(random_state)
    gsw_rej = np.zeros(B, dtype=bool)
    ms_rej = np.zeros(B, dtype=bool)
    mk_rej = np.zeros(B, dtype=bool)
    hz_rej = np.zeros(B, dtype=bool)
    fisher_rej = np.zeros(B, dtype=bool)

    for i in range(B):
        x = _gen_sample(d=d, m=m, distr=distr, choice=choice, rng=rng)
        pvals = np.empty((L, 5), dtype=float)  # gsw, ms, mk, hz, fisher
        for l in range(L):
            res = multivariate_normality_check_high_dim(
                x,
                alpha=alpha,
                print_summary=False,
                hz_bootstrap_reps=BB,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            pvals[l, 0] = res.generalized_shapiro_wilk_p
            pvals[l, 1] = res.mardia_skewness_p
            pvals[l, 2] = res.mardia_kurtosis_p
            pvals[l, 3] = res.henze_zirkler_p if not np.isnan(res.henze_zirkler_p) else 1.0
            pvals[l, 4] = res.fisher_marginal_p

        pbar = pvals.mean(axis=0)
        gsw_rej[i] = pbar[0] < alpha
        ms_rej[i] = pbar[1] < alpha
        mk_rej[i] = pbar[2] < alpha
        hz_rej[i] = pbar[3] < alpha
        fisher_rej[i] = pbar[4] < alpha

        if print_progress and ((i + 1) % max(B // 10, 1) == 0):
            print(f"[{i + 1}/{B}] completed")

    mardia_bonf = np.mean(np.logical_or(ms_rej, mk_rej))
    return MultivariatePowerStudyResult(
        d=d,
        m=m,
        B=B,
        BB=BB,
        L=L,
        distr=distr,
        choice=choice,
        alpha=alpha,
        generalized_shapiro_wilk_power=float(gsw_rej.mean()),
        mardia_skewness_power=float(ms_rej.mean()),
        mardia_kurtosis_power=float(mk_rej.mean()),
        mardia_bonferroni_power=float(mardia_bonf),
        henze_zirkler_power=float(hz_rej.mean()),
        fisher_marginal_power=float(fisher_rej.mean()),
    )


def _print_summary(result: MultivariateNormalityResult) -> None:
    lines = [
        "",
        "      Multivariate Normality Tester",
        "     -------------------------------",
        f"  Generalized SW (modified): {result.generalized_shapiro_wilk_label}",
        f"  Mardia Skewness:           {result.mardia_skewness_label}",
        f"  Mardia Kurtosis:           {result.mardia_kurtosis_label}",
        f"  Henze-Zirkler:             {result.henze_zirkler_label}",
        f"  Fisher Marginal SW:        {result.fisher_marginal_label}",
        "     -------------------------------",
    ]
    marker = "[OK]" if result.is_normal else "[X]"
    verdict = "Data Appears Multivariate Normal" if result.is_normal else "Data is Not Multivariate Normal"
    lines.append(f"{marker} {verdict}")
    print("\n".join(lines))
