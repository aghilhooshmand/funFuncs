"""Python port of RCodes/runme.R (simulation study for NN normality test paper)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from funfuncs.statistics.nn_funcs import gen, getp, getpow
from funfuncs.statistics.nn_funcs2 import adapt_thres_cov, mv_shapiro_test_adapt_thres_mod


def _matrix_inv_sqrt_from_cov(cov: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


@dataclass(frozen=True)
class RunmeResult:
    d: int
    m: int
    distr: str
    choice: str
    b: int
    bb: int
    l: int
    new_test: np.ndarray
    efr: np.ndarray
    new_sigma_inv_sqrt: np.ndarray
    new_d_inv_sqrt: np.ndarray
    mardia_skewness: float
    mardia_kurtosis: float
    mardia_bonferroni: float
    henze_zirkler: float
    royston: float | None
    ep: float | None
    mv_shapiro: float
    fisher: float

    def summary(self) -> dict[str, float | np.ndarray | None]:
        return {
            "NEW": self.new_test,
            "eFR": self.efr,
            "NEW_sigma_inv_sqrt": self.new_sigma_inv_sqrt,
            "NEW_D_inv_sqrt": self.new_d_inv_sqrt,
            "Mardia_s": self.mardia_skewness,
            "Mardia_k": self.mardia_kurtosis,
            "Mardia_Bonf": self.mardia_bonferroni,
            "HZ": self.henze_zirkler,
            "Royston": self.royston,
            "Ep": self.ep,
            "mvSW": self.mv_shapiro,
            "Fisher": self.fisher,
        }


def _mardia_pvalues(x: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s_inv = np.linalg.pinv(cov)
    a = xc @ s_inv @ xc.T
    d2 = np.sum((xc @ s_inv) * xc, axis=1)

    skew_stat = n * float(np.mean(a**3)) / 6.0
    skew_df = p * (p + 1) * (p + 2) / 6.0
    skew_p = float(stats.chi2.sf(skew_stat, df=skew_df))

    expected = p * (p + 2)
    var = 8.0 * p * (p + 2) / n
    kurt_z = (float(np.mean(d2**2)) - expected) / np.sqrt(var)
    kurt_p = float(2.0 * stats.norm.sf(abs(kurt_z)))
    return skew_p, kurt_p


def _henze_zirkler_p(x: np.ndarray, cov: np.ndarray, rng: np.random.Generator, reps: int = 200) -> float:
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s_inv = np.linalg.pinv(cov)
    d2 = np.sum((xc @ s_inv) * xc, axis=1)
    pair = xc @ s_inv @ xc.T
    dm2 = d2[:, None] + d2[None, :] - 2 * pair
    beta = (1.0 / np.sqrt(2.0)) * ((2 * p + 1) / 4.0) ** (1.0 / (p + 4.0)) * n ** (1.0 / (p + 4.0))
    term1 = np.exp(-(beta**2 / 2.0) * dm2).sum() / (n**2)
    term2 = 2 * (1 + beta**2) ** (-p / 2.0) * np.exp(-(beta**2 / (2 * (1 + beta**2))) * d2).mean()
    term3 = (1 + 2 * beta**2) ** (-p / 2.0)
    hz = float(n * (term1 - term2 + term3))

    null = np.empty(reps)
    for i in range(reps):
        z = rng.multivariate_normal(np.zeros(p), cov, size=n)
        zc = z - z.mean(axis=0)
        z_inv = np.linalg.pinv(cov)
        zd2 = np.sum((zc @ z_inv) * zc, axis=1)
        zpair = zc @ z_inv @ zc.T
        zdm2 = zd2[:, None] + zd2[None, :] - 2 * zpair
        t1 = np.exp(-(beta**2 / 2.0) * zdm2).sum() / (n**2)
        t2 = 2 * (1 + beta**2) ** (-p / 2.0) * np.exp(-(beta**2 / (2 * (1 + beta**2))) * zd2).mean()
        null[i] = n * (t1 - t2 + term3)
    return float((np.sum(null >= hz) + 1) / (reps + 1))


def _royston_p(x_white: np.ndarray) -> float:
    _, p = x_white.shape
    pvals = [stats.shapiro(x_white[:, j]).pvalue for j in range(p)]
    pvals = np.clip(pvals, 1e-300, 1.0)
    stat = float(np.sum(stats.norm.ppf(1.0 - np.array(pvals))))
    mu = 0.0
    sigma = float(np.sqrt(p))
    z = (stat - mu) / sigma
    return float(2.0 * stats.norm.sf(abs(z)))


def _ep_p(x: np.ndarray, cov: np.ndarray) -> float:
    """Doornik-Hansen style omnibus normality p-value (R ``normality.test2`` proxy)."""
    n, p = x.shape
    xc = x - x.mean(axis=0)
    s_inv = np.linalg.pinv(cov)
    a = xc @ s_inv @ xc.T
    d2 = np.sum((xc @ s_inv) * xc, axis=1)
    skew_stat = n * float(np.mean(a**3)) / 6.0
    kurt_stat = float(np.mean(d2**2))
    skew_df = p * (p + 1) * (p + 2) / 6.0
    p_skew = stats.chi2.sf(skew_stat, df=skew_df)
    expected = p * (p + 2)
    var = 8.0 * p * (p + 2) / n
    z_kurt = (kurt_stat - expected) / np.sqrt(var)
    p_kurt = 2.0 * stats.norm.sf(abs(z_kurt))
    chi = -2.0 * np.log(np.clip([p_skew, p_kurt], 1e-300, 1.0)).sum()
    return float(stats.chi2.sf(chi, df=4))


def _fisher_p(x_white: np.ndarray) -> float:
    _, p = x_white.shape
    pvals = np.clip([stats.shapiro(x_white[:, j]).pvalue for j in range(p)], 1e-300, 1.0)
    fisher = -2.0 * np.log(pvals).sum()
    return float(stats.chi2.sf(fisher, 2 * p))


def _format_p_pair(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return f"[{arr[0]:.4f}, {arr[1]:.4f}]"


def _p_decision_label(p_value: float, alpha: float) -> str:
    return "Fail to reject normality" if p_value >= alpha else "Reject normality"


def _rate_label(rate: float, alpha: float, *, is_null: bool) -> str:
    if is_null:
        if abs(rate - alpha) <= 0.03:
            return "Good size control (close to alpha)"
        if rate > alpha + 0.05:
            return "Inflated type I error (too many false alarms)"
        if rate < alpha - 0.03:
            return "Conservative (few rejections)"
        return "Acceptable for small B; use larger B for stable size"
    if rate >= 0.5:
        return "Strong power (detects non-normality well)"
    if rate >= 0.2:
        return "Moderate power"
    if rate > alpha:
        return "Some power, but not strong"
    return "Low power (hard to detect this alternative)"


def _is_null_scenario(distr: str) -> bool:
    return distr == "normal"


def _line(char: str = "=", width: int = 88) -> str:
    return char * width


def print_runme_intro() -> None:
    print()
    print(_line())
    print(" Multivariate Normality Simulation (Chen & Xia, 2021)")
    print(_line())
    print(
        "This run repeats a full replication loop like RCodes/runme.R.\n"
        "For each replication it:\n"
        "  1) simulates an (n x d) dataset,\n"
        "  2) computes test p-values (NEW, eFR, mvSW, Fisher, ...),\n"
        "  3) summarizes how often each test rejects normality at level alpha."
    )
    print()


def print_runme_header(
    *,
    d: int,
    m: int,
    distr: str,
    choice: str,
    b: int,
    bb: int,
    l: int,
    alpha: float,
) -> None:
    print_runme_intro()
    print("Simulation settings")
    print(_line("-"))
    print(f"  Dimension (d)              : {d}")
    print(f"  Sample size (m)            : {m}")
    print(f"  Distribution (distr)       : {distr}")
    print(f"  Covariance model (choice)  : {choice}")
    print(f"  Replications (B)           : {b}")
    print(f"  Bootstrap reps in getp (BB): {bb}")
    print(f"  Inner repeats (L)          : {l}")
    print(f"  Significance level (alpha) : {alpha}")
    print()
    print("Column guide (per replication)")
    print(_line("-"))
    guides = [
        ("NEW", "Paper nearest-neighbor test p-value from getp() -> Yp."),
        ("NEW(Σ^-1/2)", "NEW test after whitening with full covariance inverse square root."),
        ("NEW(D^-1/2)", "NEW test after diagonal-only whitening."),
        ("eFR", "Extended Friedman-Rafsky style p-value from getp() -> Op."),
        ("mvSW", "Modified multivariate Shapiro-Wilk p-value."),
        ("Fisher", "Fisher combination of marginal Shapiro p-values."),
    ]
    for name, text in guides:
        print(f"  {name:<14} {text}")
    print()
    print("P-value rule: p >= alpha  -> fail to reject normality;  p < alpha -> reject normality.")
    print("Each [p1, p2] pair gives two sampling p-value estimators (same as R output).")
    print()
    print(
        f"{'Rep':>4} | {'NEW':^16} | {'NEW(Σ^-1/2)':^16} | {'NEW(D^-1/2)':^16} | "
        f"{'eFR':^16} | {'mvSW':^8} | {'Fisher':^8} | Decision"
    )
    print(_line("-"))


def print_replication_row(
    rep: int,
    *,
    new_p: np.ndarray,
    new_sigma_p: np.ndarray,
    new_diag_p: np.ndarray,
    efr_p: np.ndarray,
    mvsw_p: float,
    fisher_p: float,
    alpha: float,
    verbose_detail: bool = False,
) -> None:
    new_main = float(np.asarray(new_p).ravel()[0])
    decision = _p_decision_label(new_main, alpha)
    print(
        f"{rep:4d} | "
        f"{_format_p_pair(new_p):^16} | "
        f"{_format_p_pair(new_sigma_p):^16} | "
        f"{_format_p_pair(new_diag_p):^16} | "
        f"{_format_p_pair(efr_p):^16} | "
        f"{mvsw_p:8.4f} | "
        f"{fisher_p:8.4f} | {decision}"
    )
    if not verbose_detail:
        return

    print(f"      Explanation for replication {rep}:")
    print(f"        - NEW p={new_main:.4f}: {_p_decision_label(new_main, alpha)}.")
    print(f"        - eFR p={float(np.asarray(efr_p).ravel()[0]):.4f}: {_p_decision_label(float(np.asarray(efr_p).ravel()[0]), alpha)}.")
    print(f"        - mvSW p={mvsw_p:.4f}: {_p_decision_label(mvsw_p, alpha)}.")
    print(f"        - Fisher p={fisher_p:.4f}: {_p_decision_label(fisher_p, alpha)}.")
    print()


def _format_rate(value: float | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 1:
            return f"{arr[0]:.3f}"
        return f"[{arr[0]:.3f}, {arr[1]:.3f}]"
    return f"{float(value):.3f}"


def print_runme_summary(result: RunmeResult, *, alpha: float = 0.05) -> None:
    summary = result.summary()
    is_null = _is_null_scenario(result.distr)

    print()
    print(_line())
    print(" Final summary: rejection rates")
    print(_line())
    print(
        f"Scenario: d={result.d}, m={result.m}, distr={result.distr}, choice={result.choice}, "
        f"B={result.b}, BB={result.bb}, L={result.l}"
    )
    if is_null:
        print("Null case (multivariate normal): rejection rate estimates TYPE I error (size).")
        print(f"Target: rates should be close to alpha={alpha:.2f} when B is large.")
    else:
        print("Alternative case (non-normal): rejection rate estimates POWER.")
        print("Target: higher rates mean better detection of non-normality.")
    print()

    rows: list[tuple[str, str, float | np.ndarray, str]] = [
        (
            "NEW (paper test)",
            "Nearest-neighbor test from Chen & Xia (2021); primary method in the paper.",
            summary["NEW"],
            "Main benchmark for high-dimensional normality.",
        ),
        (
            "NEW with Σ^-1/2",
            "NEW test with full covariance whitening plugged in.",
            summary["NEW_sigma_inv_sqrt"],
            "Sensitivity check for covariance preprocessing.",
        ),
        (
            "NEW with D^-1/2",
            "NEW test with diagonal-only whitening.",
            summary["NEW_D_inv_sqrt"],
            "Checks robustness when only scaling is used.",
        ),
        (
            "eFR",
            "Extended Friedman-Rafsky graph test (comparison method).",
            summary["eFR"],
            "Can have size distortion when d is large.",
        ),
        (
            "Mardia skewness",
            "Classical multivariate skewness test.",
            summary["Mardia_s"],
            "Often weak or unstable in high dimensions.",
        ),
        (
            "Mardia kurtosis",
            "Classical multivariate kurtosis test.",
            summary["Mardia_k"],
            "Can be overly sensitive in high dimensions.",
        ),
        (
            "Mardia Bonferroni",
            "Reject if either Mardia skewness or kurtosis rejects at alpha/2.",
            summary["Mardia_Bonf"],
            "Combined Mardia decision rule.",
        ),
        (
            "Henze-Zirkler (HZ)",
            "Characteristic-function based normality test.",
            summary["HZ"],
            "Useful benchmark; may struggle when d >> n.",
        ),
        (
            "mvSW",
            "Modified multivariate Shapiro-Wilk with adaptive covariance.",
            summary["mvSW"],
            "Strong in low/moderate d, weaker as d grows.",
        ),
        (
            "Fisher",
            "Combines marginal Shapiro p-values across dimensions.",
            summary["Fisher"],
            "Simple high-dimensional benchmark.",
        ),
    ]
    if summary["Royston"] is not None:
        rows.append(
            (
                "Royston",
                "Multivariate extension of Shapiro-Wilk (only when d < m).",
                summary["Royston"],
                "Included for low-dimensional comparison.",
            )
        )
    if summary["Ep"] is not None:
        rows.append(
            (
                "Ep (Doornik-Hansen)",
                "Omnibus skewness+kurtosis combination (only when d < m).",
                summary["Ep"],
                "Included for low-dimensional comparison.",
            )
        )

    print(f"{'Test':<22} | {'Reject rate':^13} | {'Assessment':<36} | What it measures")
    print(_line("-"))
    for name, measures, rate, note in rows:
        if isinstance(rate, np.ndarray):
            primary = float(np.asarray(rate).ravel()[0])
        else:
            primary = float(rate)
        assessment = _rate_label(primary, alpha, is_null=is_null)
        print(f"{name:<22} | {_format_rate(rate):^13} | {assessment:<36} | {measures}")
    print(_line("-"))
    print()
    print("How to read this table")
    print(_line("-"))
    print("  - Reject rate = (# replications with p < alpha) / B")
    print("  - Under normal data: good tests have reject rate near alpha (e.g. 0.05).")
    print("  - Under non-normal data: good tests have high reject rate (high power).")
    print("  - With small B (like 5), rates are noisy; use B=1000 for paper-style tables.")
    print()


def run_simulation(
    *,
    d: int = 100,
    m: int = 100,
    distr: str = "normal",
    choice: str = "Sig3",
    b: int = 1000,
    bb: int = 500,
    l: int = 1,
    alpha: float = 0.05,
    random_state: int | None = None,
    print_progress: bool = True,
) -> RunmeResult:
    """
    Full simulation loop from RCodes/runme.R.

    Returns rejection rates (power/size) for NEW test, eFR, mvSW, Fisher, Mardia, HZ, etc.
    """
    rng = np.random.default_rng(random_state)

    ypm = np.zeros((b, 2))
    opm = np.zeros((b, 2))
    yypm = np.zeros((b, 2))
    ydpm = np.zeros((b, 2))

    pv_mv_shapiro = np.zeros(b)
    pv_mardia_s = np.zeros(b)
    pv_mardia_k = np.zeros(b)
    pv_hz = np.zeros(b)
    pv_royston = np.full(b, np.nan)
    pv_ep = np.full(b, np.nan)
    pv_fisher = np.zeros(b)

    if print_progress:
        print_runme_header(d=d, m=m, distr=distr, choice=choice, b=b, bb=bb, l=l, alpha=alpha)
    verbose_detail = print_progress and b <= 20

    for ii in range(b):
        x = gen(d=d, m=m, distr=distr, choice=choice, rng=rng)
        atx = adapt_thres_cov(x)
        inv_sqrt = _matrix_inv_sqrt_from_cov(atx)
        xx = x @ inv_sqrt
        xd = x @ np.diag(np.diag(inv_sqrt))

        temp = getp(x, l=l, bb=bb, rng=rng)
        temp2 = getp(xx, l=l, bb=bb, rng=rng)
        temp3 = getp(xd, l=l, bb=bb, rng=rng)

        ypm[ii] = temp.yp
        yypm[ii] = temp2.yp
        ydpm[ii] = temp3.yp
        opm[ii] = temp.op

        pv_mv_shapiro[ii] = mv_shapiro_test_adapt_thres_mod(x)["p_value"]
        m_s, m_k = _mardia_pvalues(x, atx)
        pv_mardia_s[ii] = m_s
        pv_mardia_k[ii] = m_k
        pv_hz[ii] = _henze_zirkler_p(x, atx, rng)

        if d < m:
            pv_royston[ii] = _royston_p(xx)
            pv_ep[ii] = _ep_p(x, atx)

        pv_fisher[ii] = _fisher_p(xx)

        if print_progress:
            print_replication_row(
                ii + 1,
                new_p=ypm[ii],
                new_sigma_p=yypm[ii],
                new_diag_p=ydpm[ii],
                efr_p=opm[ii],
                mvsw_p=float(pv_mv_shapiro[ii]),
                fisher_p=float(pv_fisher[ii]),
                alpha=alpha,
                verbose_detail=verbose_detail,
            )

    mardia_bonf = float(np.mean((pv_mardia_s < alpha / 2) | (pv_mardia_k < alpha / 2)))

    result = RunmeResult(
        d=d,
        m=m,
        distr=distr,
        choice=choice,
        b=b,
        bb=bb,
        l=l,
        new_test=getpow(ypm, alpha=alpha),
        efr=getpow(opm, alpha=alpha),
        new_sigma_inv_sqrt=getpow(yypm, alpha=alpha),
        new_d_inv_sqrt=getpow(ydpm, alpha=alpha),
        mardia_skewness=float(getpow(pv_mardia_s.reshape(-1, 1), alpha=alpha)[0]),
        mardia_kurtosis=float(getpow(pv_mardia_k.reshape(-1, 1), alpha=alpha)[0]),
        mardia_bonferroni=mardia_bonf,
        henze_zirkler=float(getpow(pv_hz.reshape(-1, 1), alpha=alpha)[0]),
        royston=float(getpow(pv_royston.reshape(-1, 1), alpha=alpha)[0]) if d < m else None,
        ep=float(getpow(pv_ep.reshape(-1, 1), alpha=alpha)[0]) if d < m else None,
        mv_shapiro=float(getpow(pv_mv_shapiro.reshape(-1, 1), alpha=alpha)[0]),
        fisher=float(getpow(pv_fisher.reshape(-1, 1), alpha=alpha)[0]),
    )

    if print_progress:
        print_runme_summary(result, alpha=alpha)

    return result


if __name__ == "__main__":
    run_simulation(
        d=20,
        m=100,
        distr="normal",
        choice="Sig1",
        b=5,
        bb=20,
        l=1,
        random_state=0,
        print_progress=True,
    )
