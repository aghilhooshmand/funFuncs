"""Comprehensive normality diagnostics (tests, shape metrics, optional plots)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad


@dataclass(frozen=True)
class NormalityResult:
    """Numeric outcomes from :func:`normality_tester`."""

    shapiro_wilk_p: float
    anderson_darling_p: float
    kolmogorov_smirnov_p: float
    jarque_bera_p: float
    skewness: float
    kurtosis: float
    is_normal: bool
    shapiro_wilk_label: str
    anderson_darling_label: str
    kolmogorov_smirnov_label: str
    jarque_bera_label: str
    skewness_label: str
    kurtosis_label: str

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "Shapiro_Wilk_p": self.shapiro_wilk_p,
            "Anderson_Darling_p": self.anderson_darling_p,
            "Kolmogorov_Smirnov_p": self.kolmogorov_smirnov_p,
            "Jarque_Bera_p": self.jarque_bera_p,
            "Skewness": self.skewness,
            "Kurtosis": self.kurtosis,
            "is_normal": self.is_normal,
        }


def _clean_numeric(data: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return arr[~np.isnan(arr)]


def _p_label(p_value: float, alpha: float = 0.05) -> str:
    return "Normal" if p_value >= alpha else "Not Normal"


def _shape_label(value: float, low: float, high: float) -> str:
    return "Within Bounds" if low < value < high else "Outside Bounds"


def normality_tester(
    data: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    skew_bounds: tuple[float, float] = (-0.5, 0.5),
    kurt_bounds: tuple[float, float] = (-2.0, 2.0),
    plot: bool = True,
    print_summary: bool = True,
) -> NormalityResult:
    """
    Run multiple normality tests and optional visual diagnostics.

    A sample is treated as approximately normal when:

    - all test p-values are >= ``alpha`` (fail to reject normality),
    - skewness lies in ``skew_bounds`` (open interval, matching the R version),
    - excess kurtosis lies in ``kurt_bounds``.

    Parameters
    ----------
    data
        Numeric sample (NaNs are dropped).
    alpha
        Significance level for hypothesis tests.
    skew_bounds, kurt_bounds
        Acceptable ranges for skewness and excess kurtosis.
    plot
        If True, draw histogram, boxplot, and Q-Q plot.
    print_summary
        If True, print a text summary to stdout.

    Returns
    -------
    NormalityResult
        Test p-values, shape metrics, labels, and overall ``is_normal`` flag.
    """
    sample = _clean_numeric(data)
    n = sample.size

    if n < 3:
        raise ValueError("At least 3 non-missing observations are required.")

    if n > 5000:
        import warnings

        warnings.warn(
            "Shapiro-Wilk is most reliable for n < 5000; results may be approximate.",
            stacklevel=2,
        )

    sw_stat, sw_p = stats.shapiro(sample)

    _, ad_p = normal_ad(sample)

    mean, std = float(sample.mean()), float(sample.std(ddof=1))
    if std == 0:
        ks_p = 0.0
    else:
        ks_p = float(stats.kstest(sample, "norm", args=(mean, std)).pvalue)

    jb_stat, jb_p = stats.jarque_bera(sample)

    skew = float(stats.skew(sample, bias=False))
    kurt = float(stats.kurtosis(sample, fisher=True, bias=False))

    sw_label = _p_label(sw_p, alpha)
    ad_label = _p_label(ad_p, alpha)
    ks_label = _p_label(ks_p, alpha)
    jb_label = _p_label(jb_p, alpha)
    skew_label = _shape_label(skew, *skew_bounds)
    kurt_label = _shape_label(kurt, *kurt_bounds)

    is_normal = (
        sw_label == ad_label == ks_label == jb_label == "Normal"
        and skew_label == "Within Bounds"
        and kurt_label == "Within Bounds"
    )

    result = NormalityResult(
        shapiro_wilk_p=float(sw_p),
        anderson_darling_p=float(ad_p),
        kolmogorov_smirnov_p=ks_p,
        jarque_bera_p=float(jb_p),
        skewness=skew,
        kurtosis=kurt,
        is_normal=is_normal,
        shapiro_wilk_label=sw_label,
        anderson_darling_label=ad_label,
        kolmogorov_smirnov_label=ks_label,
        jarque_bera_label=jb_label,
        skewness_label=skew_label,
        kurtosis_label=kurt_label,
    )

    if print_summary:
        _print_summary(result)

    if plot:
        _plot_diagnostics(sample)

    return result


def _print_summary(result: NormalityResult) -> None:
    lines = [
        "",
        "      Normality Tester Function",
        "     ---------------------------",
        f"  Shapiro-Wilk Test:       {result.shapiro_wilk_label}",
        f"  Anderson-Darling Test:   {result.anderson_darling_label}",
        f"  Kolmogorov-Smirnov Test: {result.kolmogorov_smirnov_label}",
        f"  Jarque-Bera Test:        {result.jarque_bera_label}",
        f"  Skewness:                {result.skewness_label}",
        f"  Kurtosis:                {result.kurtosis_label}",
        "     ---------------------------",
    ]
    verdict = "Data Appears to be Normal" if result.is_normal else "Data is Not Normal"
    marker = "[OK]" if result.is_normal else "[X]"
    lines.append(f"{marker} {verdict}")
    print("\n".join(lines))


def _plot_diagnostics(sample: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(sample, bins="auto", color="lightblue", edgecolor="white")
    axes[0].set_title("Histogram of Data")

    axes[1].boxplot(sample, patch_artist=True, boxprops={"facecolor": "lightblue"})
    axes[1].set_title("Boxplot of Data")

    stats.probplot(sample, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot of Data")

    fig.tight_layout()
    plt.show()
