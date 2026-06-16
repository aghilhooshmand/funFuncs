"""Analyze a real high-dimensional dataset with the Chen & Xia (2021) tests.

Usage (from project root):

    python -m examples.analyze_dataset --path DS_example/barlow_twins_class_0.pt

The script will:
- load the dataset (rows = samples, columns = variables),
- run the paper's NEW nearest-neighbor test (getp) on the data as-is by default,
- run eFR, mvSW, Mardia, HZ, Fisher,
- optionally allow PCA pre-reduction via a flag for faster but approximate runs,
- print p-values and decisions similar in style to runme.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from funfuncs.statistics import adapt_thres_cov, getp, mv_shapiro_test_adapt_thres_mod
from funfuncs.statistics.nn_runme import _fisher_p, _henze_zirkler_p, _mardia_pvalues


def _load_dataset(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    # Torch tensor (.pt) – as in DS_example
    if ext == ".pt":
        try:
            import torch  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - import-time
            raise RuntimeError(
                "This script needs torch to load .pt files. "
                "Install with `pip install torch` inside the venv."
            ) from exc

        t = torch.load(path)
        if hasattr(t, "detach"):
            t = t.detach()
        return t.cpu().numpy().astype(float)

    # Numpy arrays
    if ext in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # Take first array in .npz if multiple
            key = list(arr.keys())[0]
            arr = arr[key]
        return np.asarray(arr, dtype=float)

    # Fallback: CSV / text
    return np.loadtxt(path, delimiter=",", dtype=float)


def _p_decision(p: float, alpha: float) -> str:
    return "Fail to reject normality" if p >= alpha else "Reject normality"


def analyze_matrix(
    x: np.ndarray,
    *,
    alpha: float = 0.05,
    bb: int = 500,
    pca_components: int | None = None,
) -> None:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Dataset must be 2D: (samples, variables).")

    m, d = x.shape

    # Optional PCA pre-reduction. If pca_components is None we work in the
    # original feature space exactly as in the paper.
    if pca_components is None:
        use_pca = False
        k = d
        x_nn = x
    else:
        k = min(pca_components, m - 1, d)
        use_pca = d > k
        if use_pca:
            xc = x - x.mean(axis=0)
            _, _, vt = np.linalg.svd(xc, full_matrices=False)
            x_nn = xc @ vt[:k].T
        else:
            x_nn = x

    print("=" * 88)
    print(" Real-data multivariate normality analysis (Chen & Xia nearest-neighbor test)")
    print("=" * 88)
    print(f"Samples (m)          : {m}")
    print(f"Variables (d)        : {d}")
    print(f"Significance level   : alpha = {alpha}")
    print(f"Bootstrap reps (BB)  : {bb} in getp()")
    if use_pca:
        print(f"PCA pre-reduction    : d={d} -> k={k} components for NN test")
    print()

    # 1) Core paper test: NEW + eFR from getp
    print("Nearest-neighbor test (Chen & Xia, 2021)")
    if use_pca:
        print(f"  (applied on top-{k} PCA components)")
    print("-" * 88)
    res = getp(x_nn, l=1, bb=bb)
    new_p = float(np.asarray(res.yp).ravel()[0])
    efr_p = float(np.asarray(res.op).ravel()[0])
    print(f"  NEW p-value (Yp)           : {new_p:.4g}")
    print(f"    -> {_p_decision(new_p, alpha)}")
    print(f"  eFR p-value (Op)           : {efr_p:.4g}")
    print(f"    -> {_p_decision(efr_p, alpha)}")
    print()

    # 2) Covariance-based tests – run on PCA-reduced data for tractability
    print("Covariance-based multivariate normality tests")
    if use_pca:
        print(f"  (applied on top-{k} PCA components)")
    print("-" * 88)
    cov_est = adapt_thres_cov(x_nn)

    # mvSW (modified multivariate Shapiro-Wilk)
    mv_sw_res = mv_shapiro_test_adapt_thres_mod(x_nn)
    mv_sw_p = float(mv_sw_res["p_value"])

    # Mardia skewness / kurtosis
    mardia_s_p, mardia_k_p = _mardia_pvalues(x_nn, cov_est)

    # HZ
    hz_p = float(_henze_zirkler_p(x_nn, cov_est, np.random.default_rng(0)))

    # Fisher combination (whitened)
    eigvals, eigvecs = np.linalg.eigh(cov_est)
    eigvals = np.maximum(eigvals, 1e-12)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    x_white = (x_nn - x_nn.mean(axis=0)) @ inv_sqrt
    fisher_p = float(_fisher_p(x_white))

    print(f"  mvSW (modified Shapiro–Wilk) p : {mv_sw_p:.4g}")
    print(f"    -> {_p_decision(mv_sw_p, alpha)}")
    print(f"  Mardia skewness p              : {mardia_s_p:.4g}")
    print(f"  Mardia kurtosis p              : {mardia_k_p:.4g}")
    print(f"  Henze–Zirkler (HZ) p           : {hz_p:.4g}")
    print(f"  Fisher combined Shapiro p      : {fisher_p:.4g}")
    print()

    print("Interpretation guide")
    print("-" * 88)
    print("  p >= alpha  : fail to reject multivariate normality under that test.")
    print("  p <  alpha  : evidence against multivariate normality (reject).")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Chen & Xia high-dimensional normality tests on a real dataset.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to dataset file (e.g. DS_example/barlow_twins_class_0.pt)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--bb",
        type=int,
        default=500,
        help="Bootstrap repetitions BB used inside getp (default: 500)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        dest="pca_components",
        help=(
            "If set, use at most this many PCA components before NN/covariance "
            "tests (faster but no longer exactly the original high-d space). "
            "Default: no PCA, i.e. run in the original feature space."
        ),
    )
    args = parser.parse_args(argv)

    x = _load_dataset(args.path)
    analyze_matrix(x, alpha=args.alpha, bb=args.bb, pca_components=args.pca_components)


if __name__ == "__main__":
    main()

