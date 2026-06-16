"""Python port of RCodes/funcs.r (nearest-neighbor normality test core functions)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from funfuncs.statistics.nn_funcs2 import adapt_thres_cov, matrix_sqrtm


@dataclass(frozen=True)
class XYStatResult:
    xx: int
    yy: int
    sxy1: float
    sxy2: float


@dataclass(frozen=True)
class GetPResult:
    yp: np.ndarray
    xp: np.ndarray
    op: np.ndarray
    sp: np.ndarray
    sxy1p: np.ndarray
    sxy2p: np.ndarray


def getdis(y: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (R ``getdis``)."""
    g = np.sum(y * y, axis=1)
    dis2 = g[:, None] + g[None, :] - 2.0 * (y @ y.T)
    np.maximum(dis2, 0.0, out=dis2)
    return np.sqrt(dis2)


def _rmvnorm(n: int, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(mean, cov, size=n)


def _rmvt(n: int, cov: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    d = cov.shape[0]
    g = rng.chisquare(df, size=n)
    z = rng.multivariate_normal(np.zeros(d), cov, size=n)
    return z / np.sqrt(g[:, None] / df)


def _cov_from_choice(d: int, choice: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    i_d = np.eye(d)
    if choice == "Sig1":
        s1 = i_d.copy()
        ss = i_d.copy()
    elif choice == "Sig11":
        s1 = np.diag(rng.uniform(1.0, 5.0, size=d))
        ss = matrix_sqrtm(s1)
    elif choice == "Sig12":
        s1 = np.diag(rng.uniform(1.0, 20.0, size=d))
        ss = matrix_sqrtm(s1)
    elif choice == "Sig2":
        idx = np.arange(d)
        s1 = 0.5 ** np.abs(idx[:, None] - idx[None, :])
        ss = matrix_sqrtm(s1)
    elif choice == "Sig3":
        sigma3 = i_d.copy()
        for i in range(d - 1):
            for j in range(i + 1, d):
                if rng.random() < 0.02:
                    val = rng.random()
                    sigma3[i, j] = val
                    sigma3[j, i] = val
        de = abs(float(np.linalg.eigvalsh(sigma3).min())) + 0.05
        s1 = (sigma3 + de * i_d) / (1.0 + de)
        ss = matrix_sqrtm(s1)
    else:
        raise ValueError(f"Unsupported choice: {choice}")
    return s1, ss


def gen(
    d: int = 20,
    m: int = 100,
    distr: str = "normal",
    choice: str = "Sig1",
    chisq_df: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate data (R ``gen``)."""
    rng = np.random.default_rng() if rng is None else rng
    s1, ss = _cov_from_choice(d, choice, rng)
    mu = np.zeros(d)

    if distr == "normal":
        return _rmvnorm(m, mu, s1, rng)
    if distr == "t05":
        return _rmvt(m, s1, 0.5 * d, rng)
    if distr == "t025":
        return _rmvt(m, s1, 0.25 * d, rng)
    if distr == "t2":
        return _rmvt(m, s1, 2.0 * d, rng)
    if distr == "t4":
        return _rmvt(m, s1, 4.0 * d, rng)
    if distr == "mixture":
        ss1 = 1.0 - 1.8 / np.sqrt(d)
        ss2 = 1.0 + 1.8 / np.sqrt(d)
        half = m // 2
        x = np.vstack(
            [
                _rmvnorm(half, mu, ss1 * np.eye(d), rng),
                _rmvnorm(m - half, mu, ss2 * np.eye(d), rng),
            ]
        )
        return x @ ss
    if distr in {
        "partial_normal_01",
        "partial_normal_02",
        "partial_normal_03",
        "partial_normal_04",
        "partial_normal_05",
    }:
        frac = int(distr[-2:]) / 100.0
        prop = int(frac * d)
        x = _rmvnorm(m, mu, s1, rng)
        x[:, :prop] = _rmvt(m, np.eye(prop), d / 4.0, rng)
        return x
    if distr.startswith("chisquare"):
        if distr == "chisquare3":
            chisq_df = 3
        elif distr == "chisquare5":
            chisq_df = 5
        elif distr == "chisquare10":
            chisq_df = 10
        elif distr == "chisquare20":
            chisq_df = 20
        x0 = rng.chisquare(chisq_df, size=(m, d)) - chisq_df
        return (x0 @ ss) / np.sqrt(chisq_df * 2.0)

    raise ValueError(f"Unsupported distribution: {distr}")


def get_xy_stat(dist_m: np.ndarray) -> XYStatResult:
    """Nearest-neighbor XX/YY statistics (R ``get.XYstat``)."""
    n = dist_m.shape[0]
    m = n // 2
    nn = np.argmin(dist_m, axis=1)

    xx = int(np.sum(nn[:m] < m))
    yy = int(np.sum(nn[m:] >= m))

    counts = np.bincount(nn, minlength=n)
    share = 0.0
    a = int(counts.max()) if counts.size else 0
    if a > 1:
        for i in range(2, a + 1):
            share += comb(i, 2) * int(np.sum(counts == i))

    mutual = int(np.sum(nn[nn] == np.arange(n)))

    mu_xx = m * (m - 1) / (n - 1)
    v_xx = (
        m**2
        * (m - 1) ** 2
        / (n * (n - 1) * (n - 2) * (n - 3))
        * (n + mutual + (m - 2) / (m - 1) * share * 2 - 2 * n / (n - 1))
    )
    cov_xy = (
        m**2
        * (m - 1) ** 2
        / (n * (n - 1) * (n - 2) * (n - 3))
        * (mutual - 3 * n - 2 * share + (4 * n - 6) * n / (n - 1))
    )
    sigma = np.array([[v_xx, cov_xy], [cov_xy, v_xx]], dtype=float)

    r_vec = np.array([xx - mu_xx, yy - mu_xx], dtype=float)
    sxy1 = float(np.sum(np.abs(r_vec)))
    sxy2 = float(r_vec @ np.linalg.pinv(sigma) @ r_vec)
    return XYStatResult(xx=xx, yy=yy, sxy1=sxy1, sxy2=sxy2)


def mstree(dist_m: np.ndarray) -> np.ndarray:
    """Minimum spanning tree edges (R ``ade4::mstree``), shape (n-1, 2)."""
    mst = minimum_spanning_tree(dist_m).tocoo()
    edges = np.column_stack([mst.row, mst.col])
    edges = edges[edges[:, 0] < edges[:, 1]]
    return edges.astype(int)


def get_r(edges: np.ndarray, ids: np.ndarray) -> int:
    """Count MST edges with exactly one endpoint in ``ids`` (R ``getR``)."""
    id_set = set(int(i) for i in np.asarray(ids).ravel())
    count = 0
    for e1, e2 in edges:
        e1_in = int(e1) in id_set
        e2_in = int(e2) in id_set
        if e1_in ^ e2_in:
            count += 1
    return count


def g_tests_generalized(edges: np.ndarray, id1: np.ndarray, id2: np.ndarray) -> float:
    """
    Generalized edge-count statistic (R ``gTests::g.tests(..., 'g')``).

    Returns the number of MST edges connecting the two groups.
    """
    s1 = set(int(i) for i in np.asarray(id1).ravel())
    s2 = set(int(i) for i in np.asarray(id2).ravel())
    count = 0
    for e1, e2 in edges:
        in1 = int(e1) in s1 or int(e2) in s1
        in2 = int(e1) in s2 or int(e2) in s2
        if in1 and in2:
            count += 1
    return float(count)


def _make_pd(cov: np.ndarray) -> np.ndarray:
    d = cov.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    de = max(-float(eigvals.min()), 0.0) + 0.05
    return (cov + de * np.eye(d)) / (1.0 + de)


def _threshold_cov_from_sample(
    sample: np.ndarray,
    reference_cov: np.ndarray | None = None,
    *,
    kk: int = 5,
    cx_generate: bool = False,
) -> np.ndarray:
    """Covariance with fold-specific thresholding used inside ``getp`` Step 2."""
    m, d = sample.shape
    if reference_cov is None:
        return adapt_thres_cov(sample, kk=kk)

    yycov = np.cov(sample, rowvar=False, ddof=1)
    cc1 = (sample * sample).T @ (sample * sample) / m
    cc2 = ((sample.T @ sample) / m) * yycov
    cc3 = yycov * yycov
    t1 = cc1 - 2 * cc2 + cc3
    aa = np.sqrt(np.divide(m * (yycov * yycov), t1, out=np.zeros_like(t1), where=t1 > 0))

    lambdas = np.array([6 * j / 50 * np.sqrt(np.log(max(d, 2))) for j in range(1, 51)])
    dif = np.array([np.linalg.norm(yycov * (np.abs(aa) >= lam) - reference_cov, ord="fro") for lam in lambdas])
    thr = float(lambdas[dif == dif.min()].max())
    return _make_pd(yycov * (np.abs(aa) >= thr))


def getp(
    x: np.ndarray,
    l: int,
    bb: int,
    kk: int = 5,
    cx_generate: bool = False,
    rng: np.random.Generator | None = None,
) -> GetPResult:
    """
    Nearest-neighbor normality test p-values (R ``getp``).

  Implements Algorithm 1/2 from Chen & Xia (2021):
    - ``Yp``: NEW test based on r(Y Y)
    - ``Op``: extended Friedman-Rafsky style statistic
    - ``Xp``, ``Sp``, ``SXY1p``, ``SXY2p``: related diagnostics
    """
    rng = np.random.default_rng() if rng is None else rng
    m, d = x.shape
    xbar = x.mean(axis=0)
    xcov = adapt_thres_cov(x, kk=kk)

    ratio0v = np.zeros(l)
    xratio0v = np.zeros(l)
    sxy1_0v = np.zeros(l)
    sxy2_0v = np.zeros(l)
    ori0v = np.zeros(l)
    s0v = np.zeros(l)

    id1 = np.arange(m)
    id2 = np.arange(m, 2 * m)

    for li in range(l):
        y = _rmvnorm(m, xbar, xcov, rng)
        xy = np.vstack([x, y])
        mydist = getdis(xy)
        mydist2 = mydist.copy()
        np.fill_diagonal(mydist2, mydist.max() + 100.0)

        temp = get_xy_stat(mydist2)
        ratio0v[li] = temp.yy / m
        xratio0v[li] = temp.xx / m
        sxy1_0v[li] = temp.sxy1
        sxy2_0v[li] = temp.sxy2

        edges = mstree(mydist)
        ori0v[li] = get_r(edges, id1)
        s0v[li] = g_tests_generalized(edges, id1, id2)

    ratio0 = float(ratio0v.mean())
    xratio0 = float(xratio0v.mean())
    sxy1_0 = float(sxy1_0v.mean())
    sxy2_0 = float(sxy2_0v.mean())
    ori0 = float(ori0v.mean())
    s0 = float(s0v.mean())

    ratio = np.zeros((bb, l))
    xratio = np.zeros((bb, l))
    sxy1 = np.zeros((bb, l))
    sxy2 = np.zeros((bb, l))
    ori = np.zeros((bb, l))
    s = np.zeros((bb, l))

    if cx_generate:
        cx = _rmvnorm(m * bb, xbar, xcov, rng)
        cxcov = np.cov(cx, rowvar=False, ddof=1)
    else:
        cxcov = xcov

    for j in range(bb):
        yy = _rmvnorm(m, xbar, xcov, rng)
        yybar = yy.mean(axis=0)
        yycov = _threshold_cov_from_sample(yy, cxcov, kk=kk, cx_generate=cx_generate)

        for li in range(l):
            z = _rmvnorm(m, yybar, yycov, rng)
            yz = np.vstack([yy, z])
            mydist_yz = getdis(yz)
            mydist2_yz = mydist_yz.copy()
            np.fill_diagonal(mydist2_yz, mydist_yz.max() + 100.0)

            temp = get_xy_stat(mydist2_yz)
            ratio[j, li] = temp.yy / m
            xratio[j, li] = temp.xx / m
            sxy1[j, li] = temp.sxy1
            sxy2[j, li] = temp.sxy2

            edges = mstree(mydist_yz)
            ori[j, li] = get_r(edges, id1)
            s[j, li] = g_tests_generalized(edges, id1, id2)

    ratio_l = ratio.mean(axis=1)
    xratio_l = xratio.mean(axis=1)
    sxy1_l = sxy1.mean(axis=1)
    sxy2_l = sxy2.mean(axis=1)
    ori_l = ori.mean(axis=1)
    s_l = s.mean(axis=1)

    def _two_sided_p(obs: float, null_dist: np.ndarray, obs_first: float, null_first: np.ndarray) -> np.ndarray:
        mean_null = null_dist.mean()
        p1 = np.mean(np.abs(null_dist - mean_null) >= abs(obs - mean_null))
        p2 = np.mean(np.abs(null_first - null_first.mean()) >= abs(obs_first - null_first.mean()))
        return np.array([p1, p2])

    yp = _two_sided_p(ratio0, ratio_l, ratio0v[0], ratio[:, 0])
    xp = _two_sided_p(xratio0, xratio_l, xratio0v[0], xratio[:, 0])
    op = _two_sided_p(ori0, ori_l, ori0v[0], ori[:, 0])
    sp = np.array([np.mean(s_l >= s0), np.mean(s[:, 0] >= s0v[0])])
    sxy1p = np.array([np.mean(sxy1_l >= sxy1_0), np.mean(sxy1[:, 0] >= sxy1_0v[0])])
    sxy2p = np.array([np.mean(sxy2_l >= sxy2_0), np.mean(sxy2[:, 0] >= sxy2_0v[0])])

    return GetPResult(yp=yp, xp=xp, op=op, sp=sp, sxy1p=sxy1p, sxy2p=sxy2p)


def getpow(pm: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Rejection rate / power from p-value matrix (R ``getpow``)."""
    pm = np.asarray(pm, dtype=float)
    if pm.ndim == 1:
        pm = pm.reshape(-1, 1)
    b, l = pm.shape
    return np.array([np.mean(pm[:, i] < alpha) for i in range(l)])
