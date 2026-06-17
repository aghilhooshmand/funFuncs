# Guide — FunFuncs: Nearest-Neighbor Normality Test (Chen & Xia, 2021)

This guide explains the Python implementation of the nearest-neighbor normality test
from the paper:

> **"A Normality Test for High-dimensional Data based on a Nearest Neighbor Approach"**
> Chen & Xia, 2021.

It covers what each file does, how the R code was converted to Python, how to run
everything, and how to interpret the results.

---

## 1. Background — Why This Test?

Classical normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, etc.) are designed for
**univariate** data. When the number of features `d` is large (e.g. `d >> n`), they
either cannot be applied or have low power.

Chen & Xia (2021) propose a test that compares a real dataset `X` against a
multivariate normal sample drawn from the same estimated distribution, using
**nearest-neighbor statistics** on the combined pool. The key idea is:

- If `X` is multivariate normal, a simulated normal sample `Y` from the same
  parameters should be indistinguishable from `X`.
- The test statistic measures how often each point's nearest neighbor comes from the
  *same* group (X→X or Y→Y) vs. the other group.

---

## 2. Project Structure

```
FunFuncs/
│
├── RCodes/                        ← Original R scripts from the paper authors
│   ├── funcs.r                    ← Core algorithm (distances, statistics, getp)
│   ├── funcs2.R                   ← Adaptive covariance + modified mvShapiro
│   └── runme.R                    ← Simulation study (power/size tables)
│
├── funfuncs/statistics/           ← Python ports (one-to-one with R files)
│   ├── nn_funcs.py                ← Port of funcs.r
│   ├── nn_funcs2.py               ← Port of funcs2.R
│   ├── nn_runme.py                ← Port of runme.R
│   ├── normality.py               ← Univariate normality helper
│   └── multivariate_normality.py  ← High-dim multivariate normality helper
│
├── examples/
│   ├── normality_example.py       ← Univariate normality demo
│   ├── multivariate_normality_example.py ← Multivariate demo
│   └── analyze_dataset.py         ← Analyze a real dataset (.pt / .npy / .csv)
│
├── tests/                         ← Automated tests (pytest)
├── DS_example/                    ← Example real datasets (PyTorch .pt files)
├── requirements.txt
└── pyproject.toml
```

---

## 3. R → Python Conversion

### 3.1 `RCodes/funcs.r` → `funfuncs/statistics/nn_funcs.py`

This file contains the **core algorithm**.

| R function | Python function | What it does |
|---|---|---|
| `getdis(y)` | `getdis(y)` | Euclidean pairwise distance matrix for an (n×d) array |
| `gen(d, m, distr, choice)` | `gen(d, m, distr, choice, rng)` | Simulate a test dataset with various distributions and covariance structures |
| `get.XYstat(dist_m)` | `get_xy_stat(dist_m)` | Count how many nearest-neighbors fall within the same group (XX, YY) and compute cross-group statistics |
| `mstree(d)` | `mstree(d)` | Minimum spanning tree on a distance matrix (uses `scipy.sparse.csgraph`) |
| `get.r(edges, id1)` | `get_r(edges, id1)` | Count MST edges within a group |
| `g.tests(edges, id1, id2)` | `g_tests_generalized(edges, id1, id2)` | Generalized edge-count statistic (cross-group MST edges) |
| `getp(x, l, bb)` | `getp(x, l, bb, rng)` | **Main test function.** Returns sampling p-values for the NEW test and related statistics |
| `getpow(pm, alpha)` | `getpow(pm, alpha)` | Compute rejection rates from a matrix of p-values |

**Key Python changes from R:**

- `np.random.default_rng(seed)` replaces R's `set.seed()` — pass `rng` explicitly for reproducibility.
- `scipy.sparse.csgraph.minimum_spanning_tree` replaces R's `ape::mst`.
- Loop-based `getdis` replaced by a vectorised `(g[:,None] + g[None,:] - 2 X@X.T)` for speed.

---

### 3.2 `RCodes/funcs2.R` → `funfuncs/statistics/nn_funcs2.py`

This file provides the **covariance estimation and modified Shapiro-Wilk** used
to whiten the data before the nearest-neighbor test.

| R function | Python function | What it does |
|---|---|---|
| `adapt.thres.cov(x, kk)` | `adapt_thres_cov(x, kk)` | Adaptive-threshold covariance estimator (Cai & Liu, 2011). Cross-validated over 50 thresholds using 5-fold CV. Produces a positive-definite sparse covariance matrix. |
| `mvShapiro.Test.adapt.thres.mod(x)` | `mv_shapiro_test_adapt_thres_mod(x)` | Modified multivariate Shapiro-Wilk: whitens `x` with the adaptive covariance, then averages univariate Shapiro-Wilk W-statistics and converts to a p-value. |

**Key Python changes from R:**

- `np.linalg.eigh` + manual inverse square root replaces R's `expm::sqrtm` for whitening.
- `scipy.linalg.sqrtm` is available as a fallback via `matrix_sqrtm`.
- Division-by-zero in the threshold formula handled by `np.divide(..., where=t1 > 0)`.

---

### 3.3 `RCodes/runme.R` → `funfuncs/statistics/nn_runme.py`

This file is the **simulation study** — it repeatedly generates data, runs all tests,
and reports rejection rates (power tables).

| R block | Python equivalent | What it does |
|---|---|---|
| Outer `for` loop over B replications | `run_simulation(...)` | Generates `B` datasets, runs every test, collects p-values |
| Mardia skewness/kurtosis | `_mardia_pvalues(x, cov)` | Tests for multivariate skewness and excess kurtosis |
| Henze-Zirkler test | `_henze_zirkler_p(x, cov, rng)` | Bootstrap approximation of the HZ statistic |
| Royston combined Shapiro | `_royston_p(x_white)` | Whitened marginal Shapiro-Wilk combined via Z-score |
| Doornik-Hansen omnibus | `_ep_p(x, cov)` | Omnibus skewness+kurtosis combination |
| Fisher combined Shapiro | `_fisher_p(x_white)` | Fisher's chi-squared combination of whitened Shapiro p-values |
| Print table rows | `print_replication_row(...)` | One row of the simulation table per replication |
| Final summary | `print_runme_summary(result)` | Rejection rates and interpretation per test |

**Key Python changes from R:**

- R's `MVN` package (Mardia, HZ, Royston) is not directly available in Python, so
  these are implemented from their published formulas.
- `run_simulation` returns a `RunmeResult` dataclass so results can be accessed
  programmatically (not just printed).
- R's `gTests` package edge-count function is re-implemented as `g_tests_generalized`.

---

## 4. Setup

```bash
# 1. Clone the repo
git clone https://github.com/aghilhooshmand/funFuncs.git
cd FunFuncs

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Install the package and its dependencies
pip install --upgrade pip
pip install -e .

# For running tests:
pip install -e ".[dev]"

# For loading .pt (PyTorch) dataset files:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Dependencies installed automatically:**

| Library | Used for |
|---|---|
| `numpy` | Arrays, linear algebra, random number generation |
| `scipy` | Shapiro-Wilk, chi-squared CDF, MST, matrix square root |
| `statsmodels` | Anderson-Darling test (univariate) |
| `matplotlib` | Diagnostic plots (univariate) |

---

## 5. How to Run

### 5.1 Simulate data and run the paper's test (like `runme.R`)

```python
import numpy as np
from funfuncs.statistics import gen, getp

rng = np.random.default_rng(42)

# Simulate a 100-sample, 50-dimensional dataset from a multivariate normal
X = gen(d=50, m=100, distr="normal", choice="Sig3", rng=rng)

# Run the nearest-neighbor test (BB=200 bootstrap reps)
result = getp(X, l=1, bb=200, rng=rng)

print("NEW test p-values (Yp) :", result.yp)   # two-sided p-value pair
print("eFR test p-values (Op) :", result.op)
```

### 5.2 Full simulation study (power / size table)

```python
from funfuncs.statistics.nn_runme import run_simulation

study = run_simulation(
    d=100,       # feature dimension
    m=100,       # sample size
    distr="t4",  # alternative distribution (non-normal)
    choice="Sig3",
    b=100,       # number of replications (use 1000 for paper-quality results)
    bb=200,      # bootstrap reps inside getp
    l=1,
    alpha=0.05,
    random_state=42,
    print_progress=True,  # prints a table row for each replication
)

# Access rejection rates numerically
s = study.summary()
print("NEW rejection rate      :", s["NEW"])
print("eFR rejection rate      :", s["eFR"])
print("Henze-Zirkler rate      :", s["HZ"])
```

### 5.3 Analyze a real dataset (your `.pt` files)

```bash
# From the project root, with the venv active:
python -m examples.analyze_dataset \
  --path DS_example/clip_class_0.pt \
  --alpha 0.05 \
  --bb 200
```

For very large feature dimensions (e.g. `d=2048`) you can optionally use PCA
pre-reduction (this is *not* part of the paper algorithm, but makes it tractable):

```bash
python -m examples.analyze_dataset \
  --path DS_example/barlow_twins_class_0.pt \
  --alpha 0.05 \
  --bb 200 \
  --pca-components 50
```

### 5.4 Univariate normality (existing module)

```python
import numpy as np
from funfuncs.statistics import normality_tester

data = np.random.default_rng(0).normal(size=300)
result = normality_tester(data, plot=True, print_summary=True)
print(result.is_normal)
```

---

## 6. Available Distributions (`distr`) and Covariance Structures (`choice`)

### Distributions for `gen()` / `run_simulation()`

| `distr` value | Type | Notes |
|---|---|---|
| `"normal"` | Null hypothesis (H₀) | Multivariate Gaussian |
| `"t05"` | Alternative | Multivariate t, df = 0.5d (heavy tails) |
| `"t025"` | Alternative | Multivariate t, df = 0.25d (very heavy tails) |
| `"t2"` | Alternative | Multivariate t, df = 2d |
| `"t4"` | Alternative | Multivariate t, df = 4d (mild tails) |
| `"mixture"` | Alternative | Mixture of two Gaussians with different scales |
| `"chisquare3"` | Alternative | Centered chi-squared, df=3 (skewed) |
| `"chisquare5"` | Alternative | Centered chi-squared, df=5 |
| `"chisquare10"` | Alternative | Centered chi-squared, df=10 |
| `"chisquare20"` | Alternative | Centered chi-squared, df=20 |

> **Rule of thumb:** Use `"normal"` to verify *size* (should reject ~5% of the time
> at `alpha=0.05`). Use the alternatives to measure *power* (should reject more often).

### Covariance structures (`choice`)

| `choice` value | Description |
|---|---|
| `"Sig1"` | Identity matrix I_d |
| `"Sig11"` | Diagonal with uniform(1,5) variances |
| `"Sig12"` | Diagonal with uniform(1,20) variances |
| `"Sig2"` | AR(1) structure: `Σ_{ij} = 0.5^|i-j|` |
| `"Sig3"` | Sparse random off-diagonal entries (~2% non-zero) |

---

## 7. Understanding the Output

### From `getp` / `analyze_dataset.py`

Each result is a **pair of p-values** `[p1, p2]` — two estimators of the same quantity
(averaged over L repeats, and a single-L version). Focus on `p1`.

| Output field | What it means |
|---|---|
| `Yp` (NEW) | P-value for the paper's main NEW test. High p → data looks normal. |
| `Op` (eFR) | Extended Friedman-Rafsky test p-value. Based on MST cross-group edges. |
| `Xp` | Same idea as NEW but for XX nearest-neighbor counts. |
| `Sp` | P-value from generalized g-test edge statistic. |

**Decision rule (all tests):**

```
p-value >= alpha (0.05)  →  Fail to reject normality  (data is consistent with MV normal)
p-value <  alpha (0.05)  →  Reject normality           (evidence against MV normality)
```

### From `run_simulation` (simulation table)

Each row is one replication. The final summary prints **rejection rates** per test:

```
Under H0 (distr = "normal"):
  Rejection rate ≈ alpha        → good size control  (test is well-calibrated)
  Rejection rate >> alpha       → inflated type I error

Under alternatives (distr ≠ "normal"):
  Rejection rate → 1.0          → high power (test detects non-normality well)
  Rejection rate ≈ alpha        → low power  (test struggles with this alternative)
```

### Tests included in the simulation

| Column name | Test | Key property |
|---|---|---|
| `NEW` | Paper's nearest-neighbor test (`Yp`) | Main contribution of Chen & Xia (2021) |
| `NEW(Σ^-1/2)` | NEW after full whitening | Uses full covariance structure |
| `NEW(D^-1/2)` | NEW after diagonal whitening | Faster; uses only variances |
| `eFR` | Extended Friedman-Rafsky (`Op`) | MST-based; classic graph test |
| `mvSW` | Modified multivariate Shapiro-Wilk | Based on adaptive covariance + whitening |
| `Fisher` | Fisher combined Shapiro-Wilk | Combines marginal Shapiro p-values |
| `Mardia S` | Mardia skewness test | Chi-squared statistic on 3rd moments |
| `Mardia K` | Mardia kurtosis test | Normal Z-statistic on 4th moments |
| `HZ` | Henze-Zirkler test | Characteristic function distance |

---

## 8. Running the Tests (pytest)

```bash
pytest tests/
```

Key test files:

| File | What it tests |
|---|---|
| `tests/test_nn_rcodes.py` | `getdis`, `getp`, `adapt_thres_cov`, `mv_shapiro_test_adapt_thres_mod`, `run_simulation` smoke test |
| `tests/test_normality.py` | Univariate normality tester |
| `tests/test_multivariate_normality.py` | `multivariate_normality_check_high_dim`, power study |

---

## 9. Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Script runs for >10 minutes | Dataset too large for full-dimension run | Use `--pca-components 50` for a fast approximate run |
| `ValueError: Sample size must be between 12 and 5000` | `mv_shapiro_test_adapt_thres_mod` constraint | Use at least 12 and at most 5000 samples |
| p-values look unstable between runs | `bb` is too small | Increase `--bb` (200–500 for reliable results) |
| `adapt_thres_cov` is very slow | High-dimensional data (d ≫ 100) | Either reduce d with PCA, or reduce n to speed up the cross-validated CV loop |

---

## 10. Example: Reproduce Paper-Style Power Table

The following reproduces a single row from Table 1 of Chen & Xia (2021) — the size
experiment under the null (data is normal):

```python
from funfuncs.statistics.nn_runme import run_simulation

result = run_simulation(
    d=100,
    m=100,
    distr="normal",   # null: data IS multivariate normal
    choice="Sig3",
    b=1000,           # 1000 replications → stable rejection rates
    bb=500,           # 500 bootstrap reps inside getp
    l=1,
    alpha=0.05,
    random_state=0,
    print_progress=True,
)
```

Expected outcome: all rejection rates close to 0.05 (good size control).

For the power experiment, change `distr` to `"t4"` (mild heavy tails) or
`"mixture"` (bimodal) and observe that rejection rates for the NEW test climb
towards 1.0.


