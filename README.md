# FunFuncs

Reusable Python modules for work you repeat across projects: **statistics**, data preprocessing, analysis helpers, and more. Each topic lives in its own subpackage so you can copy the repo, install it once, and import only what you need.

## Project layout

```
FunFuncs/
├── funfuncs/
│   ├── statistics/                  # Statistical utilities
│   │   ├── normality.py             # Univariate normality tester
│   │   └── multivariate_normality.py # High-dimensional MV normality
│   └── ...                  # Future: preprocessing, analysis, ...
├── examples/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

```bash
cd "/home/aghil/Documents/my document/limerick/projects/FunFuncs"
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies and the package

```bash
pip install --upgrade pip
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Module: statistics — univariate normality

`normality_tester` automates normality checks so you do not rely on a single test. It mirrors the R workflow described in your reference implementation.

### What it runs

| Check | Role |
|--------|------|
| **Shapiro–Wilk** | Strong default for moderate sample sizes (best for *n* &lt; 5000) |
| **Anderson–Darling** | Sensitive to tails |
| **Kolmogorov–Smirnov** | Empirical vs fitted normal CDF |
| **Jarque–Bera** | Skewness and kurtosis jointly |
| **Skewness / kurtosis** | Shape diagnostics (excess kurtosis) |

### Criteria for “approximately normal”

- All test *p*-values ≥ 0.05 (fail to reject normality),
- Skewness ∈ (−0.5, 0.5),
- Excess kurtosis ∈ (−2, 2).

### Usage

```python
import numpy as np
from funfuncs.statistics import normality_tester

# Standard normal sample
normality_tester(np.random.default_rng(0).normal(size=500))

# Non-normal mixture
rng = np.random.default_rng(42)
mixed = np.concatenate([
    rng.normal(size=250),
    rng.normal(loc=5, scale=1.5, size=250),
])
result = normality_tester(mixed, plot=True, print_summary=True)

# Access numeric results (R-style list)
print(result.as_dict())
print(result.shapiro_wilk_p, result.is_normal)
```

Run the bundled example:

```bash
python examples/normality_example.py
```

## Module: statistics — multivariate normality (high-dimensional)

`multivariate_normality_check_high_dim` is a Python port inspired by your R scripts in `RCodes/` for high-dimensional data.

### What it runs

- Adaptive-threshold covariance estimate (Cai-Liu style thresholding)
- Modified generalized Shapiro-Wilk (multivariate, via whitened marginals)
- Mardia skewness and kurtosis diagnostics
- Henze-Zirkler test (bootstrap p-value approximation)
- Fisher combined p-value from marginal Shapiro tests

### Usage

```python
import numpy as np
from funfuncs.statistics import multivariate_normality_check_high_dim

rng = np.random.default_rng(0)
X = rng.multivariate_normal(np.zeros(30), np.eye(30), size=150)
result = multivariate_normality_check_high_dim(X, alpha=0.05)

print(result.is_normal)
print(result.as_dict())
```

Run the bundled multivariate example:

```bash
python examples/multivariate_normality_example.py
```

### Power study loop (R `runme.R` style)

You can run a simulation loop with `B`, `BB`, and `L` to estimate rejection rates (power/size):

```python
from funfuncs.statistics import multivariate_power_study_high_dim

study = multivariate_power_study_high_dim(
    d=100,
    m=100,
    distr="normal",   # or mixture, t05, t025, t2, t4, chisquare3/5/10/20
    choice="Sig3",    # Sig1, Sig11, Sig12, Sig2, Sig3
    B=200,            # outer replications
    BB=150,           # inner bootstrap reps
    L=1,              # repeated p-value estimates per replication
    alpha=0.05,
    random_state=42,
)
print(study.as_dict())
```

### Univariate options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.05` | Significance level for tests |
| `skew_bounds` | `(-0.5, 0.5)` | Acceptable skewness (open interval) |
| `kurt_bounds` | `(-2, 2)` | Acceptable excess kurtosis |
| `plot` | `True` | Histogram, boxplot, Q–Q plot |
| `print_summary` | `True` | Text report to stdout |

### Multivariate options

| Function | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| `multivariate_normality_check_high_dim` | `alpha` | `0.05` | Significance level for decision labels |
| `multivariate_normality_check_high_dim` | `hz_bootstrap_reps` | `150` | Bootstrap replications used for HZ p-value |
| `multivariate_normality_check_high_dim` | `random_state` | `None` | Random seed for reproducible HZ bootstrap |
| `multivariate_normality_check_high_dim` | `print_summary` | `True` | Print compact text summary |
| `multivariate_power_study_high_dim` | `d` | `100` | Feature dimension |
| `multivariate_power_study_high_dim` | `m` | `100` | Sample size per replication |
| `multivariate_power_study_high_dim` | `distr` | `"normal"` | Data generator (`normal`, `mixture`, `t05`, `t025`, `t2`, `t4`, `chisquare3/5/10/20`) |
| `multivariate_power_study_high_dim` | `choice` | `"Sig3"` | Covariance structure (`Sig1`, `Sig11`, `Sig12`, `Sig2`, `Sig3`) |
| `multivariate_power_study_high_dim` | `B` | `200` | Number of outer simulation replications |
| `multivariate_power_study_high_dim` | `BB` | `150` | Inner bootstrap reps (for HZ calibration) |
| `multivariate_power_study_high_dim` | `L` | `1` | Repeated p-value estimates per replication |
| `multivariate_power_study_high_dim` | `alpha` | `0.05` | Rejection threshold for power/size |
| `multivariate_power_study_high_dim` | `random_state` | `None` | Seed for reproducible simulation |
| `multivariate_power_study_high_dim` | `print_progress` | `False` | Print progress while running long studies |

## Tests

```bash
pytest
```

## Planned modules

- **Data preprocessing** — cleaning, encoding, scaling
- **Analysis** — exploratory and reporting helpers
- Additional domains as needed for your other projects

## Dependencies

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) — Shapiro, KS, Jarque–Bera, skew, kurtosis
- [statsmodels](https://www.statsmodels.org/) — Anderson–Darling (*normal_ad*)
- [Matplotlib](https://matplotlib.org/) — diagnostic plots

## Publish to GitHub

### 1. Create a new empty repository on GitHub

Create a repo (for example: `FunFuncs`) from [GitHub New Repository](https://github.com/new). Do not add README/license/gitignore there, since this project already has files.

### 2. Initialize git locally (first time only)

```bash
cd "/home/aghil/Documents/my document/limerick/projects/FunFuncs"
git init
git add .
git commit -m "Initial commit: add FunFuncs statistics modules"
```

### 3. Link local repo to GitHub and push

Replace `<your-username>` with your GitHub username:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/FunFuncs.git
git push -u origin main
```

If your remote already exists, use:

```bash
git remote set-url origin https://github.com/<your-username>/FunFuncs.git
git push -u origin main
```

### 4. Next updates

For future changes:

```bash
git add .
git commit -m "your message"
git push
```

## License

Add a license file if you plan to share or publish this repository.
