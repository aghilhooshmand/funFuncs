"""Example usage of the normality tester (mirrors the R workflow)."""

import numpy as np

from funfuncs.statistics import normality_tester


def mixture_distribution(n: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.concatenate(
        [
            rng.normal(loc=0, scale=1, size=half),
            rng.normal(loc=5, scale=1.5, size=n - half),
        ]
    )


if __name__ == "__main__":
    print("=== Normal data (n=500) ===")
    normal_sample = np.random.default_rng(0).normal(size=500)
    normality_tester(normal_sample)

    print("\n=== Mixture distribution (non-normal) ===")
    mixed_sample = mixture_distribution()
    normality_tester(mixed_sample)
