"""Example usage for high-dimensional multivariate normality checks."""

import numpy as np

from funfuncs.statistics import multivariate_normality_check_high_dim


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    print("=== Multivariate normal sample ===")
    x_normal = rng.multivariate_normal(np.zeros(20), np.eye(20), size=150)
    normal_result = multivariate_normality_check_high_dim(x_normal)
    print(normal_result.as_dict())

    print("\n=== Multivariate non-normal sample (mixture) ===")
    x_mix = np.vstack(
        [
            rng.multivariate_normal(np.zeros(20), 0.7 * np.eye(20), size=75),
            rng.multivariate_normal(0.8 * np.ones(20), 1.4 * np.eye(20), size=75),
        ]
    )
    mix_result = multivariate_normality_check_high_dim(x_mix)
    print(mix_result.as_dict())
