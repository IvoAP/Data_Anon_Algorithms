"""
Local Differential Privacy — Gaussian mechanism.

Numerical QIs: perturbed with N(0, sigma^2) noise, where
  sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon.
Categorical QIs: same randomised response as Laplace (the Gaussian mechanism
  applies to the noise distribution; categorical perturbation uses RR with
  the same epsilon budget).
"""

import numpy as np
import pandas as pd


def apply_df_gauss(
    df: pd.DataFrame,
    epsilon: float,
    delta: float,
    qi_cols: list,
    numerical_cols: list,
) -> pd.DataFrame:
    rng = np.random.default_rng()
    result = df.copy()

    for col in qi_cols:
        if col in numerical_cols:
            lo, hi = float(df[col].min()), float(df[col].max())
            sensitivity = hi - lo
            if sensitivity > 0:
                sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
                noise = rng.normal(0.0, sigma, size=len(df))
                result[col] = np.clip(df[col].astype(float) + noise, lo, hi).round(2)
        else:
            vals = df[col].unique()
            n = len(vals)
            if n < 2:
                continue
            p_keep = np.exp(epsilon) / (np.exp(epsilon) + n - 1)
            keep_mask = rng.random(len(df)) < p_keep
            rand_idx = rng.integers(0, n - 1, size=len(df))

            val_to_idx = {v: i for i, v in enumerate(vals)}
            new_col = df[col].to_numpy(dtype=object).copy()
            for i, (keep, orig) in enumerate(zip(keep_mask, df[col])):
                if not keep:
                    true_idx = val_to_idx[orig]
                    shifted = rand_idx[i] if rand_idx[i] < true_idx else rand_idx[i] + 1
                    new_col[i] = vals[shifted]
            result[col] = new_col

    return result
