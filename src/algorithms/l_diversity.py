"""
l-Diversity 

Builds on k-Mondrian partitions and enforces that each equivalence class
contains at least l distinct values of the sensitive attribute.
Classes that fail the criterion are merged together into a single class.
"""

import pandas as pd
from .k_mondrian import _mondrian_partition, _generalize


def _satisfies_l(partition: pd.DataFrame, sensitive_col: str, l: int) -> bool:
    return partition[sensitive_col].nunique() >= l


def _enforce_l_diversity(
    partitions: list,
    l: int,
    k: int,
    sensitive_col: str,
) -> list:
    good, bad = [], []
    for p in partitions:
        (good if _satisfies_l(p, sensitive_col, l) else bad).append(p)

    if not bad:
        return good

    merged = pd.concat(bad, ignore_index=True)
    good.append(merged)
    return good


def apply_l_diversity(
    df: pd.DataFrame,
    k: int,
    l: int,
    qi_cols: list,
    numerical_cols: list,
    sensitive_col: str,
) -> pd.DataFrame:
    partitions = _mondrian_partition(df, df, k, qi_cols, numerical_cols)
    partitions = _enforce_l_diversity(partitions, l, k, sensitive_col)
    return pd.concat(
        [_generalize(p, qi_cols, numerical_cols) for p in partitions],
        ignore_index=True,
    )
