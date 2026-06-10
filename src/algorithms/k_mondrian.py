"""
k-Mondrian multidimensional k-anonymity.

Recursively partitions the dataset into equivalence classes each containing
at least k records, choosing at each step the QI column with the largest
normalised span.
"""

import pandas as pd
from typing import List


def _normalized_spans(
    df_global: pd.DataFrame,
    partition: pd.DataFrame,
    qi_cols: list,
    numerical_cols: list,
) -> dict:
    spans = {}
    for col in qi_cols:
        if col in numerical_cols:
            global_range = df_global[col].max() - df_global[col].min()
            spans[col] = (partition[col].max() - partition[col].min()) / global_range if global_range else 0.0
        else:
            spans[col] = partition[col].nunique() / max(df_global[col].nunique(), 1)
    return spans


def _split(partition: pd.DataFrame, col: str, numerical_cols: list):
    if col in numerical_cols:
        median = partition[col].median()
        return partition[partition[col] <= median], partition[partition[col] > median]
    vals = sorted(partition[col].unique())
    mid = len(vals) // 2
    return (
        partition[partition[col].isin(vals[:mid])],
        partition[partition[col].isin(vals[mid:])],
    )


def _mondrian_partition(
    df_global: pd.DataFrame,
    partition: pd.DataFrame,
    k: int,
    qi_cols: list,
    numerical_cols: list,
) -> List[pd.DataFrame]:
    if len(partition) < 2 * k:
        return [partition]

    spans = _normalized_spans(df_global, partition, qi_cols, numerical_cols)
    best_col = max(spans, key=spans.get)

    if spans[best_col] == 0:
        return [partition]

    left, right = _split(partition, best_col, numerical_cols)
    if len(left) < k or len(right) < k:
        return [partition]

    return (
        _mondrian_partition(df_global, left, k, qi_cols, numerical_cols)
        + _mondrian_partition(df_global, right, k, qi_cols, numerical_cols)
    )


def _generalize(partition: pd.DataFrame, qi_cols: list, numerical_cols: list) -> pd.DataFrame:
    result = partition.copy()
    for col in qi_cols:
        if col in numerical_cols:
            lo, hi = partition[col].min(), partition[col].max()
            lo_s = str(int(lo)) if lo == int(lo) else str(round(lo, 2))
            hi_s = str(int(hi)) if hi == int(hi) else str(round(hi, 2))
            result[col] = f"{lo_s}-{hi_s}" if lo != hi else lo_s
        else:
            vals = sorted(partition[col].unique())
            result[col] = vals[0] if len(vals) == 1 else "|".join(str(v) for v in vals)
    return result


def apply_k_mondrian(
    df: pd.DataFrame,
    k: int,
    qi_cols: list,
    numerical_cols: list,
) -> pd.DataFrame:
    partitions = _mondrian_partition(df, df, k, qi_cols, numerical_cols)
    return pd.concat(
        [_generalize(p, qi_cols, numerical_cols) for p in partitions],
        ignore_index=True,
    )
