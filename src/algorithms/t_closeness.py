"""
t-Closeness

Extends k-Mondrian by requiring that the distribution of the sensitive
attribute within each equivalence class is no more than t away from the
overall distribution.

For categorical sensitive attributes: variation distance (half L1).
For numerical sensitive attributes: Earth Mover's Distance (L1 on CDFs).
"""

import pandas as pd
from .k_mondrian import _mondrian_partition, _generalize


def _var_distance(part: pd.Series, glob: pd.Series) -> float:
    p = part.value_counts(normalize=True)
    q = glob.value_counts(normalize=True)
    all_vals = set(p.index) | set(q.index)
    return sum(abs(p.get(v, 0.0) - q.get(v, 0.0)) for v in all_vals) / 2.0


def _emd_numerical(part: pd.Series, glob: pd.Series) -> float:
    all_vals = sorted(set(part) | set(glob))
    n = len(all_vals)
    if n <= 1:
        return 0.0
    pc = part.value_counts()
    qc = glob.value_counts()
    pn, qn = len(part), len(glob)
    pc_cum = qc_cum = emd = 0.0
    for v in all_vals:
        pc_cum += pc.get(v, 0) / pn
        qc_cum += qc.get(v, 0) / qn
        emd += abs(pc_cum - qc_cum)
    return emd / (n - 1)


def apply_t_closeness(
    df: pd.DataFrame,
    k: int,
    t: float,
    qi_cols: list,
    numerical_cols: list,
    sensitive_col: str,
) -> pd.DataFrame:
    is_num = pd.api.types.is_numeric_dtype(df[sensitive_col])
    partitions = _mondrian_partition(df, df, k, qi_cols, numerical_cols)

    def dist(p: pd.DataFrame) -> float:
        return (
            _emd_numerical(p[sensitive_col], df[sensitive_col])
            if is_num
            else _var_distance(p[sensitive_col], df[sensitive_col])
        )

    good = [p for p in partitions if dist(p) <= t]
    bad = [p for p in partitions if dist(p) > t]

    if bad:
        good.append(pd.concat(bad, ignore_index=True))

    return pd.concat(
        [_generalize(p, qi_cols, numerical_cols) for p in good],
        ignore_index=True,
    )
