import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

ALGO_LABELS = {
    "k_mondrian": "k-Mondrian (k-anonymity)",
    "l_diversity": "l-Diversity",
    "t_closeness": "t-Closeness",
    "df_laplace": "Differential Privacy - Laplace",
    "df_gauss": "Differential Privacy - Gaussian",
}


def _partition_stats(df_anon: pd.DataFrame, qi_cols: list) -> dict:
    sizes = [len(g) for _, g in df_anon.groupby(qi_cols, sort=False)]
    return {
        "n_classes": len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": round(sum(sizes) / len(sizes), 1),
    }


def save_results(
    df_original: pd.DataFrame,
    df_anon: pd.DataFrame,
    algorithm: str,
    params: dict,
    qi_cols: list,
    numerical_cols: list,
    sensitive_col: str,
    extra_stats: Optional[dict] = None,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = ALGO_LABELS.get(algorithm, algorithm)

    csv_path = RESULTS_DIR / f"{algorithm}_{ts}.csv"
    df_anon.to_csv(csv_path, index=False)

    report_path = RESULTS_DIR / f"{algorithm}_{ts}_report.txt"
    sep = "=" * 58

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n          DATA ANONYMIZATION REPORT\n{sep}\n\n")
        f.write(f"Date              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm         : {label}\n\n")

        f.write(f"Parameters\n{'-'*30}\n")
        for k, v in params.items():
            f.write(f"  {k:<24}: {v}\n")

        f.write(f"\nDataset\n{'-'*30}\n")
        f.write(f"  Original records  : {len(df_original):,}\n")
        f.write(f"  Result records    : {len(df_anon):,}\n")
        f.write(f"  Data retention    : {100.0 * len(df_anon) / len(df_original):.1f}%\n")

        f.write(f"\nQuasi-identifiers ({len(qi_cols)} selected)\n{'-'*30}\n")
        for col in qi_cols:
            kind = "numerical" if col in numerical_cols else "categorical"
            f.write(f"  - {col} ({kind})\n")

        f.write(f"\nSensitive attribute : {sensitive_col}\n")

        if algorithm in ("k_mondrian", "l_diversity", "t_closeness"):
            try:
                stats = _partition_stats(df_anon, qi_cols)
                f.write(f"\nEquivalence Classes\n{'-'*30}\n")
                f.write(f"  Total classes     : {stats['n_classes']:,}\n")
                f.write(f"  Min class size    : {stats['min_size']}\n")
                f.write(f"  Max class size    : {stats['max_size']}\n")
                f.write(f"  Avg class size    : {stats['avg_size']}\n")
            except Exception:
                pass

        if extra_stats:
            f.write(f"\nAdditional Statistics\n{'-'*30}\n")
            for k, v in extra_stats.items():
                f.write(f"  {k:<24}: {v}\n")

        f.write(f"\n{sep}\n")

    print(f"\n  Anonymized CSV  -> {csv_path}")
    print(f"  Report          -> {report_path}")
