import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import questionary
from rich.console import Console
from rich.table import Table

from utils.data_loader import load_data, ALL_QI, NUMERICAL_QI, SENSITIVE_ATTR
from utils.report import save_results
from algorithms.k_mondrian import apply_k_mondrian
from algorithms.l_diversity import apply_l_diversity
from algorithms.t_closeness import apply_t_closeness
from algorithms.df_laplace import apply_df_laplace
from algorithms.df_gauss import apply_df_gauss

console = Console()

ALGORITHMS = [
    ("k-Mondrian  (k-anonymity)",        "k_mondrian"),
    ("l-Diversity",                       "l_diversity"),
    ("t-Closeness",                       "t_closeness"),
    ("DP - Laplace  (local DP)",          "df_laplace"),
    ("DP - Gaussian (local DP)",          "df_gauss"),
]
ALGO_MAP = {label: key for label, key in ALGORITHMS}


# ── helpers ──────────────────────────────────────────────────────────────────

def _ask_int(message: str, default: int) -> int:
    raw = questionary.text(message, default=str(default)).ask()
    if raw is None:
        sys.exit(0)
    return int(raw)


def _ask_float(message: str, default: float) -> float:
    raw = questionary.text(message, default=str(default)).ask()
    if raw is None:
        sys.exit(0)
    return float(raw)


def _choose_qi() -> tuple[list, list]:
    selected = questionary.checkbox(
        "Select quasi-identifiers  (Space = toggle, Enter = confirm):",
        choices=[questionary.Choice(c, checked=True) for c in ALL_QI],
    ).ask()
    if not selected:
        console.print("[red]No quasi-identifiers selected. Exiting.[/red]")
        sys.exit(1)
    numerical = [c for c in selected if c in NUMERICAL_QI]
    return selected, numerical


# ── per-algorithm runners ─────────────────────────────────────────────────────

def _run_k_mondrian(df, qi_cols, numerical_cols):
    k = _ask_int("k  (min equivalence-class size, e.g. 5):", 5)
    params = {"k": k, "quasi_identifiers": ", ".join(qi_cols)}
    console.print(f"\n[cyan]Running k-Mondrian  k={k}…[/cyan]")
    df_anon = apply_k_mondrian(df, k, qi_cols, numerical_cols)
    return df_anon, params, {}


def _run_l_diversity(df, qi_cols, numerical_cols):
    k = _ask_int("k  (min equivalence-class size):", 5)
    l = _ask_int("l  (min distinct sensitive values per class):", 3)
    params = {
        "k": k, "l": l,
        "sensitive_attribute": SENSITIVE_ATTR,
        "quasi_identifiers": ", ".join(qi_cols),
    }
    console.print(f"\n[cyan]Running l-Diversity  k={k}  l={l}…[/cyan]")
    df_anon = apply_l_diversity(df, k, l, qi_cols, numerical_cols, SENSITIVE_ATTR)
    min_l = df_anon.groupby(qi_cols, sort=False)[SENSITIVE_ATTR].nunique().min()
    return df_anon, params, {"min_l_achieved": int(min_l)}


def _run_t_closeness(df, qi_cols, numerical_cols):
    k = _ask_int("k  (min equivalence-class size):", 5)
    t = _ask_float("t  (max allowed distance 0.0–1.0, e.g. 0.2):", 0.2)
    params = {
        "k": k, "t": t,
        "sensitive_attribute": SENSITIVE_ATTR,
        "quasi_identifiers": ", ".join(qi_cols),
    }
    console.print(f"\n[cyan]Running t-Closeness  k={k}  t={t}…[/cyan]")
    df_anon = apply_t_closeness(df, k, t, qi_cols, numerical_cols, SENSITIVE_ATTR)
    return df_anon, params, {}


def _run_df_laplace(df, qi_cols, numerical_cols):
    epsilon = _ask_float("ε  (privacy budget, e.g. 1.0 — smaller = more private):", 1.0)
    params = {
        "epsilon": epsilon,
        "mechanism": "Laplace",
        "quasi_identifiers": ", ".join(qi_cols),
    }
    console.print(f"\n[cyan]Running DP-Laplace  ε={epsilon}…[/cyan]")
    df_anon = apply_df_laplace(df, epsilon, qi_cols, numerical_cols)
    return df_anon, params, {}


def _run_df_gauss(df, qi_cols, numerical_cols):
    epsilon = _ask_float("ε  (privacy budget, e.g. 1.0):", 1.0)
    delta = _ask_float("δ  (failure probability, e.g. 1e-5):", 1e-5)
    params = {
        "epsilon": epsilon,
        "delta": delta,
        "mechanism": "Gaussian",
        "quasi_identifiers": ", ".join(qi_cols),
    }
    console.print(f"\n[cyan]Running DP-Gaussian  ε={epsilon}  δ={delta}…[/cyan]")
    df_anon = apply_df_gauss(df, epsilon, delta, qi_cols, numerical_cols)
    return df_anon, params, {}


RUNNERS = {
    "k_mondrian": _run_k_mondrian,
    "l_diversity": _run_l_diversity,
    "t_closeness": _run_t_closeness,
    "df_laplace":  _run_df_laplace,
    "df_gauss":    _run_df_gauss,
}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]DATA ANONYMIZATION ALGORITHMS[/bold blue]")
    console.print()

    df = load_data()

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_row("[bold]Dataset[/bold]",          str(Path("data/adult.csv")))
    info.add_row("[bold]Records[/bold]",           f"{len(df):,}")
    info.add_row("[bold]Columns[/bold]",           str(len(df.columns)))
    info.add_row("[bold]Sensitive attribute[/bold]", SENSITIVE_ATTR)
    console.print(info)
    console.print()

    algo_label = questionary.select(
        "Choose an anonymization algorithm:",
        choices=[label for label, _ in ALGORITHMS],
    ).ask()
    if algo_label is None:
        sys.exit(0)

    algo_key = ALGO_MAP[algo_label]
    console.print()

    qi_cols, numerical_cols = _choose_qi()
    console.print(
        f"\n  QIs selected  : [green]{', '.join(qi_cols)}[/green]"
        f"\n  Numerical QIs : [green]{', '.join(numerical_cols) or 'none'}[/green]\n"
    )

    df_anon, params, extra_stats = RUNNERS[algo_key](df, qi_cols, numerical_cols)

    console.print(f"\n[green]✓ Anonymization complete[/green] — {len(df_anon):,} records")
    console.print("\nSaving results…")
    save_results(df, df_anon, algo_key, params, qi_cols, numerical_cols, SENSITIVE_ATTR, extra_stats)
    console.print("\n[bold green]All done![/bold green]")


if __name__ == "__main__":
    main()
