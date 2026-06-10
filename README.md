# Data Anonymization Algorithms

A modular CLI tool implementing five data anonymization techniques on the [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

---

## Installation

Clone the repository and let `uv` create the virtual environment and install all dependencies:

```bash
git clone <repo-url>
cd Data_Anon_Algorithms
uv sync
```

That's it. No manual `pip install` needed.

---

## Running

```bash
uv run src\main.py
```

An interactive menu will guide you through algorithm selection, quasi-identifier selection, and parameter configuration. The anonymized dataset and a report are saved to the `results/` folder.

---

## Dataset

File: `data/adult.csv`  
Source: UCI Adult Census Income dataset (~48 000 records, 15 columns)

| Role | Columns |
|---|---|
| Numerical quasi-identifiers | `age`, `education.num`, `hours.per.week`, `capital.gain`, `capital.loss` |
| Categorical quasi-identifiers | `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country` |
| Sensitive attribute | `income` |
| Passthrough (not modified) | `fnlwgt` |

---

## Algorithms

### k-Mondrian (k-anonymity)

Recursively partitions the dataset using a multidimensional median split. At each step the algorithm picks the quasi-identifier with the largest normalised span and splits the current partition at its median (numerical) or alphabetical midpoint (categorical). Splitting stops when a partition has fewer than `2k` records. Each resulting equivalence class is generalised: numerical columns become ranges (`25-45`) and categorical columns become pipe-separated sets (`Male|Female`).

| Parameter | Description | Typical range |
|---|---|---|
| `k` | Minimum number of records per equivalence class | 2 – 100 |

Higher `k` → stronger anonymity, broader generalisations, less utility.

---

### l-Diversity

Extends k-Mondrian by adding a diversity requirement on the sensitive attribute: each equivalence class must contain at least `l` distinct values of `income`. After the Mondrian partitioning, classes that do not satisfy the criterion are merged into a single fallback class.

| Parameter | Description | Typical range |
|---|---|---|
| `k` | Minimum class size (same as k-anonymity) | 2 – 100 |
| `l` | Minimum distinct sensitive values per class | 2 – 10 |

Constraint: `l` cannot exceed the total number of distinct sensitive values (2 for the Adult dataset: `<=50K` and `>50K`).

---

### t-Closeness

Extends l-Diversity by controlling the *distribution* of the sensitive attribute, not just its diversity. Each equivalence class must have a distribution of `income` that is within distance `t` from the global distribution. Distance is measured with:

- **Variation distance** (half L1) for categorical sensitive attributes
- **Earth Mover's Distance** (L1 on CDFs) for numerical sensitive attributes

Classes that exceed the threshold are merged into a single fallback class.

| Parameter | Description | Typical range |
|---|---|---|
| `k` | Minimum class size | 2 – 100 |
| `t` | Maximum allowed distance from global distribution | 0.05 – 0.50 |

Lower `t` → stricter closeness, larger merged classes, less utility.

---

### DP – Laplace (Local Differential Privacy)

Applies the **Laplace mechanism** to each selected quasi-identifier independently (local DP, i.e. noise is added per record):

- **Numerical columns** — Laplace noise with scale `sensitivity / ε`, where `sensitivity = max − min` for each column. Result is clipped to the original range.
- **Categorical columns** — Randomised response: the true value is kept with probability `e^ε / (e^ε + k − 1)`; otherwise a uniformly random different value is substituted.

| Parameter | Description | Typical range |
|---|---|---|
| `ε` (epsilon) | Privacy budget | 0.1 – 10.0 |

Lower `ε` → stronger privacy guarantee, more noise, less utility.

---

### DP – Gaussian (Local Differential Privacy)

Applies the **Gaussian mechanism** under `(ε, δ)`-differential privacy:

- **Numerical columns** — Gaussian noise `N(0, σ²)` where `σ = sensitivity × √(2 ln(1.25/δ)) / ε`. Result is clipped to the original range.
- **Categorical columns** — Same randomised response as the Laplace variant (using the same `ε` budget).

| Parameter | Description | Typical range |
|---|---|---|
| `ε` (epsilon) | Privacy budget | 0.1 – 10.0 |
| `δ` (delta) | Failure probability (probability the guarantee does not hold) | 1e-6 – 1e-3 |

The Gaussian mechanism offers a tighter utility/privacy tradeoff than Laplace at the cost of introducing the `δ` parameter.

---

## Menu walkthrough

```
DATA ANONYMIZATION ALGORITHMS
──────────────────────────────────────────────────────────

  Dataset            data/adult.csv
  Records            48,842
  Columns            15
  Sensitive attr.    income

? Choose an anonymization algorithm:  (Use arrow keys)
 ❯ k-Mondrian  (k-anonymity)
   l-Diversity
   t-Closeness
   DP - Laplace  (local DP)
   DP - Gaussian (local DP)
```

1. **Choose algorithm** — arrow keys + Enter.

2. **Select quasi-identifiers** — a checkbox list opens with all 13 QIs pre-selected. Use `Space` to toggle individual columns, `Enter` to confirm. The same selection applies across all algorithms.

3. **Set parameters** — the tool prompts for each algorithm-specific parameter with a suggested default. Press Enter to accept the default or type a new value.

4. **Results** — two files are written to `results/`:

   | File | Contents |
   |---|---|
   | `<algorithm>_<timestamp>.csv` | Anonymized dataset |
   | `<algorithm>_<timestamp>_report.txt` | Date, technique, parameters, dataset stats, equivalence class statistics |

---

## Project structure

```
Data_Anon_Algorithms/
├── data/
│   └── adult.csv
├── src/
│   ├── main.py                   # interactive entry point
│   ├── algorithms/
│   │   ├── k_mondrian.py
│   │   ├── l_diversity.py
│   │   ├── t_closeness.py
│   │   ├── df_laplace.py
│   │   └── df_gauss.py
│   └── utils/
│       ├── data_loader.py        # dataset loading + column definitions
│       └── report.py             # CSV export + report generation
├── results/                      # created automatically on first run
├── pyproject.toml
└── README.md
```
