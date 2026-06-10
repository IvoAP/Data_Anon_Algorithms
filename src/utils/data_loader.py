import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "adult.csv"

NUMERICAL_QI = ["age", "education.num", "hours.per.week", "capital.gain", "capital.loss"]
CATEGORICAL_QI = [
    "workclass", "education", "marital.status", "occupation",
    "relationship", "race", "sex", "native.country",
]
ALL_QI = NUMERICAL_QI + CATEGORICAL_QI
SENSITIVE_ATTR = "income"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df
