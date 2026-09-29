
import pandas as pd
from sklearn.datasets import load_iris

OUT_PATH = "data/iris.csv"


def build_frame() -> pd.DataFrame:
    """Return the iris dataset as a tidy DataFrame with a `target` column."""
    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print the few things worth checking before modelling anything.

    Deliberately short. Extend it: `df.describe()`, `df.dtypes`, `df.corr()`,
    per-class means... see the pandas docs for what a DataFrame can tell you.
    """
    balance = df["target"].value_counts().sort_index().to_dict()
    missing = int(df.isna().sum().sum())
    print(f"rows={len(df)}  columns={len(df.columns)}  missing={missing}")
    print(f"class balance: {balance}")


def main() -> None:
    df = build_frame()
    summarize(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
