
import pandas as pd


def add_petal_area(df: pd.DataFrame) -> pd.DataFrame:
    """Exercise 1 - return a copy of `df` with a `petal_area` column.

    petal_area = petal_length * petal_width
    """
    raise NotImplementedError("exercise 1")


def class_balance(df: pd.DataFrame) -> dict[int, int]:
    """Exercise 2 - return how many rows each class has.

    """
    raise NotImplementedError("exercise 2 ")


def accuracy_for_seed(df: pd.DataFrame, seed: int) -> float:
    """Exercise 3 - train once with `seed` and return the test accuracy.

    Load params with `load_params()`, set `random_state` to `seed`, call
    `evaluate()`, and return metrics["accuracy"].

    """
    raise NotImplementedError("exercise 3 ")
