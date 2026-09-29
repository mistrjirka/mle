
import pytest

from data import build_frame
from features import accuracy_for_seed, add_petal_area, class_balance

pytestmark = pytest.mark.exercise


def test_add_petal_area():
    df = build_frame()
    out = add_petal_area(df)

    assert "petal_area" in out.columns
    assert "petal_area" not in df.columns, "do not modify the input DataFrame"

    row = out.iloc[0]
    assert row["petal_area"] == pytest.approx(
        row["petal_length"] * row["petal_width"]
    )


def test_class_balance():
    balance = class_balance(build_frame())

    assert isinstance(balance, dict)
    assert sum(balance.values()) == 150
    assert set(balance) == {0, 1, 2}
    assert all(isinstance(k, int) for k in balance), "keys must be plain ints"


def test_accuracy_varies_with_seed():
    """The point of the exercise: one number is not a result."""
    df = build_frame()
    scores = [accuracy_for_seed(df, seed) for seed in (0, 1, 2, 3, 4)]

    assert all(0.0 <= s <= 1.0 for s in scores)
    spread = max(scores) - min(scores)
    print(f"\naccuracy across 5 seeds: {min(scores):.3f}-{max(scores):.3f} "
          f"(spread {spread:.3f})")