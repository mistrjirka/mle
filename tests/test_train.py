from data import build_frame
from train import evaluate, load_params, split


def test_beats_the_baseline():
    _, metrics = evaluate(load_params(), build_frame())
    assert metrics["gap_over_baseline"] > 0.1, (
        f"model barely beats a constant prediction: {metrics}"
    )


def test_same_seed_same_result():
    """Two runs with the same params must agree, or no comparison is valid."""
    params = load_params()
    df = build_frame()
    _, first = evaluate(params, df)
    _, second = evaluate(params, df)
    assert first == second


def test_split_sizes_add_up():
    params = load_params()
    X_train, X_test, _, _ = split(
        build_frame(), params["test_size"], params["random_state"]
    )
    assert len(X_train) + len(X_test) == 150
