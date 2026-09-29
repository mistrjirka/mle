from data import build_frame

EXPECTED_COLUMNS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target",
}


def test_frame_shape():
    df = build_frame()
    assert len(df) == 150
    assert set(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    assert build_frame().isna().sum().sum() == 0


def test_all_classes_present():
    assert build_frame()["target"].nunique() == 3
