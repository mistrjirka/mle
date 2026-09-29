
import json

import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/iris.csv"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "metrics.json"


def load_params(path: str = "params.yaml") -> dict:
    """Read params.yaml into a plain dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def split(df: pd.DataFrame, test_size: float, random_state: int):
    """Stratified train/test split. Stratify so no class vanishes from test."""
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def evaluate(params: dict, df: pd.DataFrame):
    """Fit the model and a do-nothing baseline; return both and the metrics."""
    X_train, X_test, y_train, y_test = split(
        df, params["test_size"], params["random_state"]
    )

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    baseline_accuracy = accuracy_score(y_test, y_base)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "baseline_accuracy": round(float(baseline_accuracy), 4),
        "gap_over_baseline": round(float(accuracy - baseline_accuracy), 4),
    }
    return model, metrics


def main() -> None:
    params = load_params()
    df = pd.read_csv(DATA_PATH)
    model, metrics = evaluate(params, df)

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(
        f"accuracy {metrics['accuracy']:.3f}   "
        f"baseline {metrics['baseline_accuracy']:.3f}   "
        f"gap +{metrics['gap_over_baseline']:.3f}"
    )


if __name__ == "__main__":
    main()
