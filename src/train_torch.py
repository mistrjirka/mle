"""GPU-capable training: MLP classifier on iris, device-agnostic.

Run with:  uv run python src/train_torch.py

Uses the AMD/NVIDIA GPU whenever torch can see one, otherwise falls back
to CPU -- same code path, no flags needed. Hyperparameters come from
``params.yaml`` (test_size, random_state); the MLP shape/epochs are module
constants below so ``dvc repro`` still tracks what matters.

Outputs (kept separate from the sklearn pipeline on purpose):
    models/model_torch.pt   trained weights + scaler + metadata
    metrics_torch.json      accuracy + which device trained it
"""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/iris.csv"
MODEL_PATH = "models/model_torch.pt"
METRICS_PATH = "metrics_torch.json"

HIDDEN = 16
EPOCHS = 200
LR = 0.01


def get_device() -> torch.device:
    """Best available device. Under ROCm the AMD GPU shows up as 'cuda'."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class IrisMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_params(path: str = "params.yaml") -> dict:
    """Read params.yaml into a plain dict; fail fast with a clear message."""
    try:
        with open(path) as fh:
            return yaml.safe_load(fh)
    except OSError as exc:
        raise SystemExit(f"cannot read params file {path!r}: {exc}") from exc


def save_metrics(metrics: dict, path: str = METRICS_PATH) -> None:
    """Write metrics JSON; fail fast with a clear message."""
    try:
        with open(path, "w") as fh:
            json.dump(metrics, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write metrics file {path!r}: {exc}") from exc


def main() -> None:
    params = load_params()
    device = get_device()
    print(f"training on: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (CPU fallback -- no GPU visible to torch)")

    torch.manual_seed(params["random_state"])
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).to_numpy(dtype="float32")
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        stratify=y,
        random_state=params["random_state"],
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(y_test).long().to(device)

    model = IrisMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1)
        accuracy = (pred == yte).float().mean().item()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": np.asarray(scaler.mean_).tolist(),
            "scaler_scale": np.asarray(scaler.scale_).tolist(),
            "hidden": HIDDEN,
        },
        MODEL_PATH,
    )
    metrics = {"accuracy": round(float(accuracy), 4), "device": str(device)}
    save_metrics(metrics)

    print(f"accuracy {metrics['accuracy']:.3f} on {metrics['device']}")


if __name__ == "__main__":
    main()
