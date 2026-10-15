"""Smoke test: is there a GPU torch can use, and does compute work on it?

Run with:  uv run python src/gpu_check.py

Under ROCm (HIP) PyTorch exposes the AMD GPU through the ``torch.cuda``
API, so ``torch.cuda.is_available()`` is the check on both NVIDIA and AMD.

Radeon 880M (gfx1150) note: prebuilt ROCm wheels do not officially
support this chip. If this script reports no GPU or crashes with
``invalid device function``, that is the known unsupported-arch failure,
not a bug in this script. As an experiment you can retry with::

    HSA_OVERRIDE_GFX_VERSION=11.5.0 uv run python src/gpu_check.py
"""

import os

import torch


def main() -> None:
    print(f"torch {torch.__version__}  (hip/rocm build: {torch.version.hip})")
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print(f"cuda/HIP available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No GPU visible to torch -> training will use CPU.")
        device = torch.device("cpu")
    else:
        print(f"device 0: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")

    # Real compute, not just device listing: 1000x1000 matmul.
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    c = a @ b
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"matmul ok on {device}: result mean {c.mean().item():.4f}")


if __name__ == "__main__":
    main()
