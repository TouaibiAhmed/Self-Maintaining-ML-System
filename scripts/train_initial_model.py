"""Train and package the initial churn model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.train import train  # noqa: E402, I001


if __name__ == "__main__":
    result = train()
    print(f"Artifact path: {result['artifact_path']}")
    print("Metrics:")
    for name, value in result["metrics"].items():
        print(f"{name}: {value}")
