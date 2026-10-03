import json
from pathlib import Path
import subprocess

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    model_path = project_root / "models" / "model.joblib"
    X_validation_path = project_root / "data" / "processed" / "X_validation.csv"
    y_validation_path = project_root / "data" / "processed" / "y_validation.csv"
    metrics_path = project_root / "metrics.json"

    model = joblib.load(model_path)

    X_validation = pd.read_csv(X_validation_path)
    y_validation = pd.read_csv(y_validation_path).squeeze("columns")

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "validation_rows": len(y_validation),
        "git_commit": get_git_commit(project_root),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
