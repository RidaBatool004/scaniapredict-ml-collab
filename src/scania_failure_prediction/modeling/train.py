from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml


def load_params(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    params = load_params(project_root / "params.yaml")

    model_params = params["model"]

    random_state = model_params["random_state"]

    # Set seeds for reproducibility.
    random.seed(random_state)
    np.random.seed(random_state)

    X_train_path = project_root / "data" / "processed" / "X_train.csv"
    y_train_path = project_root / "data" / "processed" / "y_train.csv"
    model_path = project_root / "models" / "model.joblib"

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")

    model = LogisticRegression(
        C=model_params["C"],
        max_iter=model_params["max_iter"],
        class_weight=model_params["class_weight"],
        solver=model_params["solver"],
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Training rows: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Model: {model.__class__.__name__}")
    print(f"C: {model_params['C']}")
    print(f"Max iterations: {model_params['max_iter']}")
    print(f"Class weight: {model_params['class_weight']}")
    print(f"Solver: {model_params['solver']}")
    print(f"Random state: {random_state}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
