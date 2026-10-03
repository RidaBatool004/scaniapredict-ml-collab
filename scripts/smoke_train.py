from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "class"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "ci" / "aps_failure_sample.csv"
    metrics_path = project_root / "ci_metrics.md"

    df = pd.read_csv(data_path, na_values=["na"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"neg": 0, "pos": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    max_iter=100,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    assert len(predictions) == len(y_test)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= roc_auc <= 1.0

    report = f"""## CI Smoke-Train Metrics

| Metric | Value |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 | {f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## Training Details

| Detail | Value |
|---|---:|
| Training rows | {len(X_train)} |
| Validation rows | {len(X_test)} |
| Features | {X.shape[1]} |
"""

    metrics_path.write_text(report, encoding="utf-8")

    print("Smoke training passed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Metrics report: {metrics_path}")


if __name__ == "__main__":
    main()
