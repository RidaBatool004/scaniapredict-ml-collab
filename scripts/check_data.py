from pathlib import Path

import pandas as pd

EXPECTED_FEATURE_COUNT = 170
TARGET_COLUMN = "class"
EXPECTED_CLASSES = {"neg", "pos"}
MAX_NULL_RATIO = 0.95


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "ci" / "aps_failure_sample.csv"

    df = pd.read_csv(data_path, na_values=["na"])

    # Schema checks
    assert TARGET_COLUMN in df.columns, "Missing target column."
    assert len(df.columns) == EXPECTED_FEATURE_COUNT + 1, (
        f"Expected {EXPECTED_FEATURE_COUNT + 1} columns, found {len(df.columns)}."
    )

    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    assert len(feature_columns) == EXPECTED_FEATURE_COUNT

    # Target checks
    classes = set(df[TARGET_COLUMN].dropna().unique())
    assert classes <= EXPECTED_CLASSES, (
        f"Unexpected target values found: {classes - EXPECTED_CLASSES}"
    )

    assert df[TARGET_COLUMN].notna().all(), "Target column contains null values."

    # Null checks
    null_ratios = df[feature_columns].isna().mean()
    invalid_nulls = null_ratios[null_ratios > MAX_NULL_RATIO]

    assert invalid_nulls.empty, (
        f"Columns exceeding {MAX_NULL_RATIO:.0%} nulls: {invalid_nulls.to_dict()}"
    )

    # Numeric feature checks
    for column in feature_columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        non_numeric = df[column].notna() & numeric_values.isna()

        assert not non_numeric.any(), f"Non-numeric values found in feature '{column}'."

    print("Data checks passed.")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    print(f"Maximum feature null ratio: {null_ratios.max():.2%}")


if __name__ == "__main__":
    main()
