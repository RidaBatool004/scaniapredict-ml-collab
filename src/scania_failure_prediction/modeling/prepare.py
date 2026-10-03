from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml


def load_params(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    params = load_params(project_root / "params.yaml")

    preprocessing_params = params["preprocessing"]

    missing_threshold = preprocessing_params["missing_threshold"]
    imputation_strategy = preprocessing_params["imputation_strategy"]

    train_path = project_root / "data" / "interim" / "train.csv"
    validation_path = project_root / "data" / "interim" / "validation.csv"

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    target_column = params["data"]["target_column"]

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data.")

    if target_column not in validation_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in validation data.")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_validation = validation_df.drop(columns=[target_column])
    y_validation = validation_df[target_column]

    # Identify columns to remove using TRAINING data only.
    missing_fraction = X_train.isna().mean()
    columns_to_keep = missing_fraction[missing_fraction <= missing_threshold].index.tolist()

    X_train = X_train[columns_to_keep]
    X_validation = X_validation[columns_to_keep]

    # Fit preprocessing ONLY on training data.
    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputation_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_validation_processed = preprocessing_pipeline.transform(X_validation)

    X_train = pd.DataFrame(
        X_train_processed,
        columns=columns_to_keep,
        index=X_train.index,
    )

    X_validation = pd.DataFrame(
        X_validation_processed,
        columns=columns_to_keep,
        index=X_validation.index,
    )

    # Encode the target consistently.
    target_mapping = {"neg": 0, "pos": 1}

    y_train = y_train.map(target_mapping)
    y_validation = y_validation.map(target_mapping)

    if y_train.isna().any() or y_validation.isna().any():
        raise ValueError("Unexpected target value found. Expected only 'neg' and 'pos'.")

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_validation.to_csv(output_dir / "X_validation.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_validation.to_csv(output_dir / "y_validation.csv", index=False)

    joblib.dump(
        {
            "pipeline": preprocessing_pipeline,
            "columns": columns_to_keep,
        },
        models_dir / "preprocessor.joblib",
    )

    print(f"Original feature count: {len(train_df.columns) - 1}")
    print(f"Features retained: {len(columns_to_keep)}")
    print(f"Features removed: {len(train_df.columns) - 1 - len(columns_to_keep)}")
    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_validation.shape}")
    print(f"Training target distribution:\n{y_train.value_counts().sort_index()}")
    print(f"Validation target distribution:\n{y_validation.value_counts().sort_index()}")
    print("Preprocessing: median imputation + StandardScaler")
    print("Preprocessing fitted on: training data only")
    print(f"Saved processed data to: {output_dir}")
    print(f"Saved preprocessor to: {models_dir / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
