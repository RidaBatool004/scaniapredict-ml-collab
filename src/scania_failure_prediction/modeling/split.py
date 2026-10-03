from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_params(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    params = load_params(project_root / "params.yaml")

    data_params = params["data"]

    input_path = project_root / data_params["input"]
    target_column = data_params["target_column"]
    skiprows = data_params["skiprows"]
    test_size = data_params["test_size"]
    random_state = data_params["random_state"]

    output_dir = project_root / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        input_path,
        skiprows=skiprows,
        na_values=["na"],
    )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    train_df, validation_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Training class distribution:\n{train_df[target_column].value_counts()}")
    print(f"Validation class distribution:\n{validation_df[target_column].value_counts()}")
    print(f"Saved: {train_path}")
    print(f"Saved: {validation_path}")


if __name__ == "__main__":
    main()
