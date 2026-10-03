from pathlib import Path

import yaml


def test_params_file_contains_required_pipeline_settings():
    project_root = Path(__file__).resolve().parents[1]

    with (project_root / "params.yaml").open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    assert params["data"]["input"] == "data/raw/aps_failure_training_set.csv"
    assert params["data"]["target_column"] == "class"
    assert params["data"]["skiprows"] == 20
    assert params["data"]["test_size"] == 0.3
    assert params["data"]["random_state"] == 42

    assert params["model"]["type"] == "logistic_regression"
    assert params["model"]["random_state"] == 42
