import pandas as pd
import pytest

from scania_failure_prediction.cleaning import clean_sensor_data


def test_clean_sensor_data_removes_high_missing_columns():
    df = pd.DataFrame(
        {
            "good_feature": [1, 2, 3, 4],
            "bad_feature": [None, None, None, 1],
        }
    )

    result = clean_sensor_data(df, missing_threshold=0.5)

    assert "good_feature" in result.columns
    assert "bad_feature" not in result.columns


def test_clean_sensor_data_preserves_input():
    df = pd.DataFrame(
        {
            "feature": [1, None, 3],
        }
    )

    original = df.copy()

    clean_sensor_data(df, missing_threshold=0.5)

    pd.testing.assert_frame_equal(df, original)


def test_clean_sensor_data_rejects_invalid_threshold():
    df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError):
        clean_sensor_data(df, missing_threshold=1.5)
