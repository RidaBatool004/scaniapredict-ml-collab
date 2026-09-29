import pandas as pd


def clean_sensor_data(
    df: pd.DataFrame,
    missing_threshold: float = 0.95,
) -> pd.DataFrame:
    """Remove columns with excessive missing values.

    Parameters
    ----------
    df:
        Input dataframe.
    missing_threshold:
        Maximum allowed fraction of missing values in a column.

    Returns
    -------
    pd.DataFrame
        Dataframe with high-missingness columns removed.
    """
    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1")

    cleaned = df.copy()

    missing_fraction = cleaned.isna().mean()

    columns_to_drop = missing_fraction[missing_fraction > missing_threshold].index

    return cleaned.drop(columns=columns_to_drop)
