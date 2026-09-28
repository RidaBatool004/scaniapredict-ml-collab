# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd

# 1. Define file path using a raw string to handle Windows backslashes
DATA_PATH = Path(r"D:\mlops assi\scaniapredict-ml-collab\data\raw\aps_failure_training_set.csv")

# 2. Read CSV:
# - skip_blank_lines=True ignores comment blocks separated by blank lines if needed
# - na_values="na" converts all "na" strings directly into np.nan
# - comment='#' can be used if lines start with #, otherwise pandas auto-detects the header row
df = pd.read_csv(
    DATA_PATH,
    comment="T",  # Skips lines starting with "This" (the license header)
    skiprows=17,  # Alternatively, skip the first 17 lines of copyright text
    na_values="na",  # Automatically converts 'na' strings to NaN
)

print("Shape:", df.shape)
print("\nFirst 5 columns preview:")
print(df.iloc[:, :5].head())

# %% [markdown]
# # imports

# %%
# from scania_failure_prediction.cleaning import clean_sensor_data
# %% [markdown]
# # load data
# %%
# Cell 2 — Load Data Correctly
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

DATA_PATH = Path(r"D:\mlops assi\scaniapredict-ml-collab\data\raw\aps_failure_training_set.csv")

# Read skipping the header comment block (20 lines) and the separator line ('---')
df = pd.read_csv(DATA_PATH, skiprows=20, comment="-", na_values="na", low_memory=False)

# Strip any leading/trailing whitespaces from column names
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)  # Should display (60000, 171)
print(
    "First 5 columns:", df.columns.tolist()[:5]
)  # Should display ['class', 'aa_000', 'ab_000', ...]

# %%
df.info()

# %%
df.describe(include="all").T

# %%
missing = df.isna().sum().sort_values(ascending=False).to_frame("missing_count")

missing["missing_pct"] = missing["missing_count"] / len(df) * 100

missing.head(20)


# %% [markdown]
# # visualize missingness

# %%
missing_top = missing.head(20).sort_values("missing_pct")
plt.figure(figsize=(10, 6))
plt.barh(missing_top.index, missing_top["missing_pct"])
plt.xlabel("Missing values (%)")
plt.ylabel("Feature")
plt.title("Top features by missing-value percentage")
plt.tight_layout()
plt.show()

# %%
print("Columns in DataFrame:", df.columns.tolist()[:10])

TARGET_COLUMN = "class"

if TARGET_COLUMN in df.columns:
    print("\nTarget counts:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))
    print("\nTarget normalized:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))
else:
    print(f"Error: Column '{TARGET_COLUMN}' not found in columns: {df.columns.tolist()}")

# %%
df[TARGET_COLUMN].value_counts(dropna=False)

# %%
df[TARGET_COLUMN].value_counts(normalize=True)

# %% [markdown]
# # Plot class distribution

# %%
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x=TARGET_COLUMN)

plt.title("Target class distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %%
class_distribution = df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(2)

class_distribution

# %% [markdown]
# # Numerical feature analysis

# %%
numeric_columns = df.select_dtypes(include="number").columns

print(f"Number of numerical features: {len(numeric_columns)}")

# %%
df[numeric_columns].describe().T.head(20)

# %% [markdown]
# # Feature distributions

# %%
selected_features = numeric_columns[:6]

df[selected_features].hist(
    figsize=(14, 10),
    bins=30,
)

plt.suptitle("Distributions of selected sensor features")
plt.tight_layout()
plt.show()
