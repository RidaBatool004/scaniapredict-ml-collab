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
#     display_name: scania-failure-prediction (3.11.x)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # imports

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scania_failure_prediction.cleaning import clean_sensor_data

# %% [markdown]
# # load data

# %%
# Cell 2 — Load Data Correctly


DATA_PATH = Path("../data/raw/aps_failure_training_set.csv")

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

# %%


PROJECT_ROOT = (
    Path.cwd().resolve().parents[0] if Path.cwd().name == "notebooks" else Path.cwd().resolve()
)
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# %%


cleaned_df = clean_sensor_data(
    df,
    missing_threshold=0.95,
)

print("Original shape:", df.shape)
print("Cleaned shape:", cleaned_df.shape)

# %%
removed_columns = sorted(set(df.columns) - set(cleaned_df.columns))

print(f"Removed {len(removed_columns)} high-missingness columns.")
print(removed_columns[:20])
