import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "dropout.csv")

print("Fetching UCI dataset 697 ...")
ds = fetch_ucirepo(id=697)
X = ds.data.features
y = ds.data.targets

print(f"\nShape: {X.shape}")
print(f"\nColumn names ({len(X.columns)}):")
for i, c in enumerate(X.columns, 1):
    print(f"  {i:>2}. {c}")

print("\nDtypes:")
print(X.dtypes.value_counts())

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values:", int(X.isnull().sum().sum()))

print("\nFeature ranges (note how wildly the scales differ -- this is the")
print("concrete reason StandardScaler is required for LR, kNN and NB):")
print(X.describe().T[["min", "max", "mean", "std"]].round(2).to_string())

os.makedirs(OUT_DIR, exist_ok=True)
full = X.copy()
full["Target"] = y
full.to_csv(OUT_PATH, index=False)

print(f"\nSaved {OUT_PATH}: {full.shape[0]} rows, {full.shape[1]} columns")
