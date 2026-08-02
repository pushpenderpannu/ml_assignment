import os

DATA_PATH = os.path.join("data", "dropout.csv")
MODEL_DIR = "model"
TEST_CSV = "test_data.csv"
TARGET_COL = "Target"
TEST_SIZE = 0.2


NOMINAL_COLS = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run download_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)

    print(f"\nFeatures: {X.shape[1]} (assignment minimum is 12)")
    print(f"Instances: {X.shape[0]} (assignment minimum is 500)")
    print(f"Missing values: {int(X.isnull().sum().sum())}")

    print("\nClass distribution:")
    counts = y.value_counts()
    for label, count in counts.items():
        print(f"  {label:<12} {count:>5}  ({count / len(y):.1%})")

    baseline = counts.max() / len(y)
    print(f"\nMajority-class baseline accuracy: {baseline:.4f}")
    print("Any model scoring below this is worse than always guessing the")
    print("largest class. Quote this number in your observations.")

    return X, y, baseline

