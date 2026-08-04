import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 1987

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

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Input record count: {X.shape[0]}")
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



def encode_nominals(X):
    """One-hot encode integer-coded nominal columns.
    """
    present = [c for c in NOMINAL_COLS if c in X.columns]
    missing = [c for c in NOMINAL_COLS if c not in X.columns]

    if missing:
        print("\nNote: these expected nominal columns were not found "
              "(check exact spelling in your CSV):")
        for c in missing:
            print(f"  - {c}")

    if not present:
        print("\nNo nominal columns encoded; using features as-is.")
        return X

    print(f"\nOne-hot encoding {len(present)} nominal column(s):")
    for c in present:
        print(f"  {c:<28} {X[c].nunique():>3} distinct values")

    X_enc = pd.get_dummies(X, columns=present, prefix=present, dtype=float)
    print(f"Feature count: {X.shape[1]} -> {X_enc.shape[1]}")
    return X_enc


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train, X_test):

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("\nScaler fitted on training data only "
          f"({X_train.shape[1]} features).")
    return scaler, X_train_s, X_test_s



def build_models():

    k = 21  

    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=2000,          
                C=1.0,                 
                class_weight="balanced", 
                random_state=SEED,
            ),
            "scaled": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=8,            
                min_samples_leaf=15,    
                criterion="gini",
                class_weight="balanced",
                random_state=SEED,
            ),
            "scaled": False,
        },
        "K-Nearest Neighbours": {
            "model": KNeighborsClassifier(
                n_neighbors=k,
                weights="distance",    
                metric="minkowski", p=2,
            ),
            "scaled": True,
        },
        "Naive Bayes (Gaussian)": {
            "model": GaussianNB(
                var_smoothing=1e-8,     
            ),
            "scaled": True,
        },
        "Random Forest (Ensemble)": {
            "model": RandomForestClassifier(
                n_estimators=300,       
                max_depth=14,
                min_samples_leaf=3,
                max_features="sqrt",    
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
            "scaled": False,
        },
    }

def compute_metrics(y_true, y_pred, proba, n_classes):
    average = "binary" if n_classes == 2 else "macro"

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr",
                                average="macro")
    except (ValueError, TypeError):
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average,
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average,
                               zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def heading_separater(message):
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    heading_separater("STEP 1: LOAD DATA")
    X, y, baseline = load_data()

    heading_separater("STEP 2: ENCODE NOMINAL COLUMNS")
    X = encode_nominals(X)

    heading_separater("STEP 3: ENCODE TARGET AND SPLIT")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    n_classes = len(encoder.classes_)

    print(f"Classes: {list(encoder.classes_)} -> {list(range(n_classes))}")

    X_train, X_test, y_train, y_test = split_data(X, y_enc)

    heading_separater("STEP 4: SCALE FEATURES")

    scaler, X_train_s, X_test_s = fit_scaler(X_train, X_test)

    heading_separater("STEP 5: TRAIN MODELS")

    specs = build_models()

    filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbours": "knn.pkl",
        "Naive Bayes (Gaussian)": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
    }

    results = {}
    train_scores = {}

    for name, spec in specs.items():
        model = spec["model"]
        Xtr = X_train_s if spec["scaled"] else X_train
        Xte = X_test_s if spec["scaled"] else X_test

        model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        results[name] = compute_metrics(y_test, y_pred, proba, n_classes)


        train_scores[name] = accuracy_score(y_train, model.predict(Xtr))

        joblib.dump(model, os.path.join(MODEL_DIR, filenames[name]))
        print(f"  {name:<26} test acc {results[name]['Accuracy']:.4f} | "
              f"train acc {train_scores[name]:.4f} -> {filenames[name]}")

    heading_separater("STEP 6: SAVE ARTIFACTS")
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(list(X_train.columns),
                os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("  scaler.pkl, label_encoder.pkl, feature_columns.pkl")


    test_df = X_test.copy()
    test_df[TARGET_COL] = encoder.inverse_transform(y_test)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"  {TEST_CSV} ({test_df.shape[0]} rows, {test_df.shape[1]} cols)")

    heading_separater("COMPARISON TABLE  (paste into README.md)")
    table = pd.DataFrame(results).transpose()
    print(table.round(4).to_string())

    print("\nMarkdown:\n")
    print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
    print("|---|---|---|---|---|---|---|")
    for name, row in table.iterrows():
        print(f"| {name} | {row['Accuracy']:.4f} | {row['AUC']:.4f} | "
              f"{row['Precision']:.4f} | {row['Recall']:.4f} | "
              f"{row['F1']:.4f} | {row['MCC']:.4f} |")

    heading_separater("NOTES FOR YOUR OBSERVATIONS SECTION")
    
    print(f"Majority-class baseline accuracy: {baseline:.4f}")
    print(f"Best accuracy: {table['Accuracy'].idxmax()} "
          f"({table['Accuracy'].max():.4f})")
    print(f"Best MCC:      {table['MCC'].idxmax()} "
          f"({table['MCC'].max():.4f})")
    print("\nTrain/test accuracy gap (large gap = overfitting):")
    for name in table.index:
        gap = train_scores[name] - table.loc[name, "Accuracy"]
        flag = "  <-- overfitting" if gap > 0.10 else ""
        print(f"  {name:<26} {gap:+.4f}{flag}")

    rf = specs["Random Forest (Ensemble)"]["model"]
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns).nlargest(10)
    print("\nTop 10 features by Random Forest importance:")
    for feat, imp in importances.items():
        print(f"  {imp:.4f}  {feat}")



if __name__ == "__main__":
    main()