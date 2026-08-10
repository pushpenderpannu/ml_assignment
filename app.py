import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Configuration

MODEL_DIR = "model"
TARGET_COL = "Target"

MODEL_REGISTRY = {
    "Logistic Regression": {"file": "logistic_regression.pkl", "scaled": True},
    "Decision Tree": {"file": "decision_tree.pkl", "scaled": False},
    "K-Nearest Neighbours": {"file": "knn.pkl", "scaled": True},
    "Naive Bayes (Gaussian)": {"file": "naive_bayes.pkl", "scaled": True},
    "Random Forest (Ensemble)": {"file": "random_forest.pkl", "scaled": False},
}

st.set_page_config(
    page_title="2025ac05706 Student Dropout Classifier",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_artifacts():

    missing = []

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    columns_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    for path in (scaler_path, encoder_path, columns_path):
        if not os.path.exists(path):
            missing.append(path)

    models = {}
    for label, spec in MODEL_REGISTRY.items():
        path = os.path.join(MODEL_DIR, spec["file"])
        if os.path.exists(path):
            models[label] = joblib.load(path)
        else:
            missing.append(path)

    if missing:
        return None, None, None, None, missing

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    feature_columns = joblib.load(columns_path)

    return models, scaler, encoder, feature_columns, []



def validate_upload(df, feature_columns):
    """Return a list of human-readable problems with the uploaded frame."""
    problems = []

    if TARGET_COL not in df.columns:
        problems.append(
            f"Missing the '{TARGET_COL}' column. Evaluation metrics require "
            "ground-truth labels, so the uploaded CSV must include it."
        )

    absent = [c for c in feature_columns if c not in df.columns]
    if absent:
        preview = ", ".join(absent[:5])
        suffix = f" (+{len(absent) - 5} more)" if len(absent) > 5 else ""
        problems.append(f"Missing {len(absent)} feature column(s): {preview}{suffix}")

    if df.empty:
        problems.append("The uploaded file contains no rows.")

    return problems


def prepare_features(df, feature_columns):
    return df[feature_columns].copy()


# Metrics


def safe_auc(y_true_enc, proba, n_classes):

    try:
        if n_classes == 2:
            return roc_auc_score(y_true_enc, proba[:, 1])
        return roc_auc_score(
            y_true_enc, proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        return np.nan


def compute_metrics(y_true_enc, y_pred_enc, proba, n_classes):
    average = "binary" if n_classes == 2 else "macro"

    return {
        "Accuracy": accuracy_score(y_true_enc, y_pred_enc),
        "AUC": safe_auc(y_true_enc, proba, n_classes),
        "Precision": precision_score(
            y_true_enc, y_pred_enc, average=average, zero_division=0
        ),
        "Recall": recall_score(
            y_true_enc, y_pred_enc, average=average, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true_enc, y_pred_enc, average=average, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true_enc, y_pred_enc),
    }


def evaluate(model, X_input, y_true_enc, n_classes):
    """Run one model and return its metrics, predictions and probabilities."""
    y_pred_enc = model.predict(X_input)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)
    else:
        proba = None

    if proba is None:
        metrics = compute_metrics(y_true_enc, y_pred_enc, None, n_classes)
        metrics["AUC"] = np.nan
    else:
        metrics = compute_metrics(y_true_enc, y_pred_enc, proba, n_classes)

    return metrics, y_pred_enc, proba


def plot_confusion_matrix(y_true_enc, y_pred_enc, class_names):
    cm = confusion_matrix(y_true_enc, y_pred_enc)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_metric_comparison(results_df, metric):
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = results_df.sort_values(metric, ascending=True)
    ax.barh(ordered.index, ordered[metric], color="#4C72B0")
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"{metric} across models")
    for i, value in enumerate(ordered[metric]):
        if not np.isnan(value):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig



def main():
    st.title("2025ac05706 Student Dropout & Academic Success Classifier")
    st.caption(
        "Five classification models trained on the UCI *Predict Students' "
        "Dropout and Academic Success* dataset. Upload the held-out test "
        "split to evaluate any model on unseen data."
    )

    models, scaler, encoder, feature_columns, missing = load_artifacts()

    if missing:
        st.error("Required model artifacts were not found.")
        st.write("The following files are missing:")
        for path in missing:
            st.code(path)
        st.info(
            "Run `model/train_models.ipynb` to generate the artifacts, then "
            "commit the resulting `.pkl` files to the repository."
        )
        st.stop()

    class_names = list(encoder.classes_)
    n_classes = len(class_names)


    st.sidebar.header("Configuration")

    uploaded = st.sidebar.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help=(
            f"Must contain the {len(feature_columns)} feature columns plus a "
            f"'{TARGET_COL}' column with true labels."
        ),
    )

    selected_model = st.sidebar.selectbox(
        "Select model", list(models.keys())
    )

    show_all = st.sidebar.checkbox(
        "Compare all models", value=True,
        help="Evaluate every model on the uploaded data and show a summary table.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Dataset")
    st.sidebar.write(f"Classes: {n_classes}")
    st.sidebar.write(f"Features: {len(feature_columns)}")
    st.sidebar.caption(", ".join(class_names))

    # -- Landing state -----------------------------------------------------
    if uploaded is None:
        st.info("Upload `test_data.csv` from the sidebar to begin.")

        left, right = st.columns(2)
        with left:
            st.subheader("Models available")
            for label, spec in MODEL_REGISTRY.items():
                note = "standardised input" if spec["scaled"] else "raw input"
                st.write(f"- **{label}** — {note}")
        with right:
            st.subheader("Metrics reported")
            st.write(
                "- Accuracy\n- AUC (one-vs-rest, macro)\n- Precision (macro)\n"
                "- Recall (macro)\n- F1 Score (macro)\n"
                "- Matthews Correlation Coefficient"
            )
        return

    # -- Read and validate -------------------------------------------------
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return

    problems = validate_upload(df, feature_columns)
    if problems:
        st.error("The uploaded file cannot be evaluated.")
        for problem in problems:
            st.write(f"- {problem}")
        return

    st.success(f"Loaded {len(df)} rows and {df.shape[1]} columns.")

    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(10), use_container_width=True)
        st.write("**Class distribution**")
        st.dataframe(
            df[TARGET_COL].value_counts().rename("Count").to_frame(),
            use_container_width=False,
        )

    # -- Transform ---------------------------------------------------------
    X_raw = prepare_features(df, feature_columns)

    X_scaled = scaler.transform(X_raw)

    try:
        y_true_enc = encoder.transform(df[TARGET_COL].astype(str))
    except ValueError as exc:
        st.error(
            f"The '{TARGET_COL}' column contains labels the model was not "
            f"trained on: {exc}"
        )
        return

    def features_for(label):
        return X_scaled if MODEL_REGISTRY[label]["scaled"] else X_raw

    # -- Selected model ----------------------------------------------------
    st.header(f"Results — {selected_model}")

    metrics, y_pred_enc, proba = evaluate(
        models[selected_model], features_for(selected_model),
        y_true_enc, n_classes,
    )

    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        display = "n/a" if np.isnan(value) else f"{value:.4f}"
        col.metric(name, display)

    if np.isnan(metrics["AUC"]):
        st.caption(
            "AUC is unavailable — this usually means the uploaded subset does "
            "not contain every class."
        )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion_matrix(y_true_enc, y_pred_enc, class_names))

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_true_enc, y_pred_enc,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True,
        )

    with st.expander("Per-row predictions"):
        out = pd.DataFrame({
            "True label": encoder.inverse_transform(y_true_enc),
            "Predicted label": encoder.inverse_transform(y_pred_enc),
        })
        if proba is not None:
            for i, name in enumerate(class_names):
                out[f"P({name})"] = proba[:, i].round(4)
        out["Correct"] = out["True label"] == out["Predicted label"]
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # -- All models --------------------------------------------------------
    if show_all:
        st.divider()
        st.header("Model comparison")

        rows = {}
        progress = st.progress(0.0, text="Evaluating models...")
        for i, label in enumerate(models, start=1):
            model_metrics, _, _ = evaluate(
                models[label], features_for(label), y_true_enc, n_classes
            )
            rows[label] = model_metrics
            progress.progress(i / len(models), text=f"Evaluated {label}")
        progress.empty()

        results_df = pd.DataFrame(rows).transpose()

        st.dataframe(
            results_df.style.format("{:.4f}", na_rep="n/a")
            .highlight_max(axis=0, color="#d4edda"),
            use_container_width=True,
        )

        metric_choice = st.selectbox(
            "Chart a metric", list(results_df.columns), index=0
        )
        st.pyplot(plot_metric_comparison(results_df, metric_choice))

        best = results_df["MCC"].idxmax()
        st.success(
            f"Highest MCC on this data: **{best}** "
            f"({results_df.loc[best, 'MCC']:.4f}). MCC is the most reliable "
            "single summary here because it accounts for all confusion-matrix "
            "cells and is not inflated by class imbalance."
        )

        st.download_button(
            "Download comparison table as CSV",
            results_df.to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()