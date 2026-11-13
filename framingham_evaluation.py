import pandas as pd
import joblib
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ============================================================
# Load Test Data
# ============================================================

def load_test_data():

    results_path = Path(__file__).resolve().parent.parent / "results"

    X_test = pd.read_csv(
        results_path / "framingham_encoded_test.csv"
    )

    y_test = pd.read_csv(
        results_path / "framingham_y_test.csv"
    ).squeeze()

    return X_test, y_test


# ============================================================
# Load Models
# ============================================================

def load_models():

    model_path = Path(__file__).resolve().parent.parent / "models"

    rf_model = joblib.load(
        model_path / "framingham_random_forest.pkl"
    )

    xgb_model = joblib.load(
        model_path / "framingham_xgboost.pkl"
    )

    return rf_model, xgb_model


# ============================================================
# Evaluate Model
# ============================================================

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {

        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)

    }

    return metrics, y_pred


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(rf_metrics, xgb_metrics):

    comparison = pd.DataFrame({

        "Metric": list(rf_metrics.keys()),
        "Random Forest": list(rf_metrics.values()),
        "XGBoost": list(xgb_metrics.values())

    })

    results_path = Path(__file__).resolve().parent.parent / "results"

    comparison.to_csv(

        results_path / "framingham_model_comparison.csv",

        index=False

    )

    print("\nModel comparison saved successfully.\n")

    print(comparison)


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(model_name, y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot(cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    figures_path.mkdir(exist_ok=True)

    plt.savefig(

        figures_path / f"{model_name}_framingham_confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# ============================================================
# ROC Curve
# ============================================================

def plot_roc(rf_model, xgb_model, X_test, y_test):

    plt.figure(figsize=(6,6))

    RocCurveDisplay.from_estimator(

        rf_model,

        X_test,

        y_test,

        name="Random Forest"

    )

    RocCurveDisplay.from_estimator(

        xgb_model,

        X_test,

        y_test,

        name="XGBoost",

        ax=plt.gca()

    )

    plt.title("Framingham ROC Curve")

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    plt.savefig(

        figures_path / "framingham_roc_curve.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# ============================================================
# Best Model Selection
# ============================================================

def select_best_model(rf_metrics, xgb_metrics):

    print("\n" + "=" * 60)
    print("BEST MODEL SELECTION")
    print("=" * 60)

    if xgb_metrics["Accuracy"] > rf_metrics["Accuracy"]:

        print("Best Model : XGBoost")
        print(f"Accuracy   : {xgb_metrics['Accuracy']:.4f}")

    elif rf_metrics["Accuracy"] > xgb_metrics["Accuracy"]:

        print("Best Model : Random Forest")
        print(f"Accuracy   : {rf_metrics['Accuracy']:.4f}")

    else:

        print("Both models achieved the same accuracy.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FRAMINGHAM MODEL EVALUATION")
    print("=" * 70)

    X_test, y_test = load_test_data()

    rf_model, xgb_model = load_models()

    rf_metrics, rf_pred = evaluate(

        rf_model,

        X_test,

        y_test

    )

    xgb_metrics, xgb_pred = evaluate(

        xgb_model,

        X_test,

        y_test

    )

    save_metrics(

        rf_metrics,

        xgb_metrics

    )

    plot_confusion_matrix(

        "Random_Forest",

        y_test,

        rf_pred

    )

    plot_confusion_matrix(

        "XGBoost",

        y_test,

        xgb_pred

    )

    plot_roc(

        rf_model,

        xgb_model,

        X_test,

        y_test

    )

    select_best_model(

        rf_metrics,

        xgb_metrics

    )

    print("\nEvaluation Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()