import pandas as pd
from pathlib import Path
import joblib

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

    X_test = pd.read_csv(results_path / "cleveland_encoded_test.csv")
    y_test = pd.read_csv(results_path / "cleveland_y_test.csv").squeeze()

    return X_test, y_test


# ============================================================
# Load Models
# ============================================================

def load_models():

    model_path = Path(__file__).resolve().parent.parent / "models"

    rf_model = joblib.load(model_path / "random_forest.pkl")

    xgb_model = joblib.load(model_path / "xgboost.pkl")

    return rf_model, xgb_model


# ============================================================
# Evaluate One Model
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

    return metrics, y_pred, y_prob


# ============================================================
# Save Comparison Table
# ============================================================

def save_metrics(rf_metrics, xgb_metrics):

    comparison = pd.DataFrame({

        "Metric": rf_metrics.keys(),

        "Random Forest": rf_metrics.values(),

        "XGBoost": xgb_metrics.values()

    })

    results_path = Path(__file__).resolve().parent.parent / "results"

    comparison.to_csv(

        results_path / "model_comparison.csv",

        index=False

    )

    print("\nModel comparison saved successfully.")

    print("\n", comparison)

    return comparison


# ============================================================
# Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(model_name, y_test, y_pred):

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    figures_path.mkdir(exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(

        figures_path / f"{model_name.lower()}_confusion_matrix.png"

    )

    plt.close()


# ============================================================
# Plot ROC Curves
# ============================================================

def plot_roc(

        rf_model,

        xgb_model,

        X_test,

        y_test

):

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    plt.figure(figsize=(8,6))

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

        name="XGBoost"

    )

    plt.title("ROC Curve Comparison")

    plt.savefig(

        figures_path / "roc_curve.png"

    )

    plt.close()
    # ============================================================
# Select Best Model
# ============================================================

def select_best_model(rf_metrics, xgb_metrics):

    print("\n" + "=" * 60)
    print("BEST MODEL SELECTION")
    print("=" * 60)

    if xgb_metrics["Accuracy"] > rf_metrics["Accuracy"]:

        print("Best Model : XGBoost")
        print(f"Accuracy   : {xgb_metrics['Accuracy']:.4f}")

        return "XGBoost"

    elif rf_metrics["Accuracy"] > xgb_metrics["Accuracy"]:

        print("Best Model : Random Forest")
        print(f"Accuracy   : {rf_metrics['Accuracy']:.4f}")

        return "Random Forest"

    else:

        print("Both models achieved the same accuracy.")

        return "Tie"


# ============================================================
# Main
# ============================================================

def main():

    print("="*70)

    print("MODEL EVALUATION")

    print("="*70)

    X_test, y_test = load_test_data()

    rf_model, xgb_model = load_models()

    rf_metrics, rf_pred, rf_prob = evaluate(

        rf_model,

        X_test,

        y_test

    )

    xgb_metrics, xgb_pred, xgb_prob = evaluate(

        xgb_model,

        X_test,

        y_test

    )

    save_metrics(

        rf_metrics,

        xgb_metrics

    )
# ============================
    # ADD THESE 4 LINES
    # ============================

    best_model = select_best_model(

        rf_metrics,

        xgb_metrics

    )

    plot_confusion_matrix(

        "RF",

        y_test,

        rf_pred

    )
    plot_confusion_matrix(

        "RF",

        y_test,

        rf_pred

    )

    plot_confusion_matrix(

        "XGB",

        y_test,

        xgb_pred

    )

    plot_roc(

        rf_model,

        xgb_model,

        X_test,

        y_test

    )

    print("\nEvaluation Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()