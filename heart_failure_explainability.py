import pandas as pd
import joblib
from pathlib import Path

import shap
import matplotlib.pyplot as plt


# ============================================================
# Load Model and Data
# ============================================================

def load_model_and_data():

    results_path = Path(__file__).resolve().parent.parent / "results"
    models_path = Path(__file__).resolve().parent.parent / "models"

    model = joblib.load(
        models_path / "heart_failure_random_forest.pkl"
    )

    X_train = pd.read_csv(
        results_path / "heart_failure_encoded_train.csv"
    )

    return model, X_train


# ============================================================
# SHAP Explainability
# ============================================================

def generate_shap(model, X_train):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train)

    figures_path = Path(__file__).resolve().parent.parent / "figures"
    figures_path.mkdir(exist_ok=True)

    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_train,

        show=False

    )

    plt.savefig(

        figures_path / "heart_failure_shap_summary.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_train,

        plot_type="bar",

        show=False

    )

    plt.savefig(

        figures_path / "heart_failure_shap_bar.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("\nSHAP figures saved successfully.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("HEART FAILURE SHAP EXPLAINABILITY")
    print("=" * 70)

    model, X_train = load_model_and_data()

    generate_shap(

        model,

        X_train

    )

    print("\nExplainability Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()