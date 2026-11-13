import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# Load Model
# ============================================================

def load_model():

    model_path = Path(__file__).resolve().parent.parent / "models"

    model = joblib.load(model_path / "xgboost.pkl")

    return model


# ============================================================
# Load Encoded Test Data
# ============================================================

def load_data():

    results_path = Path(__file__).resolve().parent.parent / "results"

    X_test = pd.read_csv(results_path / "cleveland_encoded_test.csv")

    return X_test


# ============================================================
# Rename Features
# ============================================================

def rename_features(X):

    feature_names = []

    for i in range(X.shape[1]):

        feature_names.append(f"Latent_Feature_{i+1}")

    X.columns = feature_names

    return X


# ============================================================
# SHAP Analysis
# ============================================================

def perform_shap_analysis(model, X):

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    return shap_values


# ============================================================
# Save SHAP Summary Plot
# ============================================================

def save_summary_plot(shap_values, X):

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    figures_path.mkdir(exist_ok=True)

    plt.figure(figsize=(10,6))

    shap.summary_plot(

        shap_values,

        X,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        figures_path / "shap_summary.png",

        dpi=300

    )

    plt.close()

    print("SHAP Summary Plot Saved.")


# ============================================================
# Save SHAP Bar Plot
# ============================================================

def save_bar_plot(shap_values, X):

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    plt.figure(figsize=(10,6))

    shap.summary_plot(

        shap_values,

        X,

        plot_type="bar",

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        figures_path / "shap_bar.png",

        dpi=300

    )

    plt.close()

    print("SHAP Bar Plot Saved.")


# ============================================================
# Print Feature Importance
# ============================================================

def print_feature_importance(shap_values, X):

    importance = abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance": importance

    })

    importance_df = importance_df.sort_values(

        by="Importance",

        ascending=False

    )

    print("\n" + "="*60)

    print("FEATURE IMPORTANCE")

    print("="*60)

    print(importance_df)

    results_path = Path(__file__).resolve().parent.parent / "results"

    importance_df.to_csv(

        results_path / "shap_feature_importance.csv",

        index=False

    )


# ============================================================
# Main
# ============================================================

def main():

    print("="*70)
    print("SHAP EXPLAINABILITY")
    print("="*70)

    model = load_model()

    X = load_data()

    X = rename_features(X)

    shap_values = perform_shap_analysis(

        model,

        X

    )

    save_summary_plot(

        shap_values,

        X

    )

    save_bar_plot(

        shap_values,

        X

    )

    print_feature_importance(

        shap_values,

        X

    )

    print("\nExplainability Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()