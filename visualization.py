import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# Load Model Comparison Results
# ============================================================

def load_results():

    results_path = Path(__file__).resolve().parent.parent / "results"

    df = pd.read_csv(results_path / "model_comparison.csv")

    return df


# ============================================================
# Create Figures Folder
# ============================================================

def create_figures_folder():

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    figures_path.mkdir(exist_ok=True)

    return figures_path


# ============================================================
# Plot Individual Metric
# ============================================================

def plot_metric(df, metric_name, filename, figures_path):

    row = df[df["Metric"] == metric_name]

    rf = row["Random Forest"].values[0]
    xgb = row["XGBoost"].values[0]

    plt.figure(figsize=(6,5))

    plt.bar(
        ["Random Forest", "XGBoost"],
        [rf, xgb]
    )

    plt.ylim(0, 1)

    plt.ylabel(metric_name)

    plt.title(f"{metric_name} Comparison")

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(

        figures_path / filename,

        dpi=300

    )

    plt.close()

    print(f"{filename} Saved Successfully.")


# ============================================================
# Plot Overall Comparison
# ============================================================

def plot_all_metrics(df, figures_path):

    metrics = df["Metric"]

    rf_scores = df["Random Forest"]

    xgb_scores = df["XGBoost"]

    x = range(len(metrics))

    width = 0.35

    plt.figure(figsize=(10,6))

    plt.bar(

        [i-width/2 for i in x],

        rf_scores,

        width,

        label="Random Forest"

    )

    plt.bar(

        [i+width/2 for i in x],

        xgb_scores,

        width,

        label="XGBoost"

    )

    plt.xticks(x, metrics)

    plt.ylabel("Score")

    plt.title("Overall Model Performance Comparison")

    plt.legend()

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(

        figures_path / "overall_metrics_comparison.png",

        dpi=300

    )

    plt.close()

    print("overall_metrics_comparison.png Saved Successfully.")


# ============================================================
# Main
# ============================================================

def main():

    print("="*70)

    print("MODEL VISUALIZATION")

    print("="*70)

    df = load_results()

    figures_path = create_figures_folder()

    plot_metric(

        df,

        "Accuracy",

        "accuracy_comparison.png",

        figures_path

    )

    plot_metric(

        df,

        "Precision",

        "precision_comparison.png",

        figures_path

    )

    plot_metric(

        df,

        "Recall",

        "recall_comparison.png",

        figures_path

    )

    plot_metric(

        df,

        "F1-score",

        "f1score_comparison.png",

        figures_path

    )

    plot_metric(

        df,

        "ROC-AUC",

        "roc_auc_comparison.png",

        figures_path

    )

    plot_all_metrics(

        df,

        figures_path

    )

    print("\nVisualization Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()