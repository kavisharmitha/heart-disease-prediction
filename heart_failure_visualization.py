import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# Load Model Comparison Results
# ============================================================

def load_results():

    results_path = Path(__file__).resolve().parent.parent / "results"

    df = pd.read_csv(
        results_path / "heart_failure_model_comparison.csv"
    )

    return df


# ============================================================
# Accuracy Comparison
# ============================================================

def accuracy_plot(df):

    accuracy = df[df["Metric"] == "Accuracy"]

    plt.figure(figsize=(6,5))

    plt.bar(

        ["Random Forest", "XGBoost"],

        [

            accuracy["Random Forest"].values[0],

            accuracy["XGBoost"].values[0]

        ]

    )

    plt.ylabel("Accuracy")

    plt.title("Heart Failure Model Accuracy Comparison")

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    figures_path.mkdir(exist_ok=True)

    plt.savefig(

        figures_path / "heart_failure_accuracy_comparison.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# ============================================================
# Metric Comparison
# ============================================================

def metrics_plot(df):

    plt.figure(figsize=(8,5))

    metrics = df["Metric"]

    rf = df["Random Forest"]

    xgb = df["XGBoost"]

    x = range(len(metrics))

    width = 0.35

    plt.bar(

        [i - width/2 for i in x],

        rf,

        width,

        label="Random Forest"

    )

    plt.bar(

        [i + width/2 for i in x],

        xgb,

        width,

        label="XGBoost"

    )

    plt.xticks(x, metrics)

    plt.ylabel("Score")

    plt.title("Heart Failure Performance Comparison")

    plt.legend()

    figures_path = Path(__file__).resolve().parent.parent / "figures"

    plt.savefig(

        figures_path / "heart_failure_metrics_comparison.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("HEART FAILURE VISUALIZATION")
    print("=" * 70)

    df = load_results()

    accuracy_plot(df)

    metrics_plot(df)

    print("\nVisualization Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()