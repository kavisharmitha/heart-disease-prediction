import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Load Encoded Features and Target Labels
# ============================================================

def load_data():

    results_path = Path(__file__).resolve().parent.parent / "results"

    X_train = pd.read_csv(results_path / "cleveland_encoded_train.csv")
    X_test = pd.read_csv(results_path / "cleveland_encoded_test.csv")

    y_train = pd.read_csv(results_path / "cleveland_y_train.csv").squeeze()
    y_test = pd.read_csv(results_path / "cleveland_y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


# ============================================================
# Train Random Forest
# ============================================================

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if len(model.classes_) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = "Not Applicable"

    print("\n" + "=" * 60)
    print("RANDOM FOREST RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    return y_pred


# ============================================================
# Save Model
# ============================================================

def save_model(model):

    model_path = Path(__file__).resolve().parent.parent / "models"

    model_path.mkdir(exist_ok=True)

    joblib.dump(

        model,

        model_path / "random_forest.pkl"

    )

    print("\nRandom Forest Model Saved Successfully.")


# ============================================================
# Save Predictions
# ============================================================

def save_predictions(y_pred):

    results_path = Path(__file__).resolve().parent.parent / "results"

    pd.DataFrame(

        y_pred,

        columns=["Predicted"]

    ).to_csv(

        results_path / "rf_predictions.csv",

        index=False

    )

    print("Predictions Saved Successfully.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    model = train_random_forest(

        X_train,

        y_train

    )

    y_pred = evaluate_model(

        model,

        X_test,

        y_test

    )

    save_model(model)

    save_predictions(y_pred)

    print("\nRandom Forest Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()