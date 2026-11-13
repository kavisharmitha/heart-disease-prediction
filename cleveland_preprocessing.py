# ===============================================
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# Load Cleveland Harmonized Dataset
# ============================================================

def load_dataset():

    file_path = Path(__file__).resolve().parent.parent / "results" / "cleveland_harmonized.csv"

    if not file_path.exists():
        print("ERROR : cleveland_harmonized.csv not found.")
        return None

    df = pd.read_csv(file_path)

    return df


# ============================================================
# Dataset Information
# ============================================================

def dataset_info(df):

    print("\n" + "=" * 70)
    print("CLEVELAND DATASET")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nData Types")
    print(df.dtypes)


# ============================================================
# Remove Duplicates
# ============================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"\nDuplicates Removed : {before-after}")

    return df


# ============================================================
# Handle Missing Values
# ============================================================

def handle_missing_values(df):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if "target" in numeric_cols:
        numeric_cols.remove("target")

    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if len(numeric_cols) > 0:

        num_imputer = SimpleImputer(strategy="median")

        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if len(categorical_cols) > 0:

        cat_imputer = SimpleImputer(strategy="most_frequent")

        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    print("Missing Values Handled Successfully.")

    return df


# ============================================================
# Encode Categorical Features
# ============================================================

def encode_categorical(df):

    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) == 0:

        print("No Categorical Columns Found.")

        return df

    encoder = LabelEncoder()

    for col in categorical_cols:

        df[col] = encoder.fit_transform(df[col])

    print("Categorical Encoding Completed.")

    return df


# ============================================================
# Feature Scaling
# ============================================================

def scale_features(df):

    scaler = StandardScaler()

    feature_cols = df.drop(columns=["target"]).columns

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("StandardScaler Applied Successfully.")

    return df


# ============================================================
# Save Preprocessed Dataset
# ============================================================

def save_dataset(df):

    output_path = Path(__file__).resolve().parent.parent / "results"

    df.to_csv(output_path / "cleveland_preprocessed.csv", index=False)

    print("Preprocessed Dataset Saved Successfully.")


# ============================================================
# Train Test Split
# ============================================================

def split_dataset(df):

    X = df.drop(columns=["target"])

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,
        random_state=42,
        stratify=y

    )

    output_path = Path(__file__).resolve().parent.parent / "results"

    X_train.to_csv(output_path / "cleveland_X_train.csv", index=False)
    X_test.to_csv(output_path / "cleveland_X_test.csv", index=False)

    y_train.to_frame().to_csv(output_path / "cleveland_y_train.csv", index=False)
    y_test.to_frame().to_csv(output_path / "cleveland_y_test.csv", index=False)

    print("\nTrain-Test Split Completed.")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")


# ============================================================
# Main Function
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("CLEVELAND PREPROCESSING PIPELINE")
    print("=" * 70)

    df = load_dataset()

    if df is None:
        return

    if df.empty:
        print("Dataset is Empty.")
        return

    dataset_info(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = encode_categorical(df)

    df = scale_features(df)

    save_dataset(df)

    split_dataset(df)

    print("\n" + "=" * 70)
    print("CLEVELAND PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()