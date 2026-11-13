import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# Load Heart Failure Dataset
# ============================================================

def load_dataset():

    data_path = Path(__file__).resolve().parent.parent / "data"

    df = pd.read_csv(data_path / "heart_failure.csv")

    return df


# ============================================================
# Dataset Information
# ============================================================

def dataset_info(df):

    print("=" * 70)
    print("HEART FAILURE DATASET")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())


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

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    if numerical_columns:

        num_imputer = SimpleImputer(strategy="median")

        df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

    if categorical_columns:

        cat_imputer = SimpleImputer(strategy="most_frequent")

        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    print("Missing Values Handled Successfully.")

    return df


# ============================================================
# Encode Categorical Columns
# ============================================================

def encode_data(df):

    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    if len(categorical_columns) == 0:

        print("No Categorical Columns Found.")

        return df

    for col in categorical_columns:

        df[col] = encoder.fit_transform(df[col])

    print("Categorical Variables Encoded Successfully.")

    return df


# ============================================================
# Normalize Features
# ============================================================

def normalize_data(df):

    scaler = StandardScaler()

    X = df.drop(columns=["DEATH_EVENT"])

    df[X.columns] = scaler.fit_transform(X)

    print("StandardScaler Applied Successfully.")

    return df


# ============================================================
# Split Features and Target
# ============================================================

def split_features_target(df):

    X = df.drop(columns=["DEATH_EVENT"])

    y = df["DEATH_EVENT"]

    print("Features and Target Separated.")

    return X, y


# ============================================================
# Train-Test Split
# ============================================================

def split_train_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("Train-Test Split Completed.")

    print(f"Training Samples : {len(X_train)}")

    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# Save Files
# ============================================================

def save_files(df, X_train, X_test, y_train, y_test):

    results = Path(__file__).resolve().parent.parent / "results"

    df.to_csv(results / "heart_failure_preprocessed.csv", index=False)

    X_train.to_csv(results / "heart_failure_X_train.csv", index=False)

    X_test.to_csv(results / "heart_failure_X_test.csv", index=False)

    y_train.to_csv(results / "heart_failure_y_train.csv", index=False)

    y_test.to_csv(results / "heart_failure_y_test.csv", index=False)

    print("Files Saved Successfully.")


# ============================================================
# Main
# ============================================================

def main():

    print("="*70)
    print("HEART FAILURE PREPROCESSING")
    print("="*70)

    df = load_dataset()

    dataset_info(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = encode_data(df)

    df = normalize_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    save_files(df, X_train, X_test, y_train, y_test)

    print("\nHeart Failure preprocessing completed successfully.")


# ============================================================

if __name__ == "__main__":

    main()