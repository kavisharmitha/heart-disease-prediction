import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import zscore

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():

    data_path = Path(__file__).resolve().parent.parent / "data"

    df = pd.read_csv(data_path / "framingham.csv")

    return df


# ============================================================
# Select Features
# ============================================================

def select_features(df):

    columns = [

        "AGE",
        "SEX",
        "TOTCHOL",
        "SYSBP",
        "DIABP",
        "CURSMOKE",
        "CIGPDAY",
        "BMI",
        "DIABETES",
        "BPMEDS",
        "HEARTRTE",
        "GLUCOSE",
        "HDLC",
        "LDLC",
        "educ",
        "CVD"

    ]

    df = df[columns]

    return df


# ============================================================
# Dataset Information
# ============================================================

def dataset_info(df):

    print("\n" + "=" * 70)
    print("FRAMINGHAM DATASET")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

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

    numerical_columns = df.columns.drop("CVD")

    imputer = SimpleImputer(strategy="median")

    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    print("\nMissing Values Handled Successfully.")

    return df


# ============================================================
# Remove Outliers (Z-score)
# ============================================================

def remove_outliers(df):

    numerical_columns = df.columns.drop("CVD")

    z_scores = np.abs(zscore(df[numerical_columns]))

    df = df[(z_scores < 3).all(axis=1)]

    print(f"Dataset Shape After Outlier Removal : {df.shape}")

    return df


# ============================================================
# Feature Scaling
# ============================================================

def scale_features(df):

    X = df.drop(columns=["CVD"])

    y = df["CVD"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(

        X_scaled,

        columns=X.columns

    )

    return X_scaled, y


# ============================================================
# Train-Test Split
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    return X_train, X_test, y_train, y_test


# ============================================================
# Save Files
# ============================================================

def save_files(df, X_train, X_test, y_train, y_test):

    results_path = Path(__file__).resolve().parent.parent / "results"

    results_path.mkdir(exist_ok=True)

    df.to_csv(

        results_path / "framingham_preprocessed.csv",

        index=False

    )

    X_train.to_csv(

        results_path / "framingham_X_train.csv",

        index=False

    )

    X_test.to_csv(

        results_path / "framingham_X_test.csv",

        index=False

    )

    y_train.to_csv(

        results_path / "framingham_y_train.csv",

        index=False

    )

    y_test.to_csv(

        results_path / "framingham_y_test.csv",

        index=False

    )

    print("\nPreprocessed Files Saved Successfully.")


# ============================================================
# Main
# ============================================================

def main():

    df = load_dataset()

    df = select_features(df)

    dataset_info(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = remove_outliers(df)

    X, y = scale_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    save_files(

        df,

        X_train,

        X_test,

        y_train,

        y_test

    )

    print("\nFramingham Preprocessing Completed Successfully.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()