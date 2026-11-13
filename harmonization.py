# ============================================================
# Clinical Feature Harmonization Framework (CFHF)
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# Load Datasets
# ============================================================

def load_datasets():

    data_path = Path(__file__).resolve().parent.parent / "data"

    datasets = {

        "Cleveland": pd.read_csv(data_path / "cleveland.csv"),

        "Framingham": pd.read_csv(data_path / "framingham.csv"),

        "Heart Failure": pd.read_csv(data_path / "heart_failure.csv")

    }

    return datasets


# ============================================================
# Standardize Column Names
# ============================================================

def standardize_column_names(datasets):

    column_mapping = {

        # -------------------------
        # Age
        # -------------------------
        "Age": "age",

        # -------------------------
        # Gender
        # -------------------------
        "Gender": "sex",
        "gender": "sex",
        "male": "sex",

        # -------------------------
        # Blood Pressure
        # -------------------------
        "trestbps": "blood_pressure",
        "RestingBP": "blood_pressure",
        "resting_bp": "blood_pressure",
        "sysBP": "blood_pressure",

        # -------------------------
        # Cholesterol
        # -------------------------
        "chol": "cholesterol",
        "totChol": "cholesterol",
        "Cholesterol": "cholesterol",

        # -------------------------
        # Heart Rate
        # -------------------------
        "thalach": "max_heart_rate",
        "MaxHR": "max_heart_rate",

        # -------------------------
        # Blood Sugar
        # -------------------------
        "fbs": "fasting_blood_sugar",
        "glucose": "glucose",

        # -------------------------
        # Target Variable
        # -------------------------
        "target": "target",
        "HeartDisease": "target",
        "TenYearCHD": "target",
        "DEATH_EVENT": "target"

    }

    standardized = {}

    for name, df in datasets.items():

        df = df.rename(columns=column_mapping)

        standardized[name] = df

    return standardized
# ============================================================
# Dataset Profiling
# ============================================================

def profile_datasets(datasets):

    print("\n" + "=" * 70)
    print("DATASET PROFILE")
    print("=" * 70)

    for name, df in datasets.items():

        print(f"\nDataset : {name}")
        print("-" * 50)

        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        print("\nColumn Names")
        print(df.columns.tolist())

        print("\nData Types")
        print(df.dtypes)

        print("\nMissing Values")
        print(df.isnull().sum())

        print("\nFirst Five Rows")
        print(df.head())

        print("\n" + "=" * 70)


# ============================================================
# Compare Features Across Datasets
# ============================================================

def compare_features(datasets):

    all_features = set()

    # Collect every unique column
    for df in datasets.values():
        all_features.update(df.columns)

    comparison = pd.DataFrame(index=sorted(all_features))

    # Check feature availability
    for name, df in datasets.items():

        comparison[name] = comparison.index.isin(df.columns)

    comparison = comparison.replace({
        True: "✓",
        False: "✗"
    })

    return comparison


# ============================================================
# Save Feature Comparison Report
# ============================================================

def save_feature_report(comparison):

    results_path = Path(__file__).resolve().parent.parent / "results"

    results_path.mkdir(exist_ok=True)

    comparison.to_csv(results_path / "feature_comparison.csv")

    print("\nFeature comparison report saved successfully.")
    # ============================================================
# Clinical Feature Mapping
# ============================================================

def create_feature_mapping():

    feature_mapping = {

        "age": ["age"],

        "sex": ["sex"],

        "blood_pressure": ["blood_pressure"],

        "cholesterol": ["cholesterol"],

        "max_heart_rate": ["max_heart_rate"],

        "fasting_blood_sugar": ["fasting_blood_sugar", "glucose"],

        "target": ["target"]

    }

    return feature_mapping


# ============================================================
# Create Unified Feature Schema
# ============================================================

def create_unified_schema(feature_mapping):

    schema = list(feature_mapping.keys())

    return schema


# ============================================================
# Harmonize a Single Dataset
# ============================================================

def harmonize_dataset(df, feature_mapping, schema):

    harmonized_df = pd.DataFrame()

    for unified_feature in schema:

        found = False

        for column in feature_mapping[unified_feature]:

            if column in df.columns:

                harmonized_df[unified_feature] = df[column]

                found = True

                break

        if not found:

            harmonized_df[unified_feature] = pd.NA

    return harmonized_df


# ============================================================
# Harmonize All Datasets
# ============================================================

def harmonize_all_datasets(datasets):

    feature_mapping = create_feature_mapping()

    schema = create_unified_schema(feature_mapping)

    harmonized = {}

    for name, df in datasets.items():

        harmonized[name] = harmonize_dataset(
            df,
            feature_mapping,
            schema
        )

    return harmonized
# ============================================================
# Save Harmonized Datasets
# ============================================================

def save_harmonized_datasets(harmonized):

    results_path = Path(__file__).resolve().parent.parent / "results"

    results_path.mkdir(exist_ok=True)

    for name, df in harmonized.items():

        filename = name.lower().replace(" ", "_") + "_harmonized.csv"

        df.to_csv(results_path / filename, index=False)

    print("\nHarmonized datasets saved successfully!")


# ============================================================
# Save Unified Schema
# ============================================================

def save_unified_schema(schema):

    results_path = Path(__file__).resolve().parent.parent / "results"

    schema_df = pd.DataFrame({
        "Unified_Features": schema
    })

    schema_df.to_csv(results_path / "unified_schema.csv", index=False)

    print("Unified schema saved successfully!")


# ============================================================
# Main Function
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("Clinical Feature Harmonization Framework (CFHF)")
    print("=" * 70)

    # -------------------------------------------------------
    # Step 1: Load datasets
    # -------------------------------------------------------

    datasets = load_datasets()

    print("\nDatasets loaded successfully!")

    # -------------------------------------------------------
    # Step 2: Standardize column names
    # -------------------------------------------------------

    datasets = standardize_column_names(datasets)

    print("Column names standardized.")

    # -------------------------------------------------------
    # Step 3: Dataset profiling
    # -------------------------------------------------------

    profile_datasets(datasets)

    # -------------------------------------------------------
    # Step 4: Feature comparison
    # -------------------------------------------------------

    comparison = compare_features(datasets)

    print("\nFeature Comparison Table\n")

    print(comparison)

    save_feature_report(comparison)

    # -------------------------------------------------------
    # Step 5: Clinical feature mapping
    # -------------------------------------------------------

    feature_mapping = create_feature_mapping()

    # -------------------------------------------------------
    # Step 6: Unified Schema
    # -------------------------------------------------------

    schema = create_unified_schema(feature_mapping)

    save_unified_schema(schema)

        # -------------------------------------------------------
    # Step 7: Harmonize datasets
    # -------------------------------------------------------

    harmonized = harmonize_all_datasets(datasets)

    save_harmonized_datasets(harmonized)

    # -------------------------------------------------------
    # Step 8: Harmonization Quality Assessment
    # -------------------------------------------------------

    report = calculate_harmonization_score(harmonized)

    print("\nHarmonization Quality Report\n")
    print(report)

    save_harmonization_report(report)

    print("\nClinical Feature Harmonization Completed Successfully!")
    print("=" * 70)
    # ============================================================
# Harmonization Quality Assessment
# ============================================================

def calculate_harmonization_score(harmonized):

    report = []

    for dataset_name, df in harmonized.items():

        total_features = df.shape[1]

        available_features = df.notna().any().sum()

        missing_features = total_features - available_features

        coverage = (available_features / total_features) * 100

        report.append({
            "Dataset": dataset_name,
            "Total Features": total_features,
            "Available Features": available_features,
            "Missing Features": missing_features,
            "Coverage (%)": round(coverage, 2)
        })

    report_df = pd.DataFrame(report)

    return report_df
# ============================================================
# Save Harmonization Report
# ============================================================

def save_harmonization_report(report):

    results_path = Path(__file__).resolve().parent.parent / "results"

    report.to_csv(
        results_path / "harmonization_report.csv",
        index=False
    )

    print("\nHarmonization report saved successfully!")


# ============================================================

if __name__ == "__main__":
    main()