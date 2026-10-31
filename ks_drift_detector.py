import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "request_rate"
]

ALPHA = 0.05


def load_dataset(file_path):
    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    return data


def detect_data_drift(reference_data, current_data, alpha=ALPHA):

    # Validate required columns
    for dataset_name, dataset in [
        ("Reference", reference_data),
        ("Production", current_data)
    ]:
        missing = set(FEATURES) - set(dataset.columns)

        if missing:
            raise ValueError(
                f"{dataset_name} dataset missing columns: {sorted(missing)}"
            )

    # Calculate Bonferroni threshold
    adjusted_alpha = alpha / len(FEATURES)

    results = []

    for feature in FEATURES:

        reference_values = pd.to_numeric(
            reference_data[feature], errors="coerce"
        ).dropna()

        current_values = pd.to_numeric(
            current_data[feature], errors="coerce"
        ).dropna()

        if len(reference_values) < 2 or len(current_values) < 2:
            raise ValueError(
                f"Feature '{feature}' needs at least 2 valid values "
                "in both datasets."
            )

        result = ks_2samp(
            reference_values,
            current_values,
            alternative="two-sided",
            method="asymp"
        )

        potential_drift = result.pvalue < adjusted_alpha

        results.append({
            "feature": feature,
            "reference_samples": len(reference_values),
            "production_samples": len(current_values),
            "ks_statistic": round(result.statistic, 4),
            "p_value": round(result.pvalue, 6),
            "adjusted_alpha": round(adjusted_alpha, 6),
            "potential_drift": potential_drift
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    base_path = Path(__file__).parent

    reference_path = base_path / "monitoring" / "reference_data.csv"
    production_path = base_path / "monitoring" / "production_data.csv"

    reference_data = load_dataset(reference_path)
    production_data = load_dataset(production_path)

    report = detect_data_drift(
        reference_data,
        production_data
    )

    print("\nPRODUCTION DATA DRIFT REPORT")
    print("=" * 100)
    print(report.to_string(index=False))

    print("\nMonitoring completed successfully.")