
from datetime import datetime
import pandas as pd
from pathlib import Path

from data_quality import validate_data
from scipy.stats import ks_2samp

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


def run_drift_detection(reference_data, production_data):

    adjusted_alpha = ALPHA / len(FEATURES)
    results = []

    for feature in FEATURES:

        reference_values = pd.to_numeric(
            reference_data[feature], errors="coerce"
        ).dropna()

        production_values = pd.to_numeric(
            production_data[feature], errors="coerce"
        ).dropna()

        result = ks_2samp(
            reference_values,
            production_values,
            alternative="two-sided",
            method="asymp"
        )

        results.append({
            "feature": feature,
            "ks_statistic": round(result.statistic, 4),
            "p_value": round(result.pvalue, 6),
            "adjusted_alpha": round(adjusted_alpha, 6),
            "potential_drift": result.pvalue < adjusted_alpha
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    base_path = Path(__file__).parent

    reference_path = base_path / "reference_data.csv"
    production_path = base_path / "production_data.csv"

    reference_data = load_dataset(reference_path)
    production_data = load_dataset(production_path)

    print("\nSTEP 1: DATA QUALITY VALIDATION")
    print("=" * 80)

    quality_report = validate_data(production_data)

    print(quality_report.to_string(index=False))

    # Quality gate
    feature_failures = quality_report[
        quality_report["feature"] != "ALL_RECORDS"
    ]

    if (feature_failures["status"] != "VALID").any():
        print("\nDATA QUALITY FAILED")
        print("Drift detection stopped.")
        raise SystemExit(1)

    print("\nDATA QUALITY PASSED")

    print("\nSTEP 2: DATA DRIFT DETECTION")
    print("=" * 80)

    drift_report = run_drift_detection(
        reference_data,
        production_data
    )

    print(drift_report.to_string(index=False))
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add quality status to drift report
    quality_status = quality_report.set_index("feature")["status"]

    drift_report["quality_status"] = drift_report["feature"].map(
        quality_status
    )

    # Generate report directory
    report_directory = base_path / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)

    # Generate report filename
    report_filename = f"monitoring_report_{timestamp}.csv"

    report_path = report_directory / report_filename

    # Save monitoring report
    drift_report.to_csv(report_path, index=False)

    print("\nMONITORING REPORT SAVED")
    print(report_path)

    print("\nMONITORING COMPLETED")
