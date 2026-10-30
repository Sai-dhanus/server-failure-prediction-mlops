
import pandas as pd
from scipy.stats import ks_2samp


FEATURES = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "request_rate",
]


def detect_data_drift(reference_data, current_data, alpha=0.05):
    """
    Compare reference and current feature distributions
    using the two-sample Kolmogorov-Smirnov test.

    alpha is the chosen significance level.
    """

    missing_reference = set(FEATURES) - set(reference_data.columns)
    missing_current = set(FEATURES) - set(current_data.columns)

    if missing_reference:
        raise ValueError(
            f"Missing reference features: {sorted(missing_reference)}"
        )

    if missing_current:
        raise ValueError(
            f"Missing current features: {sorted(missing_current)}"
        )

    results = []

    for feature in FEATURES:

        reference_values = reference_data[feature].dropna()
        current_values = current_data[feature].dropna()

        if len(reference_values) < 2 or len(current_values) < 2:
            raise ValueError(
                f"Feature '{feature}' needs at least two "
                "non-missing values in each dataset."
            )

        result = ks_2samp(
            reference_values,
            current_values,
            alternative="two-sided",
            method="auto",
        )

        results.append({
            "feature": feature,
            "reference_samples": len(reference_values),
            "current_samples": len(current_values),
            "ks_statistic": round(result.statistic, 4),
            "p_value": round(result.pvalue, 6),
            "potential_drift": result.pvalue < alpha,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    # Illustrative reference data for practice.
    # Replace these values with representative project data
    # when integrating the detector into your actual workflow.
    reference_data = pd.DataFrame({
        "cpu_usage": [25, 35, 40, 45, 55],
        "memory_usage": [30, 40, 45, 50, 55],
        "disk_usage": [35, 40, 45, 50, 55],
        "request_rate": [100, 150, 200, 250, 300],
    })

    # Illustrative production-like data.
    current_data = pd.DataFrame({
        "cpu_usage": [82, 85, 90, 95, 98],
        "memory_usage": [65, 70, 75, 80, 85],
        "disk_usage": [40, 42, 45, 48, 50],
        "request_rate": [700, 750, 800, 850, 900],
    })

    report = detect_data_drift(
        reference_data,
        current_data,
        alpha=0.05,
    )

    print("\nKS-BASED DATA DRIFT REPORT")
    print("=" * 85)
    print(report.to_string(index=False))

    print("\nNote: Potential drift does not automatically mean")
    print("that model performance has degraded.")