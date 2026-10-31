import pandas as pd
from scipy.stats import ks_2samp

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "request_rate"
]


def detect_data_drift(reference_data, current_data, alpha=0.05):

    # Step 1: Validate columns
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

    # Step 2: Calculate Bonferroni threshold
    number_of_tests = len(FEATURES)
    adjusted_alpha = alpha / number_of_tests

    results = []

    # Step 3: Perform KS test for each feature
    for feature in FEATURES:

        reference_values = reference_data[feature].dropna()
        current_values = current_data[feature].dropna()

        if len(reference_values) < 2 or len(current_values) < 2:
            raise ValueError(
                f"Feature '{feature}' needs at least two non-missing "
                "values in each dataset."
            )

        result = ks_2samp(
            reference_values,
            current_values,
            alternative="two-sided",
            method="auto"
        )

        # Step 4: Compare p-value against adjusted threshold
        potential_drift = result.pvalue < adjusted_alpha

        results.append({
            "feature": feature,
            "reference_samples": len(reference_values),
            "current_samples": len(current_values),
            "ks_statistic": round(result.statistic, 4),
            "p_value": round(result.pvalue, 6),
            "adjusted_alpha": round(adjusted_alpha, 6),
            "potential_drift": potential_drift
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    reference_data = pd.DataFrame({
        "cpu_usage": [25, 35, 40, 45, 55],
        "memory_usage": [30, 40, 45, 50, 55],
        "disk_usage": [35, 40, 45, 50, 55],
        "request_rate": [100, 150, 200, 250, 300],
    })

    current_data = pd.DataFrame({
        "cpu_usage": [82, 85, 90, 95, 98],
        "memory_usage": [65, 70, 75, 80, 85],
        "disk_usage": [40, 42, 45, 48, 50],
        "request_rate": [700, 750, 800, 850, 900],
    })

    report = detect_data_drift(
        reference_data,
        current_data,
        alpha=0.05
    )

    print("\nKS-BASED DATA DRIFT REPORT")
    print("=" * 100)
    print(report.to_string(index=False))

    print("\nBonferroni correction applied.")
    print("Original alpha: 0.05")
    print(f"Adjusted alpha: {0.05 / len(FEATURES):.6f}")