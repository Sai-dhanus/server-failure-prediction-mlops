import pandas as pd


# Reference data: representative training-data values
reference_data = pd.DataFrame({
    "cpu_usage": [25, 35, 40, 45, 55],
    "memory_usage": [30, 40, 45, 50, 55],
    "disk_usage": [35, 40, 45, 50, 55],
    "request_rate": [100, 150, 200, 250, 300],
})


# Current data: illustrative production-like values
current_data = pd.DataFrame({
    "cpu_usage":[30, 35, 40, 45, 50],
    "memory_usage": [65, 70, 75, 80, 85],
    "disk_usage": [40, 42, 45, 48, 50],
    "request_rate": [700, 750, 800, 850, 900],
})


def detect_mean_drift(reference, current, threshold=0.5):

    results = []

    for feature in reference.columns:

        reference_mean = reference[feature].mean()
        current_mean = current[feature].mean()
        reference_std = reference[feature].std()

        if reference_std == 0:
            print(
                f"Cannot calculate standardized drift "
                f"for {feature}: reference standard deviation is zero."
            )
            continue

        drift_score = (
            abs(current_mean - reference_mean) / reference_std
        )

        drift_detected = drift_score > threshold

        results.append({
            "feature": feature,
            "reference_mean": round(reference_mean, 2),
            "current_mean": round(current_mean, 2),
            "drift_score": round(drift_score, 2),
            "potential_drift": drift_detected,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    report = detect_mean_drift(
        reference_data,
        current_data,
        threshold=0.5,
    )

    print("\nDATA DRIFT REPORT")
    print("=" * 75)
    print(report.to_string(index=False))