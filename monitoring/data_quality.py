
import pandas as pd

FEATURE_RULES = {
    "cpu_usage": (0, 100),
    "memory_usage": (0, 100),
    "disk_usage": (0, 100),
    "request_rate": (0, None)
}


def validate_data(data):

    report = []

    duplicate_count = data.duplicated().sum()
    

    for feature, (minimum, maximum) in FEATURE_RULES.items():

        if feature not in data.columns:
            report.append({
                "feature": feature,
                "missing_values": None,
                "non_numeric_values": None,
                "out_of_range_values": None,
                "status": "MISSING COLUMN"
            })
            continue

        original_values = data[feature]

        # Identify originally missing values
        missing_count = original_values.isna().sum()

        # Convert values to numeric
        numeric_values = pd.to_numeric(
            original_values,
            errors="coerce"
        )

        # Identify non-numeric values
        non_numeric_count = (
            original_values.notna() & numeric_values.isna()
        ).sum()

        # Identify out-of-range values
        invalid_count = (
            (numeric_values < minimum) |
            (numeric_values > maximum)
        ).sum()

        status = (
            "INVALID"
            if missing_count > 0
            or non_numeric_count > 0
            or invalid_count > 0
            else "VALID"
        )

        report.append({
            "feature": feature,
            "missing_values": int(missing_count),
            "non_numeric_values": int(non_numeric_count),
            "out_of_range_values": int(invalid_count),
            "status": status
        })

    report.append({
        "feature": "ALL_RECORDS",
        "missing_values": 0,
        "non_numeric_values": 0,
        "out_of_range_values": 0,
        "status": f"{duplicate_count} DUPLICATES"
    })

    return pd.DataFrame(report)
