
from scipy.stats import ks_2samp


# Reference data
reference_data = [10, 20, 30, 40, 50]

# Current production-like data
current_data = [30, 30, 30, 30, 30]


# Compare the distributions
result = ks_2samp(reference_data, current_data)

print("KS DRIFT DETECTION REPORT")
print("-" * 35)
print(f"KS statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.6f}")

significance_level = 0.05

if result.pvalue < significance_level:
    print("Result: Statistically significant distribution difference")
else:
    print("Result: Insufficient evidence of a distribution difference")