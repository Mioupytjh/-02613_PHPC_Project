import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "task12_results.csv"

df = pd.read_csv(RESULTS_FILE)

# Clean column names in case there are spaces
df.columns = [c.strip() for c in df.columns]

avg_mean_temp = df["mean_temp"].mean()
avg_std_temp = df["std_temp"].mean()
n_above_18 = (df["pct_above_18"] >= 50).sum()
n_below_15 = (df["pct_below_15"] >= 50).sum()

print("Task 12 results")
print("----------------")
print(f"Number of buildings: {len(df)}")
print(f"Average mean temperature: {avg_mean_temp:.4f}")
print(f"Average temperature standard deviation: {avg_std_temp:.4f}")
print(f"Buildings with at least 50% area above 18°C: {n_above_18}")
print(f"Buildings with at least 50% area below 15°C: {n_below_15}")

plt.figure(figsize=(7, 5))
plt.hist(df["mean_temp"], bins=40)
plt.xlabel("Mean temperature [°C]")
plt.ylabel("Number of buildings")
plt.title("Distribution of mean temperatures")
plt.grid(True)
plt.savefig("task12_mean_temperature_histogram.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
plt.hist(df["std_temp"], bins=40)
plt.xlabel("Temperature standard deviation [°C]")
plt.ylabel("Number of buildings")
plt.title("Distribution of temperature standard deviations")
plt.grid(True)
plt.savefig("task12_temperature_std_histogram.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved figures:")
print("task12_mean_temperature_histogram.png")
print("task12_temperature_std_histogram.png")