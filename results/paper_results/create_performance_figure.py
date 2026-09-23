import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    "results/paper_results/tables/final_model_comparison.csv"
)

models = df["Model"].tolist()
accuracy = [float(x.strip("%")) for x in df["Accuracy"]]
macro_f1 = [float(x.strip("%")) for x in df["Macro F1"]]
mcc = df["MCC"].astype(float).tolist()

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - width, accuracy, width, label="Accuracy")
ax.bar(x, macro_f1, width, label="Macro F1")
ax.bar(x + width, [v * 100 for v in mcc], width, label="MCC")

ax.set_ylabel("Performance (%)")
ax.set_title("Performance Comparison of Classification Models")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylim(90, 100)
ax.legend()

for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

plt.tight_layout()

plt.savefig(
    "results/paper_results/figures/model_performance_comparison.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print("Created model_performance_comparison.png")
