import pandas as pd
import matplotlib.pyplot as plt

csv = "results/paper_results/tables/final_model_comparison.csv"
df = pd.read_csv(csv)

fig, ax = plt.subplots(figsize=(18, 3.8))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
    if row == 4:
        cell.set_text_props(weight="bold")

plt.tight_layout()
plt.savefig(
    "results/paper_results/figures/final_model_comparison_table.png",
    dpi=400,
    bbox_inches="tight"
)
plt.close()

print("Created final_model_comparison_table.png")
