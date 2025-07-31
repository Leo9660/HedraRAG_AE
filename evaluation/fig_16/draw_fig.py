import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import string
import os

# Load data
csv_path = "ablation1_result.csv"  # ← Replace with your actual CSV path
df = pd.read_csv(csv_path)

# Determine baseline method
df = pd.read_csv(csv_path, header=None, names=["nprobe", "nprobe_mini", "rps", "latency"])

df["method"] = np.where(df["nprobe"] == df["nprobe_mini"], "Faiss", "HedraRAG")

# Set up parameters
dataset_list = [128, 512]
methods = ['Faiss', 'HedraRAG']
colors = ['green', 'orange']
markers = ['o', '^']

fig, axs = plt.subplots(1, 2, figsize=(6, 2.4))
letters = string.ascii_lowercase

for i, ax in enumerate(axs):
    nprobe_value = dataset_list[i]
    sub_df = df[df["nprobe"] == nprobe_value]

    rps_values = sorted(sub_df["rps"].unique())
    xaxis = range(len(rps_values))

    for j, method in enumerate(methods):
        y_values = []
        for rps in rps_values:
            rows = sub_df[(sub_df["method"] == method) & (sub_df["rps"] == rps)]
            latency_avg = rows["latency"].mean() if not rows.empty else np.nan
            y_values.append(latency_avg)

        ax.plot(
            xaxis,
            y_values,
            marker=markers[j],
            linestyle='-',
            linewidth=2,
            color=colors[j],
            label=method if i == 0 else None
        )

    ax.set_xlabel("Arrival Rate (req/s)", fontsize=10)
    ax.set_ylabel("Average Latency (s)", fontsize=10)
    ax.set_title(f"({letters[i]}) nprobe = {nprobe_value}", fontsize=12, fontweight="bold")
    ax.set_xticks(xaxis, rps_values)
    ax.grid(True, linestyle='--', alpha=0.7)
    if i == 0:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

fig.legend(
    handles=[
        plt.Line2D([0], [0], color=colors[i], marker=markers[i], linestyle='-', linewidth=2, label=methods[i])
        for i in range(len(methods))
    ],
    labels=methods,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.11),
    ncol=2,
    fontsize=12
)

plt.tight_layout()
import os
os.makedirs("../output_figure", exist_ok=True)
plt.savefig("../output_figure/fig_16.pdf", bbox_inches="tight")
plt.show()