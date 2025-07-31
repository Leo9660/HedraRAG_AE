import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

accuracy_df = pd.read_csv("ablation2_accuracy.csv", header=None)
latency_df = pd.read_csv("ablation2_latency.csv", header=None)

def classify_baseline(method, spec_size):
    if method == "heteRAG" and int(spec_size) == 0:
        return "Non-spec"
    elif method == "heteRAG" and int(spec_size) > 0:
        return "HedraRAG"
    elif method == "RAGCache":
        return "PipeRAG"
    else:
        return None

accuracy_df['baseline'] = accuracy_df.apply(lambda row: classify_baseline(row[0], row[1]), axis=1)
latency_df['baseline'] = latency_df.apply(lambda row: classify_baseline(row[0], row[1]), axis=1)

accuracy_df = accuracy_df.dropna()
latency_df = latency_df.dropna()

baseline_order = ['Non-spec', 'PipeRAG', 'HedraRAG']
colors = ['green', 'purple', 'orange']

titles = ["MultiStep, RPS=4", "IRG, RPS=4", "MultiStep, RPS=8", "IRG, RPS=8"]
letters = ['a', 'c', 'b', 'd']
rps_list = [4, 4, 8, 8]

fig, axs = plt.subplots(2, 2, figsize=(6, 4.8))

for idx, ax in enumerate(axs.flat):
    rps = rps_list[idx]

    acc_points = []
    lat_points = []
    labels = []

    for i, b in enumerate(baseline_order):
        acc_row = accuracy_df[(accuracy_df['baseline'] == b) & (accuracy_df[2] == rps)]
        lat_row = latency_df[(latency_df['baseline'] == b) & (latency_df[2] == rps)]

        if not acc_row.empty and not lat_row.empty:
            acc = acc_row.iloc[0, 3]
            lat = lat_row.iloc[0, 3]

            acc_points.append(acc)
            lat_points.append(lat)
            labels.append(b)

            ax.scatter(acc, lat, color=colors[i], s=100, marker='o',
                       edgecolor='dimgray', linewidth=1.2)

            offset_x = 0.01
            offset_y = 0.03
            ax.text(acc + offset_x, lat + offset_y, b,
                    fontsize=8, verticalalignment='bottom', horizontalalignment='left')

    # Axis setup
    ax.set_xlabel("Speculation Accuracy", fontsize=10)
    ax.set_ylabel("Average Latency (s)", fontsize=10)
    ax.set_title(f"({letters[idx]}) {titles[idx]}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    # Padding
    if acc_points and lat_points:
        x_min, x_max = min(acc_points), max(acc_points)
        y_min, y_max = min(lat_points), max(lat_points)
        ax.set_xlim(x_min - 0.02, x_max + 0.02)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)

plt.tight_layout()
import os
os.makedirs("../output_figure", exist_ok=True)
plt.savefig(f"../output_figure/fig_17.pdf", dpi=300)