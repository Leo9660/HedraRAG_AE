import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import csv
from matplotlib.patches import Patch

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.6

file_list = ["offline_result.csv"]

all_data = {}
for file_path in file_list:
    if not os.path.exists(file_path):
        continue

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            baseline = row[0].strip().lower()
            workflow = row[1].strip().lower()
            try:
                if len(row) == 7:
                    nprobe = int(row[3])
                    latency = float(row[6])
                else:
                    continue

                all_data.setdefault(baseline, {}).setdefault(workflow, {})[nprobe] = latency

            except:
                continue

workflow_list = ["sequential", "hyde", "recomp", "multistep", "irg"]
name_list = ["One-shot", "HyDE", "RECOMP", "Multistep", "IRG"]
nprobe_list = [128, 256, 512]
index_types = ["langchain", "flashrag", "hedrarag"]
name_types = ["LangChain", "FlashRAG", "HedraRAG"]
colors = ['#6a8fd3', '#6fc18d', '#f1a873']
hatches = ['', '\\\\', '//']
letters = string.ascii_lowercase

fig, axs = plt.subplots(3, 1, figsize=(5, 4.5), sharex=True)

for i, ax in enumerate(axs):
    x = np.arange(len(workflow_list))
    total_width = 0.6
    bar_width = total_width / len(index_types)

    for j, baseline in enumerate(index_types):
        bar_vals = []
        for wf in workflow_list:
            hete_val = all_data.get("hedrarag", {}).get(wf, {}).get(nprobe_list[i], np.nan)
            this_val = all_data.get(baseline, {}).get(wf, {}).get(nprobe_list[i], np.nan)
            print(hete_val, this_val)
            if baseline == "hedrarag":
                ratio = 1.0 if not np.isnan(hete_val) else np.nan
            else:
                ratio = this_val / hete_val if not np.isnan(this_val) and not np.isnan(hete_val) else np.nan
            bar_vals.append(ratio)
        offset = (j - len(index_types) / 2) * bar_width + bar_width / 2
        ax.bar(
            x + offset, bar_vals,
            width=bar_width * 0.9,
            label=name_types[j],
            color=colors[j],
            edgecolor='black',
            linewidth=0.6,
            hatch=hatches[j]
        )

    ax.set_ylabel("Total time", fontsize=10)
    ax.set_title(f"({letters[i]}) nprobe = {nprobe_list[i]}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(name_list, fontsize=10)
    ax.set_ylim(0, 2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.legend(
    handles=[
        Patch(facecolor=colors[i], edgecolor='black', linewidth=0.6, hatch=hatches[i], label=name_types[i])
        for i in range(len(index_types))
    ],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    fontsize=11
)

speedup_stats = {baseline: [] for baseline in ["FlashRAG", "LangChain"]}

for nprobe in nprobe_list:
    for wf in workflow_list:
        hete_val = all_data.get("HedraRAG", {}).get(wf, {}).get(nprobe, np.nan)

        for baseline in ["FlashRAG", "LangChain"]:
            base_val = all_data.get(baseline, {}).get(wf, {}).get(nprobe, np.nan)
            if not np.isnan(hete_val) and not np.isnan(base_val) and base_val > 0:
                speedup = base_val / hete_val
                speedup_stats[baseline].append(speedup)

for baseline in ["FlashRAG", "LangChain"]:
    values = speedup_stats[baseline]
    if values:
        mean_speedup = np.mean(values)
        max_speedup = np.max(values)
        print(f"[{baseline}] Avg speedup over: {mean_speedup:.2f}x, Max speedup: {max_speedup:.2f}x")
    else:
        print(f"[{baseline}] No valid data for speedup calculation.")

plt.tight_layout(rect=[0, 0, 1, 0.97])
import os
os.makedirs("../output_figure", exist_ok=True)
plt.savefig(f"../output_figure/fig_13.pdf", bbox_inches="tight", dpi=300)
plt.show()
