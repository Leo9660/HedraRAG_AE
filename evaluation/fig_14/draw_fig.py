import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from collections import defaultdict

file_list = ["concurrent_result.csv"]

all_data = defaultdict(lambda: defaultdict(dict))

for file_path in file_list:
    if not os.path.exists(file_path):
        continue

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 9:
                continue
            baseline = row[0].strip()
            wf1 = row[1].strip().lower()
            wf2 = row[2].strip().lower()
            try:
                rps = int(row[5])
                latency = float(row[7])
                key = tuple(sorted((wf1, wf2)))
                if rps not in all_data[baseline][key] or latency < all_data[baseline][key][rps]:
                    all_data[baseline][key][rps] = latency
            except Exception:
                continue

print(all_data)

workflow_pairs = sorted({k for b in all_data.values() for k in b})
baseline_order = ["FlashRAG", "HedraRAG"]
workflow_pairs = [('recomp', 'sequential'), ('hyde', 'recomp'), ('multistep', 'recomp'), ('irg', 'multistep')]
namelist = ["One-shot + RECOMP", "HYDE + RECOMP", "Multistep + RECOMP", "Multistep + IRG"]

fig, axs = plt.subplots(2, 2, figsize=(6, 4))
axs = axs.flatten()

baseline_styles = {
    "LangChain": {"marker": 'o', "color": 'blue'},
    "FlashRAG": {"marker": '^', "color": 'green'},
    "HedraRAG": {"marker": 'd', "color": 'orange'},
}

for idx, pair in enumerate(workflow_pairs[:4]):
    ax = axs[idx]
    for baseline in baseline_order:
        if pair in all_data[baseline]:
            items = sorted(all_data[baseline][pair].items())
            rps_list = [r for r, _ in items]
            latency_list = [l for _, l in items]
            style = baseline_styles[baseline]
            ax.plot(rps_list, latency_list, marker=style["marker"], color=style["color"], linewidth=2, label=baseline)

    ax.set_title(f"({string.ascii_lowercase[idx]}) {namelist[idx]}", fontsize=11, fontweight ="bold")
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Latency (s/query)")
    ax.set_ylim(0, 10)
    ax.grid(True, linestyle='--', alpha=0.6)

baseline_name = ["FlashRAG", "HedraRAG"]

fig.legend(
    handles=[plt.Line2D([0], [0], marker=baseline_styles[b]["marker"], color=baseline_styles[b]["color"],
                        label=n, linewidth=2) for b, n in zip(baseline_order, baseline_name)],
    labels=baseline_name,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.03),
    ncol=3,
    fontsize=11
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
import os
os.makedirs("../output_figure", exist_ok=True)
plt.savefig("../output_figure/fig_14.pdf", bbox_inches="tight")
