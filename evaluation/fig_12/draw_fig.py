import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import csv

file_list = ["test_result.csv"]

x_dict = {
    # "irg128":(0, 28),
    # "irg256":(0, 14),
    # "irg512":(0, 10),
    # "multistep128":(0, 28),
    # "multistep256":(0, 18),
    # "multistep512":(0, 16),
    # "recomp128":(0, 28),
    # "recomp256":(0, 28),
    # "recomp512":(0, 24),
    # "hyde512":(0, 40),
    # "sequential128":(0, 60),
    # "hyde128":(0, 60),
}

all_data = {}

for file_path in file_list:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            baseline = row[0].strip()
            workflow = row[1].strip()
            workflow = workflow.lower()
            if workflow == "iterative":
                workflow = "irg"
            try:
                if len(row) == 5:
                    rps = int(row[3])
                    nprobe = int(row[2])
                    latency = float(row[4])
                else:
                    continue

                all_data.setdefault(baseline, {}).setdefault(workflow, {}).setdefault(nprobe, {})
                if rps not in all_data[baseline][workflow][nprobe] or latency < all_data[baseline][workflow][nprobe][rps]:
                    all_data[baseline][workflow][nprobe][rps] = latency
            except Exception as e:
                print(f"Error parsing row: {row}, err: {e}")

plot_data = {}

for baseline, wf_data in all_data.items():
    for workflow, nprobe_data in wf_data.items():
        best = max(nprobe_data.values(), key=lambda x: len(x))
        sorted_rps = sorted(best.keys())[:10]
        latency = [best[r] for r in sorted_rps]
        plot_data.setdefault(workflow, {})[baseline] = (np.array(sorted_rps), np.array(latency))

baseline_styles = {
    "LangChain": {"marker": 'o', "color": 'blue'},
    "flashrag": {"marker": 's', "color": 'green'},
    "vLLM + faiss": {"marker": '^', "color": 'purple'},
    "FlashRAG": {"marker": '^', "color": 'purple'},
    "HedraRAG": {"marker": 'd', "color": 'orange'},
}


workflow_order = ["sequential", "hyde", "recomp", "multistep", "irg"]
workflow_name = ["One-shot", "HyDE", "RECOMP", "Multistep", "IRG"]
workflow_set = {wf for b in all_data.values() for wf in b}
workflow_list = [wf for wf in workflow_order if wf in workflow_set]

nprobe_list = sorted({npb for b in all_data.values() for wf in b.values() for npb in wf})

baseline_styles = {
    "LangChain": {"marker": 'o', "color": '#4c72b0'},
    "FlashRAG": {"marker": '^', "color": '#55a868'},
    "HedraRAG": {"marker": 'd', "color": '#dd8452'},
}

baseline_order = ["LangChain", "FlashRAG", "HedraRAG"]
rps_ticks = np.linspace(50, 500, 10)

fig, axs = plt.subplots(len(nprobe_list), len(workflow_list), figsize=(15, 7.5), sharey=True)
letters = string.ascii_lowercase

for i, nprobe in enumerate(nprobe_list):
    for j, workflow in enumerate(workflow_list):
        ax = axs[i, j]

        for baseline in baseline_order:
            if baseline in all_data:
                print(baseline, all_data[baseline])
            if baseline in all_data and \
               workflow in all_data[baseline] and \
               nprobe in all_data[baseline][workflow]:
                rps_latency_map = all_data[baseline][workflow][nprobe]

                sorted_items = sorted((r, l) for r, l in rps_latency_map.items() if r != 0)

                rps_list = [r for r, _ in sorted_items]
                latency_list = [l for _, l in sorted_items]
            else:
                rps_list = []
                latency_list = []

            style = baseline_styles.get(baseline, {"marker": 'x', "color": 'gray'})
            ax.plot(rps_list, latency_list,
                    marker=style["marker"],
                    linestyle='-',
                    linewidth=2,
                    color=style["color"],
                    label=baseline)

        if i == len(nprobe_list) - 1:
            ax.set_xlabel("Request Rate (req/s)")
        if j == 0:
            ax.set_ylabel(f"Latency (s/query)")
        if f"{workflow}{nprobe}" in x_dict:
            my_dict = x_dict[f"{workflow}{nprobe}"]
            ax.set_xlim(my_dict[0], my_dict[1])
        ax.set_ylim(0, 10)

        ax.set_title(f"({letters[i * len(workflow_list) + j]}) {workflow_name[j]}, nprobe={nprobe}",
             fontsize=12, fontweight="bold", rotation=0)

        ax.grid(True, linestyle='--', alpha=0.7)

baseline_name = ["LangChain", "FlashRAG", "HedraRAG"]

handles = []
labels = []
for i, baseline in enumerate(baseline_order):
    style = baseline_styles[baseline]
    handles.append(plt.Line2D([0], [0], marker=style["marker"], color=style["color"],
                               linestyle='-', linewidth=2, label=baseline_name[i]))
    labels.append(baseline_name[i])

fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.03),
           ncol=len(baseline_order), fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.98])
import os
os.makedirs("../output_figure", exist_ok=True)
plt.savefig("../output_figure/fig_12.pdf", bbox_inches="tight")
