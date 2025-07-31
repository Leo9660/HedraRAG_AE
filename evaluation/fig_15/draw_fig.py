import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import FuncFormatter

# Load CSVs
df_13b = pd.read_csv("result_13b.csv", header=None, names=["baseline", "workflow", "nprobe", "rps", "latency"])
df_30b = pd.read_csv("result_30b.csv", header=None, names=["baseline", "workflow", "nprobe", "rps", "latency"])

# Filter by nprobe = 128
df_13b = df_13b[df_13b["nprobe"] == 128]
df_30b = df_30b[df_30b["nprobe"] == 128]

# Shared axis config
model_name = ["Llama2-13B", "OPT-30B"]
workflow_names = ["Multistep", "IRG"]
baseline_name = ["LangChain", "FlashRAG", "HedraRAG"]
rps = range(4, 30, 4)

baseline_styles = {
    "LangChain": {"marker": 'o', "color": 'blue'},
    "FlashRAG": {"marker": '^', "color": 'green'},
    "HedraRAG": {"marker": 'd', "color": 'orange'}
}

fig, axs = plt.subplots(2, 2, figsize=(6, 4.5))
lines = []
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
label_idx = 0

# Model to dataframe mapping
model_df_map = {
    "Llama2-13B": df_13b,
    "OPT-30B": df_30b
}

for row_idx, workflow in enumerate(workflow_names):
    for col_idx, model in enumerate(model_name):
        ax = axs[row_idx][col_idx]
        df = model_df_map[model]

        for bl in baseline_name:
            style = baseline_styles.get(bl, {"marker": 'x', "color": 'gray'})
            df_bl = df[(df["workflow"] == workflow) & (df["baseline"] == bl)]
            df_bl = df_bl.set_index("rps").reindex(rps).sort_index()
            latencies = df_bl["latency"].to_numpy()

            print(rps, latencies)

            line, = ax.plot(
                rps,
                latencies,
                label=bl,
                marker=style["marker"],
                color=style["color"],
                linestyle='-',
                linewidth=2
            )
            if row_idx == 0 and col_idx == 0:
                lines.append(line)

        subplot_label = subplot_labels[label_idx]
        ax.set_title(f"{subplot_label} {model}, {workflow}", fontsize=11, fontweight='bold')
        label_idx += 1

        if col_idx == 0:
            ax.set_ylabel("Latency (s/query)", fontsize=10)
        ax.set_xlabel("Request Rate (req/s)", fontsize=10)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))
        ax.grid(True, linestyle='--', alpha=0.6)

fig.legend(
    handles=lines,
    labels=baseline_name,
    loc='upper center',
    ncol=3,
    fontsize=11,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs("../output_figure", exist_ok=True)
plt.savefig("../output_figure/fig_15.pdf", bbox_inches="tight")