#!/bin/bash

# Master script to run all RAG workflows with multiple nprobe values
# Usage: ./master_run_all.sh

# List of workflows
WORKFLOWS=(
    "sequential"
    "hyde"
    "recomp"
    "iterative"
    "multistep"
)

# nprobe values
NPROBES=(128 256 512)
ROOT_DIR="$(pwd)"
cd ../../LangChain/
SCRIPT="./run_rag_expts_2.sh"

# Loop through each workflow and nprobe
for WF in "${WORKFLOWS[@]}"; do
    for NP in "${NPROBES[@]}"; do
        echo "=============================================="
        echo "Running workflow: $WF with nprobe: $NP"
        echo "=============================================="
        bash $SCRIPT $WF $NP
        echo "Completed: $WF with nprobe: $NP"
        echo
    done
done

echo "All workflows completed successfully!"
cd ../evaluation/fig_12