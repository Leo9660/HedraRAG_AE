#!/bin/bash

# Usage: ./run_rag_expts_2.sh <workflow> <nprobe>
# Example: ./run_rag_expts_2.sh sequential 512

WORKFLOW=$1
NPROBE=$2
SCRIPT="run_baseline_expts_2.py"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_LOGGING_LEVEL=DEBUG
if [[ "$WORKFLOW" == "sequential" ]]; then
    DATADIR="nq"
    echo "Running SEQUENTIAL RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow sequential --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "hyde" ]]; then
    DATADIR="hotpotqa"
    echo "Running HYDE RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow hyde --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "recomp" ]]; then
    DATADIR="hotpotqa"
    echo "Running RECOMP RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow recomp --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "iterative" ]]; then
    DATADIR="2wikimultihopqa"
    echo "Running ITERATIVE RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow iterative --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "multistep" ]]; then
    DATADIR="2wikimultihopqa"
    echo "Running MULTISTEP RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow multistep --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "sequential_offline" ]]; then
    DATADIR="nq"
    echo "Running SEQUENTIAL OFFLINE RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow sequential_offline --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "hyde_offline" ]]; then
    DATADIR="hotpotqa"
    echo "Running HYDE OFFLINE RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow hyde_offline --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "recomp_offline" ]]; then
    DATADIR="hotpotqa"
    echo "Running RECOMP OFFLINE RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow recomp_offline --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "iterative_offline" ]]; then
    DATADIR="2wikimultihopqa"
    echo "Running ITERATIVE RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow iterative_offline --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "multistep_offline" ]]; then
    DATADIR="2wikimultihopqa"
    echo "Running MULTISTEP RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow multistep_offline --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "seqrec" ]]; then
    DATADIR="nq"
    echo "Running SEQREC RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow seqrec --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "hyderec" ]]; then
    DATADIR="hotpotqa"
    echo "Running HYDEREC RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow hyderec --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "recmulti" ]]; then
    DATADIR="2wikimultihopqa"
    echo "Running RECMULTI RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow recmulti --nprobe $NPROBE --data_dir $DATADIR

elif [[ "$WORKFLOW" == "multiiter" ]]; then
    DATADIR="2wikimultihopqa"
    echo "Running MULTIITER RAG with nprobe=$NPROBE on dataset=$DATADIR"
    python $SCRIPT --workflow multiiter --nprobe $NPROBE --data_dir $DATADIR

else
    echo "Invalid workflow: $WORKFLOW"
    echo "Usage: ./run_rag_expts_2.sh <sequential|hyde|recomp> <nprobe>"
    exit 1
fi