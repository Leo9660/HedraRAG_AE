#!/bin/bash

OUTPUT_FILE="test_result.csv"

declare -a NPROBE_LIST=(128)
declare -a NPROBE_MINIBATCH_LIST=(32)

WORKFLOWS=("IRG" "Multistep")
DATASETS=("2wikimultihopqa" "2wikimultihopqa")

model_name1=meta-llama/Llama-2-13b-chat-hf
model_name2=facebook/opt-iml-30b
file_name1=result_13b.csv
file_name2=result_30b.csv

for i in "${!WORKFLOWS[@]}"; do
  WORKFLOW=${WORKFLOWS[$i]}
  DATASET=${DATASETS[$i]}

  for ((i=0; i<${#NPROBE_LIST[@]}; i++)); do

    nprobe=${NPROBE_LIST[$i]}

    echo "[[[Running FlashRAG: workflow=$WORKFLOW, dataset=$DATASET]]]"
    CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_sequential_online_rps.py \
        --model_path "$model_name1" \
        --gpu_id 0 \
        --spec_total_size 0 \
        --spec_method heteRAG \
        --nprobe $nprobe \
        --nprobe_minibatch $nprobe \
        --data_dir "$DATASET" \
        --rag_workflow "$WORKFLOW" \
        --write_file "$file_name1" \
        --index_path "$index_path" \
        --corpus_path "$corpus_path"

    echo "[[[Running FlashRAG: workflow=$WORKFLOW, dataset=$DATASET]]]"
    CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_sequential_online_rps.py \
        --model_path "$model_name2" \
        --gpu_id 0 \
        --spec_total_size 0 \
        --spec_method heteRAG \
        --nprobe $nprobe \
        --nprobe_minibatch $nprobe \
        --data_dir "$DATASET" \
        --rag_workflow "$WORKFLOW" \
        --write_file "$file_name2" \
        --index_path "$index_path" \
        --corpus_path "$corpus_path"
  done
done
