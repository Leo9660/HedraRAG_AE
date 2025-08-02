#!/bin/bash

OUTPUT_FILE="test_result.csv"

declare -a NPROBE_LIST=(128 256 512)
declare -a NPROBE_MINIBATCH_LIST=(32 32 64)

WORKFLOWS=("Sequential" "RECOMP" "HyDE" "IRG" "Multistep")
DATASETS=("nq" "hotpotqa" "hotpotqa" "2wikimultihopqa" "2wikimultihopqa")

spec=8

for i in "${!WORKFLOWS[@]}"; do
  WORKFLOW=${WORKFLOWS[$i]}
  dataset=${DATASETS[$i]}

  for ((i=0; i<${#NPROBE_LIST[@]}; i++)); do
    nprobe=${NPROBE_LIST[$i]}
    minibatch=${NPROBE_MINIBATCH_LIST[$i]}

    echo "[[[Running HedraRAG: workflow=$WORKFLOW, dataset=$dataset, nprobe=$nprobe, minibatch=$minibatch, spec_total_size=$spec]]]"
    CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_serve_online_rps.py \
      --gpu_id 0 \
      --spec_total_size $spec \
      --spec_method heteRAG \
      --nprobe $nprobe \
      --nprobe_minibatch $minibatch \
      --data_dir $dataset \
      --rag_workflow "$WORKFLOW" \
      --write_file "$OUTPUT_FILE" \
      --index_path "$index_path" \
      --corpus_path "$corpus_path"
  done

done