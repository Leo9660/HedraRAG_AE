#!/bin/bash
source ../../data.conf

OUTPUT_FILE="offline_result.csv"

declare -a NPROBE_LIST=(128 256 512)
declare -a NPROBE_MINIBATCH_LIST=(32 32 64)
declare -a SPEC_TOTAL_SIZE_LIST=(8)
declare -a RETRIEVAL_BATCH_LIST=(64)

WORKFLOWS=("Sequential" "RECOMP" "HyDE" "IRG" "Multistep")
DATASETS=("nq" "hotpotqa" "hotpotqa" "2wikimultihopqa" "2wikimultihopqa")

for i in "${!WORKFLOWS[@]}"; do
  WORKFLOW=${WORKFLOWS[$i]}
  dataset=${DATASETS[$i]}

  for ((j=0; j<${#NPROBE_LIST[@]}; j++)); do
    nprobe=${NPROBE_LIST[$j]}
    minibatch=${NPROBE_MINIBATCH_LIST[$j]}

      for ((k=0; k<${#RETRIEVAL_BATCH_LIST[@]}; k++)); do
        retrieval_batchsize=${RETRIEVAL_BATCH_LIST[$k]}

        for spec in "${SPEC_TOTAL_SIZE_LIST[@]}"
        do
          echo "[[[Running HedraRAG: workflow=$WORKFLOW, dataset=$dataset, nprobe=$nprobe, minibatch=$minibatch, spec_total_size=$spec]]]"
          CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_serve_offline.py \
            --model_path "$model_path" \
            --gpu_id 0 \
            --spec_total_size $spec \
            --spec_method heteRAG \
            --nprobe $nprobe \
            --nprobe_minibatch $minibatch \
            --request_per_second 0 \
            --data_dir $dataset \
            --rag_workflow "$WORKFLOW" \
            --retrieval_batchsize $retrieval_batchsize \
            --write_file "$OUTPUT_FILE"
        done
      done
  done

  for ((j=0; j<${#NPROBE_LIST[@]}; j++)); do
    nprobe=${NPROBE_LIST[$j]}
    echo "[[[Running: workflow=$WORKFLOW dataset=$dataset, nprobe=$nprobe]]]"
    CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_sequential_offline.py \
      --model_path "$model_path" \
      --gpu_id 0 \
      --spec_method heteRAG \
      --nprobe $nprobe \
      --nprobe_minibatch $nprobe \
      --request_per_second 0 \
      --data_dir $dataset \
      --rag_workflow "$WORKFLOW" \
      --write_file "$OUTPUT_FILE"
  done

done
