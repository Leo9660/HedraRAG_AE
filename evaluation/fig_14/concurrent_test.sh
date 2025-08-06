#!/bin/bash
source ../../data.conf

OUTPUT_FILE="concurrent_result.csv"

declare -a NPROBE_LIST=(128)
declare -a NPROBE_MINIBATCH_LIST=(32)
declare -a SPEC_TOTAL_SIZE_LIST=(8)
declare -a RETRIEVAL_BATCH_LIST=(64)

WORKFLOWS1=("Sequential" "HyDE" "RECOMP" "Multistep")
WORKFLOWS2=("RECOMP" "RECOMP" "Multistep" "IRG")
DATASETS=("nq" "hotpotqa" "2wikimultihopqa" "2wikimultihopqa")

for i in "${!WORKFLOWS1[@]}"; do
  WORKFLOW1=${WORKFLOWS1[$i]}
  WORKFLOW2=${WORKFLOWS2[$i]}
  dataset=${DATASETS[$i]}

  for ((j=0; j<${#NPROBE_LIST[@]}; j++)); do
    nprobe=${NPROBE_LIST[$j]}
    minibatch=${NPROBE_MINIBATCH_LIST[$j]}

      for ((k=0; k<${#RETRIEVAL_BATCH_LIST[@]}; k++)); do
        retrieval_batchsize=${RETRIEVAL_BATCH_LIST[$k]}

        echo "[[[Running: workflow=$WORKFLOW1+$WORKFLOW2, dataset=$dataset, nprobe=$nprobe, minibatch=$minibatch, spec_total_size=0]]]"
        CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_serve_concurrent_rps.py \
          --model_path "$model_path" \
          --gpu_id 0 \
          --spec_total_size 0 \
          --spec_method heteRAG \
          --nprobe $nprobe \
          --nprobe_minibatch $nprobe \
          --request_per_second 0 \
          --data_dir $dataset \
          --rag_workflow1 "$WORKFLOW1" \
          --rag_workflow2 "$WORKFLOW2" \
          --retrieval_batchsize $retrieval_batchsize \
          --write_file "$OUTPUT_FILE"
      done

      echo "[[[Running: workflow=$WORKFLOW, dataset=$dataset, nprobe=$nprobe, minibatch=$minibatch, spec_total_size=0]]]"
      CUDA_VISIBLE_DEVICES=0 python ../../HedraRAG/test/test_sequential_concurrent_rps.py \
        --model_path "$model_path" \
        --gpu_id 0 \
        --spec_total_size 0 \
        --spec_method heteRAG \
        --nprobe $nprobe \
        --nprobe_minibatch $nprobe \
        --request_per_second 0 \
        --data_dir $dataset \
        --rag_workflow1 "$WORKFLOW1" \
        --rag_workflow2 "$WORKFLOW2" \
        --write_file "$OUTPUT_FILE"
  done
done