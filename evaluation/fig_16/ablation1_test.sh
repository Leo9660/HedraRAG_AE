#!/bin/bash
source ../../data.conf

OUTPUT_FILE="ablation1_result.csv"

echo "[[[Running benchmark with nprobe=128]]]"
python ../../HedraRAG/test/test_ablation1.py --model_path "$model_path" --gpu_id 0 --nprobe 128 --nprobe_minibatch 128 --write_file "$OUTPUT_FILE" --index_path "$index_path" --corpus_path "$corpus_path"
python ../../HedraRAG/test/test_ablation1.py --model_path "$model_path" --gpu_id 0 --nprobe 128 --nprobe_minibatch 64 --write_file "$OUTPUT_FILE" --index_path "$index_path" --corpus_path "$corpus_path"

echo "[[[Running benchmark with nprobe=512]]]"
python ../../HedraRAG/test/test_ablation1.py --model_path "$model_path" --gpu_id 0 --nprobe 512 --nprobe_minibatch 512 --write_file "$OUTPUT_FILE" --index_path "$index_path" --corpus_path "$corpus_path"
python ../../HedraRAG/test/test_ablation1.py --model_path "$model_path" --gpu_id 0 --nprobe 512 --nprobe_minibatch 64 --write_file "$OUTPUT_FILE" --index_path "$index_path" --corpus_path "$corpus_path"
