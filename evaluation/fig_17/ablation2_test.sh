MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
RETRIEVER_PATH="intfloat/e5-large-v2"

export CUDA_VISIBLE_DEVICES=7

OUTPUT_FILE="ablation2_latency.csv"
OUTPUT_FILE2="ablation2_accuracy.csv"

echo "[[[[HedraRAG, None speculation]]]] RPS=4"
python ../../HedraRAG/test/test_ablation2.py \
    --model_path "$MODEL_PATH" \
    --retriever_path "$RETRIEVER_PATH" \
    --gpu_id 0 \
    --spec_total_size 0 \
    --spec_method heteRAG \
    --nprobe 128 \
    --nprobe_minibatch 32 \
    --request_per_second 4 \
    --write_file "$OUTPUT_FILE" \
    --spec_output_file "$OUTPUT_FILE2"

echo "[[[[HedraRAG, Speculative execution]]]]  RPS=4"
python ../../HedraRAG/test/test_ablation2.py \
    --model_path "$MODEL_PATH" \
    --retriever_path "$RETRIEVER_PATH" \
    --gpu_id 0 \
    --spec_total_size 8 \
    --spec_method heteRAG \
    --nprobe 128 \
    --nprobe_minibatch 32 \
    --request_per_second 4 \
    --write_file "$OUTPUT_FILE" \
    --spec_output_file "$OUTPUT_FILE2"

echo "[[[[RAGCache]]]] RPS=4"
python ../../HedraRAG/test/test_ablation2.py \
    --model_path "$MODEL_PATH" \
    --retriever_path "$RETRIEVER_PATH" \
    --gpu_id 0 \
    --spec_method RAGCache \
    --nprobe 128 \
    --nprobe_minibatch 32 \
    --request_per_second 4 \
    --write_file "$OUTPUT_FILE" \
    --spec_output_file "$OUTPUT_FILE2"

echo "[[[[HedraRAG, None speculation]]]] RPS=8"
python ../../HedraRAG/test/test_ablation2.py \
    --model_path "$MODEL_PATH" \
    --retriever_path "$RETRIEVER_PATH" \
    --gpu_id 0 \
    --spec_total_size 0 \
    --spec_method heteRAG \
    --nprobe 128 \
    --nprobe_minibatch 32 \
    --request_per_second 8 \
    --write_file "$OUTPUT_FILE" \
    --spec_output_file "$OUTPUT_FILE2"

echo "[[[[HedraRAG, Speculative execution]]]] RPS=8"
python ../../HedraRAG/test/test_ablation2.py \
    --model_path "$MODEL_PATH" \
    --retriever_path "$RETRIEVER_PATH" \
    --gpu_id 0 \
    --spec_total_size 8 \
    --spec_method heteRAG \
    --nprobe 128 \
    --nprobe_minibatch 32 \
    --request_per_second 8 \
    --write_file "$OUTPUT_FILE" \
    --spec_output_file "$OUTPUT_FILE2"

echo "[[[[RAGCache]]]] RPS=8"
python ../../HedraRAG/test/test_ablation2.py \
    --model_path "$MODEL_PATH" \
    --retriever_path "$RETRIEVER_PATH" \
    --gpu_id 0 \
    --spec_method RAGCache \
    --nprobe 128 \
    --nprobe_minibatch 32 \
    --request_per_second 8 \
    --write_file "$OUTPUT_FILE" \
    --spec_output_file "$OUTPUT_FILE2"
