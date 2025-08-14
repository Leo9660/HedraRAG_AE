corpus_path=/data/wiki_dataset/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
save_dir=/data/wiki_dataset/corpora/index/

cd HedraRAG

python -m heterag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/e5-large-v2 \
    --corpus_path "$corpus_path" \
    --save_dir "$save_dir" \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type IVF4096_HNSW32,Flat \
    --save_embedding \
    --resume