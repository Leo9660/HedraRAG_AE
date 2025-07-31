def main():

    from heterag import Config, HeteragExecutor, SequentialRAGraph, get_dataset
    import numpy as np
    import faiss
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retriever_path", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--request_per_second", type=int, default=16)
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--write_file", default=None)
    parser.add_argument("--spec_total_size", type=int, default=1)

    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--nprobe_minibatch", type=int, default=32)
    parser.add_argument("--spec_method", type=str, default="heteRAG")

    parser.add_argument("--speculative_retrieval", type=bool, default=True)
    parser.add_argument("--gpu_onloading", action="store_true")
    parser.add_argument("--rag_workflow", type=str, default="Multistep")
    parser.add_argument("--data_dir", type=str, default="2wikimultihopqa")
    parser.add_argument("--retrieval_batchsize", type=int, default=32)

    parser.add_argument("--index_path", type=str, default="/data/index_0319/wiki/IVF4096/ivf.index")
    parser.add_argument("--corpus_path", type=str, default="/data/wiki_dataset/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl")

    args = parser.parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "data_source": "huggingface",
        "split_name": "dev",
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "nprobe": args.nprobe,
        "retrieval_topk": 1,
        "model2path": {"e5": args.retriever_path, "llama3-8B-instruct": args.model_path},
        "generator_model": "llama3-8B-instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc", "rouge-1", "rouge-2", "rouge-l"],
        "gpu_id": args.gpu_id,
        "retrieval_batch_size": args.retrieval_batchsize,
        "return_embedding": True,
        "framework": "vllm",
        "use_fp16": True,
        "pooling_method": "mean",
        "generator_max_input_len": 4096,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": False,
        "generation_params": {'max_tokens': 128},
        "task_batch": 64,
        "iter_num": 2,
        "print_log": True,
        "continuous_retrieval": True,
        "nprobe_minibatch": args.nprobe_minibatch,
        "speculative_retrieval": args.speculative_retrieval,
        "spec_method": args.spec_method,
        "spec_total_size": args.spec_total_size,
        "gpu_onloading": args.gpu_onloading
    }

    config = Config(config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["test"][0:1024]

    questions = []
    for item in test_data:
        questions.append(item.question)

    string_list = questions

    executor = HeteragExecutor(config)

    #warm up
    executor.add_requests_string(string_list, workflow = args.rag_workflow)
    while executor.execute() < len(string_list):
        continue

    start_time_list = [0 for i in range(len(string_list))]

    t1 = time.time()

    current_idx = 0

    while executor.execute() < len(string_list):

        current_time = time.time() - t1
        
        query_list = []
        while current_idx < len(string_list) and current_time >= start_time_list[current_idx]:
            query_list.append(string_list[current_idx])
            current_idx += 1

        if len(query_list):
            executor.add_requests_string(query_list, workflow = args.rag_workflow)

    t2 = time.time()
    loop_time = t2 - t1

    query_latency = []
    for key, values in executor.request_time_dict.items():
        query_latency.append(values - start_time_list[key] - t1)
    print(f"[Average query latency]: {np.mean(query_latency)} s")
    average_query_latency = np.mean(query_latency)

    executor.finalize()

    for i in range(min(5, len(string_list))):
        request = executor.output_list[i]

        print(f"Answer {request.id}: {request.answer}")

    import csv

    split_minibatch = config["nprobe"] / config["nprobe_minibatch"]
    if args.write_file is not None:
        with open(args.write_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["HedraRAG", args.rag_workflow, args.data_dir, args.nprobe, args.nprobe_minibatch, average_query_latency, loop_time])

if __name__ == "__main__":
    main()

