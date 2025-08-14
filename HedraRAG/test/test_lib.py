from heterag import Config, HeteragExecutor, SequentialRAGraph, get_dataset
import numpy as np
import argparse
import time

def main():

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

    parser.add_argument("--index_path", type=str, default="/data/index_0319/wiki/IVF4096/ivf.index")
    parser.add_argument("--corpus_path", type=str, default="/data/wiki_dataset/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl")

    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "data_source": "huggingface",
        "split_name": "dev",
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "nprobe": args.nprobe,
        "retrieval_topk": args.topk,
        "model2path": {"e5": args.retriever_path, "llama3-8B-instruct": args.model_path},
        "generator_model": "llama3-8B-instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc", "rouge-1", "rouge-2", "rouge-l"],
        "gpu_id": args.gpu_id,
        "retrieval_batch_size": 512,
        "return_embedding": True,
        "framework": "vllm",
        "use_fp16": True,
        "pooling_method": "mean",
        "generator_max_input_len": 4096,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
        "generation_params": {'max_tokens': 128},
        "task_batch": 32,
        "iter_num": 2,
        "print_log": True,
        "continuous_retrieval": False,
        "nprobe_minibatch": args.nprobe_minibatch,
        "speculative_retrieval": args.speculative_retrieval,
        "spec_method": args.spec_method,
        "spec_total_size": args.spec_total_size,
        "gpu_onloading": args.gpu_onloading
    }

    config = Config(config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["test"]

    questions = []
    for item in test_data:
        questions.append(item.question)
    string_list = questions

    executor = HeteragExecutor(config)

    # example 
    # executor.add_requests_string(string_list, workflow = "Sequential" | "HyDE" | "RECOMP" | "Multistep" | "IRG")
    executor.add_requests_string(string_list, workflow = args.rag_workflow)

    t1 = time.time()
    start_time_list = [t1 for _ in range(len(string_list))]

    while executor.execute() < len(string_list):
        pass

    t2 = time.time()

    query_latency = []
    for key, values in executor.request_time_dict.items():
        query_latency.append(values - start_time_list[key])
    print(f"Average query latency: {np.mean(query_latency)} s")

    executor.finalize()

    for i in range(min(10, len(string_list))):
        request = executor.output_list[i]
        print(f"Answer {request.id}: {request.answer}")

    print("[average word]", executor.generate_words / len(string_list))

if __name__ == "__main__":
    main()
