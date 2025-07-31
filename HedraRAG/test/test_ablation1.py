import multiprocessing as mp

def main():

    from heterag import Config, HeteragExecutor, SequentialRAGraph, get_dataset, Request
    from heterag.retriever.utils import load_corpus, load_docs
    from heterag.executor.retrieval_worker import retrieval_worker, retrieval_worker_heterag
    from heterag.utils import TaskID
    import numpy as np
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retriever_path", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--request_per_second", type=float, default=16)
    parser.add_argument("--nprobe", type=int, default=512)
    parser.add_argument("--nprobe_minibatch", type=int, default=128)
    parser.add_argument("--subset_num", type=int, default=8)
    parser.add_argument("--write_file", default=None)

    parser.add_argument("--index_path", type=str, default="/data/index_0319/wiki/IVF4096/ivf.index")
    parser.add_argument("--corpus_path", type=str, default="/data/wiki_dataset/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl")

    args = parser.parse_args()

    config_dict = {
        "data_dir": "2wikimultihopqa",
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
        "retrieval_batch_size": 32,
        "return_embedding": False,
        "framework": "vllm",
        "use_fp16": True,
        "pooling_method": "mean",
        "generator_max_input_len": 4096,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": False,
        "generation_params": {'max_tokens': 128},
        "max_retrieval_batch": 64,
        "continuous_retrieval": True,
        "nprobe_minibatch": args.nprobe_minibatch,
        "use_llm": False,
        "gpu_onloading": False,
        "speculative_retrieval": False,
        "subset_num": args.subset_num
    }

    config = Config(config_dict=config_dict)
    
    executor = HeteragExecutor(config)

    all_split = get_dataset(config)
    test_data = all_split["test"][0:512]

    questions = []
    for item in test_data:
        questions.append(item.question)

    string_list = questions

    rps_list = range(4, 30, 4)

    # warm up
    start_time_list = [0 for _ in range(len(string_list))]
    executor.faiss_benchmark(string_list, start_time_list)

    # run rps

    for rps in rps_list:
        executor.reinit(config)

        np.random.seed(2025)
        if args.request_per_second != 0:
            interval = 1.0 / args.request_per_second
            intervals = np.random.exponential(scale=interval, size=len(string_list))
            start_time_list = np.cumsum(intervals).tolist()
        else:
            start_time_list = [0 for _ in range(len(string_list))]

        final_time = executor.faiss_benchmark(string_list, start_time_list)

        import csv

        if args.write_file is not None:
            with open(args.write_file, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([args.nprobe, args.nprobe_minibatch, rps, np.mean(final_time)])
    
    executor.finalize()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()