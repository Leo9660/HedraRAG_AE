from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import FAISS
from datasets import load_dataset
from encoder import Encoder
from transformers import pipeline
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import time
import torch
from langchain.llms import OpenAI
from utils import *
import csv
from langchain_community.llms import VLLM
import argparse

from sequential_rag import sequential_rag
from hyde_rag import hyde_rag
from recomp_rag import recomp_rag
from iterative_rag import iterative_rag
from multistep_rag import multistep_rag

from sequential_offline import sequential_rag_offline
from hyde_offline import hyde_rag_offline
from recomp_offline import recomp_rag_offline
from iterative_offline import iterative_rag_offline
from multistep_offline import multistep_rag_offline

def load_corpus(corpus_path: str, hf=False):
    if hf:
        corpus = load_dataset("json", data_files=corpus_path)
        corpus = corpus['train']
    return corpus

def load_retriever(faiss_index_path: str, nprobe: int = 128):
    encoder = Encoder(
        model_name="e5-large-v2",
        model_path="intfloat/e5-large-v2",
        pooling_method="mean",
        max_length=512,
        use_fp16=False
    )
    index = faiss.read_index(faiss_index_path)
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = nprobe
        print("NPROBE HAS BEEN SET TO: ",index.nprobe)
    else:
        print("Warning: This index is not an IVF index; 'nprobe' has no effect.")   
    return index, encoder

def get_llm():
    return VLLM(
        # model="meta-llama/Llama-3.1-8B-Instruct",
        #model="meta-llama/Llama-2-13b-chat-hf",
        #model = "/home/vibha/models/Llama-2-13b-chat",
        model="facebook/opt-iml-30b",
        trust_remote_code=True,
        max_new_tokens=200,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
    )


def hybrid_dual_rag(
    queries: list[str],
    encoder,
    retriever,
    corpus,
    llm,
    workflow1,
    workflow2,
    write_file: str = None,
    request_per_second: int = 16,
    top_k: int = 1,
    nprobe: int = 512
):
    results = []
    query_latency = []
    start_time_list = [i / request_per_second for i in range(len(queries))]

    current_idx = 0
    t1 = time.time()

    while current_idx < len(queries):
        t_batch_start = time.time()
        current_time = t_batch_start - t1
        batch_even = []
        batch_odd = []
        batch_even_idx = []
        batch_odd_idx = []

        while current_idx < len(queries) and current_time >= start_time_list[current_idx]:
            if current_idx % 2 == 0:
                batch_even.append(queries[current_idx])
                batch_even_idx.append(current_idx)
            else:
                batch_odd.append(queries[current_idx])
                batch_odd_idx.append(current_idx)
            current_idx += 1
            current_time = time.time() - t1

        if not batch_even and not batch_odd:
            time.sleep(0.001)
            continue

        batch_results = []
        t_batch_exec_start = time.time()

        if batch_even:
            w2_out = workflow2(
                queries=batch_even,
                encoder=encoder,
                retriever=retriever,
                corpus=corpus,
                llm=llm,
                top_k=top_k,
                request_per_second=request_per_second,
                write_file=None,
                nprobe=nprobe
            )
            batch_results.extend(zip(batch_even_idx, w2_out))

        if batch_odd:
            w1_out = workflow1(
                queries=batch_odd,
                encoder=encoder,
                retriever=retriever,
                corpus=corpus,
                llm=llm,
                top_k=top_k,
                request_per_second=request_per_second,
                write_file=None,
                nprobe=nprobe
            )
            batch_results.extend(zip(batch_odd_idx, w1_out))

        t_batch_exec_end = time.time()

        for idx, res in batch_results:
            launch_time = t1 + start_time_list[idx]
            latency = t_batch_exec_end - launch_time
            query_latency.append(latency)
            results.append(res)

    avg_latency = float(np.mean(query_latency))
    print(f"\n[HYBRID] Average query latency: {avg_latency:.4f} s")

    if write_file:
        with open(write_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hybrid"+str(workflow1)+str(workflow2), nprobe, request_per_second, avg_latency])

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=str, required=True, choices=["sequential", "hyde", "recomp", "iterative", "multistep", "sequential_offline", "hyde_offline", "recomp_offline", "iterative_offline", "multistep_offline", "seqrec", "hyderec", "recmulti", "multiiter"])
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "data_source": "huggingface",
        "split_name": "dev",
        "index_path": "/dataset/index_0319/wiki/IVF4096/ivf.index",
        "corpus_path":  "/../../../dataset/wiki_dataset/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl",
        "nprobe": args.nprobe,
        "retrieval_topk": 1,
        "generator_model": "llama3-8B-instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc", "rouge-1", "rouge-2", "rouge-l"],
        "retrieval_batch_size": 512,
        "return_embedding": True,
        "framework": "vllm",
        "use_fp16": True,
        "pooling_method": "mean",
        "generator_max_input_len": 4096,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
        "generation_params": {'max_tokens': 128},
        "task_batch": 64,
        "iter_num": 2,
        "print_log": True,
        "continuous_retrieval": True,
        "nprobe_minibatch": 64,
        "speculative_retrieval": False
    }

    config = Config(config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["test"][:100]
    questions = [item.question for item in test_data]

    faiss_index_path = config_dict["index_path"]
    corpus_path = config_dict["corpus_path"]
    retriever, encoder = load_retriever(faiss_index_path, args.nprobe)
    corpus = load_corpus(corpus_path, hf=True)
    llm = get_llm()

    if args.nprobe == 512:
        rps_list = range(1, 11, 1)
    elif args.nprobe == 256:
        rps_list = range(2, 22, 2)
    else:
        rps_list = range(4, 44, 4)
    
    for rps in rps_list:
        if args.workflow == "sequential":
            sequential_rag(encoder=encoder, retriever=retriever, corpus=corpus, llm=llm, queries=questions, write_file="sequential.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe)

        elif args.workflow == "hyde":
            hyde_rag(queries=questions, write_file="hyde.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)

        elif args.workflow == "recomp":
            recomp_rag(queries=questions, write_file="recomp.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)
        
        elif args.workflow == "iterative":
            iterative_rag(queries=questions, write_file="iterative.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)

        elif args.workflow == "multistep":
            multistep_rag(queries=questions, write_file="multistep.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)
        
        elif args.workflow == "sequential_offline":
            sequential_rag_offline(encoder=encoder, retriever=retriever, corpus=corpus, llm=llm, queries=questions, write_file="sequential_offline.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe)

        elif args.workflow == "hyde_offline":
            hyde_rag_offline(queries=questions, write_file="hyde_offline.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)

        elif args.workflow == "recomp_offline":
            recomp_rag_offline(queries=questions, write_file="recomp_offline.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)
        
        elif args.workflow == "iterative_offline":
            iterative_rag_offline(queries=questions, write_file="iterative_offline.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)

        elif args.workflow == "multistep_offline":
            multistep_rag_offline(queries=questions, write_file="multistep_offline.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)
        
        elif args.workflow == "seqrec":
            hybrid_dual_rag(queries=questions, workflow1 = sequential_rag, workflow2 = recomp_rag, write_file="seqrec.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)

        elif args.workflow == "hyderec":
            hybrid_dual_rag(queries=questions, workflow1 = hyde_rag, workflow2 = recomp_rag, write_file="hydrec.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)

        elif args.workflow == "recmulti":
            hybrid_dual_rag(queries=questions, workflow1 = recomp_rag, workflow2 = multistep_rag, write_file="recmulti.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)
         
        elif args.workflow == "multiiter":
            hybrid_dual_rag(queries=questions, workflow1 = multistep_rag, workflow2 = iterative_rag, write_file="multiiter.csv", request_per_second=rps, top_k=1, nprobe=args.nprobe, encoder=encoder, retriever=retriever, corpus=corpus, llm=llm)
        

if __name__ == "__main__":
    main()
