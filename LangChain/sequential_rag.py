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

DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_SYS_TMPL = (
    "The original question is as follows: {initial_query}\n"
    "We have an opportunity to answer some, or all of the question from a knowledge source. "
    "Previous reasoning steps are provided below.\n"
    "Given the previous reasoning, return a question that can be answered."
    "This question can be the same as the original question, "
    "or this question can represent a subcomponent of the overall question."
    "It should not be irrelevant to the original question.\n"
    "If we cannot extract more information from the context, provide 'None' as the answer. "
    "Some examples are given below: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Previous reasoning: None\n"
    "Next question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: How many Grand Slam titles does Novak Djokovic have?"
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Australian Open - includes biographical information for each winner\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "- How many Grand Slam titles does Novak Djokovic have?\n"
    "- Novak Djokovic has 24 Grand Slam titles.\n"
    "New question: None"
    "\n\n"
    "Only output the question, do not output any other word.\n"
)

DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_USR_TMPL = (
    "Question: {question}\n"
    "Knowledge source context: {reference}\n"
    "Previous reasoning: {prev_reasoning}\n"
    "New question: "
)

def load_corpus(corpus_path: str, hf=False):
    if hf:
        corpus = load_dataset("json", data_files=corpus_path)
        corpus = corpus['train']
    return corpus

def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]

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
    else:
        print("Warning: This index is not an IVF index; 'nprobe' has no effect.")   
    return index, encoder

def get_llm():
    return VLLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_new_tokens=200,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
    )

def sequential_rag(
    queries: list[str],
    encoder,
    retriever,
    corpus,
    llm,
    write_file: str = None,
    request_per_second: int = 16,
    top_k: int = 1,
    nprobe: int = 512
):
    prompt_template = PromptTemplate(
        input_variables=["batch_results", "query"],
        template="Answer the question based on the given documents."
        " Please give a complete sentence and a single answer. Do not output other words after the answer. "
        "\nThe following are given documents:\n{batch_results}\nQuestion: {query}",
    )
    chain = prompt_template | llm

    results = []
    query_latency = []

    start_time_list = [i / request_per_second for i in range(len(queries))]
    current_idx = 0
    t1 = time.time()

    while current_idx < len(queries):
        current_time = time.time() - t1
        query_batch = []
        batch_start_indices = []

        while current_idx < len(queries) and current_time >= start_time_list[current_idx]:
            query_batch.append(queries[current_idx])
            batch_start_indices.append(current_idx)
            current_idx += 1
            current_time = time.time() - t1

        if not query_batch:
            time.sleep(0.001)
            continue

        encode_start = time.time()
        query_vecs = encoder.encode(query_batch)
        scores, idxs = retriever.search(np.array(query_vecs), k=top_k)
        encode_end = time.time()
        print(f"[Batch @ idx {batch_start_indices[0]}] Encoding time: {encode_end - encode_start:.4f} s")

        input_dicts = []
        for j in range(len(query_batch)):
            doc_idxs = np.array(idxs[j]).reshape(-1)
            batch_results = load_docs(corpus, doc_idxs)
            input_dicts.append({
                "query": query_batch[j],
                "batch_results": batch_results,
            })

        gen_start = time.time()
        answers = chain.batch(input_dicts)
        gen_end = time.time()
        print(f"[Batch @ idx {batch_start_indices[0]}] Generation time: {gen_end - gen_start:.4f} s")

        q_end = time.time()
        for j, answer in enumerate(answers):
            launch_time = t1 + start_time_list[batch_start_indices[j]]
            latency = q_end - launch_time
            query_latency.append(latency)

            results.append({
                "question": query_batch[j],
                "answer": answer,
            })

    avg_latency = float(np.mean(query_latency))
    print(f"\n[Sequential (Rate-Limited)] Average query latency: {avg_latency:.4f} s")

    if write_file:
        with open(write_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Langchain","sequential", nprobe, request_per_second, avg_latency])

    return results
