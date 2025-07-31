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

def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def iterative_rag(
    queries: list[str],
    encoder,
    retriever,
    corpus,
    llm,
    write_file: str = None,
    max_iter: int = 2,
    request_per_second: int = 16,
    top_k: int = 1,
    nprobe: int = 512
):
    prompt_template = PromptTemplate(
        input_variables=["reference", "user_prompt"],
        template="{reference}\n{user_prompt}"
    )
    chain = prompt_template | llm

    results = []
    query_latency = []
    start_time_list = [i / request_per_second for i in range(len(queries))]
    t1 = time.time()
    current_idx = 0

    current_queries = queries[:]
    final_answers = ["" for _ in queries]

    while current_idx < len(queries):
        t_batch_start = time.time()
        current_time = t_batch_start - t1
        query_batch = []
        batch_start_indices = []

        while current_idx < len(queries) and current_time >= start_time_list[current_idx]:
            query_batch.append(current_queries[current_idx])
            batch_start_indices.append(current_idx)
            current_idx += 1
            current_time = time.time() - t1

        if not query_batch:
            time.sleep(0.001)
            continue

        for _ in range(max_iter):
            query_vecs = encoder.encode(query_batch)
            scores, idxs = retriever.search(query_vecs, k=top_k)

            input_dicts = []
            for j in range(len(query_batch)):
                docs = load_docs(corpus, np.array(idxs[j]).reshape(-1))
                reference_str = "\n".join([doc["text"] for doc in docs])
                user_prompt = f"Question: {queries[batch_start_indices[j]]}, Paragraph: {query_batch[j]}"

                input_dicts.append({
                    "reference": f"Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words.\nThe following are given documents.\n\n{reference_str}",
                    "user_prompt": user_prompt
                })

            responses = chain.batch(input_dicts)
            for j, response in enumerate(responses):
                query_batch[j] = f"{queries[batch_start_indices[j]]} Based on this: {response}"
                final_answers[batch_start_indices[j]] = response

        q_end = time.time()
        for j in range(len(query_batch)):
            latency = q_end - (t1 + start_time_list[batch_start_indices[j]])
            query_latency.append(latency)
            results.append({
                "question": queries[batch_start_indices[j]],
                "answer": final_answers[batch_start_indices[j]],
            })

    t2 = time.time()
    avg_latency = np.mean(query_latency)
    print(f"\nAverage query latency: {avg_latency:.4f} s")

    if write_file:
        with open(write_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iterative", nprobe, request_per_second, avg_latency])

    return results
