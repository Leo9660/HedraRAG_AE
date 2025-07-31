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

def multistep_rag(
    queries: list[str],
    encoder,
    retriever,
    corpus,
    llm,
    request_per_second=1,
    write_file: str = None,
    max_steps: int = 2,
    top_k: int = 1,
    nprobe: int = 512
):
    system_prompt = DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_SYS_TMPL
    user_prompt_template = DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_USR_TMPL

    subq_chain = PromptTemplate(
        input_variables=["prompt"],
        template="{prompt}"
    ) | llm

    answer_prompt = PromptTemplate(
        input_variables=["batch_results", "query"],
        template="Answer the question based on the given documents."
                 " Please give a complete sentence and a single answer. Do not output other words after the answer. "
                 "\nThe following are given documents:\n{batch_results}\nQuestion: {query}"
    )
    answer_chain = answer_prompt | llm

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

        full_answers = ["" for _ in query_batch]
        prev_reasonings = ["None" for _ in query_batch]
        original_queries = query_batch[:]

        for _ in range(max_steps):
            sub_prompts = [
                system_prompt.format(initial_query=original_queries[i]) +
                user_prompt_template.format(
                    question=original_queries[i],
                    reference=full_answers[i],
                    prev_reasoning=prev_reasonings[i]
                )
                for i in range(len(query_batch))
            ]

            subquestions = subq_chain.batch([{"prompt": p} for p in sub_prompts])

            input_dicts = []
            for i, subq in enumerate(subquestions):
                if subq.strip() == "None":
                    input_dicts.append({"query": original_queries[i], "batch_results": ""})
                    continue

                query_vec = encoder.encode([subq])[0].reshape(1, -1)
                scores, idxs = retriever.search(query_vec, k=top_k)
                docs = load_docs(corpus, np.array(idxs).reshape(-1))
                batch_results = "\n".join([doc["text"] for doc in docs])
                input_dicts.append({"query": subq, "batch_results": batch_results})
                prev_reasonings[i] += f"- {subq}\n"

            answers = answer_chain.batch(input_dicts)

            for i, ans in enumerate(answers):
                full_answers[i] += f"{ans}\n"
                prev_reasonings[i] += f"- {ans}\n"

        q_end = time.time()
        for i in range(len(query_batch)):
            launch_time = t1 + start_time_list[batch_start_indices[i]]
            latency = q_end - launch_time
            query_latency.append(latency)
            results.append({"question": query_batch[i], "answer": full_answers[i].strip()})

    avg_latency = float(np.mean(query_latency))
    print(f"\nMultistep Average query latency: {avg_latency:.4f} s")

    if write_file:
        with open(write_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["multistep", nprobe, request_per_second, avg_latency])

    return results
