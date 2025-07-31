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

def iterative_rag_offline(
    queries: list[str],
    encoder,
    retriever,
    corpus,
    llm,
    request_per_second = 1,
    write_file: str = None,
    max_iter: int = 2,
    top_k: int = 1,
    nprobe: int = 512
):
    prompt_template = PromptTemplate(
        input_variables=["reference", "user_prompt"],
        template="{reference}\n{user_prompt}"
    )
    chain = prompt_template | llm

    results = []

    t1 = time.time()

    current_queries = queries[:]
    final_answers = ["" for _ in queries]

    for _ in range(max_iter):
        query_vecs = encoder.encode(current_queries)
        scores, idxs = retriever.search(query_vecs, k=top_k)

        input_dicts = []
        for j in range(len(current_queries)):
            docs = load_docs(corpus, np.array(idxs[j]).reshape(-1))
            reference_str = "\n".join([doc["text"] for doc in docs])
            user_prompt = f"Question: {queries[j]}, Paragraph: {current_queries[j]}"

            input_dicts.append({
                "reference": f"Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words.\nThe following are given documents.\n\n{reference_str}",
                "user_prompt": user_prompt
            })

        responses = chain.batch(input_dicts)
        for j, response in enumerate(responses):
            current_queries[j] = f"{queries[j]} Based on this: {response}"
            final_answers[j] = response

    t2 = time.time()
    total_latency = t2 - t1
    #avg_latency = total_latency / len(queries)
    print(f"\nIterative Average query latency: {total_latency:.4f} s")

    for j, answer in enumerate(final_answers):
        results.append({
            "question": queries[j],
            "answer": answer,
        })

    if write_file:
        with open(write_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iterative", nprobe, len(queries), total_latency])

    return results
