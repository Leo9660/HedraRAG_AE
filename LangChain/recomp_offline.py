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


def recomp_rag_offline(
    queries: list[str],
    encoder,
    retriever,
    corpus,
    llm,
    request_per_second = 1,
    write_file: str = None,
    top_k: int = 1,
    nprobe: int = 512
):
    summarize_prompt = PromptTemplate(
        input_variables=["reference"],
        template="Please generate a sentence for the following paragraph, based on the information of given documents, do not output other words.\nThe following are given documents.\n\n{reference}"
    )
    summarize_chain = summarize_prompt | llm

    answer_prompt = PromptTemplate(
        input_variables=["summary", "query"],
        template="Use the following summary of documents to answer the question.\n\nSummary: {summary}\nQuestion: {query}"
    )
    answer_chain = answer_prompt | llm

    results = []
    query_latency = []

    t1 = time.time()

    query_vecs = encoder.encode(queries)
    scores, idxs = retriever.search(np.array(query_vecs), k=top_k)

    summaries = []
    for j in range(len(queries)):
        doc_idxs = np.array(idxs[j]).reshape(-1)
        batch_results = load_docs(corpus, doc_idxs)
        reference_str = "\n".join([doc["text"] for doc in batch_results])
        summaries.append(reference_str)

    summarized_outputs = summarize_chain.batch([{"reference": s} for s in summaries])

    input_dicts = []
    for j in range(len(queries)):
        input_dicts.append({
            "summary": summarized_outputs[j],
            "query": queries[j],
        })

    answers = answer_chain.batch(input_dicts)

    t2 = time.time()
    total_latency = t2 - t1
    #avg_latency = total_latency / len(queries)
    print(f"\nRec total query latency:     {total_latency:.4f} s")

    for j, answer in enumerate(answers):
        results.append({
            "question": queries[j],
            "answer": answer,
        })

    if write_file:
        with open(write_file, "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["recomp", nprobe, "-", total_latency])

    return results