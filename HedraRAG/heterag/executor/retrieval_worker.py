import transformers
import time
from heterag.ragraph.ragraph import *
from heterag.executor.request import *
from heterag.executor.retrieval_worker import *
import importlib
from typing import Union, List
import numpy as np
import csv

def get_retriever(config):
    if config["continuous_retrieval"] == False:
        print("Using old engine")
        if config["retrieval_method"] == "e5":
            return getattr(importlib.import_module("heterag.retriever"), "DenseEngine")(config)
        else:
            raise ValueError("Not supported retriever")
    else:
        print("gpu_onloading is", config["gpu_onloading"])
        if config["speculative_retrieval"] == True:
            if config["gpu_onloading"]:
                if config["retrieval_method"] == "e5":
                    return getattr(importlib.import_module("heterag.retriever"), "DenseHeteEngineSpecOnloading")(config)
                else:
                    raise ValueError("Not supported retriever")
            else:
                if config["retrieval_method"] == "e5":
                    return getattr(importlib.import_module("heterag.retriever"), "DenseHeteEngineSpec")(config)
                else:
                    raise ValueError("Not supported retriever")
        else:
            if config["retrieval_method"] == "e5":
                return getattr(importlib.import_module("heterag.retriever"), "DenseHeteEngine")(config)
            else:
                raise ValueError("Not supported retriever")        
            

# A separate process for database retrieval
def retrieval_worker(config, task_queue, result_queue, worker_log = False, return_emb = False, request_per_second = 0):

    if (worker_log):
        print("Old Retrieval engine initiating!")

    retriever = get_retriever(config)

    result_queue.put("Retrieval engine initiated")

    if (worker_log):
        print("Retrieval engine start!")
    
    max_batch = config["max_retrieval_batch"] if config["max_retrieval_batch"] is not None else 64

    num_retrieval = 0
    retrieval_time = 0

    worker_running = True

    loop_start_time = time.time()

    while (worker_running):

        input_query = []
        query_id_list = []

        recv_task = None
        while not task_queue.empty() and len(input_query) < max_batch:
            recv_task = task_queue.get()
            if recv_task is not None:
                if (recv_task == "RETRIEVAL END"):
                    worker_running = False
                elif (recv_task == "REINIT"):
                    config = task_queue.get()
                    retriever.reinit(config)
                else:
                    input_query.extend(recv_task[0])
                    query_id_list.extend(recv_task[1])

        if len(input_query):

            start_time = time.time()

            retrieval_results = retriever._batch_search(input_query, query_id_list)
            
            num_retrieval += 1

            result_queue.put((query_id_list, retrieval_results))

            end_time = time.time()
            retrieval_time += end_time - start_time

    loop_end_time = time.time()

    retriever.show_inter_diff()

    avg_depulicate_rate = 0.0
    avg_num = 0
    for rid, emb_info in retriever.request_dict.items():
        if len(emb_info.centroid_idx) > 1:
            depulicate_rate = 0.0
            for iter_id in range(len(emb_info.centroid_idx) - 1):
                c1 = emb_info.centroid_idx[iter_id]
                c2 = emb_info.centroid_idx[iter_id + 1]
                depulicate_rate += len(set(c1) & set(c2)) / len(c2)
            avg_depulicate_rate += depulicate_rate / (len(emb_info.centroid_idx) - 1)
            avg_num += 1
    if avg_num > 0:
        avg_depulicate_rate /= avg_num
    print(f"average duplicate rate {avg_depulicate_rate}")

    if worker_log:
        print("Total retrieval time: {:5f}s, retrieval count: {:d}".format(retrieval_time, num_retrieval))
        print(f"Total loop time: {loop_end_time - loop_start_time} s.")

        print("largest_cluster", retriever.largest_cluster_ranking)
        print("Average largest cluster ID", np.mean(retriever.largest_cluster_ranking))
        print("Larger cluster id", np.count_nonzero(np.array(retriever.largest_cluster_ranking) > retriever.index.nprobe / 2))

        print(f"total new doc search {retriever.answer_in_doc_search}")
        print(f"[clusters] answer is in old top 20 {retriever.all_answer_in_doc}, avg distance {np.mean(retriever.all_answer_in_doc_list)}")
        print(f"[clusters] answer is in old top 20 hit centroid {retriever.in_old_hit_centroid_top20}, avg distance {np.mean(retriever.in_old_hit_centroid_top20_list)}")
        print(f"[clusters] answer is in last retrieval centroid {retriever.in_old_centroid_set} avg distance {np.mean(retriever.in_old_centroid_set_list)}")

        sorted_values = [v for k, v in sorted(retriever.skewness_dict.items(), key=lambda item: item[1])]
        print(f"skewness dict check {np.sum(sorted_values)}")

    retriever.show_time_profile()

    if retriever.return_embedding:
        print("total cluster", retriever.total_iter_clusters)
        print("total overlap cluster",  retriever.total_overlap_clusters)
        print("total far cluster", retriever.total_far_clusters)
        print("total far distance", retriever.total_far_dist)

# A separate process for database retrieval
def retrieval_worker_heterag(config, task_queue, result_queue, worker_log = False, return_emb = False, request_per_second = 0):
    if (worker_log):
        print("Retrieval engine initiating!")

    retriever = get_retriever(config)

    result_queue.put("Retrieval engine initiated")

    if (worker_log):
        print("Retrieval engine start!")
    
    max_batch = config["max_retrieval_batch"] if config["max_retrieval_batch"] is not None else 64

    num_retrieval = 0
    retrieval_time = 0

    worker_running = True

    loop_start_time = time.time()

    while (worker_running):

        input_query = []
        query_id_list = []

        recv_task = None
        while not task_queue.empty() and len(input_query) < max_batch:
            recv_task = task_queue.get()
            if recv_task is not None:
                if (recv_task == "RETRIEVAL END"):
                    worker_running = False
                elif (recv_task == "REINIT"):
                    config = task_queue.get()
                    retriever.reinit(config)
                else:
                    input_query.extend(recv_task[0])
                    query_id_list.extend(recv_task[1])
        
        if len(input_query):

            if "IVF" in retriever.indextype:
                retriever.add_requests(input_query, query_id_list, [retriever.index.nprobe for _ in range(len(input_query))])
            elif "IndexFlat" in retriever.indextype:
                retriever.add_requests(input_query, query_id_list, [0 for _ in range(len(input_query))])
            else:
                raise ValueError("Not implemented!")

        start_time = time.time()

        qid_list, retrieval_results = retriever.step()

        end_time = time.time()
        retrieval_time += end_time - start_time

        if (len(qid_list)):
            num_retrieval += 1
            # for taskid in qid_list:
            result_queue.put((qid_list, retrieval_results))

    loop_end_time = time.time()

    if worker_log:
        print("Total retrieval time: {:5f}s, retrieval count: {:d}".format(retrieval_time, num_retrieval))
        print(f"Total loop time: {loop_end_time - loop_start_time} s.")
    
    if config["spec_output_file"] is not None:
        print("writing", config["spec_output_file"])
        with open(config["spec_output_file"], 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([retriever.spec_method, retriever.spec_total_size, config["request_per_second"], 1 - retriever.wrong_spec / retriever.finished_requests])

    # retriever.show_time_profile()