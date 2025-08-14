# generator initialization built upon FlashRAG
from abc import ABC, abstractmethod
from heterag.retriever.utils import *
from heterag.retriever.encoder import Encoder, STEncoder
from heterag.retriever.engine import BaseEngine
from heterag.utils import TaskID
from typing import Union, List, Dict
import faiss
import time
import numpy as np
from heapq import heappush, heappop

class DenseHeteEngine(BaseEngine):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        self.indextype = self.index.__class__.__name__
        if config["faiss_gpu"]:
            raise ValueError("GPU retrieval currently not implemented!")

        if (config["corpus_source"] == "hugging face"):
            self.corpus = load_corpus(self.corpus_path, hf = True)
        else:
            self.corpus = load_corpus(self.corpus_path)

        if config["use_sentence_transformer"]:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=config["retrieval_model_path"],
                max_length=config["retrieval_query_max_length"],
                use_fp16=config["retrieval_use_fp16"],
            )
        else:
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=config["retrieval_model_path"],
                pooling_method=config["retrieval_pooling_method"],
                max_length=config["retrieval_query_max_length"],
                use_fp16=config["retrieval_use_fp16"],
            )
        self.topk = config["retrieval_topk"]
        self.batch_size = self.config["retrieval_batch_size"]

        self.reinit(config)

    def reinit(self, config):

        if "IVF" in self.indextype:

            if "nprobe" in self.config:
                self.index.nprobe = config["nprobe"]
            else:
                self.index.nprobe = 32
            self.total_nprobe = self.index.nprobe

            if "nprobe_minibatch" in self.config:
                self.nprobe_minibatch = config["nprobe_minibatch"]
                self.index.nprobe = self.nprobe_minibatch

        elif "IndexFlat" in self.indextype:

            if config["subset_num"]:
                self.subset_num = config["subset_num"]
            else:
                self.subset_num = 4

            self.ntotal = self.index.ntotal
            self.subset_size = int((self.index.ntotal + self.subset_num - 1) / self.subset_num)

            self.current_subset = 0

        else:
            raise ValueError("Not implemented")

        self.idxs_cache = dict()

        self.request_dict = {}

        # count when new centroid dist increases, the closest distance also increases.
        self.total_iter_clusters = 0
        self.total_overlap_clusters = 0
        self.total_far_clusters = 0
        self.total_far_dist = 0

        # request list
        self.new_request_list = []

        # time profile
        self.time_profile = True
        self.new_request_time = 0
        self.process_batch_time = 0
        self.search_time = 0
        self.save_and_load_time = 0

        self.latency = []
        self.finished_requests = 0
        self.finished_search = 0
        self.request_batch_size = 0

        # total batch workload
        self.total_batch_workload = []
        self.search_time_dict = {}

        # self.heterag_profile()

    def heterag_profile(self):
        profile_string = ["Hello, this is a profiler." for _ in range(32)]

        if "IVF" in self.indextype:
            t1 = time.time()

            profile_emb = self.encoder.encode(profile_string, profile = True)
            profile_emb = self.encoder.encode(profile_string, profile = True)

            t2 = time.time()

            profile_distances, profile_centroids = self.index.quantizer.search(profile_emb, self.index.nprobe)

            t3 = time.time()

            scores, idx, cmin_dist, cmin_id = self.index.search_preassigned_heterag(profile_emb, self.topk, profile_centroids, profile_distances)

            t4 = time.time()

            print(f"encoding {t2 - t1}")
            print(f"quantizer {t3 - t2}")
            print(f"search {t4 - t3}")
        
        elif "IndexFlat" in self.indextype:

            t1 = time.time()

            profile_emb = self.encoder.encode(profile_string, profile = True)
            profile_emb = self.encoder.encode(profile_string, profile = True)

            t2 = time.time()

            profile_distances, profile_centroids = self.index.search(profile_emb, k=self.topk)

            t3 = time.time()

            print(f"encoding {t2 - t1}")
            print(f"search {t3 - t2}")

        else:
            raise ValueError("Not implemented")


    def add_requests(self, query_list: List[str], task_id_list: List[TaskID], num: List[int]):     

        for query, taskid, nprobe in zip(query_list, task_id_list, num):

            if "IVF" in self.indextype:
                new_request = RetrievalTask(query, taskid, 0, self.topk, nprobe = nprobe, start_time = time.time())
            elif "IndexFlat" in self.indextype:
                new_request = RetrievalTask(query, taskid, 0, self.topk, start_time = time.time())
            else:
                raise ValueError("Not implemented")

            self.new_request_list.append(new_request)
    
    def show_time_profile(self):
        if self.time_profile:
            print(f"new request {self.new_request_time}")
            print(f"process batch {self.process_batch_time}")
            print(f"search {self.search_time}")
            print(f"save and load {self.save_and_load_time}")

            self.avg_latency = np.mean(self.latency)
            print(f"average latency {self.avg_latency}")
            print(f"[average batch] {self.request_batch_size / self.finished_search}")

    
    def step(self):

        new_size = len(self.new_request_list)

        if self.time_profile:
            t1 = time.time()

        if "IVF" in self.indextype:

            # calculate the preassigned clusters
            if new_size:
                new_list = self.new_request_list[:new_size]
                self.new_request_list = self.new_request_list[new_size:]

                if self.time_profile:
                    t1 = time.time()

                new_emb_list = []
                for request in new_list:
                    new_emb_list.append(request.query)
                new_batch_emb = self.encoder.encode(new_emb_list)

                # now assume all the nprobes are the same
                centroid_distances, batch_assigned_centroids = self.index.quantizer.search(new_batch_emb, self.total_nprobe)

                cluster_size_list = self.index.get_cluster_size_heterag(batch_assigned_centroids)

                for request, assigned_centroid, centroid_distance, query_emb, cluster_size in zip(new_list, batch_assigned_centroids, centroid_distances, new_batch_emb, cluster_size_list):
                    if request.taskid.id not in self.request_dict:
                        self.request_dict[request.taskid.id] = request
                    request.add_clusters(assigned_centroid, dist = centroid_distance)
                    request.update_emb(query_emb)
                    request.update_start_time(time.time())
                    request.update_cluster_size(cluster_size)

            if self.time_profile:
                t2 = time.time()
                self.new_request_time += t2 - t1

            # maybe optimized
            query_emb = []
            cluster_to_process = []
            cluster_dist_to_process = []
            topk_id = []
            topk_score = []
            qid_list = []
            batch_workload = []
            for qid, request in self.request_dict.items():
                query_emb.append(request.emb)
                new_clusters, new_dist, new_size = request.pop_clusters(self.nprobe_minibatch)

                cluster_to_process.append(new_clusters)
                cluster_dist_to_process.append(new_dist)
                topk_id.append(request.topk_id)
                topk_score.append(request.topk_score)
                qid_list.append(qid)
                batch_workload.append(new_size)
                if len(query_emb) >= self.batch_size:
                    break
            if len(batch_workload):
                self.total_batch_workload.append(batch_workload)

            batch_results = []
            taskid_list = []
            if len(query_emb):
                self.request_batch_size += len(query_emb)
                query_emb = np.array(query_emb)
                topk_score = np.array(topk_score)
                topk_id = np.array(topk_id)

                if self.time_profile:
                    t3 = time.time()
                    self.process_batch_time += t3 - t2

                if self.time_profile:
                    t3 = time.time()

                self.index.search_preassigned_heterag(query_emb, self.topk, cluster_to_process, cluster_dist_to_process, D = topk_score, I = topk_id, init_heap = False)
                self.finished_search += 1

                if self.time_profile:
                    t4 = time.time()
                    current_search_time = t4 - t3
                    self.search_time += t4 - t3

                idx_list = []
                for qid, qtopkid, qtopkscore in zip(qid_list, topk_id, topk_score):
                    if self.time_profile:
                        self.request_dict[qid].search_time += current_search_time
                    if self.request_dict[qid].is_empty:
                        idx_list.append(qtopkid)
                        self.request_dict[qid].update_end_time(time.time())
                        taskid_list.append(self.request_dict[qid].taskid)

                        self.latency.append(self.request_dict[qid].end_time - self.request_dict[qid].start_time)
                        self.finished_requests += 1
                        self.search_time_dict[qid] = self.request_dict[qid].search_time
                        self.request_dict.pop(qid)
                    else:
                        self.request_dict[qid].topk_id = qtopkid
                        self.request_dict[qid].topk_score = qtopkscore

                flat_idxs = np.array(idx_list).reshape(-1)
                batch_results = load_docs(self.corpus, flat_idxs)
                batch_results = [batch_results[i * self.topk : (i + 1) * self.topk] for i in range(len(idx_list))]

                if self.time_profile:
                    t5 = time.time()
                    self.save_and_load_time += t5 - t4


        elif "IndexFlat" in self.indextype:

            if new_size:
                new_list = self.new_request_list[:new_size]
                self.new_request_list = self.new_request_list[new_size:]

                if self.time_profile:
                    t1 = time.time()

                new_emb_list = []
                for request in new_list:
                    new_emb_list.append(request.query)
                new_batch_emb = self.encoder.encode(new_emb_list)

                for request, query_emb in zip(new_list, new_batch_emb):
                    if request.taskid.id not in self.request_dict:
                        self.request_dict[request.taskid.id] = request
                    request.update_emb(query_emb)
                    request.update_start_time(time.time())

            # maybe optimized
            query_emb = []
            topk_id = []
            topk_score = []
            qid_list = []
            for qid, request in self.request_dict.items():
                query_emb.append(request.emb)
                topk_id.append(request.topk_id)
                topk_score.append(request.topk_score)
                qid_list.append(qid)

            if self.time_profile:
                t2 = time.time()
                self.new_request_time += t2 - t1
            
            batch_results = []
            taskid_list = []
            if len(query_emb):
                self.request_batch_size += len(query_emb)
                query_emb = np.array(query_emb)
                topk_score = np.array(topk_score)
                topk_id = np.array(topk_id)

                if self.time_profile:
                    t3 = time.time()
                    self.process_batch_time += t3 - t2

                subset_start = self.current_subset * self.subset_size
                subset_end = subset_start + self.subset_size
                new_scores, new_ids = self.index.search_with_idx(query_emb, subset_start, subset_end, self.topk)

                if self.time_profile:
                    t4 = time.time()
                    self.search_time += t4 - t3

                idx_list = []
                for qid, qoldid, qoldscore, qnewid, qnewscore in zip(qid_list, topk_id, topk_score, new_scores, new_ids):
                    self.request_dict[qid].update_iter_num(1)

                    qtopkid, qtopkscore = topk_merge(qoldscore, qnewscore, qoldid, qnewid, self.topk)

                    if self.request_dict[qid].subset_iter == self.subset_num:
                        idx_list.append(qtopkid)
                        self.request_dict[qid].update_end_time(time.time())
                        taskid_list.append(self.request_dict[qid].taskid)

                        self.latency.append(self.request_dict[qid].end_time - self.request_dict[qid].start_time)
                        self.finished_requests += 1
                        self.request_dict.pop(qid)
                    else:
                        self.request_dict[qid].topk_id = qtopkid
                        self.request_dict[qid].topk_score = qtopkscore

                flat_idxs = np.array(idx_list).reshape(-1)
                batch_results = load_docs(self.corpus, flat_idxs)
                batch_results = [batch_results[i * self.topk : (i + 1) * self.topk] for i in range(len(idx_list))]

                if self.time_profile:
                    t5 = time.time()
                    self.save_and_load_time += t5 - t4

                self.current_subset = (self.current_subset + 1) % self.subset_num

        else:
            raise ValueError("Not implemented!")

        return taskid_list, batch_results