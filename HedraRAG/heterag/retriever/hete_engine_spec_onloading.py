# generator initialization built upon FlashRAG
from abc import ABC, abstractmethod
from heterag.retriever.utils import *
from heterag.retriever.encoder import Encoder, STEncoder
from heterag.retriever.engine import BaseEngine
from heterag.retriever.gpu_engine import HeteRAGGPUEngine
from heterag.utils import TaskID
from typing import Union, List, Dict
import faiss
import time
import numpy as np
import sys
from heapq import heappush, heappop
import copy

class DenseHeteEngineSpecOnloading(BaseEngine):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        # print("reading", self.index_path)
        self.index = faiss.read_index(self.index_path)
        self.nlist = self.index.nlist
        self.indextype = self.index.__class__.__name__
        # print(f"index type {self.indextype}")
        # print("IVF" in self.indextype)
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
        self.large_topk = self.topk

        self.batch_size = self.config["retrieval_batch_size"]
        self.total_cluster = 0
        self.onload_cluster = 0

        if "IVF" in self.indextype:

            if "nprobe" in self.config:
                self.index.nprobe = config["nprobe"]
            else:
                self.index.nprobe = 32
            self.total_nprobe = self.index.nprobe

            if "nprobe_minibatch" in self.config:
                self.nprobe_minibatch = config["nprobe_minibatch"]
                self.index.nprobe = self.nprobe_minibatch

            self.cluster_cache = [CacheStatus() for _ in range(self.nlist)]

            self.gpu_engine = HeteRAGGPUEngine(self.index)
            self.gpu_engine.onload_clusters_from_csv("/workspace/PipeRAG/heteRAG/test/sorted_cluster.csv", config)

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
        self.history_info_dict = {}

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

        self.heterag_profile()

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
                new_request = RetrievalTask(query, taskid, 0, self.large_topk, nprobe = nprobe, start_time = time.time())
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
            print(f"average batch {self.request_batch_size / self.finished_search}")

            # print("workload distribution", self.total_batch_workload)
    
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

                    # Use historical info for retrieval optimization
                    qid = request.taskid.id
                    if qid in self.history_info_dict:
                        old_assigned_centroid = self.history_info_dict[qid].assigned_cluster
                        hit_cluster = self.history_info_dict[qid].hit_cluster
                        # rerank
                        hit_set = set(hit_cluster)
                        old_set = set(old_assigned_centroid)
                        # print("[set]", hit_set, old_set)
                        indices = np.arange(len(assigned_centroid))
                        sorted_indices = sorted(
                            indices,
                            key=lambda i: (0 if assigned_centroid[i] in hit_set else 1 if assigned_centroid[i] in old_set else 2))
                        
                        assigned_centroid = assigned_centroid[indices]
                        centroid_distance = centroid_distance[indices]

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
            # cluster_to_process = []
            cluster_dist_to_process = []
            topk_id = []
            topk_score = []
            qid_list = []
            batch_workload = []
            gpu_cluster_to_process = []
            cpu_cluster_to_process = []

            for qid, request in self.request_dict.items():
                # print("qid", qid, "request", request)
                query_emb.append(request.emb)
                new_clusters, new_dist, new_size = request.pop_clusters(self.nprobe_minibatch)

                # print("debug new_clusters", new_clusters, new_dist, new_size)
                cpu_clusters = []; cpu_dist = []; cpu_size = []
                gpu_clusters = []
                for cluster, dist in zip(new_clusters, new_dist):
                    if self.gpu_engine.has_cluster(cluster):
                        gpu_clusters.append(cluster)
                    else:
                        cpu_clusters.append(cluster)
                        cpu_dist.append(dist)
                cpu_size.append(new_size)

                self.total_cluster += len(new_clusters)
                self.onload_cluster += len(gpu_clusters)

                cpu_cluster_to_process.append(cpu_clusters)
                gpu_cluster_to_process.append(gpu_clusters)
                cluster_dist_to_process.append(cpu_dist)
                batch_workload.append(cpu_size)

                topk_id.append(request.topk_id)
                topk_score.append(request.topk_score)
                qid_list.append(qid)

            # if len(query_emb):
            #     self.gpu_engine.show_clusters()
            #     print(f"{len(query_emb)}, {len(cpu_clusters)} on cpu and {len(gpu_clusters)} on gpu")
            #     print("cpu clusters", cpu_cluster_to_process)
            #     print("gpu clusters", gpu_cluster_to_process)

            if len(batch_workload):
                self.total_batch_workload.append(batch_workload)

            batch_results = []
            taskid_list = []
            if len(query_emb):
                self.request_batch_size += len(query_emb)
                # print(f"len(query_emb) is {len(query_emb)}")
                # print(cluster_to_process)
                query_emb = np.array(query_emb)
                # cluster_dist_to_process = np.array(cluster_dist_to_process)
                # cluster_to_process = np.array(cluster_to_process)
                # cpu_cluster_to_process = np.array(cpu_cluster_to_process)
                # gpu_cluster_to_process = np.array(gpu_cluster_to_process)
                topk_score = np.array(topk_score)
                topk_id = np.array(topk_id)

                if self.time_profile:
                    t3 = time.time()
                    self.process_batch_time += t3 - t2

                # print("old", topk_score, topk_id)
                if self.time_profile:
                    t3 = time.time()

                self.gpu_engine.gpu_search_async(query_emb, gpu_cluster_to_process, 1)

                cpu_topk_score, cpu_topk_id, cmin_dist, cmin_id = \
                    self.index.search_preassigned_heterag(query_emb, self.large_topk, cpu_cluster_to_process, cluster_dist_to_process, D = topk_score, I = topk_id, init_heap = False)

                gpu_topk_score, gpu_topk_id = self.gpu_engine.gpu_search_finalize_async()

                # print("new cpu topk", cpu_topk_score, cpu_topk_id)
                # print("new gpu topk", gpu_topk_score, gpu_topk_id)

                topk_score = []; topk_id = []
                for i in range(len(query_emb)):
                    if (len(gpu_cluster_to_process[i])):
                        this_topk, this_topk_id = topk_merge(cpu_topk_score[i], cpu_topk_id[i], gpu_topk_score[i], gpu_topk_id[i], 1)
                        topk_score.append(this_topk)
                        topk_id.append(this_topk_id)
                    else:
                        topk_score.append(cpu_topk_score[i])
                        topk_id.append(cpu_topk_id[i])

                # print("final topk", topk_score, topk_id)
                
                self.finished_search += 1

                if self.time_profile:
                    t4 = time.time()
                    self.search_time += t4 - t3

                idx_list = []
                for qid, qtopkid_large, qtopkscore_large, qcmin_dist in zip(qid_list, topk_id, topk_score, cmin_dist):
                    # for key, item in self.request_dict.items():
                    #     print("key", key, "item", item.taskid)
                    qtopkid = qtopkid_large[:self.topk]
                    qtopkscore = qtopkscore_large[:self.topk]

                    this_request = self.request_dict[qid]

                    this_request.update_cmin_dist(qcmin_dist)

                    if this_request.is_empty:
                        idx_list.append(qtopkid)
                        this_request.update_end_time(time.time())

                        new_taskid = this_request.taskid
                        new_taskid.begin_spec = False
                        new_taskid.need_regeneration = this_request.need_correction()
                        # if this_request.need_correction():
                        #     this_request.taskid.need_regeneration = True
                        
                        taskid_list.append(new_taskid)

                        # print(f"sending correct task need regen = {new_taskid.need_regeneration} {new_taskid.begin_spec}")

                        self.latency.append(this_request.end_time - this_request.start_time)
                        self.finished_requests += 1

                        hit_cluster = []
                        for cid, cluster_dist in enumerate(this_request.cmin_dist):
                            if cluster_dist <= qtopkscore_large[-1]:
                                hit_cluster.append(this_request.original_clusters[cid])

                        if qid not in self.history_info_dict:
                            self.history_info_dict[qid] = EmbeddingInfo()
                        self.history_info_dict[qid].update_assigned_cluster(this_request.original_clusters)
                        self.history_info_dict[qid].update_hit_cluster(hit_cluster)

                        self.request_dict.pop(qid)
                    else:
                        this_request.topk_id = qtopkid_large
                        this_request.topk_score = qtopkscore_large

                # heteRAG: speculative retrieval, sending one retrieval results for each step
                # to do: the number
                if len(idx_list) < 1:
                    min_topk = sys.float_info.max
                    min_qid = -1

                    for qid in qid_list:
                        this_request = self.request_dict[qid]
                        if this_request.spec_begin == False and this_request.topk_id[self.topk - 1] < min_topk:
                            min_topk = this_request.topk_id[self.topk - 1]
                            min_qid = qid
                    
                    if min_qid != -1:
                        self.request_dict[min_qid].spec_exec()
                        idx_list.append(self.request_dict[min_qid].spec_idx)
                        new_taskid = self.request_dict[min_qid].taskid
                        # new_taskid = copy.deepcopy(self.request_dict[min_qid].taskid)
                        new_taskid.is_spec = True
                        new_taskid.begin_spec = True
                        taskid_list.append(new_taskid)

                    #     print(f"sending a speculative retrieval {new_taskid.id}")
                    # print("ahahaha one step")

                flat_idxs = np.array(idx_list).reshape(-1)
                batch_results = load_docs(self.corpus, flat_idxs)
                batch_results = [batch_results[i * self.topk : (i + 1) * self.topk] for i in range(len(idx_list))]

                if self.time_profile:
                    t5 = time.time()
                    self.save_and_load_time += t5 - t4

                # print(f"get {len(batch_results)} results!")

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
                # print(f"len(query_emb) is {len(query_emb)}")
                # print(cluster_to_process)
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
        
        # if len(batch_results):
        #     print("doc", batch_results)

        return taskid_list, batch_results

class CacheStatus:
    def __init__(self):
        self.access_time = 0
        self.onGPU = False
        self.GPUaddr = -1
        self.finished_transfer = False
        

