# generator initialization built upon FlashRAG
from abc import ABC, abstractmethod
from heterag.retriever.utils import *
from heterag.retriever.encoder import Encoder, STEncoder
from heterag.prompt import get_content
from typing import Union, List, Dict
import faiss
import time
import numpy as np
from heapq import heappush, heappop

class BaseEngine(ABC):
    """Base engine for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config["retrieval_method"]
        self.topk = config["retrieval_topk"]

        self.index_path = config["index_path"]
        self.corpus_path = config["corpus_path"]

        self.batch_size = self.config["retrieval_batch_size"]
        self.return_embedding = self.config["return_embedding"]

        # self.use_reranker = config["use_reranker"]
        # if self.use_reranker:
        #     self.reranker = get_reranker(config)
    
    # @abstractmethod
    # def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
    #     r"""Retrieve topk relevant documents in corpus.

    #     Return:
    #         list: contains information related to the document, including:
    #             contents: used for building index
    #             title: (if provided)
    #             text: (if provided)

    #     """

    #     pass

    # @abstractmethod
    # def _batch_search(self, query_list, num, return_score, eval_cache):
    #     pass

class DenseEngine(BaseEngine):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
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
        
        self.reinit(config)
    
    def reinit(self, config):
        self.config = config
        self.topk = config["retrieval_topk"]
        self.batch_size = self.config["retrieval_batch_size"]

        print(f"Index type {self.index.__class__.__name__}")

        # if (isinstance(self.index, faiss.swigfaiss_avx2.IndexIVFFlat)):
        if "nprobe" in self.config:
            self.index.nprobe = config["nprobe"]
        else:
            self.index.nprobe = 16
        
        self.idxs_cache = dict()

        self.request_dict = {}

        # count when new centroid dist increases, the closest distance also increases.
        self.total_iter_clusters = 0
        self.total_overlap_clusters = 0
        self.total_far_clusters = 0
        self.total_far_dist = 0

        # the largest cluster which the topk document is in
        self.largest_cluster_ranking = []
        self.overlapped_rate = []
        self.overlapped_rate_top20 = []

        # document similarity search
        self.answer_in_doc_search = 0
        self.answer_in_doc_search_list = []
        self.all_answer_in_doc = 0
        self.all_answer_in_doc_list = []

        # old query similarity search
        self.answer_in_old_query_search = 0
        self.answer_in_old_query_search_list = []
        self.answer_not_in_old_query_search_list = []

        # answer in both search
        self.answer_in_both = 0

        # old query total search
        self.total_repeat_search = 0

        # time profile
        self.latency = []
        self.finished_requests = 0
        self.request_batch_size = 0
        self.in_old_centroid_but_not_in_local_buffer = 0
        self.in_old_centroid_and_in_local_buffer = 0
        self.in_old_centroid_top20_but_not_in_local_buffer = 0
        self.in_old_centroid_top20_and_in_local_buffer = 0

        self.in_old_hit_centroid_top20 = 0
        self.in_old_centroid_set = 0
        self.in_old_hit_centroid_top20_list = []
        self.in_old_centroid_set_list = []

        # test multi-request skewness
        self.skewness_dict = {i: 0 for i in range(self.index.nlist)}

        # profile document dist
        self.top1 = []
        self.top5 = []
        self.top20 = []

        # profile early termination
        self.old_termination_point = []
        self.new_termination_point = []
        self.new_termination_point2 = []
        self.new_termination_point3 = []

        self.larger_topk = config["larger_topk"] if config["larger_topk"] is not None else 20
        print(f"[Reordering setup]: topk as {self.topk} larger_topk as {self.larger_topk}")

        # self.simulate_onload_cluster(config)
    
    def simulate_onload_cluster(self, config):

        self.cluster_size = self.index.get_cluster_size_heterag([range(0, self.index.nlist)])[0]

        available_gpu_memory = (1 - config["gpu_memory_utilization"]) * 0.6 * 80
        
        gpu_cluster_memory = 0
        gpu_cluster_to_load = []
        cluster_idx = 0
        cluster_max = 1024
        while (cluster_idx < self.index.nlist and cluster_idx < cluster_max and \
        cluster_idx < len(kv_array) and gpu_cluster_memory < available_gpu_memory):
            cluster_id = kv_array[cluster_idx][0]
            gpu_cluster_memory += self.cluster_size[cluster_id] * self.index.d * 4 / 1024 / 1024 / 1024
            gpu_cluster_to_load.append(cluster_id)
            cluster_idx += 1
        
        print(f"onloading {cluster_idx} clusters")
        self.onload_clusters(gpu_cluster_to_load)
        self.onload_cluster = gpu_cluster_to_load
        self.onload_cluster_hit = 0
        self.total_cluster_search_num = 0

    def _search(self, query: str, num: int = None, return_score=False, eval_cache=False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], query_id_list = [], num: int = None, return_score=False, eval_cache=False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        batch_size = self.batch_size

        results = []
        results_emb = EmbeddingInfo()
        scores = []

        encode_time = 0
        search_time = 0
        post_time = 0
        

        for start_idx in range(0, len(query_list), batch_size):

            t1 = time.time()

            query_batch = query_list[start_idx : start_idx + batch_size]
            query_id_batch = query_id_list[start_idx : start_idx + batch_size]

            batch_emb = self.encoder.encode(query_batch)

            t2 = time.time()
            encode_time += t2 - t1

            batch_scores, batch_idxs, cluster_min, cluster_lid = self.index.search_with_cluster_id(batch_emb, k=num)

            if self.return_embedding:
                profile_search_scores, profile_search_idxs = self.index.search(batch_emb, k=20)
                for profile_search_score in profile_search_scores:
                    self.top1.append(profile_search_score[0])
                    self.top5.append(profile_search_score[4])
                    self.top20.append(profile_search_score[19])

            t3 = time.time()
            search_time += t3 - t2

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            scores.extend(batch_scores)
            results.extend(batch_results)


            if (self.return_embedding):
                results_emb.update(query_emb = batch_emb, retrieval_score = batch_scores)

            t4 = time.time()
            post_time += t4 - t3

            centroid_distances, batch_assigned_centroids = self.index.quantizer.search(batch_emb, self.index.nprobe)
            for centroid in batch_assigned_centroids:
                for centroid_id in centroid:
                    if centroid_id not in self.skewness_dict:
                        self.skewness_dict[centroid_id] = 1
                    else:
                        self.skewness_dict[centroid_id] += 1

            if (self.return_embedding):
                if isinstance(self.index, faiss.IndexIVFFlat):

                    centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)

                    centroid_distances2, batch_assigned_centroids2 = self.index.quantizer.search(batch_emb, self.index.nlist)

                    for taskid, query_str, query_emb, assigned_centroid, batch_data, batch_idx, topk_score, centroid_distance, cluster_min_distance \
                        in zip(query_id_batch, query_list, batch_emb, batch_assigned_centroids, batch_results, batch_idxs, batch_scores, centroid_distances, cluster_min):

                        for centroid in assigned_centroid:
                            self.total_cluster_search_num += 1
                            if centroid in self.onload_cluster:
                                self.onload_cluster_hit += 1

                        last_topk_score = topk_score[-1]

                        # update cluster ranking
                        largest_cluster_ranking = 0
                        for min_cid, min_dist in enumerate(cluster_min_distance):
                            if min_dist <= last_topk_score:
                                largest_cluster_ranking = min_cid
                        self.largest_cluster_ranking.append(largest_cluster_ranking)

                        if not taskid.id in self.request_dict:
                            self.request_dict[taskid.id] = EmbeddingInfo()
                        else:
                            old_assigned_centroid = self.request_dict[taskid.id].centroid_idx[-1]
                            old_centroid_distance = self.request_dict[taskid.id].centroid_distance[-1]

                            # find common centroids
                            common_centroids = np.intersect1d(old_assigned_centroid, assigned_centroid)

                            ordered_common_elements = [x for x in old_assigned_centroid if x in common_centroids]
                            idx_old = [np.where(old_assigned_centroid == x)[0][0] for x in ordered_common_elements]
                            idx_new = [np.where(assigned_centroid == x)[0][0] for x in ordered_common_elements]


                            old_query_emb = self.request_dict[taskid.id].query_emb

                            cid_new_no_overlapped = [x for x in assigned_centroid if x not in common_centroids]
                            cid_new_no_overlapped_id = [i for i, x in enumerate(assigned_centroid) if x not in common_centroids]
                            cdist_test = []
                            for cid in cid_new_no_overlapped:
                                cdist_test.append(fvec_L2sqr(query_emb, centroids[cid]))
                            cdist_test = []
                            for cid in cid_new_no_overlapped:
                                cdist_test.append(fvec_L2sqr(old_query_emb, centroids[cid]))

                            delta_vector = query_emb - self.request_dict[taskid.id].query_emb
                            cangle_test = []
                            
                            for cid in cid_new_no_overlapped:
                                query_centoid_delta = centroids[cid] - old_query_emb
                                cangle_test.append(fvec_inner_product(delta_vector[0], query_centoid_delta[0]))

                            cdist_test = []
                            for cid in cid_new_no_overlapped_id:
                                cdist_test.append(cluster_min_distance[cid])

                            dist_old = old_centroid_distance[idx_old]
                            dist_new = centroid_distance[idx_new]

                            
                            diff_1 = np.array(dist_new) - np.array(dist_old)
                            diff_2 = np.array(centroid_distance)[idx_new] - np.array(old_centroid_distance)[idx_old]
        
                            mask_1 = diff_1 > 0
                            mask_2 = diff_2 > 0

                            self.total_iter_clusters += len(old_centroid_distance)
                            self.total_overlap_clusters += len(mask_1)
                            self.total_far_clusters += np.sum(mask_1)
                            self.total_far_dist += np.sum(mask_1 & mask_2)

                            new_old_query_dist = fvec_L2sqr(np.array(query_emb), np.array(old_query_emb))


                            target_cluster = []
                            for min_cid, min_dist in enumerate(cluster_min_distance):
                                if min_dist <= last_topk_score:
                                    target_cluster.append(assigned_centroid[min_cid])
                            overlapped_useful_cluster_number = 0
                            for cluster in target_cluster:
                                if cluster in old_assigned_centroid:
                                    overlapped_useful_cluster_number += 1
                            in_old_centroid = False
                            if overlapped_useful_cluster_number == len(target_cluster):
                                in_old_centroid = True
                                self.in_old_centroid_set += 1
                                self.in_old_centroid_set_list.append(new_old_query_dist)
                            self.overlapped_rate.append(overlapped_useful_cluster_number / len(target_cluster))
                            

                            old_batch_idx = self.request_dict[taskid.id].doc_idx
                            old_flat_idxs = sum(old_batch_idx, [])
                            old_batch_results = load_docs(self.corpus, old_flat_idxs)
                            old_doc = []
                            for doc in old_batch_results:
                                old_doc.append(get_content(doc))

                            self.total_repeat_search += num
                            answer_in_doc = False
                            answer_in_query = False

                            old_search_scores, old_search_idxs = self.index.search(np.array(old_query_emb), k=20)
                            in_old_query_top20 = 0
                            for the_idx in batch_idx:
                                if the_idx in old_search_idxs:
                                    in_old_query_top20 += 1
                                    self.answer_in_old_query_search += 1
                                    self.answer_in_old_query_search_list.append(fvec_L2sqr(np.array(query_emb),
                                    np.array(old_query_emb)))
                                else:
                                    self.answer_not_in_old_query_search_list.append(fvec_L2sqr(np.array(query_emb),
                                    np.array(old_query_emb)))

                            if in_old_query_top20 == num:
                                answer_in_query = True
        
                            old_dist_debug, old_centroid_debug = self.index.quantizer.search(np.array(old_query_emb), self.index.nprobe)
                            old_search_scores, old_search_idxs, old_cluster_min, old_cluster_lid = self.index.search_with_cluster_id(np.array(old_query_emb), k=self.larger_topk)

                            last_topk_score = old_search_scores[0][-1]
                            old_assigned_centroid_top20 = []
                            for old_cluster_min_distance in old_cluster_min:
                                for min_cid, min_dist in enumerate(old_cluster_min_distance):
                                    if min_dist <= last_topk_score:
                                        old_assigned_centroid_top20.append(old_assigned_centroid[min_cid])
                            overlapped_useful_cluster_number = 0
                            for cluster in target_cluster:
                                if cluster in old_assigned_centroid_top20:
                                    overlapped_useful_cluster_number += 1

                            in_old_centroid_top20 = False
                            if overlapped_useful_cluster_number == len(target_cluster):
                                in_old_centroid_top20 = True
                                self.in_old_hit_centroid_top20 += 1
                                self.in_old_hit_centroid_top20_list.append(new_old_query_dist)
                            self.overlapped_rate_top20.append(overlapped_useful_cluster_number / len(target_cluster))

                            # old doc top20
                            in_old_doc_top20 = 0
                            for the_idx in batch_idx:
                                if the_idx in old_search_idxs:
                                    in_old_doc_top20 += 1
                                    self.answer_in_doc_search += 1
                                    self.answer_in_doc_search_list.append((1, largest_cluster_ranking))
                            if in_old_doc_top20 == len(batch_idx):
                                answer_in_doc = True
                                self.all_answer_in_doc += 1
                                self.all_answer_in_doc_list.append(new_old_query_dist)

                            
                            if in_old_centroid and not answer_in_doc and not answer_in_query:
                                self.in_old_centroid_but_not_in_local_buffer += 1
                            if in_old_centroid and (answer_in_doc or answer_in_query):
                                self.in_old_centroid_and_in_local_buffer += 1
                            if in_old_centroid_top20 and not answer_in_doc and not answer_in_query:
                                self.in_old_centroid_top20_but_not_in_local_buffer += 1
                            if in_old_centroid_top20 and (answer_in_doc or answer_in_query):
                                self.in_old_centroid_top20_and_in_local_buffer += 1
                            
                            if answer_in_doc and answer_in_query:
                                self.answer_in_both += 1

                            self.old_termination_point.append(largest_cluster_ranking + 1)
                            if answer_in_doc:
                                self.new_termination_point.append(0)
                                self.new_termination_point2.append(0)
                                self.new_termination_point3.append(0)
                            else:
                                self.new_termination_point3.append(largest_cluster_ranking + 1)

                                et_hit_centroid = set(old_assigned_centroid_top20)
                                et_nohit_centroid = set(old_assigned_centroid)
                                indices = np.arange(len(assigned_centroid))
                                sorted_indices = sorted(indices, key=lambda i: (0 if assigned_centroid[i] in et_hit_centroid else 1 if assigned_centroid[i] in et_nohit_centroid else 2))
                                sorted_assigned_centroid = assigned_centroid[sorted_indices]


                                sorted_largest = 0
                                for i, cluster_id in enumerate(sorted_assigned_centroid):
                                    if cluster_id in target_cluster:
                                        sorted_largest = i
                                self.new_termination_point.append(sorted_largest + 1)

                                sorted_indices = sorted(indices, key=lambda i: (0 if assigned_centroid[i] in et_hit_centroid else 1))
                                sorted_assigned_centroid = assigned_centroid[sorted_indices]

                                sorted_largest = 0
                                for i, cluster_id in enumerate(sorted_assigned_centroid):
                                    if cluster_id in target_cluster:
                                        sorted_largest = i
                                if new_old_query_dist < 0.3:
                                    self.new_termination_point2.append(sorted_largest + 1)
                                else:
                                    self.new_termination_point2.append(largest_cluster_ranking + 1)

                        self.request_dict[taskid.id].update(query_emb = [query_emb], 
                        centroid_idx = [assigned_centroid], 
                        centroid_distance = [centroid_distance], 
                        topk_score = [topk_score], 
                        largest_cluster = [largest_cluster_ranking],
                        doc_idx = [batch_idx])

            if (eval_cache):
                for batch_idx in batch_idxs:
                    for idxs in batch_idx[0:1]:
                        if (idxs in self.idxs_cache):
                            self.idxs_cache[idxs] += 1
                        else:
                            self.idxs_cache[idxs] = 1
            
        

        if return_score:
            return results, scores
        else:
            return results
    
    def show_inter_diff(self):
        inter_dis = []
        for taskid, request in self.request_dict.items():
            inter_dis.append(request.show_inter_stage_diff(taskid, metric = 1))
        print("[similarity] average inter dis", np.mean(inter_dis))
        print("[similarity] average top1", np.mean(self.top1))
        print("[similarity] average top5", np.mean(self.top5))
        print("[similarity] average top20", np.mean(self.top20))

        print("average overlap rate", np.mean(self.overlapped_rate))
        print("average overlap rate top20", np.mean(self.overlapped_rate_top20))
        count = np.sum(np.array(inter_dis) < 0.15)
        print(f"< 0.15, {count}")

    def show_time_profile(self):
        print("Not implemented")