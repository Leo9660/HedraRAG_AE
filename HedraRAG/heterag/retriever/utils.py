import json
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from heterag.utils import TaskID
import numpy as np
import time
import copy

class RetrievalTask:
    def __init__(self, query: str, taskid: TaskID, index_id: int, topk: int, nprobe: int = 0, start_time = 0):
        self.query = query
        self.taskid = taskid
        self.index_id = index_id
        self.nprobe = nprobe
        self.clusters = []
        self.original_clusters = []
        self.cluster_dist = []
        self.cluster_size = []
        self.cmin_dist = []
        self.spec_idx = []
        self.start_time = start_time
        self.subset_iter = 0
        self.search_time = 0
        self.topk_id = np.full((topk), -1, dtype=np.int64)
        self.topk_score = np.full((topk), np.finfo(np.float32).max, dtype=np.float32)
        self.is_empty = False

        # for speculative execution
        self.spec_begin = False
        self.spec_subset_num = 0
    
    def update_emb(self, emb):
        self.emb = emb
    
    def update_end_time(self, end_time):
        self.end_time = end_time

    def update_start_time(self, start_time):
        self.start_time = start_time

    def update_cluster_size(self, cluster_size):
        self.cluster_size = cluster_size

    def add_clusters(self, clusters, dist = None):
        self.clusters.extend(clusters)
        self.original_clusters.extend(clusters)
        if dist is not None:
            self.cluster_dist.extend(dist)

    def update_iter_num(self, delta: int = 1):
        self.subset_iter += delta
    
    def update_cmin_dist(self, cmin_dist):
        self.cmin_dist.extend(cmin_dist)
    
    def pop_clusters(self, num: int):
        new_clusters = self.clusters[:num]
        new_dist = self.cluster_dist[:num]
        new_size = np.sum(self.cluster_size[:num])

        self.clusters = self.clusters[num:]
        self.cluster_dist = self.cluster_dist[num:]
        self.cluster_size = self.cluster_size[num:]

        if len(self.cluster_size) == 0 or self.cluster_size[0] == 0:
            self.is_empty = True

        self.update_iter_num(1)

        return new_clusters, new_dist, new_size

    def pop_clusters_loadbalance(self, num: int):
        new_clusters = []
        new_dist = []
        new_size = 0
        num = 0
        while num < len(self.clusters) and new_size < 20000000000:
            new_clusters.append(self.clusters[num])
            new_dist.append(self.cluster_dist[num])
            new_size += self.cluster_size[num]
            num += 1

        new_dist = self.cluster_dist[:num]

        self.clusters = self.clusters[num:]
        self.cluster_dist = self.cluster_dist[num:]
        self.cluster_size = self.cluster_size[num:]

        if len(self.cluster_size) == 0 or self.cluster_size[0] == 0:
            self.is_empty = True

        self.update_iter_num(1)

        return new_clusters, new_dist, new_size

    def spec_exec(self):
        self.spec_begin = True
        self.spec_subset_num = self.subset_iter
        self.spec_idx = copy.deepcopy(self.topk_id)
    
    def need_correction(self):
        # if speculative retrieval results are sent
        if self.spec_begin:
            # check correctness
            if np.array_equal(self.topk_id, self.spec_idx):
                return False
            else:
                return True
        return True

class EmbeddingInfo:
    def __init__(self):
        self.query_emb = []
        self.retrieval_score = []
        self.retrieval_emb = []
        self.centroid_idx = []
        self.centroid_emb = []
        self.centroid_distance = []
        self.topk_score = []
        self.largest_cluster = []
        self.doc_idx = []
        self.assigned_cluster = []
        self.hit_cluster = []

        self.start_time = time.time()
        self.end_time = 0

    def update(self, query_emb = None, retrieval_score = None, retrieval_emb = None, centroid_idx = None, centroid_emb = None, centroid_distance = None, topk_score = None, largest_cluster = None, doc_idx = None):
        if query_emb is not None:
            self.query_emb.extend(query_emb)

        if retrieval_score is not None:
            self.retrieval_score.extend(retrieval_score)
        
        if retrieval_emb is not None:
            self.retrieval_emb.extend(retrieval_emb)

        if retrieval_emb is not None:
            self.retrieval_emb.extend(retrieval_emb)

        if centroid_idx is not None:
            self.centroid_idx.extend(centroid_idx)
        
        if centroid_emb is not None:
            self.centroid_emb.extend(centroid_emb)

        if centroid_distance is not None:
            self.centroid_distance.extend(centroid_distance)
        
        if topk_score is not None:
            self.topk_score.extend(topk_score)
        
        if largest_cluster is not None:
            self.largest_cluster.extend(largest_cluster)
        
        if doc_idx is not None:
            self.doc_idx.extend(doc_idx)
    
    def update_assigned_cluster(self, assigned_cluster):
        self.assigned_cluster.extend(assigned_cluster)
    
    def update_hit_cluster(self, hit_cluster):
        self.hit_cluster.extend(hit_cluster)
    
    def update_end_time(self, end_time):
        self.end_time = end_time

    def update_start_time(self, start_time):
        self.start_time = start_time
    
    def show_inter_stage_diff(self, id, metric = 0):
        inter_dis = 0.0
        for i in range(len(self.query_emb) - 1):
            inter_dis = 0
            if metric == 0:
                inter_dis += fvec_inner_product(self.query_emb[i], self.query_emb[i + 1])
            else:
                inter_dis += fvec_L2sqr(self.query_emb[i], self.query_emb[i + 1])
        if len(self.query_emb) > 1:
            inter_dis /= (len(self.query_emb) - 1)
        return inter_dis

# a: score, i: index
def topk_merge(a1, a2, i1, i2, k):
    # 合并 value 和 index
    a_concat = np.concatenate([a1, a2])
    i_concat = np.concatenate([i1, i2])

    # 按照 value 排序（升序，值越小越相似）
    sorted_indices = np.argsort(a_concat)

    topk_indices = sorted_indices[:k]

    # 最终的 topk 值和对应索引
    a_topk = a_concat[topk_indices].astype(np.float32)
    i_topk = i_concat[topk_indices].astype(np.int64)
    return a_topk, i_topk

def fvec_inner_product(x, y):
    return np.dot(np.array(x), np.array(y))

def fvec_L2sqr(x, y):
    return np.linalg.norm(np.array(x) - np.array(y)) ** 2


def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str, hf = False):
    if corpus_path == "Tevatron/msmarco-passage-corpus":
        corpus = datasets.load_dataset(corpus_path)
        corpus = corpus['train']
    elif (hf):
        corpus = datasets.load_dataset(corpus_path)
        corpus = corpus['train']
    else:
        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    return corpus


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results