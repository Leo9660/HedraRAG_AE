import torch
from torch.utils.cpp_extension import load
import ctypes
import numpy as np
from itertools import accumulate
import time
import os

class HeteRAGGPUEngine:
    def __init__(self, index):

        cur_dir = os.path.dirname(os.path.abspath(__file__))

        self.external_search_module = load(
            name="external_search_backend",
            sources=[
                os.path.join(cur_dir, "gpu_retrieval_backend/search_kernel.cu"),
                os.path.join(cur_dir, "gpu_retrieval_backend/search_bind.cpp")
            ],
            # extra_cflags=['-O0', '-g'],
            # extra_cuda_cflags=['-O0', '-g', '-G'],
            verbose=False
        )
        self.external_search_module.gpu_retrieval_init()

        self.index = index
        self.device = torch.device("cuda")
        self.indextype = index.__class__.__name__

        if not "IVF" in self.indextype:
            raise ValueError("Only support IVF now")

        self.nlist = index.nlist
        self.d = index.d

        self.cluster_size = index.get_cluster_size_heterag([[i for i in range(self.nlist)]])[0]
        self.cluster_size_gpu = torch.tensor(self.cluster_size, device = "cuda")

        # self.access_time = [0 for _ in self.nlist]
        # self.cluster_is_on_gpu = [False for _ in self.nlist]
        # self.cluster_finished_transfer = [False for _ in self.nlist]
        self.gpu_cluster_addr = np.zeros(self.nlist, dtype='int64')

        self.gpu_cluster_dict = [cluster_status() for i in range(self.nlist)]

        # print("onloading")
        # self.onload_clusters([1, 2, 3, 4, 5, 6, 7, 8])

        # self.searching_clusters(gpu_search)

    def update_clusters(self, cluster_ids):
        for cluster_list in cluster_ids:
            for cluster_id in cluster_list:
                self.access_time += 1
    
    def has_cluster(self, cluster_id):
        return self.gpu_cluster_dict[cluster_id].is_on_gpu and not self.gpu_cluster_dict[cluster_id].in_transfer
    
    def show_clusters(self):
        print("show loaded cluster ", end='')
        for i in range(self.nlist):
            if self.gpu_cluster_dict[i].is_on_gpu:
                print(i, end=' ')
        print(" ")

    def onload_clusters_from_csv(self, file_path, config):

        file_path = "sorted_cluster.csv"
        kv_array = np.loadtxt(file_path, delimiter=",", dtype=int)
        print(f"{self.nlist} clusters in total")
        # print(self.nlist, kv_array)

        available_gpu_memory = (1 - config["gpu_memory_utilization"]) * 0.6 * 80
        
        gpu_cluster_memory = 0
        gpu_cluster_to_load = []
        cluster_idx = 0
        cluster_max = 512
        while (cluster_idx < self.nlist and cluster_idx < cluster_max and cluster_idx < len(kv_array) and gpu_cluster_memory < available_gpu_memory):
            cluster_id = kv_array[cluster_idx][0]
            gpu_cluster_memory += self.cluster_size[cluster_id] * self.index.d * 4 / 1024 / 1024 / 1024
            gpu_cluster_to_load.append(cluster_id)
            cluster_idx += 1
        
        print(f"onloading {cluster_idx} clusters")
        self.onload_clusters(gpu_cluster_to_load)

    def onload_clusters(self, cluster_list):

        # print("cluster list", cluster_list)

        cluster_size_list = self.index.get_cluster_size_heterag([cluster_list])[0]
        cluster_ptr_list = self.index.get_cluster_ptr_heterag(cluster_list)

        # print(cluster_size_list)
        # print(cluster_ptr_list)
        
        for cluster_id, cluster_ptr, cluster_size in zip(cluster_list, cluster_ptr_list, cluster_size_list):
            dict_item = self.gpu_cluster_dict[cluster_id]
            # print(dict_item.__dict__)
            if not dict_item.is_on_gpu:
                dict_item.onload_cluster(cluster_ptr, cluster_size * self.d)
                self.gpu_cluster_addr[cluster_id] = dict_item.gpu_tensor.data_ptr()

    def gpu_search(self, query, cluster_ids, topk):
        addr_search_list = []
        for cluster_list in cluster_ids:
            current_search_list = []
            for cluster_id in cluster_list:
                # print("id", cluster_id)
                # for cidx, item in enumerate(self.gpu_cluster_dict):
                #     if (item.is_on_gpu):
                #         print(f"{cidx} is on GPU")
                if not self.gpu_cluster_dict[cluster_id].is_on_gpu:
                    raise ValueError("Wrong cluster, not on GPU")
                current_search_list.append(self.gpu_cluster_addr[cluster_id])
            addr_search_list.append(current_search_list)

        cluster_num = [0] + list(accumulate(len(sub) for sub in cluster_ids))

        return self.gpu_search_wrapper(query, cluster_ids, addr_search_list, cluster_num, topk)

    def gpu_search_wrapper(self, query, id_list, addr_list, cluster_num, topk):

        if topk > 1:
            raise ValueError("topk > 1 not implemented!")

        query_tensor = torch.tensor(query, dtype=torch.float32, device=self.device)  # (N, D)
        
        N, D = query_tensor.shape

        cluster_id = torch.tensor(id_list, dtype=torch.int64, device=self.device)
        cluster_ptrs = torch.tensor(addr_list, dtype=torch.int64, device=self.device)
        cluster_num = torch.tensor(cluster_num, dtype=torch.int32, device=self.device)

        # out_scores = torch.empty((N, topk), device=self.device, dtype=torch.float32)
        # out_indices = torch.empty((N, topk), device=self.device, dtype=torch.int64)
        # now reduction on CPU
        out_scores = torch.empty((N, topk), dtype=torch.float32)
        out_indices = torch.empty((N, topk), dtype=torch.int64)

        t1 = time.time()

        self.external_search_module.external_search(
            query_tensor.data_ptr(),
            cluster_id.data_ptr(),
            cluster_ptrs.data_ptr(),
            cluster_num.data_ptr(),
            N,
            D,
            topk,
            self.cluster_size_gpu.data_ptr(),
            out_scores.data_ptr(),
            out_indices.data_ptr()
        )

        t2 = time.time()
        # print(f"search time: {t2 - t1}")

        return out_scores, out_indices
    
    def gpu_search_async(self, query, cluster_ids, topk):
        flat_cluster_ids = []
        flat_addr_search_list = []
        for cluster_list in cluster_ids:
            current_search_list = []
            for cluster_id in cluster_list:
                if not self.gpu_cluster_dict[cluster_id].is_on_gpu:
                    raise ValueError("Wrong cluster, not on GPU")
                current_search_list.append(self.gpu_cluster_addr[cluster_id])
            flat_addr_search_list.extend(current_search_list)
            flat_cluster_ids.extend(cluster_list)

        cluster_num = [0] + list(accumulate(len(sub) for sub in cluster_ids))
        # print("cluster_num", cluster_num)
        # print("cluster", flat_cluster_ids)
        # print("addr_search_list", flat_addr_search_list)

        self.gpu_search_wrapper_async(query, flat_cluster_ids, flat_addr_search_list, cluster_num, topk)

    def gpu_search_wrapper_async(self, query, id_list, addr_list, cluster_num, topk):
        """
        GPU search wrapper for querying against multiple clusters.

        Parameters:
            query (torch.Tensor): shape (N, D), float32, on CPU or GPU
            addr_list (list[int]): list of device pointers (int), each points to a [K_i, D] float32 array on GPU
            topk (int): number of top matches to return per cluster

        Returns:
            out_scores (torch.Tensor): shape (N, topk), float32, on GPU
            out_indices (torch.Tensor): shape (N, topk), int32, on GPU
        """

        if topk > 1:
            raise ValueError("topk > 1 not implemented!")

        query_tensor = torch.tensor(query, dtype=torch.float32, device=self.device)
        
        N, D = query_tensor.shape

        cluster_id = torch.tensor(id_list, dtype=torch.int64, device=self.device)
        cluster_ptrs = torch.tensor(addr_list, dtype=torch.int64, device=self.device)
        cluster_num = torch.tensor(cluster_num, dtype=torch.int32, device=self.device)

        qpb = self.external_search_module.external_search_get_blocks_per_query()
        self.block_out_scores_d = torch.empty((N, topk * qpb), device=self.device, dtype=torch.float32)
        self.block_out_indices_d = torch.empty((N, topk * qpb), device=self.device, dtype=torch.int64)
        self.last_retrieval_n = N
        self.last_topk = topk

        t1 = time.time()

        self.external_search_module.external_search_async(
            query_tensor.data_ptr(),
            cluster_id.data_ptr(),
            cluster_ptrs.data_ptr(),
            cluster_num.data_ptr(),
            N,
            D,
            topk,
            self.cluster_size_gpu.data_ptr(),
            self.block_out_scores_d.data_ptr(),
            self.block_out_indices_d.data_ptr()
        )

        t2 = time.time()
    
    def gpu_search_finalize_async(self):

        # now reduction on CPU
        out_scores = torch.empty((self.last_retrieval_n, self.last_topk), dtype=torch.float32)
        out_indices = torch.empty((self.last_retrieval_n, self.last_topk), dtype=torch.int64)

        self.external_search_module.search_finalize_async(self.last_retrieval_n,
            out_scores.data_ptr(),
            out_indices.data_ptr(),
            self.block_out_scores_d.data_ptr(),
            self.block_out_indices_d.data_ptr())

        del self.block_out_scores_d
        del self.block_out_indices_d

        return out_scores, out_indices

class cluster_status:
    def __init__(self):
        self.is_on_gpu = False
        self.in_transfer = False
        self.access_time = 0
        self.gpu_tensor = None
    
    def update_access_time(self):
        self.access_time += 1
    
    def onload_cluster(self, cluster_ptr, cluster_size):
        float_array_type = ctypes.c_float * cluster_size
        c_array = float_array_type.from_address(int(cluster_ptr))
        self.gpu_tensor = torch.tensor(c_array, dtype=torch.float32, device='cuda')
        self.is_on_gpu = True
