import transformers
import time
from heterag.ragraph.ragraph import *
from heterag.executor.request import *
from heterag.executor.retrieval_worker import *
from heterag.retriever.engine import *
import importlib
from typing import Union, List
import numpy as np

class SequentialExecutor:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]

        if config["use_llm"] != False:
            config["generator"] = "vllm"
            self.generator = get_generator(config)

        self.retriever = DenseEngine(config)
        self.reinit(config)
    
    def reinit(self, config):

        self.retriever.reinit(config)

        self.batch_size = config["batch_size"]
        self.retrieval_batch_size = config["retrieval_batch_size"]
        self.return_embedding = config["return_embedding"]

        # whether continuous retrieval
        self.continuous_retrieval = config["continuous_retrieval"]

        # initialize the retriever worker process
        # self.init_retrieve_worker()
        # start up the LLM serving engine
        self.print_log = config["print_log"]
        self.task_batch = config["task_batch"] if config["task_batch"] is not None else 64

        # ragraphs in the executor, each ragraph represents a type of RAG workflow
        self.ragraph_list = []
        self.ragraph_namedict = {}

        # current requests in execution
        self.request_dict = {}
        self.request_time_dict = {}
        self.new_request_id_list = []
        self.retrieval_id_list = []

        # output list of requests
        self.output_list = []

        # benchmark
        self.cpu_time = 0
        self.gpu_time = 0
        self.request_latency = []

        # benchmark
        self.cluster_dict = []

        # spec execution
        self.spec_execution = []
        self.spec_dict = {}
        self.remain_task = []

        self.current_gen_request_num = 0

        self.task_list = []

        if self.print_log:
            print("Server start!")

    def init_retrieve_worker(self, print_log: bool = False):
        import multiprocessing
        from multiprocessing import Process, Manager

        multiprocessing.set_start_method('spawn', force=True)
        manager = Manager()
        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()

        if self.continuous_retrieval:
            self.retrieve_p = multiprocessing.Process(target=retrieval_worker_heterag, args=(self.config, self.task_queue, self.result_queue), kwargs={"worker_log": True, "return_emb": self.return_embedding})
        else:
            self.retrieve_p = multiprocessing.Process(target=retrieval_worker, args=(self.config, self.task_queue, self.result_queue), kwargs={"worker_log": True, "return_emb": self.return_embedding})
            
        self.retrieve_p.start()

        while (self.result_queue.empty()):
            pass
        self.result_queue.get()

    def get_ragraph(self, name: str):
        graph_index = self.ragraph_namedict[name]
        return self.ragraph_list[graph_index]

    def graph_register(self, input_graphs: Union[str, List[str]]):
        if isinstance(input_graphs, str):
            input_graphs = [input_graphs]

        for graph_name in input_graphs:
            if not graph_name in self.ragraph_namedict:
                self.ragraph_namedict[graph_name] = len(self.ragraph_namedict)
                self.ragraph_list.append(get_new_ragraph(graph_name, self.config))
    
    def add_requests(self, input_requests: Union[Request, List[Request]]):
        if isinstance(input_requests, Request):
            input_requests = [input_requests]
        
        for input_request in input_requests:
            self.request_dict[input_request.id] = input_request
            self.new_request_id_list.append(input_request.id)
    
    def add_requests_string(self, input_strings: Union[str, List[str]], workflow: str = "Sequential"):
        if isinstance(input_strings, str):
            input_strings = [input_strings]
        
        request_list = []

        if workflow == "Sequential":
            for input_string in input_strings:
                request_list.append(SequentialRequest(input_string, self.config))
        elif workflow == "IRG":
            iter_num = self.config["iter_num"] if self.config["iter_num"] else 3
            for input_string in input_strings:
                request_list.append(IterRequest(input_string, self.config, iter_num))
        elif workflow == "Multistep":
            iter_num = self.config["iter_num"] if self.config["iter_num"] else 2
            for input_string in input_strings:
                request_list.append(MultistepRequest(input_string, self.config, iter_num))
        elif workflow == "RECOMP":
            for input_string in input_strings:
                request_list.append(RecompRequest(input_string, self.config))
        elif workflow == "HyDE":
            for input_string in input_strings:
                request_list.append(HyDERequest(input_string, self.config))
        else:
            raise ValueError("Wrong workflow!")

        
        self.add_requests(request_list)

    def send_retrieval_tasks(self, rtask_list, rtask_id_list):
        self.task_queue.put((rtask_list, rtask_id_list))
    
    def execute(self):

        # Add new request
        new_list = self.new_request_id_list
        self.new_request_id_list = self.new_request_id_list[len(new_list):]
        for new_request_id in new_list:

            request = self.request_dict[new_request_id]
            new_task = request.update_stage()

            if new_task == None:
                print(f"Warning: request {new_request_id} doing nothing")
            else:
                self.task_list.extend(new_task)

        rtask_list = []
        rtask_id_list = []
        iter_i = 0
        while iter_i < len(self.task_list):
            task = self.task_list[iter_i]

            if task.type == "G":
                gtask = task
                if gtask.is_spec == False:
                    self.generator.add_request(gtask.input, gtask.id)
                    self.current_gen_request_num += 1
            elif task.type == "R":
                rtask_list.append(task.input)
                rtask_id_list.append(task)
            else:
                raise valueError("Wrong task type {task.type}")

            iter_i += 1

        self.task_list = self.task_list[iter_i:]


        # generation
        gpu_time1 = time.time()

        gen_result_num = 0
        while gen_result_num < self.current_gen_request_num:
            generation_results = self.generator.step()

            for generation_result in generation_results:
                if generation_result.finished:
                    request_id = generation_result.request_id
                    output_res = generation_result.outputs[0].text

                    if request_id not in self.spec_dict:
                        metrics = generation_result.metrics
                        self.request_latency.append((metrics.finished_time - metrics.arrival_time, metrics.first_token_time - metrics.first_scheduled_time, metrics.finished_time - metrics.first_token_time))
                        request = self.request_dict[request_id]
                        new_task = request.update_stage(output_res)
                        if new_task == None:
                            self.output_list.append(request)
                            self.request_dict.pop(request_id, None)
                        else:
                            self.task_list.extend(new_task)
                        self.request_time_dict[request_id] = time.time()
                    else:
                        self.spec_dict[request_id] = output_res
                    gen_result_num += 1

        self.current_gen_request_num = 0

        gpu_time2 = time.time()
        self.gpu_time += gpu_time2 - gpu_time1


        cpu_time1 = time.time()

        # generation
        if (len(rtask_list)):
            retrieval_result = self.retriever._batch_search(rtask_list, rtask_id_list)

            for idx, (taskid, docs) in enumerate(zip(rtask_id_list, retrieval_result)):
                request = self.request_dict[taskid.id]

                new_task = request.update_stage(docs, taskid.stage)
                if new_task == None:
                    print(f"Warning: request {new_request_id} doing nothing after retrieval")
                else:
                    self.task_list.extend(new_task)


        cpu_time2 = time.time()
        self.cpu_time += cpu_time2 - cpu_time1

        return len(self.output_list)
    
    def finalize(self):
        pass

    def faiss_benchmark(self, string_list, input_time_list, request_per_second = 0, output_file = None):
        result_list = []

        current_idx = 0

        start_time = time.time()

        output_time_list = np.zeros(len(input_time_list))

        while len(result_list) < len(string_list):
            current_time = time.time() - start_time

            qid_list = []
            query_list = []
            while current_idx < len(string_list) and current_time >= input_time_list[current_idx]:
                qid_list.append(TaskID(current_idx, 0, "R", string_list[current_idx]))
                query_list.append(string_list[current_idx])
                current_idx += 1

            if len(qid_list):
                self.task_queue.put((query_list, qid_list))

            while not self.result_queue.empty():
                retrieval_result = None
                recv_results = self.result_queue.get()

                current_time = time.time()

                if isinstance(recv_results, tuple) and len(recv_results) == 2:
                    retrieval_ids = recv_results[0]
                    retrieval_result = recv_results[1]
                    result_list.extend(retrieval_ids)

                    for ids in retrieval_ids:
                        output_time_list[ids.id] = current_time - start_time
                
        self.task_queue.put("RETRIEVAL END")
        self.retrieve_p.join()

        final_time = output_time_list - input_time_list
        avg_latency = np.mean(final_time)
        print(f"Average latency: {avg_latency}")


def get_new_ragraph(name: str, config):
    if (name == "Sequential"):
        return SequentialRAGraph(config)
    elif (name == "IRG"):
        return IRGRAGraph(config)
    else:
        raise ValueError("Not supported workflow")

def get_generator(config):
    if config["generator"] == "vllm":
        return getattr(importlib.import_module("heterag.executor.generator"), "VLLM_asyncGenerator")(config)
    else:
        raise ValueError("Not supported generator")