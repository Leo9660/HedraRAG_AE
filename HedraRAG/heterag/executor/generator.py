# generator initialization built upon FlashRAG
from abc import ABC, abstractmethod

from typing import List
from copy import deepcopy
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoConfig,
)
from vllm import AsyncEngineArgs, AsyncLLMEngine, EngineArgs, LLMEngine

class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.model_path = config["generator_model_path"]

        self.max_input_len = config["generator_max_input_len"]
        self.batch_size = config["generator_batch_size"]
        self.device = config["device"]
        self.gpu_num = torch.cuda.device_count()
        self.generation_params = config["generation_params"]

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass

class VLLM_asyncGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)

        from vllm import LLM

        if "gpu_memory_utilization" not in config:
            gpu_memory_utilization = 0.85
        else:
            gpu_memory_utilization = config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            tensor_parallel_size = self.gpu_num - 1
        else:
            tensor_parallel_size = self.gpu_num

        self.lora_path = (
            None
            if "generator_lora_path" not in config
            else config["generator_lora_path"]
        )
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True

        enable_chunked_prefill = config["enable_chunked_prefill"]
        enforce_eager = config["enforce_eager"]

        engine_args = {
            "model": self.model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            # "disable_log_requests": True,
        }

        if self.use_lora:
            raise ValueError("LoRA not implemented!")
        else:
            self.model = LLMEngine.from_engine_args(EngineArgs(**engine_args))
    
    def step(self):
        return self.model.step()

    def add_request(
        self,
        input_query,
        request_id,
        return_raw_output=False,
        return_scores=False,
        **params,
    ):
        from vllm import SamplingParams

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop(
            "max_new_tokens", None
        )
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        # fix for llama3
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        if self.use_lora:
            raise valueError("Not implemented")
        else:
            results_generator = self.model.add_request(request_id,
                input_query, 
                sampling_params)

        final_output = None

        return results_generator
    
    def abort_request(self, request_id):
        self.model.abort_request([request_id])