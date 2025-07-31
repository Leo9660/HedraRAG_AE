# from FlashRAG

from typing import List
import torch
import numpy as np
from utils import load_model, pooling
import time


def parse_query(model_name, query_list, is_query=True):
    """
    processing query for different encoders
    """

    def is_zh(str):
        import unicodedata

        zh_char = 0
        for c in str:
            try:
                if "CJK" in unicodedata.name(c):
                    zh_char += 1
            except:
                continue
        if zh_char / len(str) > 0.2:
            return True
        else:
            return False

    if isinstance(query_list, str):
        query_list = [query_list]

    return query_list


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        # self.device = torch.device("cpu")
        # self.model.to(self.device)
    @torch.inference_mode(mode=True)
    def encode(self, query_list: List[str], is_query=True, profile=False) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, is_query)

        if profile:
            t1 = time.time()

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )

        inputs = {k: v.cuda() for k, v in inputs.items()}
        #inputs = {k: v.to("cpu") for k, v in inputs.items()}
        if profile:
            t2 = time.time()

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            if profile:
                t3 = time.time()
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            # if "dpr" not in self.retrieval_method:
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        if profile:
            t4 = time.time()

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        if profile:
            t5 = time.time()
            print("encoder profile")
            print(f"time 1: {t2 - t1}")
            print(f"time 2: {t3 - t2}")
            print(f"time 3: {t4 - t3}")
            print(f"time 3: {t5 - t4}")

        return query_emb


class STEncoder:
    def __init__(self, model_name, model_path, max_length, use_fp16):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model = SentenceTransformer(
            model_path, model_kwargs={"torch_dtype": torch.float16 if use_fp16 else torch.float}
        )

    @torch.inference_mode(mode=True)
    def encode(self, query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, is_query)
        query_emb = self.model.encode(
            query_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        )
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb

    @torch.inference_mode(mode=True)
    def multi_gpu_encode(self, query_list: List[str], is_query=True, batch_size=None) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, is_query)
        pool = self.model.start_multi_process_pool()
        query_emb = self.model.encode_multi_process(
            query_list, pool, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size
        )
        self.model.stop_multi_process_pool(pool)
        query_emb.astype(np.float32, order="C")

        return query_emb
