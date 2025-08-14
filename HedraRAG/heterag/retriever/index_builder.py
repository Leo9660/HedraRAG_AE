import os
import faiss
import json
import warnings
import numpy as np
from typing import cast
import shutil
import subprocess
import argparse
import datasets
import torch
from tqdm import tqdm
from heterag.retriever.utils import load_model, load_corpus, pooling
import time


class Index_Builder:
    def __init__(
        self,
        retrieval_method,
        model_path,
        corpus_path,
        save_dir,
        max_length,
        batch_size,
        use_fp16,
        pooling_method,
        faiss_type=None,
        embedding_path=None,
        save_embedding=False,
        faiss_gpu=False,
        use_sentence_transformer=False,
        bm25_backend='bm25s',
        corpus_source=None,
        resume=False,
        save_every=100000,
        ckpt_interval_sec=120,
        add_chunk=100000
    ):
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.use_sentence_transformer = use_sentence_transformer
        self.bm25_backend = bm25_backend
        self.resume = resume
        self.save_every = save_every
        self.ckpt_interval_sec = ckpt_interval_sec
        self.add_chunk = add_chunk
        self.gpu_num = torch.cuda.device_count()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)
        self.index_save_path = os.path.join(self.save_dir, f"ivf.index")
        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")
        self.ckpt_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.ckpt.json")
        if corpus_source == "huggingface":
            self.corpus = load_corpus(self.corpus_path, hf=True)
        else:
            self.corpus = load_corpus(self.corpus_path)

    @staticmethod
    def _check_dir(dir_path):
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        if self.retrieval_method == "bm25":
            if self.bm25_backend == 'pyserini':
                self.build_bm25_index_pyserini()
            elif self.bm25_backend == 'bm25s':
                self.build_bm25_index_bm25s()
            else:
                assert False, "Invalid bm25 backend!"
        else:
            self.build_dense_index()

    def build_bm25_index_pyserini(self):
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)
        shutil.copyfile(self.corpus_path, temp_file_path)
        pyserini_args = [
            "--collection",
            "JsonCollection",
            "--input",
            temp_dir,
            "--index",
            self.save_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
        ]
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)
        shutil.rmtree(temp_dir)

    def build_bm25_index_bm25s(self):
        import bm25s
        self.save_dir = os.path.join(self.save_dir, 'bm25')
        os.makedirs(self.save_dir, exist_ok=True)
        if ('contents' in self.corpus[0]):
            corpus_text = self.corpus['contents']
        elif ('title' in self.corpus[0] and 'text' in self.corpus[0]):
            corpus_text = [str1 + "\n" + str2 for str1, str2 in zip(self.corpus['title'], self.corpus['text'])]
        retriever = bm25s.BM25(corpus=self.corpus, backend='numba')
        retriever.index(corpus_text)
        retriever.save(self.save_dir, corpus=self.corpus)

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(self.embedding_save_path, shape=all_embeddings.shape, mode="w+", dtype=all_embeddings.dtype)
        length = all_embeddings.shape[0]
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i:j] = all_embeddings[i:j]
        else:
            memmap[:] = all_embeddings

    def _load_checkpoint(self):
        if os.path.exists(self.ckpt_path):
            try:
                with open(self.ckpt_path, "r") as f:
                    data = json.load(f)
                return int(data.get("next_index", 0))
            except Exception:
                warnings.warn("Failed to load checkpoint, will start from 0.", UserWarning)
        return 0

    def _save_checkpoint(self, next_index: int):
        tmp = {"next_index": int(next_index), "time": time.time()}
        with open(self.ckpt_path, "w") as f:
            json.dump(tmp, f)

    def _open_memmap(self, shape, mode):
        return np.memmap(self.embedding_save_path, shape=shape, mode=mode, dtype=np.float32)

    def _save_batch_embeddings(self, memmap_arr, start_idx, batch_embeddings):
        j = start_idx + batch_embeddings.shape[0]
        memmap_arr[start_idx:j] = batch_embeddings

    def st_encode_all(self):
        if self.gpu_num > 1:
            self.batch_size = self.batch_size * self.gpu_num
        if ('contents' in self.corpus):
            sentence_list = [item["contents"] for item in self.corpus]
        elif ('title' in self.corpus and 'text' in self.corpus):
            sentence_list = [(item['title'] + "\n" + item['text']) for item in self.corpus]
        if self.retrieval_method == "e5":
            sentence_list = [f"passage: {doc}" for doc in sentence_list]
        all_embeddings = self.encoder.encode(sentence_list, batch_size=self.batch_size)
        return all_embeddings

    def encode_all(self, resume=False, save_every=100000, ckpt_interval_sec=120):
        if self.gpu_num > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num
        corpus_size = len(self.corpus)
        streaming = self.save_embedding or resume
        all_embeddings_chunks = []
        last_ckpt_time = time.time()
        last_saved_doc = 0
        if self.use_sentence_transformer:
            hidden_size = self.encoder.model.get_sentence_embedding_dimension()
        else:
            hidden_size = self.encoder.module.config.hidden_size if self.gpu_num > 1 else self.encoder.config.hidden_size
        if streaming:
            mode = "r+" if (resume and os.path.exists(self.embedding_save_path)) else "w+"
            memmap_arr = self._open_memmap((corpus_size, hidden_size), mode)
            start_from = self._load_checkpoint() if resume else 0
            if start_from >= corpus_size:
                return None
        else:
            start_from = 0
        for start_idx in tqdm(range(start_from, corpus_size, self.batch_size), desc="Inference Embeddings:"):
            current_corpus = self.corpus[start_idx : start_idx + self.batch_size]
            if ('contents' in current_corpus):
                batch_data = current_corpus['contents']
            elif ('title' in current_corpus and 'text' in current_corpus):
                batch_data = [str1 + "\n" + str2 for str1, str2 in zip(current_corpus['title'], current_corpus['text'])]
            else:
                raise NotImplementedError("Wrong data!")
            inputs = self.tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to("cuda")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            if "T5" in type(self.encoder).__name__ or (self.gpu_num > 1 and "T5" in type(self.encoder.module).__name__):
                decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(inputs["input_ids"].device)
                output = self.encoder(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
                embeddings = output.last_hidden_state[:, 0, :]
            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
                )
                if "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
            if streaming:
                self._save_batch_embeddings(memmap_arr, start_idx, embeddings)
                wrote_docs = (start_idx + embeddings.shape[0]) - start_from
                need_ckpt_docs = (wrote_docs - last_saved_doc) >= save_every
                need_ckpt_time = (time.time() - last_ckpt_time) >= ckpt_interval_sec
                if need_ckpt_docs or need_ckpt_time:
                    memmap_arr.flush()
                    self._save_checkpoint(start_idx + embeddings.shape[0])
                    last_saved_doc = wrote_docs
                    last_ckpt_time = time.time()
            else:
                all_embeddings_chunks.append(embeddings)
        if streaming:
            memmap_arr.flush()
            self._save_checkpoint(corpus_size)
            return None
        else:
            all_embeddings = np.concatenate(all_embeddings_chunks, axis=0).astype(np.float32)
            return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        if os.path.exists(self.index_save_path):
            pass
        if self.use_sentence_transformer:
            from flashrag.retriever.encoder import STEncoder
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self.model_path,
                max_length=self.max_length,
                use_fp16=self.use_fp16,
            )
            hidden_size = self.encoder.model.get_sentence_embedding_dimension()
        else:
            self.encoder, self.tokenizer = load_model(model_path=self.model_path, use_fp16=self.use_fp16)
            hidden_size = self.encoder.config.hidden_size
        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            if self.use_sentence_transformer and not (self.save_embedding or self.resume):
                all_embeddings = self.st_encode_all()
            else:
                all_embeddings = self.encode_all(resume=self.resume, save_every=self.save_every, ckpt_interval_sec=self.ckpt_interval_sec)
            if self.save_embedding or self.resume:
                corpus_size = len(self.corpus)
                all_embeddings = self._load_embedding(self.embedding_save_path, corpus_size, hidden_size)
            else:
                if self.save_embedding:
                    self._save_embedding(all_embeddings)
            del self.corpus
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            if isinstance(all_embeddings, np.memmap) or self.add_chunk > 0:
                n = all_embeddings.shape[0]
                for i in tqdm(range(0, n, self.add_chunk), desc="FAISS add"):
                    j = min(i + self.add_chunk, n)
                    faiss_index.add(all_embeddings[i:j])
            else:
                faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            if isinstance(all_embeddings, np.memmap) or self.add_chunk > 0:
                n = all_embeddings.shape[0]
                for i in tqdm(range(0, n, self.add_chunk), desc="FAISS add"):
                    j = min(i + self.add_chunk, n)
                    faiss_index.add(all_embeddings[i:j])
            else:
                faiss_index.add(all_embeddings)
        faiss.write_index(faiss_index, self.index_save_path)


MODEL2POOLING = {"e5": "mean", "bge": "cls", "contriever": "mean", "jina": "mean"}


def main():
    parser = argparse.ArgumentParser(description="Creating index.")
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true", default=False)
    parser.add_argument("--bm25_backend", default='bm25s', choices=['bm25s','pyserini'])
    parser.add_argument("--corpus_source", default=None, type=str)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--save_every", type=int, default=100000)
    parser.add_argument("--ckpt_interval_sec", type=int, default=120)
    parser.add_argument("--add_chunk", type=int, default=100000)

    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = "mean"
        for k, v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ["mean", "cls", "pooler"]:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        use_sentence_transformer=args.sentence_transformer,
        bm25_backend=args.bm25_backend,
        corpus_source=args.corpus_source,
        resume=args.resume,
        save_every=args.save_every,
        ckpt_interval_sec=args.ckpt_interval_sec,
        add_chunk=args.add_chunk
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
