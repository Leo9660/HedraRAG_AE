import json
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
# from heterag.utils import TaskID
import numpy as np
import time
import copy
import torch
import re
import os
import yaml
import random
import datetime
from datasets import load_dataset

class Item:
    r"""A container class used to store and manipulate a sample within a dataset.
    Information related to this sample during training/inference will be stored in ```self.output```.
    Each attribute of this class can be used like a dict key(also for key in ```self.output```).

    """

    def __init__(self, item_dict):
        self.id = item_dict.get("id", None)
        self.question = item_dict.get("question", None)
        self.golden_answers = item_dict.get("golden_answers", [])
        self.choices = item_dict.get("choices", [])
        self.metadata = item_dict.get("metadata", {})
        self.output = item_dict.get("output", {})

    def update_output(self, key, value):
        r"""Update the output dict and keep a key in self.output can be used as an attribute."""
        if key in ["id", "question", "golden_answers", "output"]:
            raise AttributeError(f"{key} should not be changed")
        else:
            self.output[key] = value

    def update_evaluation_score(self, metric_name, metric_score):
        r"""Update the evaluation score of this sample for a metric."""
        if "metric_score" not in self.output:
            self.output["metric_score"] = {}
        self.output["metric_score"][metric_name] = metric_score

    def __getattr__(self, attr_name):
        if attr_name in ["id", "question", "golden_answers", "metadata", "output", "choices"]:
            return super().__getattribute__(attr_name)
        else:
            output = super().__getattribute__("output")
            if attr_name in output:
                return output[attr_name]
            else:
                raise AttributeError(f"Attribute `{attr_name}` not found")

    def to_dict(self):
        r"""Convert all information within the data sample into a dict. Information generated
        during the inference will be saved into output field.

        """
        for k, v in self.output.items():
            if isinstance(k, np.ndarray):
                self.output[k] = v.tolist()
        output = {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "output": self.output,
        }
        if self.metadata != {}:
            output["metadata"] = self.metadata

        return output


class Config:
    def __init__(self, config_file_path=None, config_dict={}):

        self.yaml_loader = self._build_yaml_loader()
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict

        self.external_config = self._merge_external_config()

        self.internal_config = self._get_internal_config()

        self.final_config = self._get_final_config()

        self._check_final_config()
        self._set_additional_key()

        self._init_device()
        self._set_seed()
        # self._prepare_dir()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_file_config(self, config_file_path: str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config

    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict

        same_keys = []
        for key, value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item

        old_dict.update(new_dict)
        return old_dict

    def _merge_external_config(self):
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)

        return external_config

    def _get_internal_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        init_config_path = os.path.join(current_path, "basic_config.yaml")
        internal_config = self._load_file_config(init_config_path)

        return internal_config

    def _get_final_config(self):
        final_config = dict()
        final_config = self._update_dict(final_config, self.internal_config)
        final_config = self._update_dict(final_config, self.external_config)

        return final_config

    def _check_final_config(self):
        # check split
        split = self.final_config["split"]
        if split is None:
            split = ["train", "dev", "test"]
        if isinstance(split, str):
            split = [split]
        self.final_config["split"] = split

    def _init_device(self):
        gpu_id = self.final_config["gpu_id"]
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            import torch

            self.final_config["device"] = torch.device("cuda")
        else:
            import torch

            self.final_config["device"] = torch.device("cpu")

    def _set_additional_key(self):
        # set dataset
        dataset_name = self.final_config["dataset_name"]
        data_dir = self.final_config["data_dir"]
        self.final_config["dataset_path"] = os.path.join(data_dir, dataset_name)

        # set model path
        retrieval_method = self.final_config["retrieval_method"]
        model2path = self.final_config["model2path"]
        model2pooling = self.final_config["model2pooling"]
        method2index = self.final_config["method2index"]

        generator_model = self.final_config["generator_model"]

        if self.final_config["index_path"] is None:
            try:
                self.final_config["index_path"] = method2index[retrieval_method]
            except:
                print("Index is empty!!")
                assert False

        self.final_config["retrieval_model_path"] = model2path.get(retrieval_method, retrieval_method)
        # TODO: not support when `retrieval_model` is path

        def set_pooling_method(method, model2pooling):
            for key, value in model2pooling.items():
                if key.lower() in method.lower():
                    return value
            return "mean"

        if self.final_config.get("retrieval_pooling_method") is None:
            self.final_config["retrieval_pooling_method"] = set_pooling_method(retrieval_method, model2pooling)

        rerank_model_name = self.final_config["rerank_model_name"]
        if self.final_config.get("rerank_model_path") is None:
            if rerank_model_name is not None:
                self.final_config["rerank_model_path"] = model2path.get(rerank_model_name, rerank_model_name)
        if self.final_config["rerank_pooling_method"] is None:
            if rerank_model_name is not None:
                self.final_config["rerank_pooling_method"] = set_pooling_method(rerank_model_name, model2pooling)

        if self.final_config.get("generator_model_path") is None:
            self.final_config["generator_model_path"] = model2path.get(generator_model, generator_model)

        if "refiner_name" in self.final_config:
            refiner_model = self.final_config["refiner_name"]
            if "refiner_model_path" not in self.final_config or self.final_config["refiner_model_path"] is None:
                self.final_config["refiner_model_path"] = model2path.get(refiner_model, None)

        # set model path in metric setting
        metric_setting = self.final_config["metric_setting"]
        metric_tokenizer_name = metric_setting.get("tokenizer_name", None)
        # from heterag.utils.constants import OPENAI_MODEL_DICT

        # if metric_tokenizer_name not in OPENAI_MODEL_DICT:
        #     metric_tokenizer_name = model2path.get(metric_tokenizer_name, metric_tokenizer_name)
        #     metric_setting["tokenizer_name"] = metric_tokenizer_name
        #     self.final_config["metric_setting"] = metric_setting

    def _prepare_dir(self):
        save_note = self.final_config["save_note"]
        current_time = datetime.datetime.now()
        self.final_config["save_dir"] = os.path.join(
            self.final_config["save_dir"],
            f"{self.final_config['dataset_name']}_{current_time.strftime('%Y_%m_%d_%H_%M')}_{save_note}",
        )
        os.makedirs(self.final_config["save_dir"], exist_ok=True)
        # save config parameters
        config_save_path = os.path.join(self.final_config["save_dir"], "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(self.final_config, f)

    def _set_seed(self):
        import torch
        import numpy as np

        seed = self.final_config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value

    def __getattr__(self, item):
        if "final_config" not in self.__dict__:
            raise AttributeError("'Config' object has no attribute 'final_config'")
        if item in self.final_config:
            return self.final_config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.final_config.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config

    def __repr__(self):
        return self.final_config.__str__()

class Dataset:
    """A container class used to store the whole dataset. Inside the class, each data sample will be stored
    in ```Item``` class.
    The properties of the dataset represent the list of attributes corresponding to each item in the dataset.
    """

    def __init__(self, config=None, dataset_path=None, data=None, sample_num=None, random_sample=False):
        self.config = config
        if (config != None):
            self.dataset_name = config["dataset_name"]
        self.dataset_path = dataset_path

        self.sample_num = sample_num
        self.random_sample = random_sample

        if data is None:
            if (config["data_source"] == "huggingface"):
                self.data = self._load_data(self.dataset_name, self.dataset_path, hf = True)
            else:
                self.data = self._load_data(self.dataset_name, self.dataset_path)
        else:
            self.data = data

    def _load_data(self, dataset_name, dataset_path, hf = False):
        """Load data from the provided dataset_path or directly download the file(TODO)."""

        if not os.path.exists(dataset_path):
            # TODO: auto download: self._download(dataset_name, dataset_path)
            pass

        data = []
        if (hf):
            if ("split_name" in self.config):
                split_name = self.config["split_name"]
            else:
                split_name = "test"
            dataset_item = list(load_dataset("RUC-NLPIR/FlashRAG_datasets", self.config["data_dir"])[split_name])
            # print(dataset_item)
            for item in dataset_item:
                data.append(Item(dict(item)))
                # print(item)
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    item_dict = json.loads(line)
                    item = Item(item_dict)
                    data.append(item)
        if self.sample_num is not None:
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[: self.sample_num]

        return data

    def update_output(self, key, value_list):
        """Update the overall output field for each sample in the dataset."""

        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    @property
    def question(self):
        # print(self.data[0]["question"])
        if (self.config and self.config["data_source"] == "hugging face"):
            return [item["question"] for item in self.data]
        else:
            return [item.question for item in self.data]

    @property
    def golden_answers(self):
        return [item.golden_answers for item in self.data]

    @property
    def id(self):
        return [item.id for item in self.data]

    @property
    def output(self):
        return [item.output for item in self.data]

    def get_batch_data(self, attr_name: str, batch_size: int):
        """Get an attribute of dataset items in batch."""

        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i : i + batch_size]
            yield [item[attr_name] for item in batch_items]

    def __getattr__(self, attr_name):
        return [item.__getattr__(attr_name) for item in self.data]

    def get_attr_data(self, attr_name):
        """For the attributes constructed later (not implemented using property),
        obtain a list of this attribute in the entire dataset.
        """
        return [item[attr_name] for item in self.data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def save(self, save_path):
        """Save the dataset into the original format."""

        def convert_to_float(d):
            return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items()}

        save_data = [convert_to_float(item.to_dict()) for item in self.data]

        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=4)

def get_dataset(config):

    all_split = config["split"]
    split_dict = {split: None for split in all_split}

    """Load dataset from config."""

    dataset_path = config["dataset_path"]

    for split in all_split:
        if (config["data_source"] == None):
            split_path = os.path.join(dataset_path, f"{split}.jsonl")
            if not os.path.exists(split_path):
                print(f"{split} file not exists!")
                continue
        elif (config["data_source"] == "huggingface"):
            split_path = "RUC-NLPIR/FlashRAG_datasets"
        else:
            raise ValueError("Not supported data_source!")

        if split in ["test", "val", "dev"]:
            split_dict[split] = Dataset(
                config, split_path, sample_num=config["test_sample_num"], random_sample=config["random_sample"]
            )
        else:
            split_dict[split] = Dataset(config, split_path)

    return split_dict

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
        # print("id", id, "inter distance", inter_dis)
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
    a_topk = a_concat[topk_indices]
    i_topk = i_concat[topk_indices]
    return a_topk, i_topk

def fvec_inner_product(x, y):
    # print(len(x), len(y))
    return np.dot(np.array(x), np.array(y))

def fvec_L2sqr(x, y):
    # print(len(x), len(y))
    return np.linalg.norm(np.array(x) - np.array(y)) ** 2


def load_model(model_path: str, use_fp16: bool = False):
    #model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("Using cuda")
    elif use_fp16:
        # If using fp16 on CPU, we need to handle that carefully (but CPU FP16 is usually not recommended)
        print("FP16 is not supported on CPU. Skipping FP16 conversion.")
    
    #model.cuda()
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
    if (hf):
        corpus = datasets.load_dataset(corpus_path)
        corpus = corpus['train']
    else:
        corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    # print(f"corpus size: {len(corpus)} documents")
    # print(f"corpus 1 {corpus[0]}")
    # print(f"corpus 1 {corpus[1]}")
    # print(f"corpus 1 {corpus[2]}")
    # print(f"corpus 1 {corpus[3]}")
    # print(f"corpus 1 {corpus[4]}")
    # print(f"corpus 1 {corpus[5]}")
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
