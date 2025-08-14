import os
import importlib
from transformers import AutoConfig
from heterag.utils.dataset import Dataset
from datasets import load_dataset

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