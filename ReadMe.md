# HedraRAG: Artifact Evaluation README

This document provides the instructions to reproduce the experimental environment for the **HedraRAG** artifact. The following components and versions are required for successful setup and evaluation.

## System Requirements

- Operating System: Ubuntu 20.04 (Linux x86_64)
- Python Version: 3.11
- CUDA Version: 12.4
- PyTorch Version: 2.5.1

We recommend using the official PyTorch Docker image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
Available at: https://hub.docker.com/r/pytorch/pytorch

## Environment Setup

1. Clone Repository

   ```git clone <your-repo-url>```

   ```cd <repo-root>```

2. Create and Activate Conda Environment (Recommended)

   ```conda create -n heterag python=3.9 -y```

   ```conda activate heterag```

3. Install Dependencies

   ```bash Dependency.sh```

6. Build HedraRAG

   ```bash Install.sh```

7. Build LangChain (baseline) [optional]

   ```cd LangChain```

   ```pip install -r requirements.txt```

## Dataset Preparation

The original paper uses a large Wikipedia page index (>100GB), which may be inconvenient for quick prototyping or evaluation. To simplify the setup, we provide a smaller pre-built index based on the MS MARCO passage corpus (~36GB) to help users efficiently build and test the pipeline.

### 1. Download Pre-built Index and Corpus

Please download the index and from the following link: **[https://doi.org/10.5281/zenodo.16663591](https://doi.org/10.5281/zenodo.16663591)**

### 2. Configure Dataset Paths

Update `data.conf` before running the pipeline:

```
export index_path=/path/to/ivf.index

export corpus_path=Tevatron/msmarco-passage-corpus

export model_path=/huggingface/model_path
```

- `index_path`: Path to the downloaded FAISS index file
- `corpus_path`: Defaults to the [Tevatron MS MARCO passage corpus](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus) on HuggingFace
- `model_path`: HuggingFace model path used for generation

### 3. Using Custom Corpus and Index (Optional)

You can also use your own corpus and corresponding index by updating the paths accordingly:

- Set `index_path` to your own FAISS index
- Set `corpus_path` to either a local path or HuggingFace dataset

If you want to build your own FAISS IVF index, we recommend using the [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2) model to encode your documents.

### 4. Build the Corpus Used in the Paper

The corpus and index used in the paper are based on Wikipedia passages up to the end of 2022, available at https://zenodo.org/records/16849723, and encoded with intfloat/e5-large-v2.

Since the index is large, we recommend building it locally. On a high-performance CPU+GPU machine, this process may take several days.

Steps:

- Download the corpus file

   Download `text-list-100-sec.jsonl` from the above link.

- Run the build script

   Use the provided `build_index.sh` and modify the first two lines:
   corpus_path=/path/to/text-list-100-sec.jsonl
   save_dir=/path/to/save_dir

   - corpus_path: Path to the downloaded `text-list-100-sec.jsonl` file
   - save_dir: Output directory; the generated `ivf.index` will be stored here

- Preprocessing and storage optimization
   - The build script supports a checkpoint mechanism for resuming. If the run fails midway, you can re-execute `build_index.sh` to resume and continue.
   - The process may consume up to 240GB of storage space and take over 30 hours on our CPU.
   - After preprocessing is complete, you can delete `emb_e5.memmap` to save storage space.

- Update the configuration
   In `data.conf`, set:
   export corpus_path=/path/to/text-list-100-sec.jsonl
   export index_path=/path/to/save_dir/ivf.index

   You can then run the paper experiments directly.

## Running Experiments

Once the environment is set up, you can run the evaluation scripts to reproduce the experimental results.

We provide a series of scripts named `run_fig[X].sh`, each corresponding to Figure [X] in the paper. These scripts execute the experiments and generate the associated plots.

All individual execution and plotting scripts are located in the `evaluation/` directory.

The final plots can be found in the `evaluation/output_figure` directory.
