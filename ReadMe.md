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

1. Before running the code, the user needs to download the pre-built index and corpus used in the experiments.  
   The dataset is available at: [<your-download-link-here>].

2. 

## Running Experiments

Once the environment is set up, you can run the evaluation scripts to reproduce the experimental results.

We provide a series of scripts named `run_fig[X].sh`, each corresponding to Figure [X] in the paper. These scripts execute the experiments and generate the associated plots.

All individual execution and plotting scripts are located in the `evaluation/` directory.

The final plots can be found in the `evaluation/output_figure` directory.
