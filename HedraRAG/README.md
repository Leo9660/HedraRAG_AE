HeteRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows
================================================================================

HeteRAG is a unified and high-performance framework for building and serving heterogeneous RAG (Retrieval-Augmented Generation) workflows. It combines a flexible graph-based abstraction (RAGraph) with a suite of system-level co-optimization techniques to maximize efficiency on CPU–GPU hybrid systems.

Key Features
------------

1. Graph-Based RAG Workflow Construction
   - Use intuitive graph primitives to define arbitrary RAG workflows.
   - Seamlessly model multi-stage pipelines, including pre-retrieval generation, iterative retrieval-generation, and conditional branching.
   - Compatible with open-source RAG stacks like LangChain, LlamaIndex, and FlashRAG.

Example:

    from HeteRAG import RAGraph
    
    g = RAGraph()
    g.add_generation(0, prompt="Generate hypothesis for {query}.", output="hypopara")
    g.add_retrieval(1, topk=5, query="hypopara", output="docs")
    g.add_generation(2, prompt="Answer {query} using {docs}.")
    g.add_edge(RAGraph.START, 0)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, RAGraph.END)

2. Supported RAG Workflows

HeteRAG natively supports a variety of advanced RAG workflows beyond simple retrieval-then-generation:

- One-shot: Classic single-retrieval, single-generation pipeline.
- HyDE: Hypothesis-first generation followed by retrieval and refinement.
- Multistep: Question decomposition and iterative answering.
- RECOMP: Retrieval compression and refinement after initial generation.

Users can easily specify the desired workflow by switching the `workflow` argument:

    executor.add_requests_string(query_list, workflow="HyDE")  # Or "Multistep", "RECOMP", etc.

3. System-Level Co-Optimization
   - Sub-Stage Pipelining: Partition generation and retrieval into finer-grained units for better CPU–GPU overlap.
   - Semantic-Aware Reordering & Speculation: Leverage intra-request similarity to enable early speculative execution with runtime validation.
   - Adaptive GPU Index Caching: Dynamically cache frequently accessed clusters in GPU memory.
   - Dynamic Graph Transformation: Apply node splitting, reordering, and edge rewiring adaptively based on workload patterns.

Quick Evaluation with Offline Execution
---------------------------------

We provide a ready-to-run test script `test_lib.py` for benchmarking performance across different workflows. Currently it's for offline execution test.

Example usage:

    python test_lib.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --retriever_path intfloat/e5-large-v2 \
      --rag_workflow Multistep \
      --gpu_id 0 \
      --nprobe 128

Features:
- Switch between Sequential, HyDE, RECOMP, Multistep workflows via --rag_workflow
- Measure query latency, CPU and GPU runtime
- Configure speculative execution and GPU index caching using --speculative_retrieval and --gpu_onloading


Evaluation Highlights
---------------------

- Up to 5× latency speedup and 3× throughput gain over LangChain / FlashRAG.
- Higher speculation accuracy vs. RaLMSpec and PipeRAG.
- Generalizes to complex workflows (Multistep, IRG, HyDE).

Development Notes
-----------------

- GPU onloading and FAISS-related enhancements have already been implemented in the internal codebase. These components are currently undergoing further optimization and structural refactoring to maximize performance and usability.
- Ongoing improvements include:
  - Adaptive GPU index cache sizing based on KV cache pressure and retrieval workload
  - New graph transformation rules for dynamic node partitioning and dependency realignment

