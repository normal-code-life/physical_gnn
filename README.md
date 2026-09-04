# README

## conda

create a new conda environment
```bash
conda create --name phy_gnn python=3.8.18
```

basic conda environment command
```bash
conda info --envs  # check conda environment
conda activate phy_gnn # activate phy_gnn conda environment
conda deactivate # deactivate phy_gnn conda environment
```

if it is your first time setup the python dependency, please run the following command to install the dependency
```bash
pip install -r requirements.txt
```

if you have import or update the python dependency, please run the following command to update the requirement file.
```bash
conda list --export --no-pip | awk -F= '{print $1"=="$2}' > requirements.txt
```

## tensorboard
you need to install both 'tensorboard' and 'tensorboardX' package. And please use the following command to check
your model
```bash
tensorboard --logdir=tmp/passive_biv/1/logs/ 
```

## Future Optimization Directions

### 1. Remove the data-pipeline and graph-generation bottleneck

Training currently does not fully utilize the GPU because the accelerator frequently waits for data. A major source of this idle time is the relatively slow process used to prepare the nodes, neighborhoods, and tensors for each graph.

Before changing the implementation, profile the pipeline as separate stages:

1. Read a prepared sample from storage.
2. Decode and transform the sample.
3. Generate or sample nodes, edges, and neighborhoods.
4. Collate samples into a batch.
5. Transfer the batch from CPU memory to GPU memory.
6. Run the model forward and backward passes.

Track samples per second, graph-generation time, DataLoader wait time, host-to-device transfer time, GPU utilization, and peak CPU/GPU memory. This establishes whether an optimization improves end-to-end throughput instead of only accelerating one isolated function.

#### Direction A: use a more efficient structured storage layout

The prepared dataset should minimize small random reads, repeated parsing, and repeated graph processing. Possible experiments include:

- Build on the edge indices already generated during data preparation by also precomputing reusable sorted-neighbor candidates, packed subgraph views, node offsets, and graph metadata.
- Store tensors in larger contiguous shards instead of many small independently accessed objects. Variable-sized graphs can use flat arrays accompanied by graph and node offset tables.
- Benchmark optimized HDF5 shards against formats such as Zarr, LMDB, or memory-mapped NumPy arrays. Compare sequential throughput, random-read latency, worker scalability, storage size, and recovery behavior.
- Choose shards large enough to reduce file-open overhead but small enough to distribute across DataLoader workers.
- Cache frequently reused metadata and normalization statistics in memory.
- Move deterministic preprocessing to the offline data-preparation stage. Keep only stochastic sampling and lightweight tensor conversion in the online training path.
- After reducing graph-generation cost, tune `num_workers`, `prefetch_factor`, `persistent_workers`, and `pin_memory` so CPU work overlaps GPU computation.

Any new storage format must preserve the mapping between each graph, its node features, target values, and precomputed edge data. It should also support deterministic regeneration and sample-level validation.

#### Direction B: generate multiple subgraphs from each full graph

Instead of producing only one training item from each full graph, generate multiple subgraphs or neighborhood samples from the same graph. Each subgraph can contain a different set of center nodes and sampled neighbors. Compatible subgraphs can then be collated as independent items, increasing the effective batch size and exposing more parallel work to the GPU.

A possible workflow is:

1. Load or decode one full graph once.
2. Reuse its node features and precomputed neighbor candidates.
3. Sample `K` different center-node sets.
4. Build `K` subgraphs with fixed or bucketed node and edge dimensions.
5. Stack compatible subgraphs into a larger GPU batch.
6. Accumulate or aggregate their losses before the optimizer step.

Important design considerations include:

- Keep all subgraphs from a source graph in the same train, validation, or test split to prevent data leakage.
- Prevent node sampling from overrepresenting large graphs or frequently selected regions.
- Record source graph IDs and sampled node IDs so results remain traceable.
- Prefer fixed-size subgraphs or size-based buckets to reduce padding and simplify collation.
- Compare true larger batches, gradient accumulation, and multiple subgraphs per loaded graph because they have different memory and optimization behavior.
- Recheck normalization and loss aggregation so each source graph contributes the intended weight.
- Retain a deterministic full-graph or fixed-subgraph validation path so metrics remain comparable between experiments.

The first prototype should expose `subgraphs_per_graph`, `center_nodes_per_subgraph`, `neighbors_per_node`, and `subgraph_batch_size` as configuration fields. Benchmark these values together because increasing any one of them can raise attention memory and computation substantially.

### 2. Use a more efficient Transformer attention implementation

The current multi-head attention layer explicitly creates the query-key score tensor, applies softmax, and multiplies it by the value tensor. This is straightforward but can become memory- and bandwidth-intensive as the neighborhood size or effective subgraph batch grows.

Investigate replacing the manual path with PyTorch's [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). PyTorch can dispatch this operation to fused implementations such as Flash Attention when the installed version, GPU architecture, tensor shape, and data type support them. The [`sdpa_kernel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) context manager can be used during benchmarks to select or compare available backends.

Recommended evaluation steps:

1. Preserve the current query, key, value, masking, dropout, and output semantics as the reference implementation.
2. Reshape graph neighborhoods into a batched attention layout without materializing unnecessary tensor copies.
3. Test FP16 and BF16 mixed precision where numerically safe, while retaining an FP32 or unfused fallback.
4. Verify forward outputs and gradients against the current implementation within an appropriate numerical tolerance.
5. Measure training-step time, attention-kernel time, peak GPU memory, maximum feasible batch size, and end-to-end samples per second.
6. Retain an automatic fallback for unsupported devices, data types, masks, or tensor shapes.

Flash Attention should be evaluated together with the subgraph batching design. More efficient attention reduces the memory cost of larger batches, while larger and more regular subgraph batches make it easier for fused GPU kernels to achieve high utilization.

### Suggested implementation order

1. Add end-to-end profiling and establish a reproducible performance baseline.
2. Extend existing edge preprocessing with packed neighbor or subgraph candidates.
3. Benchmark structured storage layouts and tune DataLoader concurrency.
4. Implement configurable multiple-subgraph generation and batching.
5. Replace manual attention with an optimized scaled dot-product attention path.
6. Tune mixed precision, subgraph size, batch size, and worker settings together.
7. Compare final throughput and model quality against the original baseline before adopting the new pipeline.
