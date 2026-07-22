# Adaptive memory planner

## Product contract

An allocation is valid only when every stage can hold, inside the worker's
live OS/device envelope:

```text
exact checkpoint weights for the stage
+ physical KV pages for the selected context tier
+ backend allocations already measured at materialization time
<= live worker memory envelope
```

The planner does not partition memory with a fixed parameter/KV ratio. The
legacy ratios remain wire-compatible for old schedulers but are ignored when
exact metadata is available.

The default service contract is a minimum of 16,384 tokens (enough for the
12,220-token OpenCode prompt plus 4,096 reserved output tokens) and a preferred
tier of 32,768. Tiers are discrete and only move during a worker generation;
desktop pressure never continuously changes a live layer assignment.

## Phase 1: deterministic planning

The scheduler resolves an immutable Hugging Face revision, then uses the
maintained `HfApi.get_safetensors_metadata()` API. That API reads safetensors
headers with HTTP Range requests and returns every tensor's exact data offsets.
The scheduler counts the bytes selected by Parallax's own shard-loading rules:

- all tensors belonging to each decoder layer, including quantization scales;
- input embedding;
- final norm and LM head;
- tied embeddings, duplicated across different endpoint workers but counted
  once on a single-node stage.

For each stable context tier, the allocator combines these checkpoint bytes
with the model's physical per-token KV geometry. Interior capacity uses the
heaviest same-length contiguous window, so the scalar DP cannot claim that an
arbitrary stage fits because a different, lighter range fits. The DP then
validates the endpoint-aware route and water-fills only within those safe
capacities.

If exact safetensors metadata is unavailable in product mode, allocation fails
closed. Formula-based parameter estimates are still available for diagnostics
and legacy unit fixtures, never as a qualified product allocation.

## Phase 2: runtime qualification

The selected context tier is part of the scheduler-to-worker serving contract.
After the shard and backend workspace are initialized:

- MLX reads the current pressure-aware process remainder and allocates at least
  the exact number of KV blocks required by the contract;
- vLLM reads the current CUDA remainder after the product device reserve and
  workspace initialization, then uses the KV specs' physical page sizes to
  allocate at least the contract;
- SGLang uses its maintained post-weight memory profiler to materialize the KV
  pool, with the prefill batch ceiling kept separate from physical pool size,
  then publishes and validates the allocator's real token capacity;
- if the measured remainder is insufficient, initialization fails before the
  worker advertises READY or KV capacity;
- the scheduler only exposes the swarm as available when executor-published KV
  telemetry proves a complete route supports the selected tier.

The previous KV fraction is now only a preferred amount above the minimum
contract. It cannot reduce the cache below the planned context.

## Stable pressure behavior

Worker envelopes are sampled from maintained OS/runtime counters at startup:
`psutil.virtual_memory().available` plus MLX counters on Apple silicon and
`cudaMemGetInfo` through PyTorch on CUDA. The system/device reserve remains
outside the planning envelope. During a generation, warning pressure pauses
admission and sustained critical pressure drains once before restart; it does
not resize layers on every sample.

Context tiers and allocation generations are deliberately coarse. A future
runtime-feedback replan may lower one tier once for an allocation epoch, but it
must be fenced and rate-limited; an executor must never oscillate between
shards in response to noisy free-memory samples.

## Download reliability

Hugging Face automatically enables `hf-xet` when installed. A real macOS lab
process was observed stuck for more than twenty minutes inside native hf-xet
threads with an unchanged incomplete file and no useful timeout. Parallax now
defaults `HF_HUB_DISABLE_XET=1` before importing `huggingface_hub`, using the
regular resumable HTTP path for normal sharded checkpoints. Operators can
explicitly opt back in with `HF_HUB_DISABLE_XET=0`.

## Primary references

- [Hugging Face safetensors metadata API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
- [Safetensors metadata parsing with HTTP Range](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [Safetensors format and exact data offsets](https://github.com/huggingface/safetensors)
- [vLLM memory profiler](https://docs.vllm.ai/en/latest/api/vllm/utils/mem_utils/)
- [vLLM available-memory qualification](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)
- [SGLang memory breakdown and tuning](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/hyperparameter_tuning.md)
- [llama.cpp automatic fit design](https://github.com/ggml-org/llama.cpp/discussions/18049)
- [MLX-LM bounded KV cache](https://github.com/ml-explore/mlx-lm)
- [Hugging Face Xet environment controls](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
- [Hugging Face home-network Xet mitigations](https://github.com/huggingface/huggingface_hub/issues/3155)

## Remaining qualification work

Before promoting this planner:

1. build commit-addressed macOS and Windows candidates;
2. qualify Qwen3-4B and Qwen3-8B with 32k then 16k fallback behavior;
3. run the 12,220 + 4,096 OpenCode E2E, abort, model switch and memory-pressure
   tests;
