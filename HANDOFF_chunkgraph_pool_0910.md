# Hand-off: chunk-wise CUDA graph with post-process + optimizer step in one graph pool (DSv4 flash, 2026-09-10)

## What this branch contains (snapshot of the tested working tree, base = branch `chunkgraph-rebase-0907` @ 82761c05e)
1. **Recompute-RNG fix** (`megatron/core/transformer/recompute_window.py`, hooks in `te_checkpoint` / `tensor_parallel.checkpoint`,
   `RandomSTE` / `RandomSTEShared` / router input-jitter): stochastic ops inside an activation checkpoint replay the forward's draw
   (seed replay + hash regeneration) instead of re-drawing in the recompute. Always on. Needed for `--recompute-granularity full` +
   `--moe-router-force-load-balancing` (and any router randomness) with the TE RNG tracker, which CUDA graphs force on.
   Unit tests: `tests/unit_tests/transformer/test_recompute_window.py`.
2. **Post-process captured into the chunk graphs** (`HybridPostProcessBlock` in `megatron/core/models/hybrid/hybrid_model.py`,
   discovery in `cuda_graphs.py`): MTP block + LM head + cross entropy become the 2nd callable of the last model chunk, captured in
   the same `_order` capture as the decoders, so they share the chunk graph pool replay-safely. Env-gated: `MCORE_CG_CAPTURE_POSTPROCESS=1`.
3. **Optimizer step as a CUDA graph in the same pool and on the same capture stream**: `--optimizer-cuda-graph` +
   `MCORE_GRAPH_EXTERNAL_POOL=1 MCORE_GRAPH_UNIFY_POOL=1` (`cuda_graphs.py`, `full_cuda_graph.py`, `optimizer/optimizer.py`).
4. **Graph-safety fixes found on the way (all needed at 128 GPUs):**
   - `MTPLossAutoScaler.set_loss_scale` rebinding a new tensor each iteration -> in-place copy (any graph containing the MTP loss backward).
   - `ChainedOptimizer.get_grad_norm` aggregation kept tensor-only (optimizer graph capture with `--clip-grad`).
   - `multi_token_prediction.py::_build_contiguous_packed_seq_roll_plan`: removed the boolean-mask compaction of duplicate cu_seqlens
     (data-dependent indexing killed capture at CP>1 with `--cp-partition-mode contiguous`).
   - `training.py`: MoE / DSA-indexer / MTP loss trackers cleared after `create_cudagraphs()` (TE warm-up passes are not training steps;
     otherwise the capture-step logged indexer loss is +52%).
5. Env-gated, default-off probes used during the analysis (inert unless set): `MCORE_RC_TAP`, `MCORE_RCFLOW_TRACE`, `MCORE_GRAD_DUMP*`,
   `MCORE_TOPK_DUMP*`, `MCORE_PHASE_MEM_PROBE`/`MCORE_PHASE_MEM_ITERS`, `MCORE_CG_SLOT_PROBE`, `MCORE_CG_FORCE_SLOTS`,
   `MCORE_CG_NO_BUFFER_REUSE`, `MCORE_GRAPH_POOL_LEND*` (memory-only lending experiment; do NOT use for training),
   `MCORE_CKPT_FWD_KERNEL_PARITY`, `MCORE_DISABLE_RECOMPUTE_WINDOW` (A/B only), `MCORE_EAGER_RECOMPUTE`, `MCORE_RC_CLONE_INPUT`.

## Enabling the best configuration (on top of the usual DSv4 flash chunk-graph config)
Already in the production args: `--cuda-graph-impl transformer_engine --cuda-graph-granularity chunk --cuda-graph-dynamic-microbatches
--cuda-graph-warmup-steps 2 --te-rng-tracker --recompute-granularity full ... --cross-entropy-fusion-impl linear`
(THD chunk graphs need `--max-seqlen-per-dp-cp-rank` and `--thd-max-packed-sequences`).

ADD:
```
# args
--optimizer-cuda-graph
# env (every rank)
export MCORE_CG_CAPTURE_POSTPROCESS=1      # MTP + LM head + CE captured with the chunk graphs (last PP stage)
export MCORE_GRAPH_EXTERNAL_POOL=1         # MCore-owned graph pool handle
export MCORE_GRAPH_UNIFY_POOL=1            # optimizer graph uses the chunk pool AND TE's capture stream (both needed, else +0.7 GB)
# optional memory breakdown (snapshot-based, restrict to a few iterations at scale)
export MCORE_PHASE_MEM_PROBE=1 MCORE_PHASE_MEM_ITERS=3,7,15
```
Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True` and `--cuda-graph-warmup-steps >= 1`
(the eager warm-up steps size the loss trackers and the compact workspaces before capture).
Eager baselines for numerics comparisons must also use `--te-rng-tracker` (mcore vs TE tracker are different routing realizations).
Not in this branch: upstream PR #5992 (compact indexer / deterministic top-k ties); `--deterministic-mode` runs but is not bit-exact for this model.

## Measured (0910)
- 128x GB200, DSv4 flash 1M seq, PP2 CP64 EP64, 15 iters: last-stage max reserved 150.5 GB vs 161.4 GB (chunk graph today) vs 142.2 GB (eager)
  -> chunk-graph overhead 19.2 -> 8.3 GB (-57%); stage 0 unchanged; 779 vs 768 (chunk) vs 754 (eager) TFLOP/s/GPU;
  loss within the run-to-run band (it1-8 |dloss| <= 0.023 vs eager; late steps inside the realization band). wandb thd_graph_test/runs/6c6qplcl.
- proxy (1 node, PP2 VPP2 EP2, seq 4096): last stage 41.0 GB vs 42.7 (chunk) vs 36.9 (eager), deterministic mode, numerics in band.
Details: `thd_chunk_mem_0901/mem_analysis/SHARED_POOL_0909.md`, `ROOT_CAUSE_recompute_rng_0908.md`.

## Runtime stack (must match; see `thd_chunk_mem_0901/launch_SOTA128OGU.sbatch` for the exact PYTHONPATH/env)
container `enroot_sqsh/mcore-moe-pytorch26.04-te2.17.0.dev0-cudnn9.22-arm64.sqsh`; TE build `te_router_tensor_0610_gb200_min_20260615`;
cuDNN frontend 1.27.0 runtime (`runtime_site_cudnn_73d8feb`) + CUTLASS DSL; flash_mla b7643bd; fast_hadamard_transform 1.1.0;
emerging_optimizers 0.2.0 (Muon); DeepEP hybrid-ep; NVRx sitecustomize shim; kernels JIT-compile on first use (~35-55 min cold at 128).
