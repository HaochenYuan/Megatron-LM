# Hand-off: chunk-wise CUDA graph pool + MOK megakernel + compact MXFP8 DSA indexer (DSv4 flash, 2026-09-12)

Snapshot of the tested working tree. Base = branch `handoff/chunkgraph-pool-0910` (2e20861dc, see `HANDOFF_chunkgraph_pool_0910.md`
for everything that was already there: recompute-RNG fix, post-process capture, optimizer graph in the chunk pool, graph-safety fixes).
NOT a mergeable PR: env-gated probes and experiment switches ship alongside the functional changes.

## What is new on top of the 0910 branch
1. **MOK megakernel backend** (upstream PR #6572 ported: `megatron/core/transformer/moe/megakernel/`, `moe_layer.py` dispatcher gate,
   `transformer_config.py` validation, `--moe-megakernel-backend mok --moe-megakernel-backend-config '{...}'`).
   - Upstream forbids MOK with per-layer CUDA graphs. Chunk-granularity capture wraps the whole MoE layer, so the check is bypassed with
     `MCORE_MOK_ALLOW_CHUNK_GRAPH=1` (only honoured when `--cuda-graph-granularity chunk`).
   - **Fix for MOK inside TE chunk graphs** (`cuda_graphs.py`): MOK returns `param.detach()` sentinel weight gradients (the real gradient is
     fused into `main_grad`). TE's `make_graphed_callables(_reuse_graph_input_output_buffers=True)` weak-refs and memcpy-clones returned param
     grads after every replay, which fails with `CUDA error: invalid argument` on MXFP8 parameters. With `gradient_accumulation_fusion` every
     returned param grad is a dummy, so we pass `clone_param_grads_on_return=False` (TE >= 2.7). `MCORE_CG_CLONE_PARAM_GRADS=1` restores the
     old behaviour. The MOK-side equivalent would be `skip_backward_post_hook = True` on MOK-registered params.
   - `recompute_window.py`: sets the CUDA device at checkpoint-window entry (cuTile mHC kernels launched from the recompute on the autograd
     thread need a current context in the 26.07 image).
   - `mok/runtime.py`: `MCORE_MOK_COPY_GRAD_OUT=1` materialises MOK's returned gradients (experiment, default off, not needed).
2. **Compact BF16/MXFP8 DSA indexer** = upstream PR #5992 (dev 0f9c777f7) ported: `csa.py`, `csa_utils/{cp_utils,fused_sparse_attention}.py`,
   new `core/quantization/indexer_quantization.py`, `TransformerConfig.dsa_indexer_precision` (the CLI flag `--dsa-indexer-precision {bf16,mxfp8}`
   is generated from the dataclass field, as on dev). Includes the capture-time compact-workspace lookup fix for CUDA graphs.
   `--dsa-cp-balance-indexer` is NOT included: it lives only in the open PR #6058 (not on dev).
3. `training.py`: dataset sizing under `--step-batch-size-schedule` (largest batch of the schedule; otherwise StopIteration at the first step-up).
4. `pretrain_hybrid.py`: env-gated hang localisation (`MCORE_HANG_DUMP_SIGNAL=1` + `MCORE_HANG_DUMP_DIR` → SIGUSR1 dumps all thread stacks
   to `stacks_rank<r>.txt`; `MCORE_HANG_DUMP_SEC` periodic variant exists but segfaulted a rank mid-dump at 128 GPUs — do not use it).
5. `fusions/linear_cross_entropy/blackwell/*.py`: alias `cute.make_fragment -> cute.make_rmem_tensor` when the CUTLASS DSL has renamed it
   (26.07 image). Untested: all 26.07 runs below used `--cross-entropy-fusion-impl native`.

## Enabling the best MOK configuration (128x GB200 DSv4 flash, PP2 CP64 EP64, 1M THD)
Args = the colleague's MOK eager config (`--moe-megakernel-backend mok --moe-megakernel-backend-config
'{"fwd_num_comm_sms":32,"bwd_num_comm_sms":32,"minibatch_size":16384,"macrobatch_size":131072,"schedule_capacity_multiplier":0.0625,"all_gather_top_experts_chunk_bytes":2048}'`,
no `--moe-expert-rank-capacity-factor`, no `--use-transformer-engine-op-fuser`, no `--moe-mlp-glu-interleave-size`) PLUS:
```
# args
--cuda-graph-impl transformer_engine --cuda-graph-granularity chunk --cuda-graph-dynamic-microbatches --cuda-graph-warmup-steps 2
--te-rng-tracker            # eager baselines must use it too (different routing realization otherwise)
--optimizer-cuda-graph
--dsa-indexer-precision mxfp8
# env (every rank)
export MCORE_MOK_ALLOW_CHUNK_GRAPH=1       # let MOK run inside chunk graphs
export MCORE_CG_CAPTURE_POSTPROCESS=1      # MTP + LM head + CE captured with the chunk graphs (last PP stage)
export MCORE_GRAPH_EXTERNAL_POOL=1         # MCore-owned graph pool handle
export MCORE_GRAPH_UNIFY_POOL=1            # optimizer graph in the chunk pool, on TE's capture stream
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
# optional: per-rank pool breakdown at a few iterations; hang localisation
export MCORE_PHASE_MEM_PROBE=1 MCORE_PHASE_MEM_ITERS=3,7,15
export MCORE_HANG_DUMP_SIGNAL=1 MCORE_HANG_DUMP_DIR=<run dir>   # then `kill -USR1 <worker pids>` on a stall
```
Native (non-MOK) path with the same graph scope needs `--moe-expert-rank-capacity-factor 1.05` (static per-rank padding for the HybridEP
dispatch capture) and, because the config validator ties them together, `--use-transformer-engine-op-fuser`.
Exact launchers: `thd_chunk_mem_0901/launch_MOK128OGU.sbatch` (MODE=graph|eager|native), `launch_MOK128SEQ.sbatch` (several modes in one
allocation), args `sota_mok_ogu_args.txt` / `sota_mok_eager_args.txt` / `sota_native_ogu_args.txt`; proxy: `pr_megatron/proxy8_mok.sbatch`.

## Measured 2026-09-12 (128x GB200, 15 iterations, mock data, native CE, wandb project `thd_graph_test`)
| run | steady s/iter | TFLOP/s/GPU | last-stage max reserved (graph pool) | first-stage max reserved |
|---|---|---|---|---|
| MOK + chunk graph + post-process + optimizer graph, unified pool | 60.2 | 857 | 123.3-124.6 GB (52.2) | 106.3-107.9 GB |
| MOK eager (same args minus the graph flags) | 61.0 | 846 | 111.7-115.7 GB | 93.4-101.4 GB |
| native + same graph scope (+ capacity factor + op fuser) | 61.6 | 838 | 155.9-156.7 GB (60.7) | 136.4 GB |
- With MOK, the graphs add only ~1.4% throughput (the megakernel already removes the launch overhead) and cost +9-12 GB reserved on the last
  stage; MOK vs native at equal graph scope saves ~32 GB on the last stage at +2.3% throughput. Capture ~126 s on rank 0 in both graph runs.
- Numerics: MOK graph == MOK eager at iteration 1 and within the realization band afterwards. MOK (MXFP8 mode) vs native diverge from
  iteration 2 (proxy: 11.90 vs 11.41 at it2; bf16 MOK matches native to 1e-3) — a MOK-side MXFP8 issue, not a graph issue.
- Proxy (2 nodes PP2 x EP4, seq 4096): MOK + full graph scope last stage 36.2 GB = MOK eager (zero graph overhead), native 34.7 GB.

## Known issues
- One 128-GPU MOK graph run (job 7082641) deadlocked in step 1 on the last PP stage (CP-group P2P of the MTP contiguous-CP roll halos and the
  CSA CP exchange inside the MTP layer waited forever; the other 59 stage-1 ranks were not in any timed-out NCCL op). Three later identical
  128-GPU runs and CP4/CP8 reproductions were clean: timing-dependent. Run with `--distributed-timeout-minutes 20` and the SIGUSR1 stack dump
  so the next occurrence is localised.
- `--dsa-indexer-precision mxfp8` requires `--dsa-kernel-backend cudnn`, ratio-4 compressed layers and Blackwell.

## Runtime stack for this snapshot
Colleague's MOK image `mcore-moe-pytorch26.07-20260904-texinrobin-hybridep14101b0-dsa-mok-ntrace-sm100-arm.sqsh` (torch 2.13 nv26.07,
TE 2.20.0.dev0, cuDNN DSA, HybridEP 14101b0, mok) used container-native (PYTHONPATH = NVRx shim + repo only; see
`thd_chunk_mem_0901/sota_worker_ctr.sh`). The 26.04 stack of the 0910 hand-off still works for the non-MOK path.
