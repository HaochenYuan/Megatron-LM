# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Full-iteration CUDA graph wrapper — Option B (multi-graph dispatch).

Captures one full-iteration CUDA graph per observed `num_microbatches` value
(K). At replay, dispatches to the K-specific graph based on the iter's actual
K. This eliminates the K_max-padding overhead paid by Option A (in the main
tree's copy of this file) under realistic THD packing where `num_microbatches`
varies per iter due to `dp_balanced` packing decisions.

See `option_b_multigraph/DESIGN_B.md` for the full specification.

Differences vs Option A:
  - No `empty_microbatch_fn`, no `THD_FI_KMAX`. Each iter processes exactly
    K_actual microbatches with no padding waste.
  - Multiple captured graphs sharing a single CUDA memory pool via
    `torch.cuda.graph_pool_handle()`.
  - First time we see a new K, capture for that K (one-time stall).
  - Optional pre-warm: capture a list of K values up-front via `k_prewarm`
    (pre-warm helper not yet implemented; users can rely on lazy capture).
  - Bound: `k_min` and `k_max` define the legal range. K outside the range
    raises `RuntimeError` rather than silently capturing.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static input buffer copy helpers (unchanged from Option A).
# ---------------------------------------------------------------------------

def copy_tensors_in_struct(src):
    """Deep-copy any tensors in a nested container (clone + .cuda())."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    if isinstance(src, list):
        return [copy_tensors_in_struct(i) for i in src]
    if isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    if isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    return src


def clone_tensors_in_struct(tgt, src):
    """Refresh tensors in `tgt` with values from `src` (in place via copy_)."""
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple: {type(src)}")
    if isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
        return
    if isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
        return
    if isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
        return
    raise Exception(f"Expect top-level container, got: {type(src)}")


class StaticBufferLoader:
    """Copy a per-microbatch dict into a per-microbatch static CUDA buffer.

    A separate StaticBufferLoader instance is owned per (stage, K) so the
    static buffers used by graph(K=4) don't collide with those used by
    graph(K=7).
    """

    def __init__(self):
        self._slots: List[Any] = []
        self._stream = torch.cuda.Stream()

    def __call__(self, inputs, microbatch: int):
        assert microbatch <= len(self._slots)
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]
        assert isinstance(inputs, dict)

        if microbatch == len(self._slots):
            with torch.cuda.stream(self._stream):
                self._slots.append(copy_tensors_in_struct(inputs))
        else:
            slot = self._slots[microbatch]
            for k in inputs.keys():
                if k not in slot:
                    if isinstance(inputs[k], torch.Tensor):
                        slot[k] = torch.empty_like(inputs[k], device='cuda')
                    else:
                        slot[k] = inputs[k]
            with torch.cuda.stream(self._stream):
                clone_tensors_in_struct(slot, inputs)
        torch.cuda.current_stream().wait_stream(self._stream)
        return self._slots[microbatch]


# ---------------------------------------------------------------------------
# Multi-graph full-iteration wrapper.
# ---------------------------------------------------------------------------

_KEY = Tuple[str, int]  # (stage, K)


class FullCudaGraphWrapper:
    """Multi-graph full-iteration CUDA graph dispatcher.

    Drop-in API match for the Option A (single-graph + K_max-padding) wrapper.
    Adds three new init kwargs:

        k_min:       lower bound on legal K (inclusive). Default 1.
        k_max:       upper bound on legal K (inclusive). Default 64.
        k_prewarm:   optional list of K values to pre-capture during warmup
                     (currently a no-op stub; lazy capture is the only path).

    Cache key is `(stage, K)`. All graphs share one CUDA memory pool.

    Lifecycle:
      - iter 0..warmup-1: eager. Builds initial RNG/optimizer state.
      - iter warmup: capture for the iter's K, then replay.
      - iter warmup+1..: if K already captured, replay; else capture.
    """

    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_steps: int = 1,
        pre_pad_fn=None,
        *,
        k_min: int = 1,
        k_max: int = 64,
        k_prewarm: Optional[List[int]] = None,
    ):
        self.forward_backward_func = forward_backward_func
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.pre_pad_fn = pre_pad_fn
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_prewarm: List[int] = list(k_prewarm) if k_prewarm else []

        # Per-(stage, K) state.
        self._graphs: Dict[_KEY, torch.cuda.CUDAGraph] = {}
        self._results: Dict[_KEY, Any] = {}
        self._loaders: Dict[_KEY, StaticBufferLoader] = {}
        self._captured_k_set: Dict[str, Set[int]] = {
            'training': set(),
            'validation': set(),
        }
        self._curr_iteration: Dict[str, int] = {'training': 0, 'validation': 0}

        # Shared CUDA graph memory pool. Lazy-init at first capture so that
        # `torch.cuda.is_available()` and CUDA context are guaranteed ready.
        self._shared_pool: Optional[Any] = None
        # One persistent capture stream reused across all prewarm captures
        # (vLLM/SGLang pattern). Lazy-init at first capture.
        self._capture_stream: Optional[torch.cuda.Stream] = None
        # Set once per stage after prewarm finishes; gates against re-prewarm.
        self._prewarm_done: Dict[str, bool] = {
            'training': False,
            'validation': False,
        }

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        for required in ('model', 'data_iterator', 'num_microbatches',
                         'seq_length', 'forward_only'):
            assert required in kwargs, f'missing required kwarg: {required}'

        K = int(kwargs['num_microbatches'])
        if not (self.k_min <= K <= self.k_max):
            raise RuntimeError(
                f"K (num_microbatches) = {K} is outside [{self.k_min}, {self.k_max}]. "
                f"Increase k_max for legitimate larger packs, or filter the data."
            )

        stage = 'validation' if kwargs['forward_only'] else 'training'
        key: _KEY = (stage, K)
        curr = self._curr_iteration[stage]

        # --- iter 0 bypass for warmup (matches Option A behavior) ---
        bypass_iter0 = bool(int(os.environ.get('THD_FI_BYPASS_STATIC_ITER0', '0')))
        skip_static = bypass_iter0 and curr < self.cuda_graph_warmup_steps

        if not skip_static:
            data_list = self._data_read(
                kwargs['data_iterator'], kwargs['model'], stage, K,
            )
            kwargs['data_iterator'] = data_list

        torch.cuda.synchronize()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # --- eager warmup ---
        if curr < self.cuda_graph_warmup_steps:
            if rank == 0:
                logger.info(f'Multi-graph: iter {curr} {stage} K={K} (eager warmup)')
            result = self.forward_backward_func(*args, **kwargs)
            self._curr_iteration[stage] = curr + 1
            return result

        # --- prewarm (vLLM/SGLang pattern): on the first capture-eligible
        # iter, capture all K in k_prewarm back-to-back into a shared pool
        # before any replay. After prewarm, no further captures occur. K
        # outside the prewarm set falls back to the eager no-graph path. ---
        if (curr == self.cuda_graph_warmup_steps
                and self.k_prewarm
                and not self._prewarm_done[stage]):
            self._do_prewarm(stage, K, args, kwargs)
            self._prewarm_done[stage] = True

        # --- capture / replay / eager fallback ---
        if key not in self._graphs:
            if self.k_prewarm and self._prewarm_done[stage]:
                # K not in prewarm: fall back to eager fbf so the iter still
                # runs and produces bitwise-correct loss/grad. We cannot
                # lazy-capture into the shared pool after replay (PyTorch's
                # cudaStreamBeginCapture refuses), and a per-K separated pool
                # would still hit the same unjoined-work failure mode after
                # any prior graph has been replayed. Eager is the only safe
                # fallback that keeps training going.
                if rank == 0:
                    logger.info(
                        f'Multi-graph: iter {curr} {stage} K={K} (eager fallback; '
                        f'K not in prewarm={sorted(self.k_prewarm)})'
                    )
                result = self.forward_backward_func(*args, **kwargs)
                self._curr_iteration[stage] = curr + 1
                return result
            # No prewarm configured → first-K lazy capture (works only for
            # the first K observed; subsequent K's will fail).
            self._capture(key, args, kwargs)

        if rank == 0:
            logger.info(f'Multi-graph: iter {curr} {stage} K={K} (replay)')
        self._graphs[key].replay()
        self._curr_iteration[stage] = curr + 1
        return self._results[key]

    # ------------------------------------------------------------------
    # Internal: capture
    # ------------------------------------------------------------------
    def _capture(self, key: _KEY, args, kwargs, *, use_shared_pool: bool = False):
        stage, K = key
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        tag = '[prewarm]' if use_shared_pool else '[lazy]'
        logger.info(f'Multi-graph{tag}: capturing {stage} K={K} graph !!!')
        torch.distributed.barrier()

        torch.cuda.synchronize()
        rsv_before = torch.cuda.memory_reserved()
        alloc_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        graph = torch.cuda.CUDAGraph()
        for _, state in get_all_rng_states().items():
            graph.register_generator_state(state)

        torch.cuda.synchronize()

        if use_shared_pool:
            if self._shared_pool is None:
                self._shared_pool = torch.cuda.graph_pool_handle()
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream()
            capture_stream = self._capture_stream
            cm_pool = self._shared_pool
        else:
            # Lazy fallback: per-K implicit pool. Works only for the very first
            # capture in the process (PyTorch refuses cudaStreamBeginCapture
            # once any previous graph has been replayed on the device).
            capture_stream = torch.cuda.Stream()
            cm_pool = None

        cm_kwargs: Dict[str, Any] = dict(
            stream=capture_stream, capture_error_mode='thread_local'
        )
        if cm_pool is not None:
            cm_kwargs['pool'] = cm_pool

        with torch.cuda.graph(graph, **cm_kwargs):
            self._results[key] = self.forward_backward_func(*args, **kwargs)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        self._graphs[key] = graph
        self._captured_k_set[stage].add(K)

        rsv_after = torch.cuda.memory_reserved()
        alloc_after = torch.cuda.memory_allocated()
        peak_alloc_during = torch.cuda.max_memory_allocated()
        if rank == 0:
            gib = 1024 ** 3
            logger.info(
                f'Multi-graph{tag}: capture done {stage} K={K} | '
                f'reserved Δ={(rsv_after - rsv_before) / gib:.2f} GiB '
                f'(before={rsv_before / gib:.2f}, after={rsv_after / gib:.2f}) | '
                f'allocated Δ={(alloc_after - alloc_before) / gib:.2f} GiB | '
                f'peak_alloc_during_capture={peak_alloc_during / gib:.2f} GiB'
            )

    # ------------------------------------------------------------------
    # Internal: prewarm (vLLM / SGLang pattern)
    #
    # At the first capture-eligible iter, capture all K in k_prewarm (plus
    # K_actual) back-to-back into the shared pool before any replay has run
    # on the device. After this, no further captures will succeed.
    #
    # Synthetic data: cycle the iter's real K_actual buffers to fill K_target
    # static slots. Captured kernels read pointers, not values — every later
    # iter's _data_read refreshes those slots in place via copy_().
    # ------------------------------------------------------------------
    def _do_prewarm(self, stage: str, K_actual: int, args, kwargs):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            logger.info(
                f'Multi-graph[prewarm]: starting for {stage}, K_actual={K_actual}, '
                f'k_prewarm={sorted(self.k_prewarm)}'
            )

        model = kwargs.get('model')
        if isinstance(model, list) and len(model) > 1:
            raise NotImplementedError(
                f'prewarm currently supports single-chunk models (no VPP); '
                f'got len(model)={len(model)}'
            )

        actual_loader = self._loaders[(stage, K_actual)]
        template_slots = list(actual_loader._slots[:K_actual])
        assert len(template_slots) == K_actual, (
            f'expected {K_actual} populated template slots, got {len(template_slots)}'
        )

        # Capture in large -> small order so smaller-K captures can reuse pool
        # allocations made by the larger-K captures (vLLM / SGLang convention).
        targets = sorted(set(self.k_prewarm) | {K_actual}, reverse=True)

        for K_t in targets:
            if (stage, K_t) in self._graphs:
                continue
            if K_t == K_actual:
                slots = list(actual_loader._slots[:K_t])
            else:
                target_loader = self._loaders.setdefault(
                    (stage, K_t), StaticBufferLoader()
                )
                for b in range(K_t):
                    target_loader(template_slots[b % K_actual], b)
                slots = list(target_loader._slots[:K_t])

            prewarm_kwargs = dict(kwargs)
            prewarm_kwargs['data_iterator'] = [iter(slots)]
            prewarm_kwargs['num_microbatches'] = K_t
            self._capture((stage, K_t), args, prewarm_kwargs, use_shared_pool=True)

        # The prewarm fbf calls accumulated grads on every parameter. Zero
        # them so the actual training iter starts from a clean grad state.
        self._zero_grads(model)

        if rank == 0:
            logger.info(
                f'Multi-graph[prewarm]: completed for {stage}, '
                f'captured K={sorted(targets)}'
            )

    @staticmethod
    def _zero_grads(model):
        if model is None:
            return
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    # ------------------------------------------------------------------
    # Internal: refresh per-(stage, K) static buffers with the iter's data.
    # No K_max padding — captured graph for this K is sized exactly for K.
    # ------------------------------------------------------------------
    def _data_read(self, data_iterator, model, stage: str, K: int):
        loader = self._loaders.setdefault((stage, K), StaticBufferLoader())

        if not isinstance(model, list) or len(model) == 1:
            iterator0 = (data_iterator if not isinstance(data_iterator, list)
                         else data_iterator[0])
            data_list: List[Any] = []
            if self.pre_pad_fn is not None:
                # Run pre_pad_fn on every rank for TP-broadcast symmetry.
                for b in range(K):
                    raw = next(iterator0) if iterator0 is not None else None
                    padded = self.pre_pad_fn(raw)
                    data_list.append(loader(padded, b))
                data_list = [iter(data_list)]
            elif iterator0 is not None:
                for b in range(K):
                    data_list.append(loader(next(iterator0), b))
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if self.pre_pad_fn is not None:
                    li: List[Any] = []
                    for b in range(K):
                        raw = next(data_iterator[i]) if data_iterator[i] is not None else None
                        padded = self.pre_pad_fn(raw)
                        li.append(loader(padded, b))
                    data_list.append(iter(li))
                elif data_iterator[i] is not None:
                    li = []
                    for b in range(K):
                        li.append(loader(next(data_iterator[i]), b))
                    data_list.append(iter(li))
                else:
                    data_list.append(None)
        return data_list

    # ------------------------------------------------------------------
    # Pre-warm stub (see DESIGN_B.md "Pre-warming"). Lazy capture is the
    # primary path; pre-warm is an optional optimization.
    # ------------------------------------------------------------------
    def prewarm(self, fake_data_factory):
        raise NotImplementedError(
            "prewarm: needs a fake_data_factory(K) -> data_iterator producing "
            "K synthetic microbatches matching the real shape contract. "
            "Lazy capture works without this; pre-warm only saves cold-start "
            "stalls when first-N-iters K-coverage is incomplete."
        )
