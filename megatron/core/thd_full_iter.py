# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""THD-aware pre-padding for the full-iteration CUDA graph wrapper.

When ``--cuda-graph-impl local --cuda-graph-scope full_iteration`` is active,
``FullCudaGraphWrapper`` copies each microbatch's dataloader output into a
per-microbatch static CUDA buffer before capturing the graph. Those buffers are
sized on the first microbatch and ``tensor.copy_`` requires identical shapes
for every subsequent microbatch. THD produces variable-length packed batches,
so the static-buffer copy fails on the second microbatch.

This module factors out the padding stage of ``pretrain_gpt.get_batch`` so it
can run **before** ``StaticBufferLoader``. The returned dict is
byte-copyable into static buffers (every tensor is at its final static shape)
and carries the sentinel key ``_thd_pre_padded = True`` so
``get_batch`` can detect it and skip the padding step.
"""

from typing import Optional

import torch

from megatron.core.datasets.data_schedule import (
    get_batch_on_this_rank_for_sequence_packing,
)
from megatron.core.packed_seq_params import pad_thd_for_cuda_graph


def _single_shot_iterator(item):
    """Wrap a single dict as a one-shot iterator.

    Used to reuse ``get_batch_on_this_rank_for_sequence_packing`` whose
    contract is to ``next()`` a dict off its ``data_iterator`` argument.
    """
    yield item


def prerun_thd_pad_batch_dict(
    raw_batch_dict,
    *,
    config,
    vpp_size: Optional[int] = None,
    mtp_on_this_rank: bool = False,
    vp_stage: Optional[int] = None,
):
    # ``get_batch_on_this_rank_for_sequence_packing`` performs TP-group
    # broadcasts (cu_seqlen_size, total_tokens, tokens, labels, …). Those
    # broadcasts MUST happen on every TP rank in lockstep. This function is
    # therefore called from ``FullCudaGraphWrapper.data_read`` on every rank,
    # not just the ones that own data.
    #
    # Contract:
    #   * On TP rank 0 (of a stage that has data): ``raw_batch_dict`` is the
    #     raw iterator output (a dict). The function wraps it in a one-shot
    #     iterator so the helper can call ``next()`` once.
    #   * On all other ranks: ``raw_batch_dict`` is ``None``. The helper is
    #     called with ``data_iterator=None`` — it then participates in the
    #     broadcasts and fills the batch from the received tensors.
    """Convert one raw packed-sequence batch dict to a static-shape padded dict.

    Parameters
    ----------
    raw_batch_dict
        Dict yielded by the sequence-packing scheduler. Variable T, variable
        number of cu_seqlens entries.
    config
        TransformerConfig (read ``max_seqlen_per_dp_cp_rank`` and
        ``thd_cuda_graph_max_num_seqs``).
    vpp_size, mtp_on_this_rank, vp_stage
        Forwarded to ``get_batch_on_this_rank_for_sequence_packing`` unchanged.

    Returns
    -------
    dict
        Contains the seven-tuple produced by ``pretrain_gpt.get_batch`` but as
        a flat dict with a ``_thd_pre_padded`` sentinel. Shape contract:

        * ``tokens``/``labels``/``loss_mask``/``position_ids``: ``[M]`` padded
          with zeros past the real sequence end (``loss_mask`` in particular
          ensures padded positions never contribute to loss).
        * ``cu_seqlens_{q,kv}[_padded]``: ``[N+1]`` where
          ``N = thd_cuda_graph_max_num_seqs``. Tail entries equal the true
          packed-sequence length so attention kernels don't touch padded
          positions.
        * ``padding_mask``: ``[1, M]`` bool, ``True`` at padded positions.
        * ``max_seqlen_q``/``max_seqlen_kv``: ints, both equal
          ``max_seqlen_per_dp_cp_rank``.
    """
    # Build the per-rank iterator argument. TP rank 0 wraps the dict; every
    # other rank passes None (the helper asserts this).
    if raw_batch_dict is None:
        rank_iterator = None
    else:
        rank_iterator = _single_shot_iterator(raw_batch_dict)

    result = get_batch_on_this_rank_for_sequence_packing(
        rank_iterator,
        vpp_size=vpp_size,
        mtp_on_this_rank=mtp_on_this_rank,
        vp_stage=vp_stage,
    )

    if result is None or len(result) < 6:
        # Non-first/last PP stage without MTP: no data to pad. Still produce a
        # dict so StaticBufferLoader has something to copy.
        return {'_thd_pre_padded': True}

    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = result[:6]

    if packed_seq_params is None:
        return {'_thd_pre_padded': True}

    tokens, labels, loss_mask, position_ids, packed_seq_params, padding_mask = (
        pad_thd_for_cuda_graph(
            tokens,
            labels,
            loss_mask,
            position_ids,
            packed_seq_params,
            max_seqlen=config.max_seqlen_per_dp_cp_rank,
            max_num_seqs=config.thd_cuda_graph_max_num_seqs,
        )
    )

    padded = {'_thd_pre_padded': True}

    # Only pack tensors that were actually produced (middle PP stages won't
    # have tokens/labels, for example).
    if tokens is not None:
        padded['tokens'] = tokens
    if labels is not None:
        padded['labels'] = labels
    if loss_mask is not None:
        padded['loss_mask'] = loss_mask
    if position_ids is not None:
        padded['position_ids'] = position_ids
    if attention_mask is not None:
        padded['attention_mask'] = attention_mask
    if padding_mask is not None:
        padded['padding_mask'] = padding_mask

    # Decompose PackedSeqParams into individual fields so every cross-graph
    # tensor is byte-copyable and every scalar lives as a python primitive
    # (StaticBufferLoader copies primitives by assignment, not tensor.copy_).
    padded['cu_seqlens_q'] = packed_seq_params.cu_seqlens_q
    padded['cu_seqlens_kv'] = packed_seq_params.cu_seqlens_kv
    padded['cu_seqlens_q_padded'] = packed_seq_params.cu_seqlens_q_padded
    padded['cu_seqlens_kv_padded'] = packed_seq_params.cu_seqlens_kv_padded
    padded['max_seqlen_q'] = packed_seq_params.max_seqlen_q
    padded['max_seqlen_kv'] = packed_seq_params.max_seqlen_kv
    padded['qkv_format'] = packed_seq_params.qkv_format

    # Sync to make sure all the TP broadcasts and CP slicing kernels above
    # have finished writing into these tensors before StaticBufferLoader
    # clones them onto its own stream. Without this, the clone can race with
    # in-flight broadcast writes and the resulting static buffer reads will
    # see uninitialized memory at attention time.
    torch.cuda.synchronize()

    return padded


# NOTE: `make_empty_microbatch_dict` (Option A's K_max-padding helper) is
# intentionally absent from this Option B copy of the tree. Option B captures
# one CUDA graph per observed K, so empty/dummy microbatches are never needed.


def make_prerun_pad_fn(config, vpp_size=None, mtp_on_this_rank=False, vp_stage=None):
    """Build a closure ``FullCudaGraphWrapper.pre_pad_fn`` can use.

    Closing over the config/vp parameters means the wrapper only has to
    forward a single raw-batch argument, matching the ``pre_pad_fn(raw) -> dict``
    signature expected by ``FullCudaGraphWrapper.data_read``.
    """
    def _fn(raw_batch):
        return prerun_thd_pad_batch_dict(
            raw_batch,
            config=config,
            vpp_size=vpp_size,
            mtp_on_this_rank=mtp_on_this_rank,
            vp_stage=vp_stage,
        )

    return _fn


def reconstruct_thd_tuple_from_prepadded_dict(batch):
    """Inverse of :func:`prerun_thd_pad_batch_dict` for ``get_batch``.

    Given a dict produced by ``prerun_thd_pad_batch_dict`` (possibly after
    round-tripping through static CUDA buffers), rebuild the seven-tuple
    ``(tokens, labels, loss_mask, attention_mask, position_ids, PackedSeqParams,
    padding_mask)`` expected by ``forward_step``.

    Invariants
    ----------
    * No ``.item()`` / GPU↔CPU sync — all tensor fields are read directly.
    * No new tensor allocation — everything is a reference to the static buffer.
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    psp = PackedSeqParams(
        qkv_format=batch.get('qkv_format', 'thd'),
        cu_seqlens_q=batch.get('cu_seqlens_q'),
        cu_seqlens_kv=batch.get('cu_seqlens_kv'),
        cu_seqlens_q_padded=batch.get('cu_seqlens_q_padded'),
        cu_seqlens_kv_padded=batch.get('cu_seqlens_kv_padded'),
        max_seqlen_q=batch.get('max_seqlen_q'),
        max_seqlen_kv=batch.get('max_seqlen_kv'),
    )

    return (
        batch.get('tokens'),
        batch.get('labels'),
        batch.get('loss_mask'),
        batch.get('attention_mask'),
        batch.get('position_ids'),
        psp,
        batch.get('padding_mask'),
    )
