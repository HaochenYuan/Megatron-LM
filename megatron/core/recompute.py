# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer

_EAGER_RC_PRINTS = 0  # rate-limit for the MCORE_EAGER_RECOMPUTE probe telemetry

te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint


def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    rotary_pos_emb: Tensor,
    attention_bias: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    padding_mask: Optional[Tensor] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
    input_ids: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Forward method with activation checkpointing.

    Args:
        extract_layer_indices (Set[int], optional): Global layer
            indices (across all pipeline stages) from which to
            extract features.
        layer_offset (int): The global layer offset for the current
            pipeline stage. Used to convert local layer indices to
            global indices when checking extract_layer_indices.

    Returns:
        If extract_layer_indices is empty: hidden_states tensor
        If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    # Lazy import to avoid a circular dependency (hybrid_block imports this module).
    # HyperConnectionHybridLayer is NOT a TransformerLayer subclass, so it must be
    # handled explicitly below: it accepts input_ids/padding_mask (needed by hash-MoE
    # routing) but not the cross-attention context/attention_bias kwargs.
    try:
        from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
    except Exception:  # pragma: no cover - hybrid model not always importable
        HyperConnectionHybridLayer = ()

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask=None
        ):
            # PROBE (env MCORE_EAGER_RECOMPUTE=1): grad is enabled here ONLY during the
            # activation-checkpoint backward RERUN (te_checkpoint runs the forward under
            # no_grad, the recompute under enable_grad). Flag the cuda-graph dispatch to
            # run EAGER during this recompute so the graph is NOT re-replayed in backward.
            _in_rc = torch.is_grad_enabled()  # True only during the checkpoint backward rerun
            from megatron.core.transformer.cuda_graphs import _rcflow_hit, _set_in_recompute

            _rcflow_hit("RCPY_custom_fwd")

            _set_in_recompute(_in_rc)
            # RC TAP (MCORE_RC_TAP=1): pair this forward/recompute window by the checkpoint
            # input address and tap the inputs the recompute must reproduce exactly.
            from megatron.core.transformer.cuda_graphs import _rc_tap, _rc_tap_begin, _rc_tap_end

            _rc_tap_begin(hidden_states, prefix=f"c{start}{end}.")
            _rc_tap("IN", hidden_states)
            _rc_tap("AUX_cu_q", getattr(packed_seq_params, "cu_seqlens_q", None))
            _rc_tap("AUX_cu_kv", getattr(packed_seq_params, "cu_seqlens_kv", None))
            _rc_tap("AUX_pad", padding_mask)
            _rc_tap("AUX_ids", input_ids)
            _eager_rc = os.environ.get("MCORE_EAGER_RECOMPUTE", "0") == "1" and _in_rc
            if _eager_rc:
                from megatron.core.transformer.cuda_graphs import _set_eager_recompute_active

                _set_eager_recompute_active(True)
            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]

                # Get appropriate inner quantization context
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    # TODO: check if fp4 is supported in this case
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                # Build the full TransformerLayer kwarg set; for non-TL
                # layers (currently MambaLayer in HybridStack) pop the kwargs
                # they don't accept and treat the return as a single tensor.
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                    input_ids=input_ids,
                )
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    elif HyperConnectionHybridLayer and isinstance(
                        layer, HyperConnectionHybridLayer
                    ):
                        # mHC layer wraps a TransformerLayer: it threads input_ids
                        # (required for hash-MoE routing) and padding_mask, but does
                        # not accept the cross-attention context kwargs. Popping only
                        # those keeps input_ids alive through full recompute.
                        for k in ("context", "context_mask", "attention_bias"):
                            layer_kwargs.pop(k, None)
                        hidden_states, context = layer(**layer_kwargs)
                    else:  # MambaLayer (HybridStack `M` slot)
                        for k in (
                            "context",
                            "context_mask",
                            "attention_bias",
                            "padding_mask",
                            "input_ids",
                        ):
                            layer_kwargs.pop(k, None)
                        hidden_states = layer(**layer_kwargs)
                        context = None

                # Some layer paths may still return a tuple (defensive).
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                _rc_tap(f"L{index}", hidden_states)
            _rc_tap_end()
            from megatron.core.transformer.cuda_graphs import _set_in_recompute as _clr_in_rc

            _clr_in_rc(False)
            if _eager_rc:
                from megatron.core.transformer.cuda_graphs import (
                    _get_eager_rc_hits,
                    _set_eager_recompute_active,
                )

                _set_eager_recompute_active(False)
                global _EAGER_RC_PRINTS
                if _EAGER_RC_PRINTS < 4:
                    _EAGER_RC_PRINTS += 1
                    print(
                        f"[EAGER_RC_PROBE] recompute chunk [{start},{end}) ran eager; "
                        f"cumulative eager-return hits={_get_eager_rc_hits()}",
                        flush=True,
                    )
            return hidden_states, context

        return custom_forward

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        # M2 PROBE (env MCORE_RC_CLONE_INPUT=1, default off/inert): the activation
        # checkpoint saves this chunk's INPUT `hidden_states` and re-supplies it at
        # recompute time. If that input aliases a CUDA-graph static buffer (the prior
        # chunk-graph block's captured output), a later graph replay -- including the
        # backward recompute itself -- can clobber the address, so the recompute reads
        # a DIFFERENT input than the forward did ("input not preserved"). Cloning forces
        # the checkpoint to save a private copy. If this closes the chunk+full mHC
        # divergence, the cause is input aliasing (M2); if not, it is the aggregate
        # OUTPUT address (M1, needs the mHC arena direct-write).
        if use_checkpoint and os.environ.get("MCORE_RC_CLONE_INPUT", "0") == "1":
            hidden_states = hidden_states.clone()
        cf = custom(start, end)
        args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask)
        if use_checkpoint:
            # Precision-aware activation checkpoint: TE under FP8/FP4,
            # tensor_parallel under BF16/FP16/FP32.
            if self.config.fp8 or self.config.fp4:
                hidden_states, context = te_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    *args,
                )
            else:
                hidden_states, context = tensor_parallel.checkpoint(
                    cf, self.config.distribute_saved_activations, *args
                )
        else:
            # Note: original block-branch no-checkpoint path omitted padding_mask
            # (relied on its default=None); restored here for consistency.
            hidden_states, context = cf(*args)

        if self.config.recompute_method == "uniform":
            if (end - 1 + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)
        else:
            if (start + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of layers and checkpoint
        # the input activation of each divided chunk.
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            chunk_end = min(
                layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank
            )
            chunk_runner(layer_idx, chunk_end, True)
            layer_idx += self.config.recompute_num_layers
    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # layers and skip the rest. Need at least one input tensor with
        # gradient computation for the re-entrant autograd engine, so under
        # FP8/FP4 we skip checkpointing while hidden_states.requires_grad
        # is False (these slots get pushed past the recompute window).
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            use_checkpoint = (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            )
            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
    else:
        raise ValueError("Invalid activation recompute method.")

    # Return intermediate hidden states if feature extraction was requested
    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states
