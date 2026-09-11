# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Recompute-consistent stochastic ops for activation checkpointing.

Activation recompute re-executes a checkpointed forward during backward and relies on the
recomputed activations being the ones the loss was computed with. For stochastic ops
(random router logits, dropout, ...) the checkpoint implementations achieve this by
restoring the CUDA RNG state before the recompute. That restore is only exact when the
saved state is a snapshot. With graph-safe RNG states (TE RNG tracker, required for CUDA
graphs) the saved "state" is a live generator object whose philox offset has advanced by
the time the recompute runs, and inside captured CUDA graphs the forward and backward graphs
re-seed their offsets independently at every replay. In both cases the recompute draws
different random numbers than the forward, so the backward is computed for a different
function than the one that produced the loss and every upstream gradient is biased.

This module provides the general remedy: a stochastic op executed inside a checkpoint
window records its random draw during the forward and replays that recorded draw during the
recompute. Recording/replaying are ordinary tensor copies, so under CUDA graphs they are
captured into the forward/backward graphs and evaluated per replay with static buffers.

The checkpoint wrappers identify a window by the address of the first checkpointed input
tensor. A checkpoint saves exactly that tensor for the recompute, so the forward and its
recompute see the same address; under CUDA graphs the address is a per-slot static buffer
that the graphs already guarantee is not reused until the matching backward retired, so the
pairing is exactly as safe as the graphs' own saved-activation liveness.

Set ``MCORE_DISABLE_RECOMPUTE_WINDOW=1`` to disable the replay (A/B testing only).
"""

import os
from collections import OrderedDict
from typing import Callable, Hashable, List, Optional, Tuple

import torch

_DISABLED = os.environ.get("MCORE_DISABLE_RECOMPUTE_WINDOW", "0") == "1"

# Stack of active checkpoint windows: [window_key, is_recompute, ordinal_counter]. Checkpoints
# can nest (e.g. a block-level full recompute containing a module-level selective checkpoint).
_WINDOWS: List[List] = []

# Recorded draws: (window_key, stash_id) -> tensor. Bounded so that eager execution, where
# window keys are ordinary allocator addresses, cannot grow it without limit.
_STASH: "OrderedDict[Tuple[int, Hashable], torch.Tensor]" = OrderedDict()
_STASH_MAX_ENTRIES = 4096


def _window_key(args) -> Optional[int]:
    for arg in args:
        if torch.is_tensor(arg):
            return int(arg.data_ptr())
    return None


def push_recompute_window(args, is_recompute: bool) -> bool:
    """Enter a checkpoint window keyed by the first tensor in ``args``.

    Returns True if a window was pushed (a matching pop is then required).
    """
    key = _window_key(args)
    if key is None:
        return False
    _WINDOWS.append([key, bool(is_recompute), 0])
    return True


def pop_recompute_window() -> None:
    """Leave the innermost checkpoint window."""
    if _WINDOWS:
        _WINDOWS.pop()


def get_recompute_window() -> Optional[Tuple[int, bool]]:
    """Return (window_key, is_recompute) for the innermost active window, else None."""
    if not _WINDOWS:
        return None
    key, is_recompute, _ = _WINDOWS[-1]
    return key, is_recompute


def in_checkpoint_forward() -> bool:
    """True while executing the (no_grad) forward pass of an activation checkpoint.

    Modules that select kernels by ``torch.is_grad_enabled()`` must treat this phase like the
    grad-enabled recompute, otherwise the recompute runs a different function than the one
    that produced the loss (the checkpoint forward runs under ``torch.no_grad()``).
    """
    return bool(_WINDOWS) and not _WINDOWS[-1][1]


# Kernel-path parity between a checkpoint forward and its recompute. Staged behind an env switch
# while it is being validated; the intended default is on.
_KERNEL_PARITY = os.environ.get("MCORE_CKPT_FWD_KERNEL_PARITY", "0") == "1"


def checkpoint_forward_uses_training_path() -> bool:
    """Whether a checkpoint forward should select the same (training) kernels as its recompute.

    Use as ``self.training and (torch.is_grad_enabled() or checkpoint_forward_uses_training_path())``
    wherever a module picks training-vs-inference kernels by grad mode.
    """
    return _KERNEL_PARITY and in_checkpoint_forward()


def checkpoint_window(forward_func: Callable) -> Callable:
    """Wrap a checkpointed function so both its forward and its recompute run in a window.

    The checkpoint implementations run the forward under ``torch.no_grad()`` and the
    recompute under ``torch.enable_grad()``, which is how the phase is detected.
    """

    def _wrapped(*args, **kwargs):
        pushed = push_recompute_window(args, torch.is_grad_enabled())
        try:
            return forward_func(*args, **kwargs)
        finally:
            if pushed:
                pop_recompute_window()

    return _wrapped


def recompute_consistent_random(
    stash_id: Optional[Hashable], generate: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """Run a stochastic generator so that a checkpoint recompute reproduces the forward.

    Outside a checkpoint window this is just ``generate()``. Inside a window, the forward
    pass records the generated tensor under (window, stash_id) and the recompute of the same
    window replays the recorded tensor instead of drawing again.

    Args:
        stash_id: Identifies the call site within a window (e.g. the layer number). Two
            stochastic ops of the same window must not share an id. ``None`` uses the
            ordinal of this call within the window, which is valid when the forward and the
            recompute issue the same sequence of stochastic calls.
        generate: Zero-argument callable producing the random tensor.
    """
    if _DISABLED or not _WINDOWS:
        return generate()
    window = _WINDOWS[-1]
    if stash_id is None:
        stash_id = ("__ordinal__", window[2])
    window[2] += 1
    key = (window[0], stash_id)
    is_recompute = window[1]
    if not is_recompute:
        value = generate()
        buf = _STASH.get(key)
        if buf is None or buf.shape != value.shape or buf.dtype != value.dtype:
            buf = torch.empty_like(value)
        else:
            _STASH.move_to_end(key)
        buf.copy_(value)
        _STASH[key] = buf
        while len(_STASH) > _STASH_MAX_ENTRIES:
            _STASH.popitem(last=False)
        return value
    buf = _STASH.get(key)
    if buf is None:
        # No recorded forward for this window; fall back to a fresh draw rather than fail.
        return generate()
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # Captured into the backward graph: the buffer must stay alive for every replay.
        return buf.clone()
    # Eager recompute: the entry is consumed exactly once; free it.
    return _STASH.pop(key)


# ---------------------------------------------------------------------------------------------
# Seed replay: record a 16-byte seed instead of the random values, and regenerate the values
# deterministically from it. This keeps the stash O(bytes) per stochastic op regardless of the
# tensor size (a [tokens, experts] fp32 router draw is 16 MiB per layer per in-flight microbatch at
# 1M sequence length; times the layers and CUDA-graph slots that is several GiB on the last stage).
# ---------------------------------------------------------------------------------------------
_SPLITMIX_GAMMA = -7046029254386353131  # 0x9E3779B97F4A7C15 as signed int64
_SPLITMIX_C1 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_SPLITMIX_C2 = -7723592293110705685  # 0x94D049BB133111EB


def _lshr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift of int64 (torch's >> is arithmetic)."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer on int64 tensors (wrapping arithmetic)."""
    x = (x ^ _lshr(x, 30)) * _SPLITMIX_C1
    x = (x ^ _lshr(x, 27)) * _SPLITMIX_C2
    return x ^ _lshr(x, 31)


def recompute_consistent_seed(stash_id: Optional[Hashable], device) -> torch.Tensor:
    """Draw (forward) or replay (recompute) a 2xint64 seed tensor for hash-based generation.

    The draw consumes the caller's current RNG stream (wrap the call in the appropriate RNG tracker
    fork), so ranks/steps get independent seeds; the stash keeps only 16 bytes per call site.
    """
    return recompute_consistent_random(
        stash_id, lambda: torch.empty(2, dtype=torch.int64, device=device).random_()
    )


def hash_uniform(seed: torch.Tensor, shape, dtype=torch.float32) -> torch.Tensor:
    """Deterministic U[0,1) tensor of ``shape`` from a 2xint64 seed (24-bit resolution)."""
    numel = 1
    for d in shape:
        numel *= int(d)
    idx = torch.arange(numel, dtype=torch.int64, device=seed.device)
    h = _splitmix64(idx * _SPLITMIX_GAMMA + seed[0]) ^ seed[1]
    u = (_lshr(h, 40)).to(torch.float32) * (1.0 / (1 << 24))
    return u.to(dtype).view(*shape)


def hash_normal(seed: torch.Tensor, shape, dtype=torch.float32) -> torch.Tensor:
    """Deterministic N(0,1) tensor of ``shape`` from a 2xint64 seed (Box-Muller)."""
    numel = 1
    for d in shape:
        numel *= int(d)
    idx = torch.arange(numel, dtype=torch.int64, device=seed.device)
    h1 = _splitmix64(idx * _SPLITMIX_GAMMA + seed[0])
    h2 = _splitmix64(h1 ^ seed[1])
    u1 = (_lshr(h1, 40)).to(torch.float32) * (1.0 / (1 << 24))
    u2 = (_lshr(h2, 40)).to(torch.float32) * (1.0 / (1 << 24))
    u1 = u1.clamp_min_(1.0 / (1 << 24))  # avoid log(0)
    z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(6.283185307179586 * u2)
    return z.to(dtype).view(*shape)
