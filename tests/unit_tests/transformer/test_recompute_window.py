# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for recompute-consistent stochastic ops inside activation checkpoints."""

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import recompute_window as rw
from tests.unit_tests.test_utilities import Utils


def _reset_stash():
    rw._STASH.clear()
    rw._WINDOWS.clear()


class TestRecomputeWindow:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        _reset_stash()

    def teardown_method(self, method):
        _reset_stash()
        Utils.destroy_model_parallel()

    def test_no_window_passthrough(self):
        a = rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda"))
        b = rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda"))
        assert not torch.equal(a, b)
        assert len(rw._STASH) == 0

    def test_window_records_and_replays(self):
        key = torch.zeros(8, device="cuda")
        assert rw.push_recompute_window((key,), is_recompute=False)
        fwd = rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda"))
        fwd_ord = rw.recompute_consistent_random(None, lambda: torch.randn(3, device="cuda"))
        rw.pop_recompute_window()
        assert rw.get_recompute_window() is None

        rw.push_recompute_window((key.detach(),), is_recompute=True)
        rc = rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda"))
        rc_ord = rw.recompute_consistent_random(None, lambda: torch.randn(3, device="cuda"))
        rw.pop_recompute_window()
        assert torch.equal(fwd, rc)
        assert torch.equal(fwd_ord, rc_ord)

    def test_nested_windows(self):
        outer = torch.zeros(8, device="cuda")
        inner = torch.zeros(8, device="cuda")
        rw.push_recompute_window((outer,), False)
        a = rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda"))
        rw.push_recompute_window((inner,), False)
        b = rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda"))
        rw.pop_recompute_window()
        rw.pop_recompute_window()
        rw.push_recompute_window((outer,), True)
        assert torch.equal(a, rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda")))
        rw.push_recompute_window((inner,), True)
        assert torch.equal(b, rw.recompute_consistent_random("x", lambda: torch.randn(4, device="cuda")))
        rw.pop_recompute_window()
        rw.pop_recompute_window()

    def test_checkpoint_recompute_replays_random_draw(self):
        """The recompute of tensor_parallel.checkpoint must see the forward's random draw."""
        draws = []

        def fn(x):
            r = rw.recompute_consistent_random("draw", lambda: torch.randn_like(x))
            draws.append(r.detach().clone())
            return (x * r).sum()

        x = torch.randn(16, device="cuda", requires_grad=True)
        out = tensor_parallel.checkpoint(fn, False, x)
        out.backward()
        assert len(draws) == 2, "forward + recompute expected"
        assert torch.equal(draws[0], draws[1])
        # Gradient must correspond to the forward's draw.
        assert torch.allclose(x.grad, draws[0])

    def test_random_ste_inside_checkpoint(self):
        from megatron.core.transformer.moe.moe_utils import apply_random_logits

        seen = []

        def fn(logits):
            r = apply_random_logits(logits, stash_id=7)
            seen.append(r.detach().clone())
            return r.sum()

        logits = torch.zeros(32, 8, device="cuda", requires_grad=True)
        tensor_parallel.checkpoint(fn, False, logits).backward()
        assert len(seen) == 2
        assert torch.equal(seen[0], seen[1])


class TestSeedReplay:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        _reset_stash()

    def teardown_method(self, method):
        _reset_stash()
        Utils.destroy_model_parallel()

    def test_hash_generators_are_deterministic_and_well_distributed(self):
        seed = torch.tensor([1234567890123, -987654321], dtype=torch.int64, device="cuda")
        a = rw.hash_normal(seed, (4096, 256))
        b = rw.hash_normal(seed, (4096, 256))
        assert torch.equal(a, b)
        assert abs(a.mean().item()) < 0.01 and abs(a.std().item() - 1.0) < 0.01
        u = rw.hash_uniform(seed, (4096, 256))
        assert u.min().item() >= 0.0 and u.max().item() < 1.0
        assert abs(u.mean().item() - 0.5) < 0.01
        other = rw.hash_normal(seed + 1, (4096, 256))
        assert not torch.equal(a, other)

    def test_seed_replay_inside_checkpoint_keeps_stash_tiny(self):
        from megatron.core.transformer.moe.moe_utils import apply_random_logits

        seen = []

        def fn(logits):
            r = apply_random_logits(logits, stash_id=3)
            seen.append(r.detach().clone())
            return r.sum()

        logits = torch.zeros(1024, 256, device="cuda", requires_grad=True)
        tensor_parallel.checkpoint(fn, False, logits).backward()
        assert len(seen) == 2 and torch.equal(seen[0], seen[1])
        # forward stashed only a 2xint64 seed, and the eager recompute consumed it
        assert len(rw._STASH) == 0
        # a fresh forward draws a different seed -> different logits
        seen2 = []

        def fn2(logits):
            r = apply_random_logits(logits, stash_id=3)
            seen2.append(r.detach().clone())
            return r.sum()

        tensor_parallel.checkpoint(fn2, False, logits).backward()
        assert not torch.equal(seen[0], seen2[0])
