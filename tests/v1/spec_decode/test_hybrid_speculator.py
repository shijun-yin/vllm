# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.spec_decode.test_suffix_speculator import _make, _propose, _set
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.hybrid.speculator import (
    _hybrid_observe_kernel,
    _hybrid_route_publish_kernel,
)

if not current_platform.is_cuda_alike():
    pytest.skip("requires a CUDA-like GPU", allow_module_level=True)

DEVICE = torch.device("cuda")


def test_splice_route_maturity_prefix_and_identical_tail():
    model = torch.tensor([[100, 101, 102]], dtype=torch.int64, device=DEVICE)
    suffix = torch.tensor([[1000, 1001]], dtype=torch.int64, device=DEVICE)
    model_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)

    def route():
        return _route_single(model_stats, suffix_stats, model=model, suffix=suffix)

    assert route()
    suffix_stats[3, :, 0] = 8
    suffix_stats[3, :, 1] = 8
    assert not route()
    model_stats[3, :, 1] = 8
    assert route()
    model_stats[3, :, 0] = 8
    assert not route()

    suffix_stats[3, :, 1] = 0
    model[0, 0] = -1
    assert not route()
    model[0, 0] = 100
    suffix.copy_(model[:, 1:])
    assert not route()


@pytest.mark.parametrize(
    ("k", "stats_k"),
    [(3, 3), (5, 5), (15, 15), (3, 5)],
    ids=["k3", "k5", "k15", "dynamic-k3-of-5"],
)
def test_splice_publish_and_observe_last_slot(k, stats_k):
    model = torch.arange(k, dtype=torch.int64, device=DEVICE).unsqueeze(0) + 100
    suffix = torch.arange(k - 1, dtype=torch.int64, device=DEVICE).unsqueeze(0) + 1000
    idx_mapping = torch.tensor([2], dtype=torch.int64, device=DEVICE)
    output = torch.zeros((1, k), dtype=torch.int64, device=DEVICE)
    last_use_suffix = torch.zeros(3, dtype=torch.bool, device=DEVICE)
    last_buckets = torch.full((3,), -1, dtype=torch.int32, device=DEVICE)
    last_valid = torch.zeros(3, dtype=torch.int32, device=DEVICE)
    model_stats = torch.zeros((18, stats_k, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, stats_k, 2), dtype=torch.int64, device=DEVICE)
    _hybrid_route_publish_kernel[(1,)](
        torch.full((1,), k - 1, dtype=torch.int32, device=DEVICE),
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        model,
        model.stride(0),
        suffix,
        suffix.stride(0),
        model_stats,
        suffix_stats,
        torch.zeros(1, dtype=torch.bool, device=DEVICE),
        idx_mapping,
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        0,
        output,
        output.stride(0),
        last_use_suffix,
        last_buckets,
        last_valid,
        MIN_MATCH=4,
        K=k,
        STATS_K=stats_k,
        K_PO2=1 << (k - 1).bit_length(),
        PREFIX=1,
        QUALITY_GATE=False,
        MIN_EXPECTED=0.0,
        EXPLORATION_INTERVAL=0,
    )
    assert output[0, 0].item() == 100
    assert output[0, 1:].cpu().tolist() == suffix[0].cpu().tolist()
    assert last_use_suffix[2].item()
    assert last_valid[2].item() == k

    _hybrid_observe_kernel[(1,)](
        idx_mapping,
        torch.zeros(1, dtype=torch.bool, device=DEVICE),
        torch.tensor([2], dtype=torch.int32, device=DEVICE),
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        model_stats,
        suffix_stats,
        last_use_suffix,
        last_buckets,
        last_valid,
        K=stats_k,
        K_PO2=1 << (stats_k - 1).bit_length(),
        PREFIX=1,
    )
    assert model_stats[3, 0].cpu().tolist() == [1, 1]
    assert model_stats[:3].sum().item() == 0
    assert model_stats[4:].sum().item() == 0
    assert suffix_stats[3, 1].cpu().tolist() == [0, 1]
    assert suffix_stats.sum().item() == 1
    assert last_valid[2].item() == 0


def _route_single(
    model_stats,
    suffix_stats,
    model=None,
    suffix=None,
    prefix=1,
    route_index=0,
    exploration_interval=0,
):
    if model is None:
        model = torch.tensor([[100, 101, 102]], dtype=torch.int64, device=DEVICE)
    if suffix is None:
        suffix = torch.tensor([[1000, 1001]], dtype=torch.int64, device=DEVICE)
    k = model.shape[1]
    stats_k = model_stats.shape[1]
    output = torch.zeros((1, k), dtype=torch.int64, device=DEVICE)
    last_use_suffix = torch.zeros(3, dtype=torch.bool, device=DEVICE)
    last_buckets = torch.full((3,), -1, dtype=torch.int32, device=DEVICE)
    last_valid = torch.zeros(3, dtype=torch.int32, device=DEVICE)
    _hybrid_route_publish_kernel[(1,)](
        torch.tensor([k - prefix], dtype=torch.int32, device=DEVICE),
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        model,
        model.stride(0),
        suffix,
        suffix.stride(0),
        model_stats,
        suffix_stats,
        torch.zeros(1, dtype=torch.bool, device=DEVICE),
        torch.tensor([2], dtype=torch.int64, device=DEVICE),
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        route_index,
        output,
        output.stride(0),
        last_use_suffix,
        last_buckets,
        last_valid,
        MIN_MATCH=4,
        K=k,
        STATS_K=stats_k,
        K_PO2=1 << (k - 1).bit_length(),
        PREFIX=prefix,
        QUALITY_GATE=False,
        MIN_EXPECTED=0.0,
        EXPLORATION_INTERVAL=exploration_interval,
    )
    return last_use_suffix[2].item()


def test_splice_model_early_rejection_does_not_block_maturity():
    model_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    model_stats[3, :, 0] = torch.tensor([8, 0, 0], device=DEVICE)
    model_stats[3, :, 1] = torch.tensor([8, 8, 0], device=DEVICE)
    suffix_stats[3, :, 0] = 8
    suffix_stats[3, :, 1] = 8

    assert _route_single(model_stats, suffix_stats)


def test_splice_compares_model_and_suffix_in_same_bucket():
    model_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    model_stats[3, :, 0] = 95
    model_stats[3, :, 1] = 100
    model_stats[0, :, 0] = 30
    model_stats[0, :, 1] = 100
    suffix_stats[3, :, 0] = 80
    suffix_stats[3, :, 1] = 100

    assert not _route_single(model_stats, suffix_stats)


def test_splice_bounded_exploration_collects_both_routes():
    model_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    model_stats[3, :, 0] = 95
    model_stats[3, :, 1] = 100
    suffix_stats[3, :, 0] = 80
    suffix_stats[3, :, 1] = 100

    assert not _route_single(model_stats, suffix_stats)
    assert _route_single(model_stats, suffix_stats, exploration_interval=1)

    model_stats[3, :, 0] = 0
    suffix_stats[3, :, 0] = 100
    assert _route_single(model_stats, suffix_stats)
    assert not _route_single(model_stats, suffix_stats, exploration_interval=1)


def test_splice_structured_output_skips_only_masked_rows():
    model_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, 3, 2), dtype=torch.int64, device=DEVICE)
    last_use_suffix = torch.zeros(2, dtype=torch.bool, device=DEVICE)
    last_buckets = torch.full((2,), 3, dtype=torch.int32, device=DEVICE)
    last_valid = torch.full((2,), 2, dtype=torch.int32, device=DEVICE)
    _hybrid_observe_kernel[(2,)](
        torch.tensor([0, 1], dtype=torch.int64, device=DEVICE),
        torch.tensor([False, True], dtype=torch.bool, device=DEVICE),
        torch.tensor([2, 2], dtype=torch.int32, device=DEVICE),
        torch.tensor([1, 1], dtype=torch.int32, device=DEVICE),
        model_stats,
        suffix_stats,
        last_use_suffix,
        last_buckets,
        last_valid,
        K=3,
        K_PO2=4,
        PREFIX=1,
    )

    assert model_stats[3, 0].cpu().tolist() == [1, 1]
    assert model_stats[3, 1].cpu().tolist() == [0, 1]
    assert model_stats[3, 2:].sum().item() == 0
    assert suffix_stats.sum().item() == 0
    assert last_valid.tolist() == [0, 0]


def run_suffix_query(
    history,
    num_tokens,
    prefix=(),
    docs=(),
    min_match=4,
    max_len=64,
    key_mode="dual",
):
    spec, req_states = _make(
        num_tokens, 8, 1, max_len, corpus_tokens=1024 if docs else 0
    )
    spec.vllm_config.speculative_config.suffix_hybrid_min_match_length = min_match
    spec.vllm_config.speculative_config.suffix_hybrid_corpus_key_mode = key_mode
    spec.corpus_key_mode = {"dual": 0, "four": 1, "four_fallback": 2}[key_mode]
    for index, doc in enumerate(docs):
        _set(req_states, 0, [], doc)
        req_states.req_id_to_index = {f"doc-{index}": 0}
        spec.on_requests_finished({f"doc-{index}"})

    _set(req_states, 0, [], history)
    before = req_states.all_token_ids.gpu.clone()
    query_prefix = torch.zeros(
        (1, max(1, len(prefix))), dtype=torch.int64, device=DEVICE
    )
    if prefix:
        query_prefix[0, : len(prefix)] = torch.tensor(
            prefix, dtype=torch.int64, device=DEVICE
        )
    drafts, num_valid = _propose(
        spec,
        req_states,
        [0],
        hybrid=(query_prefix, len(prefix), num_tokens),
    )
    assert torch.equal(req_states.all_token_ids.gpu, before)
    return (
        drafts[0, :num_tokens].tolist(),
        int(num_valid[0]),
        int(spec.match_lens[0].item()),
    )


def test_splice_lookup_changes_continuation_without_mutating_history():
    history = [1, 2, 3, 7, 8, 1, 2, 4, 9, 1, 2]
    assert run_suffix_query(history, 3, min_match=2) == ([4, 9, 1], 3, 2)
    assert run_suffix_query(history, 2, prefix=[3], min_match=2) == ([7, 8], 2, 3)


def test_splice_lookup_requires_full_tail():
    assert run_suffix_query([0, 1, 0, 1, 0], 2, prefix=[1], min_match=2) == (
        [0, 1],
        2,
        2,
    )


def test_splice_corpus_filters_partial_and_short_matches():
    partial_docs = [
        [0, 1, 0, 1, 9],
        [8, 0, 1, 20, 21],
        [8, 0, 1, 30, 31],
        [8, 0, 1, 40, 41],
    ]
    assert run_suffix_query(
        [5, 0, 1, 0], 2, prefix=[1], docs=partial_docs, min_match=2
    ) == ([40, 41], 2, 2)
    short_match_docs = [
        [0, 1, 0, 1, 20, 21],
        [0, 1, 0, 1, 30, 31],
        [0, 1, 0, 1, 40, 41],
        *([[8, 0, 1, 50, 51]] * 4),
    ]
    assert run_suffix_query(
        [5, 0, 1, 0], 2, prefix=[1], docs=short_match_docs, min_match=4
    ) == ([40, 41], 2, 4)


def test_splice_prefix_does_not_read_past_history_allocation():
    assert run_suffix_query([1, 2, 3, 4, 5, 6, 7, 8], 2, prefix=[9, 10], max_len=8) == (
        [8, 8],
        0,
        0,
    )


def test_splice_corpus_key_mode_ablation():
    dual = run_suffix_query(
        [9, 1, 2], 2, docs=[[1, 2, 30, 31]], min_match=2, key_mode="dual"
    )
    four = run_suffix_query(
        [9, 1, 2], 2, docs=[[1, 2, 30, 31]], min_match=2, key_mode="four"
    )
    fallback = run_suffix_query(
        [9, 1, 2],
        2,
        docs=[[1, 2, 30, 31]],
        min_match=2,
        key_mode="four_fallback",
    )
    assert dual == fallback == ([30, 31], 2, 2)
    assert four[1:] == (0, 0)

    for mode in ("dual", "four", "four_fallback"):
        assert run_suffix_query(
            [9, 1, 2, 3, 4],
            2,
            docs=[[1, 2, 3, 4, 30, 31]],
            min_match=2,
            key_mode=mode,
        ) == ([30, 31], 2, 4)
