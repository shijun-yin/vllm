# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.hybrid.speculator import (
    _hybrid_observe_kernel,
    _hybrid_publish_kernel,
    _hybrid_route_kernel,
)
from vllm.v1.worker.gpu.spec_decode.suffix.speculator import (
    _BUCKET_SLOTS,
    _HASH_MUL,
    _INGEST_BLOCK,
    _suffix_finalize_kernel,
    _suffix_ingest_kernel,
    _suffix_local_scan_kernel,
)

if not current_platform.is_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

DEVICE = torch.device("cuda")


def test_splice_route_maturity_prefix_and_identical_tail():
    k = 3
    prefix = 1
    model = torch.tensor([[100, 101, 102]], dtype=torch.int64, device=DEVICE)
    suffix = torch.tensor([[1000, 1001]], dtype=torch.int64, device=DEVICE)
    num_valid = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    match_lens = torch.tensor([4], dtype=torch.int32, device=DEVICE)
    sources = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    model_stats = torch.zeros((k, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, k, 2), dtype=torch.int64, device=DEVICE)
    use_suffix = torch.zeros(1, dtype=torch.bool, device=DEVICE)
    scores = torch.zeros((1, 2), dtype=torch.float32, device=DEVICE)

    def route():
        _hybrid_route_kernel[(1,)](
            num_valid,
            match_lens,
            sources,
            model,
            model.stride(0),
            suffix,
            suffix.stride(0),
            model_stats,
            suffix_stats,
            use_suffix,
            scores,
            MIN_MATCH=4,
            K=k,
            STATS_K=k,
            K_PO2=4,
            PREFIX=prefix,
            QUALITY_GATE=False,
            MIN_EXPECTED=0.0,
        )
        return use_suffix.item()

    assert route()
    suffix_stats[3, :, 0] = 8
    suffix_stats[3, :, 1] = 8
    assert not route()
    model_stats[:, 1] = 8
    assert route()
    model_stats[:, 0] = 8
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
    use_suffix = torch.ones(1, dtype=torch.bool, device=DEVICE)
    sources = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    match_lens = torch.tensor([4], dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.ones(1, dtype=torch.int32, device=DEVICE)
    output = torch.zeros((1, k), dtype=torch.int64, device=DEVICE)
    last_sources = torch.full((1, stats_k), -1, dtype=torch.int32, device=DEVICE)
    last_buckets = torch.full((1,), -1, dtype=torch.int32, device=DEVICE)
    last_valid = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _hybrid_publish_kernel[(1,)](
        model,
        model.stride(0),
        suffix,
        suffix.stride(0),
        use_suffix,
        sources,
        match_lens,
        idx_mapping,
        num_sampled,
        output,
        output.stride(0),
        last_sources,
        last_buckets,
        last_valid,
        K=k,
        STATS_K=stats_k,
        K_PO2=1 << (k - 1).bit_length(),
        PREFIX=1,
    )
    assert output[0, 0].item() == 100
    assert output[0, 1:].cpu().tolist() == suffix[0].cpu().tolist()
    assert last_sources[0, 0].item() == 0
    assert (last_sources[0, 1:k] == 1).all()
    assert last_valid.item() == k

    model_stats = torch.zeros((stats_k, 2), dtype=torch.int64, device=DEVICE)
    suffix_stats = torch.zeros((18, stats_k, 2), dtype=torch.int64, device=DEVICE)
    _hybrid_observe_kernel[(1,)](
        idx_mapping,
        torch.tensor([2], dtype=torch.int32, device=DEVICE),
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        model_stats,
        suffix_stats,
        last_sources,
        last_buckets,
        last_valid,
        K=stats_k,
        K_PO2=1 << (stats_k - 1).bit_length(),
    )
    assert model_stats[0].cpu().tolist() == [1, 1]
    assert model_stats[1:].sum().item() == 0
    assert suffix_stats[3, 1].cpu().tolist() == [0, 1]
    expected_suffix_observations = 1
    assert suffix_stats.sum().item() == expected_suffix_observations
    assert last_valid.item() == 0


def run_suffix_query(history, num_tokens, prefix=(), docs=(), min_match=4, max_len=64):
    max_reqs = 1
    block_l = (
        256
        if max_len >= 1024
        else 128
        if max_len >= 256
        else max(16, 1 << (max_len - 1).bit_length())
    )
    n_blocks = (max_len + block_l - 1) // block_l
    token_ids = torch.zeros((max_reqs, max_len), dtype=torch.int32, device=DEVICE)
    total_len = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.zeros(max_reqs, dtype=torch.int64, device=DEVICE)
    num_sampled = torch.ones(max_reqs, dtype=torch.int32, device=DEVICE)
    corpus_len = 1024 if docs else 1
    corpus = torch.full((corpus_len,), -1, dtype=torch.int32, device=DEVICE)
    num_buckets = 1 << (max(16, corpus_len // 4) - 1).bit_length()
    table = torch.full(
        (2, num_buckets, _BUCKET_SLOTS), -1, dtype=torch.int64, device=DEVICE
    )
    corpus_head = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    doc_seq = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    ingest_out = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    for doc in docs:
        token_ids.zero_()
        token_ids[0, : len(doc)] = torch.tensor(doc, dtype=torch.int32, device=DEVICE)
        total_len[0] = len(doc)
        docs_gpu = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
        _suffix_ingest_kernel[(1, (max_len + _INGEST_BLOCK - 1) // _INGEST_BLOCK)](
            token_ids,
            token_ids.stride(0),
            docs_gpu,
            1,
            total_len,
            corpus,
            corpus_len,
            corpus_head,
            doc_seq,
            ingest_out[0:1],
            ingest_out[1:2],
            table,
            HASH_MUL=_HASH_MUL,
            NUM_BUCKETS=num_buckets,
            SLOTS=_BUCKET_SLOTS,
            BLOCK=_INGEST_BLOCK,
            num_warps=4,
        )
        corpus_head.copy_(ingest_out[0:1])
        doc_seq.copy_(ingest_out[1:2])

    token_ids.zero_()
    token_ids[0, : len(history)] = torch.tensor(
        history, dtype=torch.int32, device=DEVICE
    )
    total_len[0] = len(history)
    before = token_ids.clone()
    last_sampled = torch.full(
        (max_reqs,), history[-1], dtype=torch.int64, device=DEVICE
    )
    query_prefix = torch.zeros(
        (max_reqs, max(1, len(prefix))), dtype=torch.int64, device=DEVICE
    )
    if prefix:
        query_prefix[0, : len(prefix)] = torch.tensor(
            prefix, dtype=torch.int64, device=DEVICE
        )
    scratch = torch.zeros((max_reqs, n_blocks), dtype=torch.int64, device=DEVICE)
    drafts = torch.zeros((max_reqs, num_tokens), dtype=torch.int64, device=DEVICE)
    num_valid = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    match_lens = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    sources = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    accept_stats = torch.zeros((18, num_tokens, 2), dtype=torch.int64, device=DEVICE)
    last_bucket = torch.full((max_reqs,), -1, dtype=torch.int32, device=DEVICE)
    last_valid = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)
    confidence = torch.zeros((max_reqs, num_tokens), dtype=torch.float32, device=DEVICE)
    _suffix_local_scan_kernel[(max_reqs, n_blocks)](
        token_ids,
        token_ids.stride(0),
        idx_mapping,
        total_len,
        num_sampled,
        scratch,
        scratch.stride(0),
        query_prefix,
        query_prefix.stride(0),
        max_len + 1,
        QUERY_PREFIX=len(prefix),
        REQUIRE_FULL=True,
        K=num_tokens,
        MAX_DEPTH=8,
        BLOCK=block_l,
        num_warps=4,
    )
    _suffix_finalize_kernel[(max_reqs,)](
        token_ids,
        token_ids.stride(0),
        idx_mapping,
        total_len,
        num_sampled,
        last_sampled,
        scratch,
        scratch.stride(0),
        n_blocks,
        max_len + 1,
        corpus,
        corpus_len,
        table,
        drafts,
        drafts.stride(0),
        num_valid,
        match_lens,
        sources,
        accept_stats,
        last_bucket,
        last_valid,
        confidence,
        query_prefix,
        query_prefix.stride(0),
        QUERY_PREFIX=len(prefix),
        HAS_CORPUS=bool(docs),
        ADAPTIVE=False,
        REQUIRE_FULL=True,
        MIN_MATCH=min_match,
        HASH_MUL=_HASH_MUL,
        NUM_BUCKETS=num_buckets,
        SLOTS=_BUCKET_SLOTS,
        MAX_DEPTH=8,
        DEPTH_PO2=8,
        K=num_tokens,
        K_PO2=1 << (num_tokens - 1).bit_length(),
        NB_PO2=1 << (n_blocks - 1).bit_length(),
        num_warps=4,
    )
    assert torch.equal(token_ids, before)
    return (
        drafts[0].cpu().tolist(),
        num_valid.item(),
        match_lens.item(),
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
    run_suffix_query([1, 2, 3, 4, 5, 6, 7, 8], 2, prefix=[9, 10], max_len=8)
