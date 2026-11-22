# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.hybrid.speculator import HybridSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

if not current_platform.is_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

DEVICE = torch.device("cuda")


class _DummyDrafter(DraftModelSpeculator):
    def __init__(self):
        self.num_query_per_req = 1
        self.draft_logits = None
        self.supports_mm_inputs = False
        self.pcp_manager = None

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        return

    def capture(self) -> None:
        return

    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError

    def propose(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_propose(self, *args) -> None:
        return


def _request_state(max_reqs: int, max_len: int):
    def buf(*shape):
        return SimpleNamespace(
            gpu=torch.zeros(*shape, dtype=torch.int32, device=DEVICE)
        )

    return SimpleNamespace(
        all_token_ids=buf(max_reqs, max_len),
        total_len=buf(max_reqs),
        prompt_len=SimpleNamespace(np=np.zeros(max_reqs, dtype=np.int32)),
        req_id_to_index={},
    )


def _make_hybrid(
    mode="direct", max_k=4, min_expected=None, corpus_tokens=0, min_match=4
):
    spec = SimpleNamespace(
        num_speculative_tokens=max_k,
        suffix_decoding_max_tree_depth=8,
        suffix_decoding_corpus_tokens=corpus_tokens,
        suffix_decoding_max_cached_requests=100,
        enable_adaptive_verification=False,
        suffix_hybrid_mode=mode,
        suffix_hybrid_prefix_length=1,
        suffix_hybrid_min_match_length=min_match,
        suffix_hybrid_min_expected_accepted=min_expected,
        suffix_hybrid_compact_direct=True,
        suffix_hybrid_diagnostics=False,
    )
    cfg = SimpleNamespace(
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=64),
    )
    req_states = _request_state(4, 64)
    return HybridSpeculator(cfg, DEVICE, req_states, _DummyDrafter()), req_states


def _batch(n=1):
    return SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.arange(n, dtype=torch.int64, device=DEVICE),
        has_structured_output_reqs=False,
    )


def _stats(hybrid, bucket, accepted, observed):
    hybrid.suffix_stats[bucket, :, 0] = accepted
    hybrid.suffix_stats[bucket, :, 1] = observed


def test_direct_mixed_hit_order_and_publication():
    hybrid, _ = _make_hybrid()
    batch = _batch(3)
    model = torch.arange(12, dtype=torch.int64, device=DEVICE).reshape(3, 4) + 100
    suffix = torch.arange(12, dtype=torch.int64, device=DEVICE).reshape(3, 4) + 1000
    hybrid.suffix.num_valid[:3] = torch.tensor([3, 0, 3], device=DEVICE)
    hybrid.suffix.match_lens[:3] = 4
    hybrid.suffix.sources[:3] = torch.tensor([0, 0, 1], device=DEVICE)
    hybrid._route(3, model, torch.ones(3, dtype=torch.int32, device=DEVICE))
    assert hybrid.use_suffix[:3].cpu().tolist() == [True, False, True]
    assert hybrid.route_scores[:3, 0].cpu().tolist() == pytest.approx([0.875] * 3)
    output = hybrid._publish(
        batch, model, suffix, torch.ones(3, dtype=torch.int32, device=DEVICE), 3
    )
    assert output[:, :3].cpu().tolist() == [
        [1000, 1001, 1002],
        [104, 105, 106],
        [1008, 1009, 1010],
    ]
    assert (output[:, 3:] == -1).all()
    assert hybrid.last_sources[:3, :3].cpu().tolist() == [
        [1, 1, 1],
        [0, 0, 0],
        [2, 2, 2],
    ]
    assert hybrid.last_valid[:3].cpu().tolist() == [3, 3, 3]


def test_route_diagnostics_are_opt_in():
    hybrid, _ = _make_hybrid()
    batch = _batch(1)
    model = torch.zeros((1, 4), dtype=torch.int64, device=DEVICE)
    suffix = torch.ones((1, 4), dtype=torch.int64, device=DEVICE)
    hybrid.suffix.num_valid[0] = 3
    hybrid.suffix.match_lens[0] = 4
    sampled = torch.ones(1, dtype=torch.int32, device=DEVICE)
    hybrid._route(3, model, sampled)

    hybrid._publish(batch, model, suffix, sampled, 3)
    assert hybrid.route_count == 1
    assert not hybrid.route_history.any()

    hybrid.diagnostics = True
    hybrid._publish(batch, model, suffix, sampled, 3)
    assert hybrid.route_count == 2
    assert hybrid.route_history[1, 0].item()


def test_direct_quality_gate_uses_mature_bucket_and_cold_start():
    hybrid, _ = _make_hybrid(min_expected=1.0)
    model = torch.zeros((3, 4), dtype=torch.int64, device=DEVICE)
    hybrid.suffix.num_valid[:3] = 3
    hybrid.suffix.match_lens[:3] = torch.tensor([5, 4, 4], device=DEVICE)
    hybrid.suffix.sources[:3] = torch.tensor([0, 0, 1], device=DEVICE)
    _stats(hybrid, 3, 0, 8)
    _stats(hybrid, 12, 8, 8)
    hybrid._route(3, model, torch.ones(3, dtype=torch.int32, device=DEVICE))
    assert hybrid.use_suffix[:3].cpu().tolist() == [True, False, True]


def test_splice_cold_start_fallback_and_prefix_rejection():
    hybrid, _ = _make_hybrid(mode="splice")
    batch = _batch(1)
    model = torch.tensor([[100, 101, 102, 103]], dtype=torch.int64, device=DEVICE)
    suffix = torch.tensor([[1000, 1001, 1002, 1003]], dtype=torch.int64, device=DEVICE)
    hybrid.suffix.num_valid[0] = 2
    hybrid.suffix.match_lens[0] = 4
    hybrid.suffix.sources[0] = 0
    sampled = torch.ones(1, dtype=torch.int32, device=DEVICE)

    hybrid._route(3, model, sampled)
    assert hybrid.use_suffix[0].item()
    output = hybrid._publish(batch, model, suffix, sampled, 3)
    assert output[:, :3].cpu().tolist() == [[100, 1000, 1001]]
    assert output[0, 3].item() == -1
    assert hybrid.last_sources[0, :3].cpu().tolist() == [0, 1, 1]

    _stats(hybrid, 3, 8, 8)
    hybrid._route(3, model, sampled)
    assert not hybrid.use_suffix[0].item()

    hybrid.model_stats[:, 0] = 0
    hybrid.model_stats[:, 1] = 8
    hybrid._route(3, model, sampled)
    assert hybrid.use_suffix[0].item()

    hybrid.model_stats[:, 0] = 8
    hybrid._route(3, model, sampled)
    assert not hybrid.use_suffix[0].item()

    hybrid.suffix_stats[3, :, 1] = 0
    model[0, 0] = -1
    hybrid._route(3, model, sampled)
    assert not hybrid.use_suffix[0].item()


def test_splice_identical_tail_keeps_model_attribution():
    hybrid, _ = _make_hybrid(mode="splice")
    model = torch.tensor([[100, 101, 102, 103]], dtype=torch.int64, device=DEVICE)
    hybrid.suffix.drafts[0, :2] = torch.tensor([101, 102], device=DEVICE)
    hybrid.suffix.num_valid[0] = 2
    hybrid.suffix.match_lens[0] = 4
    sampled = torch.ones(1, dtype=torch.int32, device=DEVICE)

    hybrid._route(3, model, sampled)

    assert not hybrid.use_suffix[0].item()
    output = hybrid._publish(_batch(1), model, hybrid.suffix.drafts, sampled, 3)
    assert output[0, :3].cpu().tolist() == [100, 101, 102]
    assert hybrid.last_sources[0, :3].cpu().tolist() == [0, 0, 0]


def test_observation_attributes_only_reached_positions():
    hybrid, _ = _make_hybrid()
    batch = _batch(3)
    model = torch.zeros((3, 4), dtype=torch.int64, device=DEVICE)
    suffix = torch.ones((3, 4), dtype=torch.int64, device=DEVICE)
    hybrid.suffix.num_valid[:3] = torch.tensor([3, 0, 3], device=DEVICE)
    hybrid.suffix.match_lens[:3] = 4
    hybrid.suffix.sources[:3] = torch.tensor([0, 0, 1], device=DEVICE)
    hybrid._route(3, model, torch.ones(3, dtype=torch.int32, device=DEVICE))
    hybrid._publish(
        batch, model, suffix, torch.ones(3, dtype=torch.int32, device=DEVICE), 3
    )
    hybrid.prepare_propose(
        batch,
        torch.tensor([2, 1, 4], dtype=torch.int32, device=DEVICE),
        torch.tensor([1, 1, 0], dtype=torch.int32, device=DEVICE),
        None,
    )
    assert hybrid.model_stats[:, :2].cpu().tolist() == [[0, 1], [0, 0], [0, 0], [0, 0]]
    assert hybrid.suffix_stats[3, :3].cpu().tolist() == [
        [1, 1],
        [0, 1],
        [0, 0],
    ]
    assert hybrid.suffix_stats[12, :3].cpu().tolist() == [
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    assert hybrid.last_valid[:3].sum().item() == 0


def test_structured_output_discards_pending_observation():
    hybrid, _ = _make_hybrid()
    batch = _batch(1)
    hybrid.last_valid[0] = 3
    batch.has_structured_output_reqs = True
    hybrid.prepare_propose(
        batch,
        torch.tensor([2], dtype=torch.int32, device=DEVICE),
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        None,
    )
    assert hybrid.last_valid[0].item() == 0
    assert hybrid.model_stats.sum().item() == 0
    assert hybrid.suffix_stats.sum().item() == 0


def test_splice_lookup_changes_continuation_without_mutating_history():
    hybrid, req_states = _make_hybrid(mode="splice")
    history = torch.tensor(
        [1, 2, 3, 7, 8, 1, 2, 4, 9, 1, 2], dtype=torch.int32, device=DEVICE
    )
    req_states.all_token_ids.gpu[0, : len(history)] = history
    req_states.total_len.gpu[0] = len(history)
    before = req_states.all_token_ids.gpu.clone()
    batch = _batch(1)
    sampled = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.tensor([[2]], dtype=torch.int64, device=DEVICE)

    direct, direct_valid, _, _ = hybrid.suffix.propose_hybrid(
        batch, sampled, last_sampled, 3
    )
    direct = direct.clone()
    direct_valid = direct_valid.clone()
    prefix = torch.tensor([[3, 0, 0, 0]], dtype=torch.int64, device=DEVICE)
    spliced, spliced_valid, _, _ = hybrid.suffix.propose_hybrid(
        batch, sampled, last_sampled, 2, query_prefix=prefix, prefix_len=1
    )
    assert direct_valid.item() == 3
    assert direct[0].cpu().tolist() == [4, 9, 1]
    assert spliced_valid.item() == 2
    assert spliced[0].cpu().tolist() == [7, 8]
    assert torch.equal(req_states.all_token_ids.gpu, before)


def test_splice_lookup_ignores_longer_match_without_full_tail():
    hybrid, req_states = _make_hybrid(mode="splice")
    history = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device=DEVICE)
    req_states.all_token_ids.gpu[0, : len(history)] = history
    req_states.total_len.gpu[0] = len(history)
    prefix = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64, device=DEVICE)

    draft, num_valid, match_len, _ = hybrid.suffix.propose_hybrid(
        _batch(1),
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int64, device=DEVICE),
        2,
        query_prefix=prefix,
        prefix_len=1,
    )

    assert num_valid.item() == 2
    assert match_len.item() == 2
    assert draft[0].cpu().tolist() == [0, 1]


def test_splice_corpus_ignores_partial_high_vote_chain():
    hybrid, req_states = _make_hybrid(mode="splice", corpus_tokens=1024, min_match=2)
    docs = [
        [0, 1, 0, 1, 9],
        [8, 0, 1, 20, 21],
        [8, 0, 1, 30, 31],
        [8, 0, 1, 40, 41],
    ]
    for i, doc in enumerate(docs):
        req_states.all_token_ids.gpu[0].zero_()
        req_states.all_token_ids.gpu[0, : len(doc)] = torch.tensor(
            doc, dtype=torch.int32, device=DEVICE
        )
        req_states.total_len.gpu[0] = len(doc)
        req_states.req_id_to_index[f"doc-{i}"] = 0
        hybrid.suffix.on_requests_finished([f"doc-{i}"])

    history = [5, 0, 1, 0]
    req_states.all_token_ids.gpu[0].zero_()
    req_states.all_token_ids.gpu[0, : len(history)] = torch.tensor(
        history, dtype=torch.int32, device=DEVICE
    )
    req_states.total_len.gpu[0] = len(history)
    prefix = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64, device=DEVICE)

    draft, num_valid, match_len, _ = hybrid.suffix.propose_hybrid(
        _batch(1),
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int64, device=DEVICE),
        2,
        query_prefix=prefix,
        prefix_len=1,
    )

    assert num_valid.item() == 2
    assert match_len.item() == 2
    assert draft[0].cpu().tolist() == [40, 41]


def test_hybrid_lookup_enforces_min_match_before_voting():
    hybrid, req_states = _make_hybrid(mode="splice", corpus_tokens=1024)
    docs = [
        [0, 1, 0, 1, 20, 21],
        [0, 1, 0, 1, 30, 31],
        [0, 1, 0, 1, 40, 41],
        [8, 0, 1, 50, 51],
        [8, 0, 1, 50, 51],
        [8, 0, 1, 50, 51],
        [8, 0, 1, 50, 51],
    ]
    for i, doc in enumerate(docs):
        req_states.all_token_ids.gpu[0].zero_()
        req_states.all_token_ids.gpu[0, : len(doc)] = torch.tensor(
            doc, dtype=torch.int32, device=DEVICE
        )
        req_states.total_len.gpu[0] = len(doc)
        req_states.req_id_to_index[f"doc-{i}"] = 0
        hybrid.suffix.on_requests_finished([f"doc-{i}"])

    history = [5, 0, 1, 0]
    req_states.all_token_ids.gpu[0].zero_()
    req_states.all_token_ids.gpu[0, : len(history)] = torch.tensor(
        history, dtype=torch.int32, device=DEVICE
    )
    req_states.total_len.gpu[0] = len(history)
    prefix = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64, device=DEVICE)

    draft, num_valid, match_len, _ = hybrid.suffix.propose_hybrid(
        _batch(1),
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int64, device=DEVICE),
        2,
        query_prefix=prefix,
        prefix_len=1,
    )

    assert num_valid.item() == 2
    assert match_len.item() == 4
    assert draft[0].cpu().tolist() == [40, 41]


def test_mtp_direct_compaction_reorders_decode_state():
    draft = MTPSpeculator.__new__(MTPSpeculator)
    draft.device = DEVICE
    draft.draft_tokens = torch.arange(12, device=DEVICE).reshape(4, 3)
    draft.hidden_states = torch.arange(8, device=DEVICE).reshape(4, 2)
    draft.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4, dtype=torch.int32, device=DEVICE),
        positions=torch.arange(4, dtype=torch.int64, device=DEVICE) + 10,
        seq_lens=torch.tensor([10, 20, 30, 0], dtype=torch.int32, device=DEVICE),
        query_start_loc=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device=DEVICE),
    )
    draft.sample_src_positions = torch.arange(4, device=DEVICE) + 20
    draft.last_token_indices = torch.arange(4, device=DEVICE) + 30
    draft.idx_mapping = torch.arange(4, device=DEVICE)
    draft.temperature = torch.arange(4, device=DEVICE, dtype=torch.float32)
    draft.seeds = torch.arange(4, device=DEVICE)
    input_batch = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([100, 200, 300, 400])
    )
    rejected = torch.tensor([1, 2, 3, 0], dtype=torch.int32, device=DEVICE)

    compact, compact_rejected, compact_seq_lens = draft._compact_direct_batch(
        input_batch, rejected, np.array([2, 0])
    )

    assert draft.draft_tokens[:2].cpu().tolist() == [[6, 7, 8], [0, 1, 2]]
    assert draft.hidden_states[:2].cpu().tolist() == [[4, 5], [0, 1]]
    assert draft.input_buffers.input_ids[:2].cpu().tolist() == [2, 0]
    assert draft.input_buffers.positions[:2].cpu().tolist() == [12, 10]
    assert draft.input_buffers.seq_lens.cpu().tolist() == [30, 10, 0, 0]
    assert draft.input_buffers.query_start_loc.cpu().tolist() == [0, 1, 2, 2, 2]
    assert draft.sample_src_positions[:2].cpu().tolist() == [22, 20]
    assert draft.last_token_indices[:2].cpu().tolist() == [32, 30]
    assert draft.idx_mapping.cpu().tolist() == [2, 0, -1, -1]
    assert draft.temperature[:2].cpu().tolist() == [2, 0]
    assert draft.seeds[:2].cpu().tolist() == [2, 0]
    assert compact_rejected.cpu().tolist() == [3, 1]
    assert compact_seq_lens.cpu().tolist() == [300, 100]
    assert compact.num_reqs == 2
