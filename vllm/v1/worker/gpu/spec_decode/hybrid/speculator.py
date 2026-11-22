# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.suffix.speculator import (
    _NUM_LEN_BUCKETS,
    _STATS_HALF_LIFE,
    SuffixSpeculator,
    _len_bucket,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

_MIN_OBSERVATIONS = tl.constexpr(8)


@triton.jit
def _hybrid_route_kernel(
    num_valid_ptr,
    match_lens_ptr,
    sources_ptr,
    model_drafts_ptr,
    model_drafts_stride,
    suffix_drafts_ptr,
    suffix_drafts_stride,
    model_stats_ptr,
    suffix_stats_ptr,
    use_suffix_ptr,
    scores_ptr,
    MIN_MATCH: tl.constexpr,
    K: tl.constexpr,
    STATS_K: tl.constexpr,
    K_PO2: tl.constexpr,
    PREFIX: tl.constexpr,
    QUALITY_GATE: tl.constexpr,
    MIN_EXPECTED: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.arange(0, K_PO2)
    mask = k < K
    source = tl.load(sources_ptr + b)
    match_len = tl.load(match_lens_ptr + b)
    bucket = source * _NUM_LEN_BUCKETS + _len_bucket(tl.maximum(match_len, 1))

    suffix_stat = suffix_stats_ptr + (bucket * STATS_K + k) * 2
    suffix_accepted = tl.load(suffix_stat, mask=mask, other=0).to(tl.float32)
    suffix_observed = tl.load(suffix_stat + 1, mask=mask, other=0).to(tl.float32)
    suffix_prob = (suffix_accepted + 1.0) / (suffix_observed + 2.0)

    model_stat = model_stats_ptr + k * 2
    model_accepted = tl.load(model_stat, mask=mask, other=0).to(tl.float32)
    model_observed = tl.load(model_stat + 1, mask=mask, other=0).to(tl.float32)
    model_prob = (model_accepted + 1.0) / (model_observed + 2.0)
    model_survival = tl.cumprod(model_prob, axis=0)

    prefix_survival = tl.sum(tl.where(k == PREFIX - 1, model_survival, 0.0), axis=0)
    suffix_survival = prefix_survival * tl.cumprod(
        tl.where(k < PREFIX, 1.0, suffix_prob), axis=0
    )
    splice_score = tl.sum(
        tl.where(mask & (k < PREFIX), model_survival, 0.0), axis=0
    ) + tl.sum(tl.where(mask & (k >= PREFIX), suffix_survival, 0.0), axis=0)
    model_score = tl.sum(tl.where(mask, model_survival, 0.0), axis=0)
    history_mature = (
        tl.sum(tl.where(k == PREFIX, suffix_observed, 0.0), axis=0) >= _MIN_OBSERVATIONS
    )
    model_mature = (
        tl.min(tl.where(mask & (k >= PREFIX), model_observed, float("inf")), axis=0)
        >= _MIN_OBSERVATIONS
    )
    mature = history_mature & model_mature
    better = splice_score > model_score
    if QUALITY_GATE:
        better = better & (splice_score >= MIN_EXPECTED)
    select = (~history_mature) | (mature & better)
    required = K - PREFIX
    prefix_valid = (
        tl.sum(
            (
                tl.load(
                    model_drafts_ptr + b * model_drafts_stride + k,
                    mask=k < PREFIX,
                    other=0,
                )
                < 0
            ).to(tl.int32),
            axis=0,
        )
        == 0
    )
    tail_mask = mask & (k >= PREFIX)
    model_tail = tl.load(
        model_drafts_ptr + b * model_drafts_stride + k,
        mask=tail_mask,
        other=-1,
    )
    suffix_tail = tl.load(
        suffix_drafts_ptr + b * suffix_drafts_stride + k - PREFIX,
        mask=tail_mask,
        other=-2,
    )
    different_tail = (
        tl.sum(
            tl.where(tail_mask, model_tail != suffix_tail, 0).to(tl.int32),
            axis=0,
        )
        > 0
    )
    select = select & different_tail

    valid = (
        (tl.load(num_valid_ptr + b) >= required)
        & (match_len >= MIN_MATCH)
        & prefix_valid
    )
    tl.store(use_suffix_ptr + b, valid & select)
    tl.store(scores_ptr + b * 2, splice_score)
    tl.store(scores_ptr + b * 2 + 1, model_score)


@triton.jit
def _hybrid_publish_kernel(
    model_drafts_ptr,
    model_drafts_stride,
    suffix_drafts_ptr,
    suffix_drafts_stride,
    use_suffix_ptr,
    sources_ptr,
    match_lens_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    output_ptr,
    output_stride,
    last_sources_ptr,
    last_buckets_ptr,
    last_valid_ptr,
    K: tl.constexpr,
    STATS_K: tl.constexpr,
    K_PO2: tl.constexpr,
    PREFIX: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    k = tl.arange(0, K_PO2)
    mask = k < K
    use_suffix = tl.load(use_suffix_ptr + b)
    model = tl.load(model_drafts_ptr + b * model_drafts_stride + k, mask=mask, other=-1)
    suffix_pos = k - PREFIX
    suffix = tl.load(
        suffix_drafts_ptr + b * suffix_drafts_stride + suffix_pos,
        mask=mask & (suffix_pos >= 0),
        other=-1,
    )
    take_suffix = use_suffix & (k >= PREFIX)
    out = tl.where(take_suffix, suffix, model)
    tl.store(output_ptr + b * output_stride + k, out, mask=mask)

    source = tl.load(sources_ptr + b)
    source_at_pos = tl.where(take_suffix, source + 1, 0)
    tl.store(last_sources_ptr + req * STATS_K + k, source_at_pos, mask=mask)
    match_len = tl.load(match_lens_ptr + b)
    bucket = source * _NUM_LEN_BUCKETS + _len_bucket(tl.maximum(match_len, 1))
    tl.store(last_buckets_ptr + req, tl.where(use_suffix, bucket, -1))
    tl.store(last_valid_ptr + req, tl.where(tl.load(num_sampled_ptr + b) > 0, K, 0))


@triton.jit
def _hybrid_observe_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    model_stats_ptr,
    suffix_stats_ptr,
    last_sources_ptr,
    last_buckets_ptr,
    last_valid_ptr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    valid = tl.load(last_valid_ptr + req)
    num_sampled = tl.load(num_sampled_ptr + b)
    if (valid <= 0) | (num_sampled <= 0):
        return
    accepted_count = num_sampled - 1
    verified = accepted_count + tl.load(num_rejected_ptr + b)
    bucket = tl.load(last_buckets_ptr + req)
    k = tl.arange(0, K_PO2)
    observed = (k < K) & (k < valid) & (k <= accepted_count) & (k < verified)
    accepted = k < accepted_count
    source = tl.load(last_sources_ptr + req * K + k, mask=observed, other=0)
    model_stat = model_stats_ptr + k * 2
    suffix_stat = suffix_stats_ptr + (bucket * K + k) * 2
    is_model = source == 0
    tl.atomic_add(model_stat, accepted.to(tl.int64), mask=observed & is_model)
    tl.atomic_add(model_stat + 1, 1, mask=observed & is_model)
    tl.atomic_add(suffix_stat, accepted.to(tl.int64), mask=observed & ~is_model)
    tl.atomic_add(suffix_stat + 1, 1, mask=observed & ~is_model)
    tl.store(last_valid_ptr + req, 0)


class HybridSpeculator(DraftModelSpeculator):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: "RequestState",
        drafter: DraftModelSpeculator,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.drafter = drafter
        self.suffix = SuffixSpeculator(vllm_config, device, req_states)
        spec = vllm_config.speculative_config
        assert spec is not None
        self.speculative_config = spec
        self.mode = spec.suffix_hybrid_mode
        self.prefix_len = spec.suffix_hybrid_prefix_length
        self.min_match = spec.suffix_hybrid_min_match_length
        self.min_expected = spec.suffix_hybrid_min_expected_accepted
        self.max_num_reqs = self.suffix.max_num_reqs
        self.num_speculative_steps = spec.num_speculative_tokens
        self.num_query_per_req = drafter.num_query_per_req
        self.enable_adaptive_verification = False
        self.draft_logits = drafter.draft_logits
        self.supports_mm_inputs = drafter.supports_mm_inputs
        self.pcp_manager = drafter.pcp_manager

        shape = (2 * _NUM_LEN_BUCKETS, self.num_speculative_steps, 2)
        self.suffix_stats = torch.zeros(shape, dtype=torch.int64, device=device)
        self.model_stats = torch.zeros(
            (self.num_speculative_steps, 2), dtype=torch.int64, device=device
        )
        self.last_sources = torch.full(
            (self.max_num_reqs, self.num_speculative_steps),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.last_buckets = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.last_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.use_suffix = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self.route_scores = torch.zeros(
            (self.max_num_reqs, 2), dtype=torch.float32, device=device
        )
        self.output_drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        self.num_steps = 0

    def __getattr__(self, name: str) -> Any:
        if name in {"drafter", "suffix"}:
            raise AttributeError(name)
        return getattr(self.drafter, name)

    @property
    def max_model_len(self) -> int:
        return self.vllm_config.model_config.max_model_len

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        return self.drafter.load_draft_model(target_model, target_attn_layer_names)

    def load_model(self, target_model: nn.Module) -> None:
        self.drafter.load_model(target_model)
        self.model = self.drafter.model
        self.draft_logits = self.drafter.draft_logits
        self.draft_attn_layer_names = self.drafter.draft_attn_layer_names
        self.supports_mm_inputs = self.drafter.supports_mm_inputs

    def free_unused_weights(self, target_model: nn.Module) -> None:
        self.drafter.free_unused_weights(target_model)

    def set_eplb_state(self, eplb_state) -> None:
        self.drafter.set_eplb_state(eplb_state)

    def set_attn(self, *args) -> None:
        self.drafter.pcp_manager = self.pcp_manager
        self.drafter.set_attn(*args)

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        self.drafter.init_cudagraph_manager(cudagraph_mode)
        self.suffix.init_cudagraph_manager(cudagraph_mode)

    def capture(self) -> None:
        self.drafter.capture()
        self.suffix.capture()

    def update_max_model_len(self, max_model_len: int) -> None:
        self.drafter.update_max_model_len(max_model_len)

    def shutdown(self) -> None:
        self.drafter.shutdown()
        self.suffix.shutdown()

    def add_request(self, req_index: int, new_req_data) -> None:
        self.drafter.add_request(req_index, new_req_data)
        self.suffix.add_request(req_index, new_req_data)
        self.last_valid[req_index].fill_(0)
        self.last_buckets[req_index].fill_(-1)
        self.last_sources[req_index].fill_(-1)

    def on_requests_finished(self, finished_req_ids: "Iterable[str]") -> None:
        self.drafter.on_requests_finished(finished_req_ids)
        self.suffix.on_requests_finished(finished_req_ids)

    def reset_state(self) -> None:
        self.drafter.reset_state()
        self.suffix.reset_state()
        self.model_stats.zero_()
        self.suffix_stats.zero_()
        self.last_sources.fill_(-1)
        self.last_buckets.fill_(-1)
        self.last_valid.zero_()
        self.num_steps = 0

    def prepare_propose(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        sampler: "Sampler",
    ) -> None:
        self.drafter.prepare_propose(input_batch, num_sampled, num_rejected, sampler)
        if input_batch.has_structured_output_reqs:
            self.last_valid[input_batch.idx_mapping] = 0
            return
        _hybrid_observe_kernel[(input_batch.num_reqs,)](
            input_batch.idx_mapping,
            num_sampled,
            num_rejected,
            self.model_stats,
            self.suffix_stats,
            self.last_sources,
            self.last_buckets,
            self.last_valid,
            K=self.num_speculative_steps,
            K_PO2=triton.next_power_of_2(self.num_speculative_steps),
        )
        self.num_steps += 1
        if self.num_steps % _STATS_HALF_LIFE == 0:
            self.model_stats.bitwise_right_shift_(1)
            self.suffix_stats.bitwise_right_shift_(1)

    def _route(
        self,
        k: int,
        model_drafts: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        quality_gate = self.min_expected is not None
        _hybrid_route_kernel[(model_drafts.shape[0],)](
            self.suffix.num_valid,
            self.suffix.match_lens,
            self.suffix.sources,
            model_drafts,
            model_drafts.stride(0),
            self.suffix.drafts,
            self.suffix.drafts.stride(0),
            self.model_stats,
            self.suffix_stats,
            self.use_suffix,
            self.route_scores,
            MIN_MATCH=self.min_match,
            K=k,
            STATS_K=self.num_speculative_steps,
            K_PO2=triton.next_power_of_2(k),
            PREFIX=self.prefix_len,
            QUALITY_GATE=quality_gate,
            MIN_EXPECTED=self.min_expected or 0.0,
        )

    def _publish(
        self,
        input_batch: InputBatch,
        model_drafts: torch.Tensor,
        suffix_drafts: torch.Tensor,
        num_sampled: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        self.output_drafts[: input_batch.num_reqs, k:].fill_(-1)
        _hybrid_publish_kernel[(input_batch.num_reqs,)](
            model_drafts,
            model_drafts.stride(0),
            suffix_drafts,
            suffix_drafts.stride(0),
            self.use_suffix,
            self.suffix.sources,
            self.suffix.match_lens,
            input_batch.idx_mapping,
            num_sampled,
            self.output_drafts,
            self.output_drafts.stride(0),
            self.last_sources,
            self.last_buckets,
            self.last_valid,
            K=k,
            STATS_K=self.num_speculative_steps,
            K_PO2=triton.next_power_of_2(k),
            PREFIX=self.prefix_len,
        )
        return self.output_drafts[: input_batch.num_reqs]

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: "DPSyncState | None" = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        kwargs = dict(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )
        k = (
            self.num_speculative_steps
            if num_speculative_tokens is None
            else num_speculative_tokens
        )
        if dummy_run or k == 0 or self.mode == "off":
            model_drafts = self.drafter.propose(**kwargs, num_speculative_tokens=k)
            if not dummy_run and k == 0:
                self.last_valid[input_batch.idx_mapping] = 0
            return model_drafts

        model_drafts = self.drafter.propose(**kwargs, num_speculative_tokens=k)
        if k <= self.prefix_len:
            self.use_suffix[: input_batch.num_reqs].zero_()
            return self._publish(
                input_batch, model_drafts, model_drafts, num_sampled, k
            )
        suffix_drafts, _, _, _ = self.suffix.propose_hybrid(
            input_batch,
            num_sampled,
            last_sampled,
            k - self.prefix_len,
            query_prefix=model_drafts,
            prefix_len=self.prefix_len,
        )
        self._route(k, model_drafts, num_sampled)
        return self._publish(input_batch, model_drafts, suffix_drafts, num_sampled, k)
