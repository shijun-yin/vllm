# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.suffix.speculator import (
    _NUM_LEN_BUCKETS,
    SuffixSpeculator,
    _len_bucket,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.v1.worker.gpu.states import RequestState

_MIN_OBSERVATIONS = tl.constexpr(8)


@triton.jit
def _hybrid_route_publish_kernel(
    num_valid_ptr,
    match_lens_ptr,
    sources_ptr,
    model_drafts_ptr,
    model_drafts_stride,
    suffix_drafts_ptr,
    suffix_drafts_stride,
    model_stats_ptr,
    suffix_stats_ptr,
    structured_output_mask_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    route_index,
    output_ptr,
    output_stride,
    last_use_suffix_ptr,
    last_buckets_ptr,
    last_valid_ptr,
    MIN_MATCH: tl.constexpr,
    K: tl.constexpr,
    STATS_K: tl.constexpr,
    K_PO2: tl.constexpr,
    PREFIX: tl.constexpr,
    QUALITY_GATE: tl.constexpr,
    MIN_EXPECTED: tl.constexpr,
    EXPLORATION_INTERVAL: tl.constexpr,
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

    model_stat = model_stats_ptr + (bucket * STATS_K + k) * 2
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
        tl.sum(tl.where(k == PREFIX, model_observed, 0.0), axis=0) >= _MIN_OBSERVATIONS
    )
    mature = history_mature & model_mature
    better = splice_score > model_score
    if QUALITY_GATE:
        better = better & (splice_score >= MIN_EXPECTED)
    select = (~history_mature) | (mature & better)
    if EXPLORATION_INTERVAL > 0:
        explore = mature & ((route_index + b) % EXPLORATION_INTERVAL == 0)
        select = tl.where(explore, ~select, select)
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
    use_suffix = valid & select & ~tl.load(structured_output_mask_ptr + b)
    req = tl.load(idx_mapping_ptr + b)
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
    tl.store(last_use_suffix_ptr + req, use_suffix)
    has_candidate = (tl.load(num_valid_ptr + b) >= required) & (match_len >= MIN_MATCH)
    tl.store(last_buckets_ptr + req, tl.where(has_candidate, bucket, -1))
    tl.store(last_valid_ptr + req, tl.where(tl.load(num_sampled_ptr + b) > 0, K, 0))


@triton.jit
def _hybrid_observe_kernel(
    idx_mapping_ptr,
    structured_output_mask_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    model_stats_ptr,
    suffix_stats_ptr,
    last_use_suffix_ptr,
    last_buckets_ptr,
    last_valid_ptr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    PREFIX: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    if tl.load(structured_output_mask_ptr + b):
        tl.store(last_valid_ptr + req, 0)
        return
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
    use_suffix = tl.load(last_use_suffix_ptr + req)
    model_stat = model_stats_ptr + (bucket * K + k) * 2
    suffix_stat = suffix_stats_ptr + (bucket * K + k) * 2
    valid_bucket = bucket >= 0
    is_suffix = valid_bucket & use_suffix & (k >= PREFIX)
    is_model = valid_bucket & ~is_suffix
    tl.atomic_add(model_stat, accepted.to(tl.int64), mask=observed & is_model)
    tl.atomic_add(model_stat + 1, 1, mask=observed & is_model)
    tl.atomic_add(suffix_stat, accepted.to(tl.int64), mask=observed & is_suffix)
    tl.atomic_add(suffix_stat + 1, 1, mask=observed & is_suffix)
    tl.store(last_valid_ptr + req, 0)


class SuffixSplice:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: "RequestState",
    ):
        self.device = device
        self.suffix = SuffixSpeculator(vllm_config, device, req_states)
        spec = vllm_config.speculative_config
        assert spec is not None
        self.prefix_len = spec.suffix_hybrid_prefix_length
        self.min_match = spec.suffix_hybrid_min_match_length
        self.min_expected = spec.suffix_hybrid_min_expected_accepted
        self.stats_half_life = spec.suffix_hybrid_stats_half_life
        self.exploration_interval = spec.suffix_hybrid_exploration_interval
        self.max_num_reqs = self.suffix.max_num_reqs
        self.num_speculative_steps = spec.num_speculative_tokens

        shape = (2 * _NUM_LEN_BUCKETS, self.num_speculative_steps, 2)
        self.suffix_stats = torch.zeros(shape, dtype=torch.int64, device=device)
        self.model_stats = torch.zeros(
            (2 * _NUM_LEN_BUCKETS, self.num_speculative_steps, 2),
            dtype=torch.int64,
            device=device,
        )
        self.last_use_suffix = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self.last_buckets = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.last_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.structured_output_mask = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self.output_drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        self.num_steps = 0

    def add_request(self, req_index: int, new_req_data) -> None:
        self.last_valid[req_index].fill_(0)
        self.last_buckets[req_index].fill_(-1)
        self.last_use_suffix[req_index].fill_(False)

    def on_requests_finished_with_lengths(
        self,
        finished_req_ids: "Iterable[str]",
        final_token_lengths: dict[str, int],
    ) -> None:
        self.suffix.on_requests_finished_with_lengths(
            finished_req_ids, final_token_lengths
        )

    def reset_state(self) -> None:
        self.suffix.reset_state()
        self.model_stats.zero_()
        self.suffix_stats.zero_()
        self.last_use_suffix.zero_()
        self.last_buckets.fill_(-1)
        self.last_valid.zero_()
        self.num_steps = 0

    def prepare_propose(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        num_reqs = input_batch.num_reqs
        structured_mask = self.structured_output_mask[:num_reqs]
        structured_mask.zero_()
        if input_batch.has_structured_output_reqs:
            structured_ids = input_batch.structured_output_req_ids
            if structured_ids:
                structured_mask.copy_(
                    torch.tensor(
                        [req_id in structured_ids for req_id in input_batch.req_ids],
                        dtype=torch.bool,
                        device=self.device,
                    )
                )
            else:
                structured_mask.fill_(True)
            self.last_valid[input_batch.idx_mapping[structured_mask]] = 0
        _hybrid_observe_kernel[(num_reqs,)](
            input_batch.idx_mapping,
            structured_mask,
            num_sampled,
            num_rejected,
            self.model_stats,
            self.suffix_stats,
            self.last_use_suffix,
            self.last_buckets,
            self.last_valid,
            K=self.num_speculative_steps,
            K_PO2=triton.next_power_of_2(self.num_speculative_steps),
            PREFIX=self.prefix_len,
        )
        self.num_steps += 1
        if self.num_steps % self.stats_half_life == 0:
            self.model_stats.bitwise_right_shift_(1)
            self.suffix_stats.bitwise_right_shift_(1)

    def _publish(
        self,
        input_batch: InputBatch,
        model_drafts: torch.Tensor,
        suffix_drafts: torch.Tensor,
        num_sampled: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        self.output_drafts[: input_batch.num_reqs, k:].fill_(-1)
        _hybrid_route_publish_kernel[(input_batch.num_reqs,)](
            self.suffix.num_valid,
            self.suffix.match_lens,
            self.suffix.sources,
            model_drafts,
            model_drafts.stride(0),
            suffix_drafts,
            suffix_drafts.stride(0),
            self.model_stats,
            self.suffix_stats,
            self.structured_output_mask,
            input_batch.idx_mapping,
            num_sampled,
            self.num_steps,
            self.output_drafts,
            self.output_drafts.stride(0),
            self.last_use_suffix,
            self.last_buckets,
            self.last_valid,
            MIN_MATCH=self.min_match,
            K=k,
            STATS_K=self.num_speculative_steps,
            K_PO2=triton.next_power_of_2(k),
            PREFIX=self.prefix_len,
            QUALITY_GATE=self.min_expected is not None,
            MIN_EXPECTED=self.min_expected or 0.0,
            EXPLORATION_INTERVAL=self.exploration_interval,
        )
        return self.output_drafts[: input_batch.num_reqs]

    def sync_drafts_metadata(self) -> None:
        from vllm.distributed.parallel_state import get_pcp_group

        pcp_group = get_pcp_group()
        for tensor in (
            self.last_use_suffix,
            self.last_buckets,
            self.last_valid,
        ):
            pcp_group.broadcast(tensor, src=0)

    @torch.inference_mode()
    def postprocess(
        self,
        model_drafts: torch.Tensor,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        if k == 0:
            self.last_valid[input_batch.idx_mapping] = 0
            return model_drafts
        if k <= self.prefix_len:
            return self._publish(
                input_batch, model_drafts, model_drafts, num_sampled, k
            )
        suffix_drafts = self.suffix.propose_hybrid(
            input_batch,
            num_sampled,
            last_sampled,
            k - self.prefix_len,
            query_prefix=model_drafts,
            prefix_len=self.prefix_len,
        )
        return self._publish(input_batch, model_drafts, suffix_drafts, num_sampled, k)
