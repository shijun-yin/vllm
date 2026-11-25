# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import Counter, defaultdict
from statistics import median

import pytest
import regex as re
from scipy.stats import chi2_contingency

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import StructuredOutputsParams

from .utils import compute_acceptance_rate


def check_structured_output_preserves_sampling_distribution(
    vllm_runner,
    spec_config,
    async_scheduling,
    model="Qwen/Qwen3-4B",
    num_samples=4000,
    check_target_logprobs=False,
    **runner_kwargs,
):
    regex = r'\{"city": "(Paris|Berlin|Madrid)"\}'
    cities_counts = []
    for config in (None, spec_config):
        with vllm_runner(
            model,
            speculative_config=config,
            async_scheduling=async_scheduling,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
            **runner_kwargs,
        ) as runner:
            params = [
                SamplingParams(
                    temperature=1.0,
                    max_tokens=32,
                    seed=i,
                    logprobs=0 if check_target_logprobs else None,
                    structured_outputs=StructuredOutputsParams(regex=regex),
                )
                for i in range(num_samples)
            ]
            outputs = runner.get_llm().chat(
                [[{"role": "user", "content": "Name a random European capital."}]]
                * num_samples,
                params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            if config is not None:
                assert compute_acceptance_rate(runner.get_llm().get_metrics()) > 0
            texts = [o.outputs[0].text for o in outputs]
            assert all(re.fullmatch(regex, text) for text in texts), (
                "Grammar-invalid output"
            )
            if check_target_logprobs:
                for output in outputs:
                    completion = output.outputs[0]
                    assert completion.logprobs is not None
                    for token_id, logprobs in zip(
                        completion.token_ids, completion.logprobs, strict=True
                    ):
                        assert token_id in logprobs
                        assert math.isfinite(logprobs[token_id].logprob)
            cities_counts.append(Counter(texts))
    assert_grammar_distribution_unchanged(*cities_counts)


@pytest.mark.parametrize("rejection_sample_method", ["standard", "block"])
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@single_gpu_only
def test_structured_output_preserves_sampling_distribution(
    rejection_sample_method, async_scheduling, vllm_runner
):
    check_structured_output_preserves_sampling_distribution(
        vllm_runner,
        {
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 3,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": rejection_sample_method,
        },
        async_scheduling,
    )


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@single_gpu_only
def test_splice_preserves_sampling_distribution(async_scheduling, vllm_runner):
    check_structured_output_preserves_sampling_distribution(
        vllm_runner,
        {
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 3,
            "draft_sample_method": "greedy",
            "rejection_sample_method": "standard",
            "suffix_hybrid_mode": "splice",
            "suffix_hybrid_min_match_length": 2,
            "suffix_decoding_max_tree_depth": 8,
            "suffix_decoding_corpus_tokens": 65536,
        },
        async_scheduling,
        model=("Qwen/Qwen3-4B-FP8" if current_platform.is_cuda() else "Qwen/Qwen3-4B"),
        num_samples=2000,
        check_target_logprobs=True,
        max_model_len=256,
        max_num_seqs=256,
    )


class SpliceSamplingProbe:
    def read_splice_state(self):
        splice = self.model_runner.splice
        if splice is None:
            return 0, 0
        corpus_start = splice.suffix_stats.shape[0] // 2
        return (
            int(splice.suffix_stats[..., 1].sum().item()),
            int(splice.suffix_stats[corpus_start:, ..., 1].sum().item()),
        )

    def enable_splice_trace(self):
        splice = self.model_runner.splice
        self.splice_events = []
        self.splice_pending = {}
        original_postprocess = splice.postprocess
        original_prepare = splice.prepare_propose
        original_add = splice.add_request

        def add_request(req_index, new_req_data):
            self.splice_pending.pop(req_index, None)
            return original_add(req_index, new_req_data)

        def postprocess(model_drafts, input_batch, num_sampled, last_sampled, k):
            drafts = original_postprocess(
                model_drafts, input_batch, num_sampled, last_sampled, k
            )
            if k > 0:
                req_states = self.model_runner.req_states
                for batch_index, req_index in enumerate(
                    input_batch.idx_mapping.tolist()
                ):
                    if req_index < 0 or not splice.last_use_suffix[req_index].item():
                        continue
                    self.splice_pending[req_index] = {
                        "request_id": req_states.index_to_req_id[req_index].split(
                            "-", 1
                        )[0],
                        "start": int(req_states.total_len.gpu[req_index].item())
                        - int(req_states.prompt_len.np[req_index]),
                        "draft": drafts[batch_index, :k].cpu().tolist(),
                        "prefix_len": splice.prefix_len,
                    }
            return drafts

        def prepare_propose(input_batch, num_sampled, num_rejected):
            for batch_index, req_index in enumerate(input_batch.idx_mapping.tolist()):
                event = self.splice_pending.pop(req_index, None)
                if event is None:
                    continue
                event["accepted"] = int(num_sampled[batch_index].item()) - 1
                event["rejected"] = bool(num_rejected[batch_index].item())
                self.splice_events.append(event)
            return original_prepare(input_batch, num_sampled, num_rejected)

        splice.add_request = add_request
        splice.postprocess = postprocess
        splice.prepare_propose = prepare_propose
        return True

    def read_splice_trace(self):
        return self.splice_events


def _conditional_tokens(outputs, position, prefix):
    samples = []
    for output in outputs:
        completion = output.outputs[0]
        if (
            len(completion.token_ids) > position
            and tuple(completion.token_ids[:position]) == prefix
        ):
            samples.append(completion.token_ids[position])
    return samples


def _token_logprob(output, position, token):
    completion = output.outputs[0]
    if len(completion.token_ids) <= position:
        return None
    logprobs = completion.logprobs[position]
    return logprobs[token].logprob if token in logprobs else None


def check_splice_affected_positions(outputs_by_mode, events):
    splice_outputs = {output.request_id: output for output in outputs_by_mode["splice"]}
    other_outputs = {
        mode: {output.request_id: output for output in outputs}
        for mode, outputs in outputs_by_mode.items()
        if mode != "splice"
    }
    affected = {"accepted": set(), "rejected": set()}
    event_logprobs = []
    for event in events:
        output = splice_outputs[event["request_id"]]
        token_ids = output.outputs[0].token_ids
        positions = [
            ("accepted", position)
            for position in range(event["prefix_len"], event["accepted"])
        ]
        if event["rejected"] and event["prefix_len"] <= event["accepted"] < len(
            event["draft"]
        ):
            positions.append(("rejected", event["accepted"]))
        for kind, position in positions:
            if position >= len(token_ids):
                continue
            prefix = tuple(token_ids[:position])
            affected[kind].add((position, prefix))
            sampled_token = token_ids[position]
            event_logprobs.append((kind, event, position, prefix, sampled_token))

    compared = {"accepted": 0, "rejected": 0}
    for kind, groups in affected.items():
        for position, prefix in groups:
            samples = {
                mode: _conditional_tokens(outputs, position, prefix)
                for mode, outputs in outputs_by_mode.items()
            }
            if not all(len(sample) >= 10 for sample in samples.values()):
                continue
            assert_grammar_distribution_unchanged(
                Counter(samples["target"]), Counter(samples["splice"])
            )
            assert_grammar_distribution_unchanged(
                Counter(samples["native"]), Counter(samples["splice"])
            )
            compared[kind] += 1
    assert all(compared.values()), (
        f"Insufficient affected distribution groups: {compared}"
    )

    matched = {(kind, mode): 0 for kind in affected for mode in other_outputs}
    logprob_errors = defaultdict(list)
    for kind, event, position, prefix, candidate in event_logprobs:
        splice_logprob = _token_logprob(
            splice_outputs[event["request_id"]], position, candidate
        )
        if splice_logprob is None:
            continue
        for mode, outputs in other_outputs.items():
            output = outputs[event["request_id"]]
            if tuple(output.outputs[0].token_ids[:position]) != prefix:
                continue
            other_logprob = _token_logprob(output, position, candidate)
            if other_logprob is None:
                continue
            logprob_errors[kind, mode].append(abs(splice_logprob - other_logprob))
            matched[kind, mode] += 1
    assert all(matched.values()), f"Insufficient affected logprob checks: {matched}"
    for (kind, mode), errors in logprob_errors.items():
        drift = median(errors)
        assert drift <= 0.1, f"{kind} logprob drift at {mode}: {drift}; errors={errors}"


def _splice_test_output(request_id, second_logprob):
    tokens = [10, 20, 30]
    logprobs = [-0.1, -0.15, second_logprob]
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=tokens,
        cumulative_logprob=sum(logprobs),
        logprobs=[
            {token: Logprob(logprob)}
            for token, logprob in zip(tokens, logprobs, strict=True)
        ],
        finish_reason="stop",
    )
    return RequestOutput(request_id, None, [], None, [completion], True)


def test_splice_affected_logprobs_rejects_mutation():
    outputs_by_mode = {
        "target": [_splice_test_output(f"req-{index}", -0.2) for index in range(10)],
        "native": [_splice_test_output(f"req-{index}", -0.2) for index in range(10)],
        "splice": [_splice_test_output(f"req-{index}", -0.5) for index in range(10)],
    }
    events = [
        {
            "request_id": f"req-{index}",
            "start": 0,
            "draft": [10, 20, 31],
            "accepted": 2,
            "rejected": True,
            "prefix_len": 1,
        }
        for index in range(10)
    ]
    with pytest.raises(AssertionError, match="rejected logprob drift"):
        check_splice_affected_positions(outputs_by_mode, events)


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@single_gpu_only
def test_splice_real_switch_preserves_sampling_distribution(
    async_scheduling, vllm_runner
):
    splice_config = {
        "method": "dflash",
        "model": "z-lab/Qwen3-4B-DFlash-b16",
        "num_speculative_tokens": 3,
        "draft_sample_method": "greedy",
        "rejection_sample_method": "standard",
        "suffix_hybrid_mode": "splice",
        "suffix_hybrid_min_match_length": 2,
        "suffix_decoding_max_tree_depth": 8,
        "suffix_decoding_corpus_tokens": 65536,
    }
    native_config = {
        key: value
        for key, value in splice_config.items()
        if not key.startswith("suffix_")
    }
    model = "Qwen/Qwen3-4B-FP8" if current_platform.is_cuda() else "Qwen/Qwen3-4B"
    outputs_by_mode = {}
    splice_events = []
    for mode, config in (
        ("target", None),
        ("native", native_config),
        ("splice", splice_config),
    ):
        runner_kwargs = {}
        if mode == "splice":
            runner_kwargs["worker_extension_cls"] = f"{__name__}.SpliceSamplingProbe"
        with vllm_runner(
            model,
            speculative_config=config,
            async_scheduling=async_scheduling,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
            max_model_len=256,
            max_num_seqs=256,
            **runner_kwargs,
        ) as runner:
            num_samples = 1000
            params = [
                SamplingParams(
                    temperature=1.0,
                    top_k=3,
                    max_tokens=8,
                    seed=i,
                    logprobs=3,
                )
                for i in range(num_samples)
            ]
            if mode == "splice":
                runner.llm.collective_rpc("enable_splice_trace")
            outputs = runner.get_llm().chat(
                [[{"role": "user", "content": "Name a random European capital."}]]
                * num_samples,
                params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            if config is not None:
                assert compute_acceptance_rate(runner.get_llm().get_metrics()) > 0
            if mode == "splice":
                _, corpus_observed = runner.llm.collective_rpc("read_splice_state")[0]
                assert corpus_observed > 0
                splice_events = runner.llm.collective_rpc("read_splice_trace")[0]
            outputs_by_mode[mode] = outputs
    check_splice_affected_positions(outputs_by_mode, splice_events)


def assert_grammar_distribution_unchanged(counts1: Counter[str], counts2: Counter[str]):
    outcomes = sorted(counts1.keys() | counts2.keys())
    row1 = [counts1[outcome] for outcome in outcomes]
    row2 = [counts2[outcome] for outcome in outcomes]
    _, p, _, _ = chi2_contingency([row1, row2], correction=False)
    assert p > 1e-3, (
        f"Grammar output distributions differ: {counts1=}, {counts2=}, {p=}"
    )
