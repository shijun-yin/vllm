# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import Counter

import pytest
import regex as re
from scipy.stats import chi2_contingency

from tests.utils import single_gpu_only
from vllm import SamplingParams
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


def assert_grammar_distribution_unchanged(counts1: Counter[str], counts2: Counter[str]):
    outcomes = sorted(counts1.keys() | counts2.keys())
    row1 = [counts1[outcome] for outcome in outcomes]
    row2 = [counts2[outcome] for outcome in outcomes]
    _, p, _, _ = chi2_contingency([row1, row2], correction=False)
    assert p > 1e-3, (
        f"Grammar output distributions differ: {counts1=}, {counts2=}, {p=}"
    )
