# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ..utils import (
    evaluate_llm_for_gsm8k,
    get_spec_decode_metric_value,
    get_test_prompts,
)


@pytest.fixture
def disable_vllm_compile_cache_on_rocm(request: pytest.FixtureRequest) -> None:
    if current_platform.is_rocm():
        request.getfixturevalue("disable_vllm_compile_cache")


@pytest.mark.parametrize(
    "speculative_config",
    [
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
        {"method": "suffix"},
    ],
)
@pytest.mark.usefixtures("disable_vllm_compile_cache_on_rocm")
@single_gpu_only
def test_ngram_and_suffix_correctness(
    speculative_config: dict,
    model_name: str,
    vllm_runner,
):
    with vllm_runner(
        model_name,
        block_size=None,
        # Keep LLM defaults; VllmRunner only provides lifecycle cleanup here.
        enable_chunked_prefill=None,
        speculative_config=speculative_config,
        max_model_len=4096,
        # Preserve LLM's default compilation/cudagraph configuration. Without
        # this, VllmRunner injects its reduced test-only capture sizes.
        compilation_config=CompilationConfig(),
    ) as runner:
        evaluate_llm_for_gsm8k(runner.llm)


@pytest.mark.parametrize("async_scheduling", [True], ids=["async"])
@single_gpu_only
def test_ngram_gpu_default_with_async_scheduling(
    async_scheduling: bool,
    vllm_runner,
):
    """Test ngram_gpu speculative decoding (k=3) correctness with and without
    async scheduling, validated via GSM8K accuracy.
    Uses Qwen/Qwen3-8B (ref GSM8K accuracy: 87%-92%).
    """
    qwen3_model = "Qwen/Qwen3-8B"
    with vllm_runner(
        qwen3_model,
        block_size=None,
        speculative_config={
            "method": "ngram_gpu",
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 2,
        },
        max_model_len=4096,
        async_scheduling=async_scheduling,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        # Assert the resolved async_scheduling config matches what was requested.
        assert (
            spec_runner.llm.llm_engine.vllm_config.scheduler_config.async_scheduling
            == async_scheduling
        )
        evaluate_llm_for_gsm8k(spec_runner.llm, expected_accuracy_threshold=0.8)


@single_gpu_only
def test_suffix_cross_request_acceptance(
    sampling_config: SamplingParams,
    model_name: str,
    vllm_runner,
):
    """Repeating the same prompts should let suffix decoding draft the earlier
    responses from its corpus of finished requests."""
    test_prompts = get_test_prompts(mm_enabled=False)
    k = 8

    with vllm_runner(
        model_name,
        block_size=None,
        speculative_config={"method": "suffix", "num_speculative_tokens": k},
        max_model_len=1024,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        num_drafts = []
        num_accept = []
        for _ in range(2):
            spec_runner.llm.chat(test_prompts, sampling_config)
            metrics = spec_runner.llm.get_metrics()
            num_drafts.append(
                get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
            )
            num_accept.append(
                get_spec_decode_metric_value(
                    metrics, "vllm:spec_decode_num_accepted_tokens"
                )
            )

    # Stats are cumulative; compare per-draft acceptance of the two runs.
    first = num_accept[0] / num_drafts[0]
    second = (num_accept[1] - num_accept[0]) / (num_drafts[1] - num_drafts[0])
    summary = f"accepted per draft: first={first:.2f}, second={second:.2f}"
    # Without the corpus, the repeated pass stays close to the first.
    assert second > 2 * first, summary


@single_gpu_only
def test_ngram_verification_applies_logit_bias_and_min_p(
    model_name: str,
    vllm_runner,
):
    """logit_bias and min_p apply to the target logits that verify drafts, so
    they also hold on the tokens that verification produces."""
    copy_prompt = (
        "Copy the sentence below many times.\n\n"
        + "The quick brown fox jumps over the lazy dog. " * 12
        + "The quick brown"
    )
    min_p = 0.3

    with vllm_runner(
        model_name,
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 5,
        },
        disable_log_stats=False,
    ) as runner:
        llm = runner.llm

        def generate(prompts: list[str], params: SamplingParams):
            def num_accepted() -> float:
                return get_spec_decode_metric_value(
                    llm.get_metrics(), "vllm:spec_decode_num_accepted_tokens"
                )

            before = num_accepted()
            outputs = llm.generate(prompts, params)
            return [output.outputs[0] for output in outputs], num_accepted() - before

        [fox] = llm.get_tokenizer().encode(" fox", add_special_tokens=False)
        greedy = {"temperature": 0, "max_tokens": 64}
        [unbiased], _ = generate([copy_prompt], SamplingParams(**greedy))
        assert fox in unbiased.token_ids, unbiased.text
        [banned], accepted = generate(
            [copy_prompt], SamplingParams(**greedy, logit_bias={fox: -100})
        )
        assert accepted > 0
        assert fox not in banned.token_ids, banned.text

        def num_below_min_p(outputs) -> int:
            # At temperature 1, min_p keeps x iff lp(x) >= max lp + log(min_p).
            return sum(
                lps[token].logprob
                < max(lp.logprob for lp in lps.values()) + math.log(min_p) - 1e-3
                for output in outputs
                for token, lps in zip(output.token_ids, output.logprobs)
            )

        prompts = [copy_prompt, "Write a short story about a lighthouse keeper.\n\n"]
        stochastic = {
            "temperature": 1.0,
            "max_tokens": 128,
            "ignore_eos": True,
            "seed": 0,
            "logprobs": 1,
        }
        filtered, accepted = generate(
            prompts, SamplingParams(**stochastic, min_p=min_p)
        )
        assert accepted > 0
        num_filtered_below = num_below_min_p(filtered)
        assert num_filtered_below == 0
        unfiltered, _ = generate(prompts, SamplingParams(**stochastic))
        num_unfiltered_below = num_below_min_p(unfiltered)
        assert num_unfiltered_below > 0


def check_greedy_sequence_close(expected, actual, name):
    assert len(expected) == len(actual)
    for expected_output, actual_output in zip(expected, actual, strict=True):
        expected_completion = expected_output.outputs[0]
        actual_completion = actual_output.outputs[0]
        assert len(expected_completion.token_ids) == len(actual_completion.token_ids)
        assert expected_completion.finish_reason == actual_completion.finish_reason
        check_logprobs_close(
            outputs_0_lst=[
                (
                    expected_completion.token_ids,
                    expected_completion.text,
                    expected_completion.logprobs,
                )
            ],
            outputs_1_lst=[
                (
                    actual_completion.token_ids,
                    actual_completion.text,
                    actual_completion.logprobs,
                )
            ],
            name_0="target",
            name_1=name,
        )


def check_greedy_output_close(expected, actual, name):
    assert len(expected) == len(actual)
    for expected_output, actual_output in zip(expected, actual, strict=True):
        expected_completion = expected_output.outputs[0]
        actual_completion = actual_output.outputs[0]
        assert len(expected_completion.token_ids) == len(actual_completion.token_ids)
        assert expected_completion.finish_reason == actual_completion.finish_reason
        check_logprobs_close(
            outputs_0_lst=[
                (
                    expected_completion.token_ids,
                    expected_completion.text,
                    expected_completion.logprobs,
                )
            ],
            outputs_1_lst=[
                (
                    actual_completion.token_ids,
                    actual_completion.text,
                    actual_completion.logprobs,
                )
            ],
            name_0="target",
            name_1=name,
            always_check_logprobs=True,
        )
        for position, (token_0, token_1) in enumerate(
            zip(expected_completion.token_ids, actual_completion.token_ids)
        ):
            if token_0 != token_1:
                break
            logprob_0 = expected_completion.logprobs[position][token_0].logprob
            logprob_1 = actual_completion.logprobs[position][token_1].logprob
            assert math.isclose(logprob_0, logprob_1, rel_tol=1e-3, abs_tol=1e-3), (
                f"{name}: logprob mismatch at {position}: {logprob_0} != {logprob_1}"
            )


class SpliceProbe:
    def read_splice_state(self):
        drafter = self.model_runner.speculator
        splice = self.model_runner.splice
        observed = (
            int(splice.suffix_stats[..., 1].sum().item()) if splice is not None else 0
        )
        corpus_observed = (
            int(
                splice.suffix_stats[splice.suffix_stats.shape[0] // 2 :, ..., 1]
                .sum()
                .item()
            )
            if splice is not None
            else 0
        )
        return type(drafter).__name__, observed, corpus_observed


def check_splice_llm_correctness(
    vllm_runner, model, speculative_config, common, expected_drafter
):
    prompts = [
        (
            "Repeat this exact sentence three times: The quick brown fox jumps over "
            "the lazy dog."
        ),
        "Write a haiku about rain.",
    ]

    def generate(runner):
        return runner.llm.generate(
            prompts, SamplingParams(temperature=0, max_tokens=48, logprobs=5)
        )

    with vllm_runner(model, **common) as runner:
        expected = generate(runner)

    native_config = {
        key: value
        for key, value in speculative_config.items()
        if not key.startswith("suffix_")
    }
    with vllm_runner(
        model,
        speculative_config=native_config,
        **common,
    ) as runner:
        native = generate(runner)

    with vllm_runner(
        model,
        speculative_config=speculative_config,
        worker_extension_cls=f"{__name__}.SpliceProbe",
        **common,
    ) as runner:
        cold = generate(runner)
        drafter, cold_observed, cold_corpus_observed = runner.llm.collective_rpc(
            "read_splice_state"
        )[0]
        hot = generate(runner)
        _, hot_observed, hot_corpus_observed = runner.llm.collective_rpc(
            "read_splice_state"
        )[0]

    check_greedy_sequence_close(expected, native, "native-drafter")
    check_greedy_sequence_close(expected, cold, "splice-cold")
    check_greedy_sequence_close(expected, hot, "splice-hot")
    check_greedy_output_close(native, cold, "splice-cold")
    check_greedy_output_close(native, hot, "splice-hot")
    assert drafter == expected_drafter
    assert hot_observed > cold_observed
    assert hot_corpus_observed > cold_corpus_observed


@single_gpu_only
def test_suffix_splice_mtp_llm_correctness(vllm_runner):
    check_splice_llm_correctness(
        vllm_runner,
        model="Qwen/Qwen3.5-0.8B-Base",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "suffix_hybrid_mode": "splice",
            "suffix_hybrid_min_match_length": 2,
            "suffix_decoding_max_tree_depth": 8,
            "suffix_decoding_corpus_tokens": 65536,
        },
        common=dict(
            block_size=None,
            max_model_len=1024,
            enable_chunked_prefill=None,
            enable_prefix_caching=False,
            compilation_config=CompilationConfig(),
            limit_mm_per_prompt={"image": 0, "video": 0},
        ),
        expected_drafter="MTPSpeculator",
    )


@single_gpu_only
def test_suffix_splice_dflash2_llm_correctness(vllm_runner):
    check_splice_llm_correctness(
        vllm_runner,
        model=(
            "RedHatAI/Qwen3.5-4B-quantized.w4a16"
            if current_platform.is_cuda()
            else "Qwen/Qwen3.5-4B"
        ),
        speculative_config={
            "method": "dflash",
            "model": "vllm-project/Qwen3.5-4B-DFlash2",
            "num_speculative_tokens": 7,
            "suffix_hybrid_mode": "splice",
            "suffix_hybrid_min_match_length": 2,
            "suffix_decoding_max_tree_depth": 8,
            "suffix_decoding_corpus_tokens": 65536,
        },
        common=dict(
            block_size=None,
            max_model_len=512,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            language_model_only=True,
            enable_chunked_prefill=None,
            enable_prefix_caching=False,
            compilation_config=CompilationConfig(),
            gpu_memory_utilization=0.98,
        ),
        expected_drafter="DFlash2Speculator",
    )


@single_gpu_only
def test_suffix_splice_dflash_llm_correctness(vllm_runner):
    check_splice_llm_correctness(
        vllm_runner,
        model="Qwen/Qwen3-4B-FP8" if current_platform.is_cuda() else "Qwen/Qwen3-4B",
        speculative_config={
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 3,
            "suffix_hybrid_mode": "splice",
            "suffix_hybrid_min_match_length": 2,
            "suffix_decoding_max_tree_depth": 8,
            "suffix_decoding_corpus_tokens": 65536,
        },
        common=dict(
            block_size=None,
            max_model_len=512,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            enable_chunked_prefill=None,
            enable_prefix_caching=False,
            compilation_config=CompilationConfig(),
            gpu_memory_utilization=0.9,
        ),
        expected_drafter="DFlashSpeculator",
    )
