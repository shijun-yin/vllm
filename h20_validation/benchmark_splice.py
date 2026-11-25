import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

ROOT = Path("/root/qwen35-9b-deploy")
DATASETS = (
    "blazedit",
    "speed_coding",
    "speed_rag",
    "speed_writing",
    "spec_rag",
    "spec_summarization",
)


class StageProfiler:
    def __init__(self, record_events=True):
        self.record_events = record_events
        self.events = {}
        self.intervals = {}
        self.programs = Counter()

    def clear(self):
        self.events.clear()
        self.intervals.clear()
        self.programs.clear()

    def invoke(self, name, fn, programs=0, shape=None):
        start_time = time.perf_counter()
        if self.record_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            return fn()
        finally:
            if self.record_events:
                end.record()
                self.events.setdefault(name, []).append((start, end))
            end_time = time.perf_counter()
            self.intervals.setdefault(name, []).append((start_time, end_time, shape))
            self.programs[name] += programs

    def wrap_method(self, obj, attr, name):
        original = getattr(obj, attr)

        def timed(*args, **kwargs):
            return self.invoke(name, lambda: original(*args, **kwargs))

        setattr(obj, attr, timed)

    def wrap_execute_model(self, runner):
        original = runner.execute_model

        def timed(scheduler_output, *args, **kwargs):
            shape = (
                len(scheduler_output.num_scheduled_tokens),
                scheduler_output.total_num_scheduled_tokens,
                len(scheduler_output.finished_req_ids),
            )
            return self.invoke(
                "execute_model",
                lambda: original(scheduler_output, *args, **kwargs),
                shape=shape,
            )

        runner.execute_model = timed

    def wrap_kernel(self, module, attr, name):
        original = getattr(module, attr)
        profiler = self

        class TimedKernel:
            def __getitem__(self, grid):
                launch = original[grid]
                programs = 1
                for size in grid:
                    programs *= size

                def timed(*args, **kwargs):
                    return profiler.invoke(
                        name, lambda: launch(*args, **kwargs), programs=programs
                    )

                return timed

        setattr(module, attr, TimedKernel())

    def install(self, runner):
        from vllm.v1.worker.gpu.spec_decode.hybrid import speculator as hybrid_module
        from vllm.v1.worker.gpu.spec_decode.suffix import speculator as suffix_module

        self.wrap_execute_model(runner)
        if runner.speculator is not None:
            self.wrap_method(runner.speculator, "propose", "native_propose")
        if runner.splice is None:
            return
        self.wrap_method(runner.splice, "prepare_propose", "splice_prepare")
        self.wrap_method(runner.splice, "postprocess", "splice_postprocess")
        self.wrap_method(
            runner.splice,
            "on_requests_finished_with_lengths",
            "splice_finish_host",
        )
        self.wrap_kernel(
            hybrid_module,
            "_hybrid_observe_kernel",
            "observe",
        )
        self.wrap_kernel(
            hybrid_module,
            "_hybrid_route_publish_kernel",
            "route_publish",
        )
        self.wrap_method(suffix_module, "async_tensor_h2d", "ingest_h2d")
        self.wrap_kernel(
            suffix_module,
            "_suffix_local_scan_kernel",
            "local_scan",
        )
        self.wrap_kernel(
            suffix_module,
            "_suffix_finalize_kernel",
            "corpus_finalize",
        )
        self.wrap_kernel(
            suffix_module,
            "_suffix_ingest_kernel",
            "ingest",
        )

    def snapshot(self):
        if self.record_events:
            torch.cuda.synchronize()
        result = {}
        for name, intervals in self.intervals.items():
            durations = [(end - start) * 1000 for start, end, _ in intervals]
            item = {
                "calls": len(intervals),
                "host_ms": sum(durations),
                "max_host_ms": max(durations),
                "p95_host_ms": sorted(durations)[int(0.95 * (len(durations) - 1))],
                "programs": self.programs[name],
            }
            if self.record_events:
                item["device_ms"] = sum(
                    start.elapsed_time(end) for start, end in self.events[name]
                )
            if name == "execute_model":
                gaps = [
                    max(0.0, current[0] - previous[1]) * 1000
                    for previous, current in zip(intervals, intervals[1:])
                ]
                item["gap_host_ms"] = sum(gaps)
                item["max_gap_ms"] = max(gaps, default=0.0)
                item["shape_counts"] = dict(
                    Counter(str(shape) for _, _, shape in intervals)
                )
                item["slowest"] = [
                    {
                        "call": index,
                        "host_ms": durations[index],
                        "gap_before_ms": gaps[index - 1] if index else 0.0,
                        "shape": shape,
                    }
                    for index, (_, _, shape) in sorted(
                        enumerate(intervals), key=lambda pair: durations[pair[0]]
                    )[-5:]
                ]
            result[name] = item
        self.clear()
        return result


class SpecStateExtension:
    def reset_spec_state(self):
        speculator = self.model_runner.speculator
        if speculator is not None:
            speculator.reset_state()
        if self.model_runner.splice is not None:
            self.model_runner.splice.reset_state()
        return True

    def read_hybrid_stats(self):
        splice = self.model_runner.splice
        if splice is None:
            return None
        return {
            "model_accepted": int(splice.model_stats[..., 0].sum().item()),
            "model_observed": int(splice.model_stats[..., 1].sum().item()),
            "suffix_accepted": int(splice.suffix_stats[..., 0].sum().item()),
            "suffix_observed": int(splice.suffix_stats[..., 1].sum().item()),
        }

    def enable_stage_profile(self, record_events=True):
        self.stage_profiler = StageProfiler(record_events=record_events)
        self.stage_profiler.install(self.model_runner)
        return True

    def reset_stage_profile(self):
        if hasattr(self, "stage_profiler"):
            self.stage_profiler.clear()
        return True

    def read_stage_profile(self):
        if not hasattr(self, "stage_profiler"):
            return None
        return self.stage_profiler.snapshot()


def render(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def add_prompt(prompts, tokenizer, request_id, content, max_prompt_tokens):
    text = render(tokenizer, content)
    prompt_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
    if prompt_tokens <= max_prompt_tokens:
        prompts.append({"id": request_id, "text": text, "prompt_tokens": prompt_tokens})


def load_prompts(tokenizer, dataset, count, max_prompt_tokens):
    prompts = []
    if dataset == "blazedit":
        path = (
            ROOT / "data/blazedit_5k/data/train-00000-of-00001-cd50d93ebdf7cd84.parquet"
        )
        rows = list(load_dataset("parquet", data_files=str(path), split="train"))
        random.Random(20261122).shuffle(rows)
        for index, row in enumerate(rows):
            if not 0.01 <= float(row["norm_distance"]) <= 0.25:
                continue
            content = (
                "Given a code file, please apply the change requests and generate "
                "the new file.\n\nOriginal file:\n```python\n"
                f"{row['code']}\n```\n\nChange request:\n{row['change_request']}\n\n"
                'Please generate the new code file in the "New file" section below.'
            )
            add_prompt(
                prompts,
                tokenizer,
                f"blazedit-{index}",
                content,
                max_prompt_tokens,
            )
            if len(prompts) == count:
                break
    elif dataset.startswith("spec_"):
        category = dataset.removeprefix("spec_")
        path = ROOT / "data/spec_bench/question.jsonl"
        with path.open() as handle:
            rows = [
                row
                for line in handle
                if line.strip() and (row := json.loads(line))["category"] == category
            ]
        random.Random(20261124).shuffle(rows)
        for row in rows:
            add_prompt(
                prompts,
                tokenizer,
                str(row["question_id"]),
                row["turns"][0],
                max_prompt_tokens,
            )
            if len(prompts) == count:
                break
    else:
        category = dataset.removeprefix("speed_")
        path = ROOT / "data/speed_bench/qualitative/test-00000-of-00001.parquet"
        rows = [
            row
            for row in load_dataset("parquet", data_files=str(path), split="train")
            if row["category"] == category
        ]
        random.Random(20261123).shuffle(rows)
        for row in rows:
            if row["turns"][0].startswith("FULL BENCHMARK DATA SHOULD BE FETCHED"):
                continue
            add_prompt(
                prompts,
                tokenizer,
                row["question_id"],
                row["turns"][0],
                max_prompt_tokens,
            )
            if len(prompts) == count:
                break
    if len(prompts) != count:
        raise RuntimeError(f"Only {len(prompts)} {dataset} prompts fit the limit")
    return prompts


def metric_values(llm):
    names = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    }
    return {
        metric.name: getattr(metric, "value", 0)
        for metric in llm.get_metrics()
        if metric.name in names
    }


def run_wave(llm, prompts, wave, args):
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=91000 + index,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
        )
        for index in range(len(prompts))
    ]
    before = metric_values(llm)
    if args.profile_stages or args.profile_host:
        llm.collective_rpc("reset_stage_profile")
    start_timestamp = time.time()
    start = time.perf_counter()
    outputs = llm.generate(
        [prompt["text"] for prompt in prompts], sampling_params, use_tqdm=False
    )
    seconds = time.perf_counter() - start
    end_timestamp = time.time()
    after = metric_values(llm)
    token_counts = [len(output.outputs[0].token_ids) for output in outputs]
    token_hashes = [
        hashlib.sha256(
            json.dumps(
                list(output.outputs[0].token_ids), separators=(",", ":")
            ).encode()
        ).hexdigest()
        for output in outputs
    ]
    metrics = {
        name: after.get(name, 0) - before.get(name, 0)
        for name in set(before) | set(after)
    }
    drafts = metrics.get("vllm:spec_decode_num_drafts", 0)
    draft_tokens = metrics.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens", 0)
    result = {
        "wave": wave,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "seconds": seconds,
        "output_tokens": sum(token_counts),
        "output_tokens_per_second": sum(token_counts) / seconds,
        "token_counts": token_counts,
        "token_hashes": token_hashes,
        "finish_reasons": dict(
            Counter(output.outputs[0].finish_reason for output in outputs)
        ),
        "spec_metrics": metrics,
        "accepted_per_draft": accepted / drafts if drafts else None,
        "draft_token_acceptance": accepted / draft_tokens if draft_tokens else None,
    }
    if args.native_spec_metrics:
        result["request_spec_metrics"] = [
            (
                output.outputs[0].spec_decode_metrics.to_dict()
                if output.outputs[0].spec_decode_metrics is not None
                else None
            )
            for output in outputs
        ]
    if args.profile_stages or args.profile_host:
        result["stage_profile"] = llm.collective_rpc("read_stage_profile")[0]
    print(
        "WAVE_RESULT="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "wave",
                    "seconds",
                    "output_tokens",
                    "output_tokens_per_second",
                    "accepted_per_draft",
                )
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["mtp", "dflash"], required=True)
    parser.add_argument("--mode", choices=["no_spec", "pure", "splice"], required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS[:4]))
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hot-rounds", type=int, default=2)
    parser.add_argument("--warmup-wave", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=2200)
    parser.add_argument("--num-speculative-tokens", type=int)
    parser.add_argument("--prefix-len", type=int, default=1)
    parser.add_argument("--corpus-tokens", type=int, default=65536)
    parser.add_argument(
        "--corpus-key-mode",
        choices=["dual", "four", "four_fallback"],
        default="dual",
    )
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--profile-host", action="store_true")
    parser.add_argument("--native-spec-metrics", action="store_true")
    parser.add_argument("--native-iteration-details", action="store_true")
    parser.add_argument("--jit-verbose", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.family == "mtp":
        model = (
            Path(args.target_model)
            if args.target_model
            else ROOT / "models/Qwen3.5-9B-AWQ-4bit"
        )
        draft_model = None
        k = args.num_speculative_tokens or 7
    else:
        model = (
            Path(args.target_model)
            if args.target_model
            else ROOT / "models/Qwen3-4B-FP8"
        )
        draft_model = (
            Path(args.draft_model)
            if args.draft_model
            else ROOT / "models/Qwen3-4B-DFlash-b16"
        )
        k = args.num_speculative_tokens or 15

    datasets = args.datasets.split(",")
    if unknown := set(datasets) - set(DATASETS):
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    tokenizer = AutoTokenizer.from_pretrained(str(model), revision=args.target_revision)
    prompts = {
        dataset: load_prompts(
            tokenizer, dataset, args.num_prompts, args.max_prompt_tokens
        )
        for dataset in datasets
    }
    if args.prepare_only:
        for dataset, items in prompts.items():
            print(
                dataset,
                len(items),
                "tokens",
                min(item["prompt_tokens"] for item in items),
                max(item["prompt_tokens"] for item in items),
                [item["id"] for item in items],
            )
        return

    spec = None
    if args.mode != "no_spec":
        spec = {
            "method": args.family,
            "num_speculative_tokens": k,
            "suffix_decoding_max_tree_depth": 24,
            "suffix_decoding_corpus_tokens": args.corpus_tokens,
        }
        if draft_model is not None:
            spec["model"] = str(draft_model)
        if args.draft_revision is not None:
            spec["revision"] = args.draft_revision
        if args.mode == "splice":
            spec.update(
                suffix_hybrid_mode="splice",
                suffix_hybrid_prefix_length=args.prefix_len,
                suffix_hybrid_min_match_length=4,
            )
            if args.corpus_key_mode != "dual":
                spec["suffix_hybrid_corpus_key_mode"] = args.corpus_key_mode

    llm = LLM(
        model=str(model),
        revision=args.target_revision,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=False,
        per_request_spec_decode_metrics=(
            "detailed" if args.native_spec_metrics else "none"
        ),
        enable_logging_iteration_details=args.native_iteration_details,
        jit_monitor_verbose=args.jit_verbose,
        speculative_config=spec,
        worker_extension_cls=f"{__name__}.SpecStateExtension",
        seed=1234,
    )
    if args.profile_stages or args.profile_host:
        llm.collective_rpc(
            "enable_stage_profile",
            kwargs={"record_events": args.profile_stages},
        )
    results = {}
    for dataset in datasets:
        llm.collective_rpc("reset_spec_state")
        if args.warmup_wave:
            run_wave(llm, prompts[dataset], "warmup", args)
            llm.collective_rpc("reset_spec_state")
        waves = [run_wave(llm, prompts[dataset], "cold", args)]
        waves.extend(
            run_wave(llm, prompts[dataset], f"hot-{index + 1}", args)
            for index in range(args.hot_rounds)
        )
        results[dataset] = {
            "ids": [prompt["id"] for prompt in prompts[dataset]],
            "prompt_tokens": [prompt["prompt_tokens"] for prompt in prompts[dataset]],
            "waves": waves,
            "hybrid_stats": llm.collective_rpc("read_hybrid_stats")[0],
        }
    output = {
        "family": args.family,
        "mode": args.mode,
        "model": str(model),
        "target_revision": args.target_revision,
        "draft_revision": args.draft_revision,
        "draft_model": (
            str(draft_model)
            if draft_model is not None and args.mode != "no_spec"
            else None
        ),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ignore_eos": args.ignore_eos,
        "max_tokens": args.max_tokens,
        "prefix_len": args.prefix_len,
        "corpus_tokens": args.corpus_tokens,
        "corpus_key_mode": args.corpus_key_mode,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_speculative_tokens": k if args.mode != "no_spec" else 0,
        "profile_stages": args.profile_stages,
        "profile_host": args.profile_host or None,
        "native_spec_metrics": args.native_spec_metrics or None,
        "native_iteration_details": args.native_iteration_details or None,
        "jit_verbose": args.jit_verbose or None,
        "warmup_wave": args.warmup_wave or None,
        "wsl_pin_memory": os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY"),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print("UPSTREAM_BENCH_RESULT=" + json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
