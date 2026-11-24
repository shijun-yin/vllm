import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

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
    start = time.perf_counter()
    outputs = llm.generate(
        [prompt["text"] for prompt in prompts], sampling_params, use_tqdm=False
    )
    seconds = time.perf_counter() - start
    after = metric_values(llm)
    token_counts = [len(output.outputs[0].token_ids) for output in outputs]
    metrics = {
        name: after.get(name, 0) - before.get(name, 0)
        for name in set(before) | set(after)
    }
    drafts = metrics.get("vllm:spec_decode_num_drafts", 0)
    draft_tokens = metrics.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens", 0)
    result = {
        "wave": wave,
        "seconds": seconds,
        "output_tokens": sum(token_counts),
        "output_tokens_per_second": sum(token_counts) / seconds,
        "token_counts": token_counts,
        "finish_reasons": dict(
            Counter(output.outputs[0].finish_reason for output in outputs)
        ),
        "spec_metrics": metrics,
        "accepted_per_draft": accepted / drafts if drafts else None,
        "draft_token_acceptance": accepted / draft_tokens if draft_tokens else None,
    }
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
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=2200)
    parser.add_argument("--num-speculative-tokens", type=int)
    parser.add_argument("--prefix-len", type=int, default=1)
    parser.add_argument("--corpus-tokens", type=int, default=65536)
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--prepare-only", action="store_true")
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
        speculative_config=spec,
        worker_extension_cls=f"{__name__}.SpecStateExtension",
        seed=1234,
    )
    results = {}
    for dataset in datasets:
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
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_speculative_tokens": k if args.mode != "no_spec" else 0,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print("UPSTREAM_BENCH_RESULT=" + json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
