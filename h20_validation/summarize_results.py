import argparse
import json
from pathlib import Path


def hot_metrics(result):
    waves = [wave for wave in result["waves"] if wave["wave"].startswith("hot-")]
    if not waves:
        raise ValueError("No hot waves found")
    seconds = sum(wave["seconds"] for wave in waves)
    output_tokens = sum(wave["output_tokens"] for wave in waves)
    accepted = sum(
        wave["spec_metrics"].get("vllm:spec_decode_num_accepted_tokens", 0)
        for wave in waves
    )
    drafts = sum(
        wave["spec_metrics"].get("vllm:spec_decode_num_drafts", 0) for wave in waves
    )
    return {
        "tokens_per_second": output_tokens / seconds,
        "accepted_per_draft": accepted / drafts if drafts else None,
        "hot_waves": len(waves),
    }


def stage_metrics(result):
    stages = {}
    for wave in result["waves"]:
        if not wave["wave"].startswith("hot-"):
            continue
        for name, values in wave.get("stage_profile", {}).items():
            total = stages.setdefault(
                name, {"device_ms": 0.0, "host_ms": 0.0, "calls": 0, "programs": 0}
            )
            for key in total:
                total[key] += values.get(key, 0)
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pure", type=Path)
    parser.add_argument("splice", type=Path)
    parser.add_argument("--min-suffix-observed", type=int, default=1)
    args = parser.parse_args()
    pure = json.loads(args.pure.read_text())
    splice = json.loads(args.splice.read_text())
    if pure["mode"] != "pure" or splice["mode"] != "splice":
        raise ValueError("Expected pure and splice result files")
    for key in (
        "family",
        "model",
        "draft_model",
        "target_revision",
        "draft_revision",
        "num_speculative_tokens",
        "corpus_key_mode",
        "temperature",
        "top_p",
        "top_k",
        "ignore_eos",
        "max_tokens",
        "batch_size",
        "max_model_len",
        "gpu_memory_utilization",
        "profile_stages",
        "profile_host",
        "native_spec_metrics",
        "native_iteration_details",
        "jit_verbose",
        "warmup_wave",
        "wsl_pin_memory",
    ):
        if pure.get(key) != splice.get(key):
            raise ValueError(f"Mismatched {key}: {pure.get(key)} != {splice.get(key)}")
    if pure["results"].keys() != splice["results"].keys():
        raise ValueError("Mismatched dataset sets")

    print(
        "| Dataset | Pure tok/s | Splice tok/s | Delta | Pure AL | Splice AL "
        "| Suffix observed |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in pure["results"]:
        pure_result = pure["results"][name]
        splice_result = splice["results"][name]
        if pure_result["ids"] != splice_result["ids"]:
            raise ValueError(f"{name}: request IDs differ")
        if pure_result["prompt_tokens"] != splice_result["prompt_tokens"]:
            raise ValueError(f"{name}: prompt token counts differ")
        pure_hot = hot_metrics(pure_result)
        splice_hot = hot_metrics(splice_result)
        if pure_hot["hot_waves"] != splice_hot["hot_waves"]:
            raise ValueError(f"{name}: hot wave counts differ")
        stats = splice_result["hybrid_stats"]
        observed = stats["suffix_observed"] if stats else 0
        if observed < args.min_suffix_observed:
            raise ValueError(f"{name}: only {observed} suffix observations")
        delta = splice_hot["tokens_per_second"] / pure_hot["tokens_per_second"] - 1
        pure_al = pure_hot["accepted_per_draft"]
        splice_al = splice_hot["accepted_per_draft"]
        print(
            f"| {name} | {pure_hot['tokens_per_second']:.2f} "
            f"| {splice_hot['tokens_per_second']:.2f} | {delta:+.2%} "
            f"| {pure_al:.3f} | {splice_al:.3f} | {observed} |"
        )
        if not splice.get("profile_stages"):
            continue
        pure_stages = stage_metrics(pure_result)
        splice_stages = stage_metrics(splice_result)
        print(f"\n{name} hot stages (nested intervals; do not sum device times):")
        print(
            "| Stage | Pure device ms | Splice device ms | Pure host ms "
            "| Splice host ms | Calls P/S |"
        )
        print("|---|---:|---:|---:|---:|---:|")
        for stage in sorted(pure_stages.keys() | splice_stages.keys()):
            p = pure_stages.get(stage, {"device_ms": 0.0, "host_ms": 0.0, "calls": 0})
            s = splice_stages.get(stage, {"device_ms": 0.0, "host_ms": 0.0, "calls": 0})
            print(
                f"| {stage} | {p['device_ms']:.3f} | {s['device_ms']:.3f} "
                f"| {p['host_ms']:.3f} | {s['host_ms']:.3f} "
                f"| {p['calls']}/{s['calls']} |"
            )


if __name__ == "__main__":
    main()
