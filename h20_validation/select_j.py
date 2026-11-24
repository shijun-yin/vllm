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
    return output_tokens / seconds, accepted / drafts if drafts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--pure", type=Path)
    parser.add_argument("--tie-percent", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.results]
    if any(run["mode"] != "splice" for run in runs):
        raise ValueError("All positional results must be splice runs")
    for key in (
        "model",
        "draft_model",
        "num_speculative_tokens",
        "temperature",
        "top_p",
        "top_k",
        "corpus_tokens",
        "batch_size",
    ):
        if len({run.get(key) for run in runs}) != 1:
            raise ValueError(f"Mismatched {key}")
    pure = json.loads(args.pure.read_text()) if args.pure else None
    selection = {}
    for dataset in runs[0]["results"]:
        if any(dataset not in run["results"] for run in runs):
            raise ValueError(f"Missing {dataset}")
        reference_ids = runs[0]["results"][dataset]["ids"]
        metrics = {}
        for path, run in zip(args.results, runs, strict=True):
            result = run["results"][dataset]
            if result["ids"] != reference_ids:
                raise ValueError(f"{dataset}: request IDs differ")
            rate, accepted = hot_metrics(result)
            j = run["prefix_len"]
            if j in metrics:
                raise ValueError(f"Duplicate j={j}; aggregate repeats explicitly")
            metrics[j] = {
                "tokens_per_second": rate,
                "accepted_per_draft": accepted,
                "suffix_observed": result["hybrid_stats"]["suffix_observed"],
                "source": str(path),
            }
        best_rate = max(value["tokens_per_second"] for value in metrics.values())
        threshold = best_rate * (1 - args.tie_percent / 100)
        tied = [
            j for j, value in metrics.items() if value["tokens_per_second"] >= threshold
        ]
        selected = min(tied)
        pure_rate = None
        if pure is not None:
            pure_result = pure["results"][dataset]
            if pure_result["ids"] != reference_ids:
                raise ValueError(f"{dataset}: pure request IDs differ")
            pure_rate, _ = hot_metrics(pure_result)
        selection[dataset] = {
            "selected_j": selected,
            "candidates": metrics,
            "pure_tokens_per_second": pure_rate,
            "selected_gain": (
                metrics[selected]["tokens_per_second"] / pure_rate - 1
                if pure_rate is not None
                else None
            ),
            "tie_percent": args.tie_percent,
        }
        print(
            f"{dataset}: selected j={selected}, "
            f"rates="
            + ", ".join(
                f"j{j}={value['tokens_per_second']:.2f}"
                for j, value in sorted(metrics.items())
            )
        )
    output = json.dumps(selection, indent=2)
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
