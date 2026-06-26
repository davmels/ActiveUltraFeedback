"""Compute per-model average oracle reward (overall_score) from a dataset, excluding truncated responses."""

import argparse
from collections import defaultdict
import numpy as np
from datasets import load_from_disk


def process_batch(batch):
    models = []
    scores = []
    truncated_count = 0
    total = 0
    for completions in batch["completions"]:
        for comp in completions:
            total += 1
            if comp.get("truncated", 0):
                truncated_count += 1
                continue
            models.append(comp["model"])
            scores.append(comp["overall_score"])
    return {
        "models": [models],
        "scores": [scores],
        "truncated_count": [truncated_count],
        "total": [total],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--output", type=str, default=None, help="Path to save results (default: print to stdout)")
    parser.add_argument("--num_proc", type=int, default=64)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    results = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )

    model_scores = defaultdict(list)
    total_completions = 0
    truncated_count = 0

    for row in results:
        total_completions += row["total"]
        truncated_count += row["truncated_count"]
        for model, score in zip(row["models"], row["scores"]):
            model_scores[model].append(score)

    lines = []
    lines.append(f"Dataset: {args.dataset_path}")
    lines.append(f"Rows: {len(dataset)}")
    lines.append(f"Total completions: {total_completions}")
    lines.append(f"Truncated (excluded): {truncated_count}")
    lines.append(f"Used: {total_completions - truncated_count}")
    lines.append("")
    lines.append(f"{'Model':<40} {'Count':>8} {'Mean Score':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 92)

    for model in sorted(model_scores.keys()):
        scores = np.array(model_scores[model])
        lines.append(f"{model:<40} {len(scores):>8} {scores.mean():>12.4f} {scores.std():>10.4f} {scores.min():>10.4f} {scores.max():>10.4f}")

    text = "\n".join(lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"Results written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
