"""Print truncation statistics per model from the marked dataset."""

import argparse
from collections import defaultdict, Counter

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=128)
    args = parser.parse_args()

    ds = load_from_disk(args.input_path)

    def extract_stats(example):
        models = [c["model"] for c in example["completions"]]
        truncated = [c["truncated"] for c in example["completions"]]
        return {"_models": models, "_truncated": truncated}

    mapped = ds.map(extract_stats, num_proc=args.num_proc, remove_columns=ds.column_names, desc="Extracting stats")

    model_total = Counter()
    model_trunc = Counter()
    for row in mapped:
        for m, t in zip(row["_models"], row["_truncated"]):
            model_total[m] += 1
            model_trunc[m] += t

    total_all = sum(model_total.values())
    trunc_all = sum(model_trunc.values())

    print(f"\n{'Model':<45} {'Truncated':>10} {'Total':>10} {'Rate':>8}")
    print("-" * 75)
    for model in sorted(model_total, key=lambda m: model_trunc[m], reverse=True):
        t = model_trunc[model]
        n = model_total[model]
        print(f"{model:<45} {t:>10} {n:>10} {100*t/n:>7.1f}%")
    print("-" * 75)
    print(f"{'TOTAL':<45} {trunc_all:>10} {total_all:>10} {100*trunc_all/total_all:>7.1f}%")


if __name__ == "__main__":
    main()
