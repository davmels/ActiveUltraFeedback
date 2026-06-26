"""Histogram of how many completions per row are truncated (0 .. n_comp)."""

import argparse
import numpy as np
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--min_non_truncated", type=int, default=2,
                        help="Rows with fewer than this many non-truncated completions are dropped.")
    parser.add_argument("--num_proc", type=int, default=64)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    n = len(dataset)
    n_comp = len(dataset[0]["completions"])
    assert "truncated" in dataset[0]["completions"][0], "dataset has no `truncated` field"

    def count_batch(batch):
        return {"n_trunc": [
            sum(int(c.get("truncated", 0)) for c in comps)
            for comps in batch["completions"]
        ]}

    res = dataset.map(
        count_batch, batched=True, batch_size=1000, num_proc=args.num_proc,
        remove_columns=dataset.column_names, desc="Counting truncated",
    )
    counts = np.asarray(res["n_trunc"], dtype=np.int64)
    hist = np.bincount(counts, minlength=n_comp + 1)

    print(f"Dataset: {args.dataset_path}")
    print(f"Rows: {n} | completions/row: {n_comp}\n")
    print(f"{'#truncated':>10} {'#rows':>10} {'pct':>8} {'cum_pct':>9}")
    print("-" * 40)
    cum = 0
    for k in range(n_comp + 1):
        cum += hist[k]
        print(f"{k:>10} {hist[k]:>10} {100*hist[k]/n:>7.3f}% {100*cum/n:>8.3f}%")

    # drop rows with < min_non_truncated non-truncated completions,
    # i.e. > n_comp - min_non_truncated truncated
    m = args.min_non_truncated
    trunc_threshold = n_comp - m + 1  # >= this many truncated -> dropped
    dropped = int(hist[trunc_threshold:].sum())
    print(f"\nWould drop {dropped} rows ({100*dropped/n:.3f}%) "
          f"(< {m} non-truncated, i.e. >= {trunc_threshold} truncated)")


if __name__ == "__main__":
    main()
