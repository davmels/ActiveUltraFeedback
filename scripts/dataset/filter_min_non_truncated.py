"""Keep only prompts with >= min_non_truncated non-truncated completions, and subset
features.npy to match (rows are dropped, so the memmap is re-indexed to the kept rows).

This materializes the filter run.py applies at load time, so the resulting dataset is
already clean (run.py's filter then keeps everything).

# python scripts/dataset/filter_min_non_truncated.py \
#     --dataset_path datasets/dolci_annotated_new_21models_trunc3600 \
#     --output_path datasets/dolci_annotated_new_21models_min8 \
#     --min_non_truncated 8
"""

import argparse
import os

import numpy as np
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--min_non_truncated", type=int, default=8,
                        help="Keep prompts with at least this many non-truncated completions.")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Source features.npy (default: <dataset_path>/features.npy).")
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()

    assert os.path.abspath(args.output_path) != os.path.abspath(args.dataset_path), \
        "output_path must differ from dataset_path"

    dataset = load_from_disk(args.dataset_path)
    n = len(dataset)
    assert "truncated" in dataset[0]["completions"][0], \
        "dataset has no `truncated` field; run mark_truncated.py first"
    n_comp = len(dataset[0]["completions"])
    min_keep = args.min_non_truncated

    def keep_batch(batch):
        return {"keep": [
            sum(1 for c in comps if not c.get("truncated", 0)) >= min_keep
            for comps in batch["completions"]
        ]}

    flags = dataset.map(
        keep_batch, batched=True, batch_size=1000, num_proc=args.num_proc,
        remove_columns=dataset.column_names, desc="Checking non-truncated counts",
    )["keep"]
    kept_indices = [i for i, k in enumerate(flags) if k]
    n_kept = len(kept_indices)
    print(f"Dataset: {n} rows, {n_comp} completions/row")
    print(f"Keeping {n_kept}/{n} rows ({100.0 * n_kept / n:.2f}%) with >= {min_keep} "
          f"non-truncated; dropping {n - n_kept}")

    ds_out = dataset.select(kept_indices)
    print(f"Saving filtered dataset -> {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    ds_out.save_to_disk(args.output_path)

    # subset features.npy to the kept rows
    feat_src = args.features_path or os.path.join(args.dataset_path, "features.npy")
    if not os.path.exists(feat_src):
        print(f"No features.npy at {feat_src}; skipping feature subset.")
        return

    total_floats = os.path.getsize(feat_src) // 4
    feat_dim = total_floats // (n * n_comp)
    assert feat_dim * n * n_comp == total_floats, \
        f"features.npy size {total_floats} not divisible by ({n} x {n_comp})"
    print(f"Source features: ({n}, {n_comp}, {feat_dim}) float32")

    old = np.memmap(feat_src, dtype=np.float32, mode="r", shape=(n, n_comp, feat_dim))
    feat_dst = os.path.join(args.output_path, "features.npy")
    new = np.memmap(feat_dst, dtype=np.float32, mode="w+",
                    shape=(n_kept, n_comp, feat_dim))
    kept = np.asarray(kept_indices, dtype=np.int64)
    for a in range(0, n_kept, args.chunk_size):
        b = min(a + args.chunk_size, n_kept)
        new[a:b] = old[kept[a:b]]
    new.flush()
    del new, old

    nf = os.path.getsize(feat_dst) // 4
    assert nf == n_kept * n_comp * feat_dim, "feature subset size mismatch"
    print(f"Wrote {feat_dst}: ({n_kept}, {n_comp}, {feat_dim}) float32")
    print("Done.")


if __name__ == "__main__":
    main()
