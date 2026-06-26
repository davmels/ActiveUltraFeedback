"""Create a new dataset with some models' completions removed, and rebuild features.npy to match.

Each row has `completions[]` (one per model) and the dataset has a raw float32
`features.npy` memmap of shape (N, n_comp, feat_dim) where column j aligns with
completions[j]. Removing models drops completions AND the matching feature columns,
gathered per-row so it works regardless of whether model order is consistent across rows.

# python scripts/dataset/filter_models.py \
#     --dataset_path datasets/dolci_annotated \
#     --output_path datasets/dolci_annotated_21models \
#     --remove_models EuroLLM-1.7B-Instruct Phi-4-mini-instruct Qwen2.5-0.5B-Instruct Trinity-Mini
"""

import argparse
import os

import numpy as np
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--remove_models", type=str, nargs="+", required=True,
                        help="Model names to drop (matched against model basename, case-insensitive).")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Source features.npy (default: <dataset_path>/features.npy).")
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2048)
    args = parser.parse_args()

    assert os.path.abspath(args.output_path) != os.path.abspath(args.dataset_path), \
        "output_path must differ from dataset_path"

    remove = {m.lower() for m in args.remove_models}

    def is_removed(model: str) -> bool:
        return model.split("/")[-1].lower() in remove or model.lower() in remove

    dataset = load_from_disk(args.dataset_path)
    n = len(dataset)
    row0_models = [c["model"] for c in dataset[0]["completions"]]
    orig_n_comp = len(row0_models)

    # validate every requested name matches a real model in row 0
    matched = [m for m in row0_models if is_removed(m)]
    matched_keys = {m.split("/")[-1].lower() for m in matched} | {m.lower() for m in matched}
    unmatched = [m for m in remove if m not in matched_keys]
    if unmatched:
        avail = sorted(m.split("/")[-1] for m in row0_models)
        raise ValueError(
            f"These --remove_models matched no completion in row 0: {unmatched}\n"
            f"Available models: {avail}"
        )
    n_keep = orig_n_comp - len(matched)
    print(f"Dataset: {n} rows, {orig_n_comp} models/row")
    print(f"Removing {len(matched)} models -> keeping {n_keep}/row")
    print(f"  removed (row 0): {sorted(m.split('/')[-1] for m in matched)}")

    def filter_batch(batch):
        new_comps, kept_idxs = [], []
        for comps in batch["completions"]:
            keep = [(j, c) for j, c in enumerate(comps) if not is_removed(c["model"])]
            if len(keep) != n_keep:
                got = [c["model"] for c in comps]
                raise ValueError(
                    f"Row yielded {len(keep)} kept completions, expected {n_keep}. "
                    f"Row models: {got}"
                )
            new_comps.append([c for _, c in keep])
            kept_idxs.append([j for j, _ in keep])
        return {"completions": new_comps, "kept_idx": kept_idxs}

    mapped = dataset.map(
        filter_batch,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        desc="Filtering models",
    )

    kept_idx = np.asarray(mapped["kept_idx"], dtype=np.int64)  # (N, n_keep)
    assert kept_idx.shape == (n, n_keep), f"kept_idx shape {kept_idx.shape}"

    # report whether the surviving columns are identical across all rows (fixed model order)
    consistent = bool((kept_idx == kept_idx[0]).all())
    if consistent:
        print(f"Model order is FIXED across all rows; kept columns: {kept_idx[0].tolist()}")
    else:
        n_distinct = len(np.unique(kept_idx, axis=0))
        print(f"WARNING: model order varies across rows ({n_distinct} distinct column patterns); "
              f"gathering per-row.")

    ds_out = mapped.remove_columns(["kept_idx"])
    print(f"Saving filtered dataset -> {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    ds_out.save_to_disk(args.output_path)

    # rebuild features.npy by gathering kept columns per row
    feat_src = args.features_path or os.path.join(args.dataset_path, "features.npy")
    if not os.path.exists(feat_src):
        print(f"No features.npy at {feat_src}; skipping feature rebuild.")
        return

    total_floats = os.path.getsize(feat_src) // 4
    feat_dim = total_floats // (n * orig_n_comp)
    assert feat_dim * n * orig_n_comp == total_floats, \
        f"features.npy size {total_floats} not divisible by ({n} x {orig_n_comp})"
    print(f"Source features: ({n}, {orig_n_comp}, {feat_dim}) float32")

    old = np.memmap(feat_src, dtype=np.float32, mode="r",
                    shape=(n, orig_n_comp, feat_dim))
    feat_dst = os.path.join(args.output_path, "features.npy")
    new = np.memmap(feat_dst, dtype=np.float32, mode="w+",
                    shape=(n, n_keep, feat_dim))
    for a in range(0, n, args.chunk_size):
        b = min(a + args.chunk_size, n)
        blk = np.asarray(old[a:b])                       # (cb, orig_n_comp, feat_dim)
        idx = kept_idx[a:b]                               # (cb, n_keep)
        new[a:b] = blk[np.arange(b - a)[:, None], idx, :]
    new.flush()
    del new, old

    # sanity check against run.py's filesize-based inference
    nf = os.path.getsize(feat_dst) // 4
    inferred = nf // (n * n_keep)
    assert inferred == feat_dim and inferred * n * n_keep == nf, "feature rebuild size mismatch"
    print(f"Wrote {feat_dst}: ({n}, {n_keep}, {feat_dim}) float32")
    print("Done.")


if __name__ == "__main__":
    main()
