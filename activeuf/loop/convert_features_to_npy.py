"""Write a raw float32 `features.npy` memmap next to a dataset so the loop loads features fast.

run.py reads features via:
    np.memmap(features.npy, dtype=np.float32, mode="r",
              shape=(len(dataset), n_comp, feat_dim))
i.e. RAW C-contiguous float32 bytes, no header (feat_dim is inferred from filesize).
So we must NOT use np.save (it adds a header). We write with np.memmap(mode="w+").

Rows must be in the dataset's on-disk order (pre-filter, pre-shuffle) and number
exactly len(dataset), because run.py indexes the memmap by original index.

Two sources, auto-detected:
  - feature partials dir (the `{rank}-{batch}.pt` files from compute_base_model_features) -- fastest
  - the inline `features` column of the dataset itself (fallback)

# from partials (preferred):
#   python -m activeuf.loop.convert_features_to_npy \
#       --dataset_path datasets/combined_with_smqwen_3_235b \
#       --feature_partials_path datasets/combined_with_smqwen_3_235b-feature_partials
# from inline column:
#   python -m activeuf.loop.convert_features_to_npy \
#       --dataset_path datasets/combined_with_smqwen_3_235b-features
"""

from argparse import ArgumentParser
import glob
import os

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm


def build_from_partials(partials_path):
    """Mirror combine_dataset_with_features.py: load all .pt partials, sort by
    (prompt_idx, completion_idx), reshape to (n_prompts, n_comp, feat_dim)."""
    temp_ids, features = [], []
    filepaths = sorted(glob.glob(os.path.join(partials_path, "*.pt")))
    if not filepaths:
        raise FileNotFoundError(f"No .pt partials found in {partials_path}")
    for filepath in tqdm(filepaths, desc="Loading partials"):
        x = torch.load(filepath, map_location="cpu")
        temp_ids.extend(x["temp_ids"])
        features.append(x["features"])
    features = torch.vstack(features)

    temp_ids = torch.tensor(temp_ids)
    n_comp = int(temp_ids[:, 1].max()) + 1
    sort_keys = temp_ids[:, 0] * n_comp + temp_ids[:, 1]
    features = features[sort_keys.argsort()]
    features = features.view(-1, n_comp, features.size(-1))
    return features.to(torch.float32).numpy()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", required=True,
        help="Dataset dir run.py loads (features.npy is written here).",
    )
    parser.add_argument(
        "--feature_partials_path", default=None,
        help="Dir of {rank}-{batch}.pt partials. Defaults to "
             "<dataset_path>-feature_partials if it exists; else reads the inline column.",
    )
    parser.add_argument(
        "--output_path", default=None,
        help="Where to write the .npy (default: <dataset_path>/features.npy).",
    )
    parser.add_argument("--chunk_size", type=int, default=2048,
                        help="Rows per chunk when reading the inline column.")
    args = parser.parse_args()

    out_path = args.output_path or os.path.join(args.dataset_path, "features.npy")

    ds = load_from_disk(args.dataset_path)
    n = len(ds)
    print(f"Dataset {args.dataset_path}: {n} rows")

    partials_path = args.feature_partials_path
    if partials_path is None:
        default_partials = args.dataset_path.rstrip("/") + "-feature_partials"
        if os.path.isdir(default_partials):
            partials_path = default_partials

    if partials_path is not None:
        print(f"Source: feature partials ({partials_path})")
        feats = build_from_partials(partials_path)
        n_feat, n_comp, feat_dim = feats.shape
        if n_feat != n:
            raise ValueError(
                f"Partials cover {n_feat} prompts but dataset has {n} rows. "
                f"features.npy must align 1:1 with the dataset run.py loads."
            )
        print(f"Writing {feats.shape} float32 -> {out_path}")
        mm = np.memmap(out_path, dtype=np.float32, mode="w+",
                       shape=(n, n_comp, feat_dim))
        mm[:] = feats
        mm.flush()
        del mm
    else:
        print("Source: inline `features` column")
        if "features" not in ds.column_names:
            raise ValueError(
                f"No feature partials found and no `features` column in {args.dataset_path}."
            )
        first = np.asarray(ds[0]["features"], dtype=np.float32)
        n_comp, feat_dim = first.shape
        print(f"Writing ({n}, {n_comp}, {feat_dim}) float32 -> {out_path}")
        mm = np.memmap(out_path, dtype=np.float32, mode="w+",
                       shape=(n, n_comp, feat_dim))
        ds_np = ds.with_format("numpy", columns=["features"])
        for start in tqdm(range(0, n, args.chunk_size), desc="Writing features"):
            end = min(start + args.chunk_size, n)
            mm[start:end] = np.asarray(ds_np[start:end]["features"], dtype=np.float32)
        mm.flush()
        del mm

    # Sanity check against run.py's filesize-based feat_dim inference.
    total_floats = os.path.getsize(out_path) // 4
    inferred_feat_dim = total_floats // (n * n_comp)
    assert inferred_feat_dim == feat_dim, (
        f"Filesize check failed: inferred feat_dim={inferred_feat_dim} != {feat_dim}. "
        f"File may have a header or wrong row count."
    )
    print(f"Done. run.py will memmap shape=({n}, {n_comp}, {feat_dim}).")
    print("Note: keep this features.npy in the dataset dir; mark_truncated.py copies it forward.")


if __name__ == "__main__":
    main()
