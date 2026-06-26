import argparse
import hashlib
import json
import os

from datasets import load_from_disk

from activeuf.utils import get_logger

logger = get_logger(__name__)

"""
Matches rows between two preference datasets by their shared prompt (all conversation turns
except the final assistant response). Uses MD5 hashing for O(n) matching.

Example run command:
    python -m scripts.dataset.match_prompts \
        --dataset_a /path/to/dataset_a \
        --dataset_b /path/to/dataset_b \
        --output_path /path/to/output
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_a", type=str, required=True, help="Path to the first (reference) dataset."
    )
    parser.add_argument(
        "--dataset_b", type=str, required=True, help="Path to the second dataset to reorder/match."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the matched dataset B (reordered to align with A)."
    )

    args = parser.parse_args()

    assert os.path.exists(args.dataset_a), f"Dataset A not found: {args.dataset_a}"
    assert os.path.exists(args.dataset_b), f"Dataset B not found: {args.dataset_b}"
    assert not os.path.exists(args.output_path), f"Output path already exists: {args.output_path}"

    return args


def prompt_key(row):
    turns = row["chosen"][:-1]
    return hashlib.md5(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {args}")

    logger.info("Loading dataset A")
    ds_a = load_from_disk(args.dataset_a)
    logger.info(f"Dataset A: {len(ds_a)} rows")

    logger.info("Loading dataset B")
    ds_b = load_from_disk(args.dataset_b)
    logger.info(f"Dataset B: {len(ds_b)} rows")

    logger.info("Building prompt index for dataset B")
    b_index = {}
    for i in range(len(ds_b)):
        key = prompt_key(ds_b[i])
        b_index[key] = i

    logger.info(f"Indexed {len(b_index)} unique prompts from dataset B")

    logger.info("Matching dataset A rows to dataset B")
    matched_b_indices = []
    unmatched_a_indices = []

    for i in range(len(ds_a)):
        key = prompt_key(ds_a[i])
        if key in b_index:
            matched_b_indices.append(b_index[key])
        else:
            unmatched_a_indices.append(i)

    logger.info(f"Matched: {len(matched_b_indices)} / {len(ds_a)}")
    if unmatched_a_indices:
        logger.warning(f"Unmatched rows in A: {len(unmatched_a_indices)}")
        logger.warning(f"First 5 unmatched indices: {unmatched_a_indices[:5]}")

    logger.info("Selecting and reordering dataset B to match dataset A ordering")
    matched_ds_b = ds_b.select(matched_b_indices)

    logger.info(f"Saving matched dataset ({len(matched_ds_b)} rows) to {args.output_path}")
    matched_ds_b.save_to_disk(args.output_path)

    logger.info("Done")
