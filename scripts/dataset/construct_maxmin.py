import argparse
import os

from datasets import load_from_disk

KEEP_COLUMNS = ["chosen_model", "rejected_model", "chosen_score", "rejected_score"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing model subdirectories, each with an HF dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save maxmin datasets.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for dataset.map().")
    return parser.parse_args()


def to_maxmin(example):
    prompt = example["prompt"]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return {
        "chosen": prompt + [{"role": "assistant", "content": example["oracle_best_response"]}],
        "rejected": prompt + [{"role": "assistant", "content": example["oracle_worst_response"]}],
    }


if __name__ == "__main__":
    args = parse_args()

    subdirs = sorted(
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d))
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for name in subdirs:
        input_path = os.path.join(args.base_dir, name)
        output_path = os.path.join(args.output_dir, f"{name}_maxmin")

        print(f"Processing {input_path} -> {output_path}")
        dataset = load_from_disk(input_path)

        cols_to_remove = [c for c in dataset.column_names if c not in KEEP_COLUMNS]
        maxmin_dataset = dataset.map(
            to_maxmin,
            remove_columns=cols_to_remove,
            load_from_cache_file=False,
            num_proc=args.num_proc,
            desc=f"MaxMin {name}",
        )

        maxmin_dataset.save_to_disk(output_path)
        print(f"  Saved {len(maxmin_dataset)} samples to {output_path}")
