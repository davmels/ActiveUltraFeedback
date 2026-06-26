import argparse
import os
import shutil
import tempfile
from datasets import load_from_disk


KEEP_COLUMNS = {"chosen", "chosen_model", "chosen_score", "rejected", "rejected_model", "rejected_score"}


def clean_columns():
    parser = argparse.ArgumentParser(
        description="Remove all columns except chosen/rejected pairs from datasets in subfolders."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder containing dataset subfolders.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory.")
        return

    for name in sorted(os.listdir(args.input_dir)):
        subfolder = os.path.join(args.input_dir, name)
        if not os.path.isdir(subfolder):
            continue

        print(f"Loading {name}...")
        try:
            ds = load_from_disk(subfolder)
        except Exception as e:
            print(f"  Skipping (not a valid dataset): {e}")
            continue

        cols_to_remove = [c for c in ds.column_names if c not in KEEP_COLUMNS]
        if not cols_to_remove:
            print(f"  No columns to remove, skipping.")
            continue

        print(f"  Removing columns: {cols_to_remove}")
        ds = ds.remove_columns(cols_to_remove)
        tmp_dir = tempfile.mkdtemp(dir=args.input_dir)
        ds.save_to_disk(tmp_dir)
        shutil.rmtree(subfolder)
        os.rename(tmp_dir, subfolder)
        print(f"  Saved back to {subfolder}")

    print("\nDone.")


if __name__ == "__main__":
    clean_columns()
