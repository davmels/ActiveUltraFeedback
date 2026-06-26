#!/usr/bin/env python3
"""
Generate 6 comparison tables for DeltaUCB experiments.
Each table fixes (method, dataset_size) and varies the outer loop batch size.

Model naming: deltaucb_{method}_{outerloop}_{inner}_{datasetsize}
  - method: blh, enn
  - outer loop batch: 256, 512, 1024, 2048, 4096
  - dataset size: 25000, 75000, 259922

Usage:
  python generate_tables_deltaucb.py \
    --models-dir /iopsstor/scratch/cscs/dmelikidze/models/dpo_new/parts2704 \
    --entity ActiveUF_Plus --project evals \
    --output-dir ./deltaucb_tables
"""

import os
import sys
import tempfile
import subprocess
import argparse
import shutil


METHODS = ["blh", "enn"]
DATASET_SIZES = ["25000", "75000", "259922"]
OUTER_LOOPS = ["256", "512", "1024", "2048", "4096"]

MAKE_TABLE_SCRIPT = "/iopsstor/scratch/cscs/dmelikidze/evals-post-train/make_table.py"
METRICS_FILE = "/iopsstor/scratch/cscs/dmelikidze/evals-post-train/configs/apertus/tasks_thesis_main_table.txt"

SFT_BASELINE = "baseline-Olmo-3-7B-SFT"
DPO_BASELINE = "baseline-Olmo-3-7B-DPO"


def get_matching_models(models_dir, method, dataset_size):
    pattern_suffix = f"_{dataset_size}"
    pattern_method = f"deltaucb_{method}_"
    matches = []
    for d in sorted(os.listdir(models_dir)):
        if d.startswith(pattern_method) and d.endswith(pattern_suffix):
            matches.append(d)
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--entity", default="ActiveUF_Plus")
    parser.add_argument("--project", default="evals")
    parser.add_argument("--output-dir", default="./deltaucb_tables")
    parser.add_argument("--metrics-file", default=METRICS_FILE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for method in METHODS:
        for ds_size in DATASET_SIZES:
            models = get_matching_models(args.models_dir, method, ds_size)
            if not models:
                print(f"[SKIP] No models for method={method}, dataset_size={ds_size}")
                continue

            table_name = f"deltaucb_{method}_ds{ds_size}"
            table_output = os.path.join(args.output_dir, table_name)
            os.makedirs(table_output, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Table: {table_name}")
            print(f"  method={method}, dataset_size={ds_size}")
            print(f"  Models: {models}")
            print(f"{'='*60}")

            tmpdir = tempfile.mkdtemp(prefix=f"deltaucb_{method}_{ds_size}_")
            try:
                for m in models:
                    src = os.path.join(args.models_dir, m)
                    dst = os.path.join(tmpdir, m)
                    os.symlink(src, dst)

                order_arg = sorted(models, key=lambda x: int(x.split("_")[2]))

                cmd = [
                    sys.executable, MAKE_TABLE_SCRIPT,
                    "--base-model-path", tmpdir,
                    "--entity", args.entity,
                    "--project", args.project,
                    "--metrics-file", args.metrics_file,
                    "--output", table_output,
                    "--baseline", SFT_BASELINE,
                    "--extra-baseline", DPO_BASELINE,
                    "--extra-baseline-name", "OLMo-3-7B-DPO",
                    "--raw-names",
                    "--model-order", *order_arg,
                    "--title", f"DeltaUCB {method.upper()} — Dataset Size {ds_size} (varying outer loop batch)",
                ]

                if args.debug:
                    cmd.append("--debug")

                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=not args.debug)
                if result.returncode != 0:
                    print(f"  [ERROR] make_table.py failed:")
                    print(result.stderr.decode() if result.stderr else "")
                else:
                    for ext in ["png", "csv"]:
                        src = os.path.join(table_output, f"eval_table.{ext}")
                        dst = os.path.join(table_output, f"{table_name}.{ext}")
                        if os.path.exists(src):
                            os.rename(src, dst)
                    print(f"  [OK] Output in {table_output}/")
            finally:
                shutil.rmtree(tmpdir)

    print(f"\nAll tables saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
