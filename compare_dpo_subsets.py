#!/usr/bin/env python3
"""Compare the DPO runs (3 methods x 3 subset sizes) via make_html_table.py.

Produces 3 separate HTML tables. By default it splits by subset size: one table per
{5k, 10k, 20k}, each comparing the 3 methods (Random / L1024 / L4096) at that budget.
Set SPLIT_BY = "method" to instead get one table per method, each showing the 3 sizes.

Run names are discovered from the partitions_dpo subdir names. Any extra CLI args are
passed straight through to make_html_table.py (e.g. --no-baselines, --no-split, --flat).

    python compare_dpo_subsets.py                 # 3 default tables
    python compare_dpo_subsets.py --no-baselines  # only the method columns
"""
import os
import re
import subprocess
import sys

# --- paths ---------------------------------------------------------------
PARTITIONS_DIR = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/meeting_08_09_2026/partitions2_dpo"
MAKE_HTML = "/iopsstor/scratch/cscs/dmelikidze/evals-post-train/make_html_table.py"
METRICS_FILE = "/iopsstor/scratch/cscs/dmelikidze/evals-post-train/configs/apertus/tasks_thesis_main_table.txt"
OUTPUT_PREFIX = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/dpo_comparison"

# W&B location of the DPO runs (baselines still come from make_html_table's pinned project)
WANDB_ENTITY = "ActiveUF_Plus"
WANDB_PROJECT = "evals"

# --- display config (edit these) -----------------------------------------
# method token (as it appears in the subdir name) -> (label, sort order)
METHODS = {
    "L256":      ("K256",       0),  # pool == select == 256  -> no selection
    "L1024_K256": ("K256 L1024", 1),  # select 256 from 1024
    "L4096_K256": ("K256 L4096", 2),  # select 256 from 4096
    "L1024_K256_domquota": ("K256 L1024", 1),  # domain-quota select 256 from 1024
    "L4096_K256_domquota": ("K256 L4096", 2),  # domain-quota select 256 from 4096
}
# subset size (trailing number) -> short label
SIZES = {"5000": "5k", "10000": "10k", "20000": "20k"}

# "size"   -> 3 files, one per subset size, columns = the 3 methods
# "method" -> 3 files, one per method, columns = the 3 subset sizes
SPLIT_BY = "size"

# parses e.g. ...ultrafeedback_L1024_K256_20260608-003704-221_20000
NAME_RE = re.compile(r"_ultrafeedback_(.+?)_\d{8}-[\d-]+_(\d+)$")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def discover_runs():
    """Return list of dicts: method_token, method_label, method_order, size, size_label, run_name."""
    runs = []
    for name in sorted(os.listdir(PARTITIONS_DIR)):
        m = NAME_RE.search(name)
        if not m:
            print(f"  skipping unrecognized subdir: {name}")
            continue
        method_token, size = m.groups()
        if method_token not in METHODS or size not in SIZES:
            print(f"  skipping (unmapped method/size): {name}")
            continue
        label, order = METHODS[method_token]
        runs.append({
            "method_token": method_token, "method_label": label, "method_order": order,
            "size": size, "size_label": SIZES[size], "run_name": name,
        })
    return runs


def build_tables(runs):
    """Group runs into (output_path, title, ordered [(run_name, display)]) per output file."""
    tables = []
    if SPLIT_BY == "size":
        for size, size_label in SIZES.items():
            group = sorted((r for r in runs if r["size"] == size), key=lambda r: r["method_order"])
            if not group:
                continue
            cols = [(r["run_name"], r["method_label"]) for r in group]  # column = method
            tables.append((f"{OUTPUT_PREFIX}_{size_label}.html",
                           f"DPO at {size_label} prompts: method comparison", cols))
    elif SPLIT_BY == "method":
        for token, (label, order) in sorted(METHODS.items(), key=lambda kv: kv[1][1]):
            group = sorted((r for r in runs if r["method_token"] == token), key=lambda r: int(r["size"]))
            if not group:
                continue
            cols = [(r["run_name"], r["size_label"]) for r in group]  # column = size
            tables.append((f"{OUTPUT_PREFIX}_{slug(label)}.html",
                           f"DPO {label}: subset-size scaling", cols))
    else:
        raise ValueError(f"SPLIT_BY must be 'size' or 'method', got {SPLIT_BY!r}")
    return tables


def main():
    runs = discover_runs()
    if not runs:
        print("No runs found in", PARTITIONS_DIR)
        return

    for output, title, cols in build_tables(runs):
        print(f"\n=== {title} -> {output} ===")
        for run_name, display in cols:
            print(f"  {display:<16} <- {run_name}")
        cmd = [
            sys.executable, MAKE_HTML,
            "--metrics-file", METRICS_FILE,
            "--entity", WANDB_ENTITY,
            "--project", WANDB_PROJECT,
            "--models", *[c[0] for c in cols],
            "--rename", *[f"{c[0]}={c[1]}" for c in cols],
            "--output", output,
            "--title", title,
            "--no-split",  # single table, no Training/Test toggle
            *sys.argv[1:],  # passthrough: --no-baselines, --more-details, ...
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
