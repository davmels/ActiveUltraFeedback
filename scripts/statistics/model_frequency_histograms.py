import os
from collections import Counter
from datasets import load_from_disk, load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

STATS_COLUMNS = ["chosen_model", "rejected_model", "chosen_score", "rejected_score"]

#python scripts/statistics/model_frequency_histograms.py single --data_dir ./datasets/
def generate_model_frequency_histograms(
    data_dir, output_dir, single_file=False, title_suffix="", num_proc=None,
):
    if num_proc is None:
        num_proc = os.cpu_count()

    # if num_proc and num_proc > 1:
    #     disable_caching()

    if single_file:
        files_to_process = [data_dir]
    else:
        files_to_process = sorted(os.listdir(data_dir))

    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(files_to_process, desc="Datasets"):
        if single_file:
            data_path = fname
        else:
            data_path = os.path.join(data_dir, fname)

        print(f"\nProcessing: {fname}")

        try:
            try:
                dataset = load_from_disk(data_path)
            except Exception:
                dataset = load_dataset(data_path)

            dataset = dataset["train"] if "train" in dataset else dataset

            missing = [c for c in STATS_COLUMNS if c not in dataset.column_names]
            if missing:
                print(f"  Skipping {data_path}: missing columns {missing}")
                continue

            # Drop all columns we don't need (speeds up all subsequent ops)
            cols_to_remove = [c for c in dataset.column_names if c not in STATS_COLUMNS]
            if cols_to_remove:
                dataset = dataset.remove_columns(cols_to_remove)

            total = len(dataset)
            print(f"  Loaded {total} rows, filtering...")

            # Filter out identical model pairs using num_proc
            filtered = dataset.filter(
                lambda row: row["chosen_model"] != row["rejected_model"],
                num_proc=num_proc,
                desc="  Filtering identical",
            )
            identical_counter = total - len(filtered)

            # Extract columns as lists (fast on arrow-backed dataset)
            chosen_models = filtered["chosen_model"]
            rejected_models = filtered["rejected_model"]
            chosen_scores = filtered["chosen_score"]
            rejected_scores = filtered["rejected_score"]

            avg_chosen = sum(chosen_scores) / len(chosen_scores)
            avg_rejected = sum(rejected_scores) / len(rejected_scores)
            print(f"  Avg chosen score:     {avg_chosen:.4f}")
            print(f"  Avg rejected score:   {avg_rejected:.4f}")
            print(f"  Avg score difference: {avg_chosen - avg_rejected:.4f}")
            print(f"  Identical models:     {identical_counter}")

            chosen_counts = Counter(chosen_models)
            rejected_counts = Counter(rejected_models)

            chosen_total = sum(chosen_counts.values())
            rejected_total = sum(rejected_counts.values())
            chosen_pct = {k: v / chosen_total * 100 for k, v in chosen_counts.most_common()}
            rejected_pct = {k: v / rejected_total * 100 for k, v in rejected_counts.most_common()}

            score_stats = {
                "avg_chosen": avg_chosen,
                "avg_rejected": avg_rejected,
                "avg_diff": avg_chosen - avg_rejected,
            }
            _plot_model_distributions(
                chosen_pct,
                rejected_pct,
                os.path.join(output_dir, f"distribution_{os.path.basename(data_path)}.png"),
                title_suffix=title_suffix,
                score_stats=score_stats,
            )
        except Exception as e:
            print(f"  Error processing {data_path}: {e}. Skipping.")
            continue


def _plot_model_distributions(
    chosen_counter_sorted,
    rejected_counter_sorted,
    filename,
    title_suffix="",
    score_stats=None,
):
    chosen_models = list(chosen_counter_sorted.keys())
    chosen_values = list(chosen_counter_sorted.values())
    rejected_models = list(rejected_counter_sorted.keys())
    rejected_values = list(rejected_counter_sorted.values())

    plot_title = "Model Selection Distribution"
    if title_suffix:
        plot_title += " " + title_suffix

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(28, max(6, len(chosen_models), len(rejected_models)) * 0.4),
    )
    fig.suptitle(plot_title, fontsize=20)

    bars1 = axes[0].barh(chosen_models, chosen_values, color="green", alpha=0.8)
    axes[0].set_xlabel("Percentage (%)", fontsize=14)
    axes[0].set_title("Chosen Model Distribution", fontsize=16)
    axes[0].invert_yaxis()
    for bar, value in zip(bars1, chosen_values):
        axes[0].text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=12,
        )
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    bars2 = axes[1].barh(rejected_models, rejected_values, color="red", alpha=0.8)
    axes[1].set_xlabel("Percentage (%)", fontsize=14)
    axes[1].set_title("Rejected Model Distribution", fontsize=16)
    axes[1].invert_yaxis()
    for bar, value in zip(bars2, rejected_values):
        axes[1].text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=12,
        )
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    if score_stats:
        stats_text = (
            f"Avg chosen score: {score_stats['avg_chosen']:.4f}\n"
            f"Avg rejected score: {score_stats['avg_rejected']:.4f}\n"
            f"Avg score difference: {score_stats['avg_diff']:.4f}"
        )
        fig.text(
            0.5, -0.02, stats_text,
            ha="center", va="top", fontsize=13,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
        )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def _compute_dataset_stats(data_path, num_proc=None):
    """Load a dataset and return (chosen_pct, rejected_pct) dicts sorted by frequency."""
    if num_proc is None:
        num_proc = os.cpu_count()

    # if num_proc and num_proc > 1:
    #     disable_caching()

    try:
        dataset = load_from_disk(data_path)
    except Exception:
        dataset = load_dataset(data_path)

    dataset = dataset["train"] if "train" in dataset else dataset

    cols_to_remove = [c for c in dataset.column_names if c not in STATS_COLUMNS]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    filtered = dataset.filter(
        lambda row: row["chosen_model"] != row["rejected_model"],
        num_proc=num_proc,
    )

    chosen_counts = Counter(filtered["chosen_model"])
    rejected_counts = Counter(filtered["rejected_model"])
    chosen_scores = filtered["chosen_score"]
    rejected_scores = filtered["rejected_score"]

    chosen_total = sum(chosen_counts.values())
    rejected_total = sum(rejected_counts.values())
    chosen_pct = {k: v / chosen_total * 100 for k, v in chosen_counts.most_common()}
    rejected_pct = {k: v / rejected_total * 100 for k, v in rejected_counts.most_common()}

    avg_chosen = sum(chosen_scores) / len(chosen_scores)
    avg_rejected = sum(rejected_scores) / len(rejected_scores)
    score_stats = {
        "avg_chosen": avg_chosen,
        "avg_rejected": avg_rejected,
        "avg_diff": avg_chosen - avg_rejected,
    }

    return chosen_pct, rejected_pct, score_stats


def generate_multi_dataset_grid(
    data_paths, labels, output_path, title="", num_proc=None, ncols=3,
):
    """Plot chosen/rejected distributions for multiple datasets in a grid.

    Each cell gets a pair of horizontal bar charts (chosen + rejected) side by side.
    Layout is automatically computed as ceil(n / ncols) rows x ncols columns.
    """
    all_stats = []
    kept_labels = []
    for path, label in zip(data_paths, labels):
        print(f"Processing: {label}")
        try:
            all_stats.append(_compute_dataset_stats(path, num_proc))
            kept_labels.append(label)
        except Exception as e:
            print(f"  Error processing {path}: {e}. Skipping.")

    if not all_stats:
        print("No datasets could be processed.")
        return

    labels = kept_labels
    n = len(all_stats)
    nrows = (n + ncols - 1) // ncols

    max_models = max(
        max(len(c), len(r)) for c, r, _ in all_stats
    )
    cell_height = max(4, max_models * 0.35)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14 * ncols, cell_height * nrows),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=22, y=1.0)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        if idx >= n:
            ax.axis("off")
            continue

        chosen_pct, rejected_pct, score_stats = all_stats[idx]

        models = list(chosen_pct.keys())
        chosen_vals = [chosen_pct.get(m, 0) for m in models]
        rejected_vals = [rejected_pct.get(m, 0) for m in models]

        y_pos = range(len(models))
        bar_height = 0.35

        bars_c = ax.barh(
            [y - bar_height / 2 for y in y_pos], chosen_vals,
            height=bar_height, color="green", alpha=0.8, label="Chosen",
        )
        bars_r = ax.barh(
            [y + bar_height / 2 for y in y_pos], rejected_vals,
            height=bar_height, color="red", alpha=0.8, label="Rejected",
        )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(models, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Percentage (%)")
        ax.set_title(labels[idx], fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        stats_text = (
            f"Avg chosen: {score_stats['avg_chosen']:.4f}  "
            f"Avg rejected: {score_stats['avg_rejected']:.4f}  "
            f"Avg diff: {score_stats['avg_diff']:.4f}"
        )
        ax.text(
            0.5, -0.12, stats_text,
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
        )

    # Share x-axis limits within each column
    for col in range(ncols):
        col_axes = [axes[row][col] for row in range(nrows) if row * ncols + col < n]
        if col_axes:
            max_xlim = max(ax.get_xlim()[1] for ax in col_axes)
            for ax in col_axes:
                ax.set_xlim(0, max_xlim)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def _discover_datasets_for_prefix(dirs, prefix):
    """Discover datasets matching a prefix across multiple directories.

    Returns a list of (path, label, sort_key) tuples. Directories are expected
    to contain folders like {prefix}_orig_{size} or {prefix}_prompts_{size}.
    The numeric suffix is extracted for sorting.
    """
    results = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.startswith(prefix):
                path = os.path.join(d, name)
                # Extract numeric suffix for sorting
                parts = name.rsplit("_", 1)
                size = int(parts[-1]) if parts[-1].isdigit() else 0
                results.append((path, name, size))
    return results


def generate_compare_grid(
    model_dirs, prefix, output_path, title="", num_proc=None,
):
    """Auto-discover datasets with a given prefix from model_dirs.

    Top row: *_orig_* datasets (sorted by size).
    Bottom row: *_prompts_* datasets (sorted by size).
    """
    entries = _discover_datasets_for_prefix(model_dirs, prefix)

    orig = sorted([(p, l, s) for p, l, s in entries if "_orig_" in l], key=lambda x: x[2])
    prompts = sorted([(p, l, s) for p, l, s in entries if "_prompts_" in l], key=lambda x: x[2])

    # Top row = orig, bottom row = prompts
    ordered = orig + prompts
    data_paths = [p for p, l, s in ordered]
    labels = [l for p, l, s in ordered]
    ncols = max(len(orig), len(prompts))

    if not data_paths:
        print(f"No datasets found with prefix '{prefix}' in {model_dirs}")
        return

    print(f"Found {len(orig)} orig + {len(prompts)} prompts datasets for '{prefix}'")
    generate_multi_dataset_grid(data_paths, labels, output_path, title, num_proc, ncols)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate model frequency histograms from dataset directory."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Original single-dataset mode
    single = subparsers.add_parser("single", help="Process one dataset or directory of datasets")
    single.add_argument("--data_dir", type=str, required=True)
    single.add_argument("--single_file", action="store_true")
    single.add_argument("--output_dir", type=str, default=".")
    single.add_argument("--title_suffix", type=str, default="")
    single.add_argument("--num_proc", type=int, default=128)

    # Multi-dataset grid mode
    multi = subparsers.add_parser("grid", help="Compare multiple datasets in a grid plot")
    multi.add_argument("--datasets", nargs="+", required=True, help="Paths to datasets")
    multi.add_argument("--labels", nargs="+", required=True, help="Label for each dataset")
    multi.add_argument("--output", type=str, required=True, help="Output image path")
    multi.add_argument("--title", type=str, default="")
    multi.add_argument("--ncols", type=int, default=3)
    multi.add_argument("--num_proc", type=int, default=128)

    # Compare mode: auto-discover by prefix
    compare = subparsers.add_parser("compare", help="Compare orig vs prompts for a method prefix")
    compare.add_argument("--model_dirs", nargs="+", required=True, help="Directories containing dataset folders")
    compare.add_argument("--prefix", type=str, required=True, help="Method prefix, e.g. delta_ucb")
    compare.add_argument("--output", type=str, required=True, help="Output image path")
    compare.add_argument("--title", type=str, default="")
    compare.add_argument("--num_proc", type=int, default=128)

    args = parser.parse_args()

    if args.command == "single":
        generate_model_frequency_histograms(
            args.data_dir, args.output_dir, args.single_file, args.title_suffix, args.num_proc,
        )
    elif args.command == "grid":
        assert len(args.datasets) == len(args.labels), "Must have one label per dataset"
        generate_multi_dataset_grid(
            args.datasets, args.labels, args.output, args.title, args.num_proc, args.ncols,
        )
    elif args.command == "compare":
        generate_compare_grid(
            args.model_dirs, args.prefix, args.output, args.title, args.num_proc,
        )
    else:
        parser.print_help()