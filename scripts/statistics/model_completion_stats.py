"""Per-model statistics over a `prompt + completions[]` dataset.

For each model that produced completions, compute:
  - response token-length stats (count, mean, std, min, max)
  - judge-score stats (mean, std, var, min, max) from `final_score`
and render bar charts (length and score, with std error bars) plus a CSV.

Token length is measured on the assistant `response_text` with the given tokenizer
(default Skywork-Reward-V2-Qwen3-4B, same as truncation marking).

# python scripts/statistics/model_completion_stats.py \
#     --dataset_path datasets/dolci_annotated \
#     --output_dir plots/dolci_model_stats
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="Skywork/Skywork-Reward-V2-Qwen3-4B")
    parser.add_argument("--score_key", type=str, default="final_score")
    parser.add_argument("--exclude_truncated", action="store_true",
                        help="Skip completions flagged truncated (if the field exists).")
    parser.add_argument("--num_proc", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=os.environ.get("HF_HOME", None)
    )

    score_key = args.score_key
    exclude_truncated = args.exclude_truncated

    def process_batch(batch):
        models, lengths, scores = [], [], []   # included completions (for len/score)
        tmodels, truncs = [], []               # ALL completions (for truncation counts)
        for prompt, completions in zip(batch["prompt"], batch["completions"]):
            messages, comp_models, comp_scores = [], [], []
            for comp in completions:
                tr = int(comp.get("truncated", 0))
                tmodels.append(comp["model"])
                truncs.append(tr)
                if exclude_truncated and tr:
                    continue
                messages.append(
                    list(prompt)
                    + [{"role": "assistant", "content": comp["response_text"]}]
                )
                comp_models.append(comp["model"])
                comp_scores.append(float(comp[score_key]))
            if not messages:
                continue
            # full chat-templated conversation length (prompt + response), as in truncation marking
            enc = tokenizer.apply_chat_template(messages, tokenize=True)
            models.extend(comp_models)
            lengths.extend(len(ids) for ids in enc)
            scores.extend(comp_scores)
        return {
            "models": [models], "lengths": [lengths], "scores": [scores],
            "tmodels": [tmodels], "truncs": [truncs],
        }

    dataset = load_from_disk(args.dataset_path)
    results = dataset.map(
        process_batch,
        batched=True,
        batch_size=200,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing + collecting",
    )

    # accumulate per model
    len_by_model = defaultdict(list)
    score_by_model = defaultdict(list)
    trunc_by_model = defaultdict(int)
    total_by_model = defaultdict(int)
    for row in results:
        for m, l, s in zip(row["models"], row["lengths"], row["scores"]):
            len_by_model[m].append(l)
            score_by_model[m].append(s)
        for m, t in zip(row["tmodels"], row["truncs"]):
            total_by_model[m] += 1
            trunc_by_model[m] += t

    models = sorted(total_by_model.keys())
    stats = []
    for m in models:
        lens = np.asarray(len_by_model[m] or [np.nan], dtype=np.float64)
        scs = np.asarray(score_by_model[m] or [np.nan], dtype=np.float64)
        n_total = total_by_model[m]
        n_trunc = trunc_by_model[m]
        stats.append({
            "model": m,
            "count": n_total,
            "n_trunc": n_trunc,
            "trunc_pct": 100.0 * n_trunc / n_total if n_total else 0.0,
            "trunc_zero": 0.0,  # for the (errorless) truncation bar plot
            "len_mean": lens.mean(), "len_std": lens.std(),
            "len_min": lens.min(), "len_max": lens.max(),
            "score_mean": scs.mean(), "score_std": scs.std(), "score_var": scs.var(),
            "score_min": scs.min(), "score_max": scs.max(),
        })

    total_comps = sum(total_by_model.values())
    total_trunc = sum(trunc_by_model.values())
    print(f"\nTotal completions: {total_comps} | truncated: {total_trunc} "
          f"({100.0 * total_trunc / total_comps:.2f}%)")

    # CSV
    csv_path = os.path.join(args.output_dir, "model_completion_stats.csv")
    cols = ["model", "count", "n_trunc", "trunc_pct", "len_mean", "len_std",
            "len_min", "len_max", "score_mean", "score_std", "score_var",
            "score_min", "score_max"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in stats:
            f.write(",".join(
                str(s[c]) if c in ("model", "count", "n_trunc") else f"{s[c]:.4f}"
                for c in cols
            ) + "\n")
    print(f"Wrote {csv_path}")

    # also echo a readable table (sorted by truncation fraction, worst first)
    print(f"\n{'Model':<40} {'N':>7} {'Trunc':>7} {'Trunc%':>7} {'LenMean':>9} "
          f"{'LenStd':>8} {'ScoreMean':>10} {'ScoreStd':>9}")
    print("-" * 102)
    for s in sorted(stats, key=lambda s: s["trunc_pct"], reverse=True):
        print(f"{s['model']:<40} {s['count']:>7} {s['n_trunc']:>7} {s['trunc_pct']:>6.2f}% "
              f"{s['len_mean']:>9.1f} {s['len_std']:>8.1f} "
              f"{s['score_mean']:>10.4f} {s['score_std']:>9.4f}")

    # --- plots ---
    def barh(metric_mean, metric_std, title, xlabel, fname, sort_desc=True):
        order = sorted(stats, key=lambda s: s[metric_mean], reverse=sort_desc)
        names = [s["model"] for s in order]
        means = [s[metric_mean] for s in order]
        errs = [s[metric_std] for s in order]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(names))))
        y = np.arange(len(names))
        ax.barh(y, means, xerr=errs, color="#4C72B0", ecolor="#888", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        out = os.path.join(args.output_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    barh("len_mean", "len_std", "Mean conversation token length per model (±std)",
         "tokens (prompt + response, chat-templated)", "length_per_model.png")
    barh("score_mean", "score_std", f"Mean {score_key} per model (±std)",
         score_key, "score_per_model.png")
    barh("trunc_pct", "trunc_zero", "Truncated completions per model",
         "% truncated", "truncated_per_model.png")


if __name__ == "__main__":
    main()
