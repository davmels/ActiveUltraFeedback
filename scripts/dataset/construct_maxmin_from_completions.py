"""Construct the maxmin DPO dataset directly from (prompt, completions).

For each prompt, pick chosen = highest-overall_score and rejected = lowest among
the NON-TRUNCATED completions, computed from the `completions` column itself (not
from any precomputed oracle_* columns). This reproduces the active-DPO loop's
selection at threshold C=+inf (best vs worst): first occurrence on score ties,
and the (0, 1) pair when all non-truncated scores are equal -- matching
select_pair_for_prompt / valid_completions exactly.

Run this on the OUTPUT of refilter_truncation.py so the `truncated` flag already
folds in the max_length rule; then loop (C=+inf) and this dataset train on the
identical pairs.

Run:
    python scripts/dataset/construct_maxmin_from_completions.py \
        --input_path  datasets/dolci_annotated_new_21models_min8_maxlen4096 \
        --output_path datasets/dolci_maxmin_dpo_v2
"""

import argparse

from datasets import load_from_disk


def as_messages(prompt):
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_proc", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(args.input_path)
    has_pid = "prompt_id" in dataset.column_names

    # need >= 2 non-truncated to form a pair (no-op on a refiltered >=8 input)
    dataset = dataset.filter(
        lambda batch: [sum(1 for c in comps if not c.get("truncated", 0)) >= 2
                       for comps in batch["completions"]],
        batched=True, num_proc=args.num_proc, desc="drop prompts with < 2 non-truncated",
    )

    out_keys = ["chosen", "rejected", "chosen_model", "rejected_model",
                "chosen_score", "rejected_score"] + (["prompt_id"] if has_pid else [])

    def to_maxmin(batch):
        out = {k: [] for k in out_keys}
        n = len(batch["prompt"])
        pids = batch["prompt_id"] if has_pid else [None] * n
        for prompt, comps, pid in zip(batch["prompt"], batch["completions"], pids):
            valid = [c for c in comps if not c.get("truncated", 0)]
            scores = [float(c["overall_score"]) for c in valid]
            best = max(range(len(valid)), key=scores.__getitem__)    # first argmax on ties
            worst = min(range(len(valid)), key=scores.__getitem__)   # first argmin on ties
            if best == worst:                # all non-truncated scores equal -> (0, 1), as the loop does
                best, worst = 0, 1
            msgs = as_messages(prompt)
            out["chosen"].append(msgs + [{"role": "assistant", "content": valid[best]["response_text"]}])
            out["rejected"].append(msgs + [{"role": "assistant", "content": valid[worst]["response_text"]}])
            out["chosen_model"].append(valid[best].get("model"))
            out["rejected_model"].append(valid[worst].get("model"))
            out["chosen_score"].append(scores[best])
            out["rejected_score"].append(scores[worst])
            if has_pid:
                out["prompt_id"].append(pid)
        return out

    dataset = dataset.map(
        to_maxmin, batched=True, remove_columns=dataset.column_names,
        num_proc=args.num_proc, desc="construct maxmin from completions",
    )
    print(f"maxmin rows: {len(dataset)}")
    dataset.save_to_disk(args.output_path)                    # no num_proc on save
    print(f"saved -> {args.output_path}")


if __name__ == "__main__":
    main()
