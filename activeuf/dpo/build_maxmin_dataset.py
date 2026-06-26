"""Build a MaxMin preference dataset for the standard-DPO baseline.

For each prompt, the chosen response is the highest-reward non-truncated completion
and the rejected is the lowest-reward one (best-vs-worst = the widest actual-reward
gap). Output rows have conversational `chosen`/`rejected` (prompt + assistant turn),
which activeuf.dpo.training consumes directly (it keeps only those two columns and
lets TRL extract the shared prompt).

Run (CPU, one-off):
    python -m activeuf.dpo.build_maxmin_dataset \
        --inputs_path datasets/dolci_annotated_new_21models_min8 \
        --output_path datasets/dolci_maxmin_dpo
"""

import argparse

from datasets import load_from_disk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--min_non_truncated", type=int, default=2)
    args = p.parse_args()

    ds = load_from_disk(args.inputs_path)
    print(f"Loaded {len(ds)} prompts from {args.inputs_path}")

    def build(ex):
        valid = [
            (c["response_text"], float(c["overall_score"]))
            for c in ex["completions"]
            if not c.get("truncated", 0)
        ]
        pm = ex["prompt"] if isinstance(ex["prompt"], list) \
            else [{"role": "user", "content": ex["prompt"]}]
        if len(valid) >= args.min_non_truncated:
            best = max(valid, key=lambda x: x[1])
            worst = min(valid, key=lambda x: x[1])
            chosen_text, rejected_text = best[0], worst[0]
            keep = chosen_text != rejected_text          # need two distinct responses
        else:
            chosen_text = rejected_text = ""
            keep = False
        return {
            "prompt_id": ex["prompt_id"],
            "chosen": pm + [{"role": "assistant", "content": chosen_text}],
            "rejected": pm + [{"role": "assistant", "content": rejected_text}],
            "keep": keep,
        }

    ds = ds.map(build, remove_columns=ds.column_names)
    ds = ds.filter(lambda e: e["keep"]).remove_columns(["keep"])

    ds.save_to_disk(args.output_path)
    print(f"Saved {len(ds)} MaxMin pairs to {args.output_path}")


if __name__ == "__main__":
    main()
