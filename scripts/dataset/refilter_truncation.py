"""Re-flag `truncated` by the DPO max_length rule, then drop sparse prompts.

Single source of truth for the active-DPO loop and the maxmin DPO dataset: a
completion is marked `truncated` if it was ALREADY truncated, OR if its full
prompt+completion chat sequence exceeds `max_length` under the model tokenizer
(the exact check the DPO baseline's prepare_dataset_for_dpo applies). The flag
is monotonic (True stays True). After re-flagging, prompts with fewer than
`min_non_truncated` non-truncated completions are removed.

Reconstruct the maxmin DPO dataset from the OUTPUT of this script so both
pipelines select best/worst over the same non-truncated candidate set.

Run (on a compute node, in the right env):

    python scripts/dataset/refilter_truncation.py \
        --input_path  datasets/dolci_annotated_new_21models_min8 \
        --output_path datasets/dolci_annotated_new_21models_min8_maxlen4096 \
        --model_path  /iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--allenai--Olmo-3-7B-Instruct-SFT/snapshots/e1452fc572d51966ff4aaeb25118b891eb93e549
"""

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from datasets import load_from_disk
from transformers import AutoTokenizer


def as_messages(prompt):
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--min_non_truncated", type=int, default=8)
    p.add_argument("--num_proc", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = load_from_disk(args.input_path)

    def reflag(batch):
        out = []
        for prompt, comps in zip(batch["prompt"], batch["completions"]):
            msgs = as_messages(prompt)
            updated = []
            for c in comps:
                full_len = len(tokenizer.apply_chat_template(
                    msgs + [{"role": "assistant", "content": c["response_text"]}],
                    tokenize=True, add_generation_prompt=False,
                ))
                over = full_len > args.max_length
                c = dict(c)                                   # copy; keep all other fields
                c["truncated"] = int(bool(c.get("truncated", 0)) or over)
                updated.append(c)
            out.append(updated)
        return {"completions": out}

    dataset = dataset.map(
        reflag, batched=True, num_proc=args.num_proc,
        desc=f"re-flag truncated by max_length={args.max_length}",
    )

    def keep_row(batch):
        return [sum(1 for c in comps if not c.get("truncated", 0)) >= args.min_non_truncated
                for comps in batch["completions"]]

    before = len(dataset)
    dataset = dataset.filter(
        keep_row, batched=True, num_proc=args.num_proc,
        desc=f"drop prompts with < {args.min_non_truncated} non-truncated",
    )
    print(f"prompts: {before} -> {len(dataset)}")

    dataset.save_to_disk(args.output_path)                    # no num_proc on save
    print(f"saved -> {args.output_path}")


if __name__ == "__main__":
    main()
