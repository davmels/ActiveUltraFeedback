"""Mark completions that would be truncated at max_length when chat-templated and tokenized."""

import os
import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-4B")
    parser.add_argument("--max_length", type=int, default=3900)
    parser.add_argument("--num_proc", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=os.environ.get("HF_HOME", None),
    )

    max_length = args.max_length

    def mark_truncated(example):
        prompt = example["prompt"]
        completions = example["completions"]
        truncated_flags = []

        for comp in completions:
            messages = list(prompt) + [{"role": "assistant", "content": comp["response_text"]}]
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            truncated_flags.append(len(token_ids) >= max_length)

        new_completions = []
        for comp, flag in zip(completions, truncated_flags):
            new_comp = dict(comp)
            new_comp["truncated"] = int(flag)
            new_completions.append(new_comp)

        return {"completions": new_completions}

    ds = load_from_disk(args.input_path)
    print(f"Loaded dataset with {len(ds)} rows")

    ds = ds.map(mark_truncated, num_proc=args.num_proc, writer_batch_size=500, desc="Marking truncated")
    print("Map done, saving...")

    ds.save_to_disk(args.output_path)
    print(f"Saved to {args.output_path}")

    print("Done. To check stats, run:")
    print(f"  python -c \"from datasets import load_from_disk; ds=load_from_disk('{args.output_path}'); s=ds[:100]['completions']; t=sum(c['truncated'] for row in s for c in row); print(f'First 100 rows: {{t}}/{{sum(len(r) for r in s)}} truncated')\"")


    # Copy features.npy if it exists
    features_src = os.path.join(args.input_path, "features.npy")
    if os.path.exists(features_src):
        import shutil
        features_dst = os.path.join(args.output_path, "features.npy")
        shutil.copy2(features_src, features_dst)
        print(f"Copied features.npy")


if __name__ == "__main__":
    main()
