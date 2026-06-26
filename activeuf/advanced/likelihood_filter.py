"""
Compute per-token log-likelihoods for completions in a preference dataset,
grouped by source model. Outputs statistics (mean, std, median, min, max)
for each model's completions.

Usage:
    python -m activeuf.advanced.likelihood_filter \
        --dataset-path /path/to/combined_annotated \
        --model-path /path/to/model \
        --max-length 4096 \
        --output-path likelihood_stats.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_prompt_messages(messages: list[dict]) -> list[dict]:
    """Extract prompt messages by removing the last assistant turn."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return messages[:i]
    return messages


def compute_completion_logprobs(
    model,
    tokenizer,
    prompt_messages: list[dict],
    response_text: str,
    max_length: int,
    device: torch.device,
) -> dict:
    """
    Compute per-token log-likelihoods for only the completion portion.

    Returns dict with:
        - sum_logprob: sum of log-probs over completion tokens
        - mean_logprob: mean log-prob per completion token
        - num_tokens: number of completion tokens
        - token_logprobs: list of per-token logprobs
    """
    # Tokenize prompt-only (with generation prompt to match how completion starts)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Build full conversation: prompt + assistant response
    full_messages = prompt_messages + [{"role": "assistant", "content": response_text}]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Completion tokens start after prompt
    completion_start = len(prompt_ids)
    num_completion_tokens = len(full_ids) - completion_start

    if num_completion_tokens <= 0:
        return {
            "sum_logprob": 0.0,
            "mean_logprob": 0.0,
            "num_tokens": 0,
            "token_logprobs": [],
        }

    # Truncate if needed
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        num_completion_tokens = len(full_ids) - completion_start
        if num_completion_tokens <= 0:
            return {
                "sum_logprob": 0.0,
                "mean_logprob": 0.0,
                "num_tokens": 0,
                "token_logprobs": [],
            }

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[0, completion_start - 1 : len(full_ids) - 1, :]
    shift_labels = input_ids[0, completion_start : len(full_ids)]

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    token_logprobs_list = token_log_probs.cpu().tolist()
    sum_lp = sum(token_logprobs_list)
    mean_lp = sum_lp / len(token_logprobs_list)

    return {
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "num_tokens": len(token_logprobs_list),
        "token_logprobs": token_logprobs_list,
    }


def compute_statistics(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute completion log-likelihood statistics per source model."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the HF dataset on disk (combined_annotated).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the evaluator model (used to compute logprobs).",
    )
    parser.add_argument("--max-length", type=int, default=4096, help="Max token length for input.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of rows to process (for testing).")
    parser.add_argument("--output-path", type=str, default="likelihood_stats.json", help="Output JSON path.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)
    data = ds[args.split]

    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))

    print(f"Processing {len(data)} rows...")

    # Key: model_name -> list of per-row results
    model_results = defaultdict(list)
    skipped = 0

    for i in tqdm(range(len(data)), desc="Computing logprobs"):
        row = data[i]

        # Extract prompt from chosen messages (remove last assistant turn)
        messages = row["chosen"]
        clean_messages = [{"role": m["role"], "content": m["content"] or ""} for m in messages]
        prompt_msgs = extract_prompt_messages(clean_messages)

        if not prompt_msgs:
            skipped += 1
            continue

        # Parse annotations to get each model's response
        annotations = json.loads(row["annotations"]) if isinstance(row["annotations"], str) else row["annotations"]

        for annotation in annotations:
            ann_model = annotation["model"]
            ann_response = annotation["response"]

            if not ann_response:
                skipped += 1
                continue

            result = compute_completion_logprobs(
                model=model,
                tokenizer=tokenizer,
                prompt_messages=prompt_msgs,
                response_text=ann_response,
                max_length=args.max_length,
                device=device,
            )

            if result["num_tokens"] == 0:
                skipped += 1
                continue

            model_results[ann_model].append({
                "row_idx": i,
                "sum_logprob": result["sum_logprob"],
                "mean_logprob": result["mean_logprob"],
                "num_tokens": result["num_tokens"],
            })

    # Compute aggregate statistics
    print(f"\nProcessed {len(data)} rows ({skipped} skipped)")
    print("=" * 80)

    output = {"evaluator_model": args.model_path, "per_model_stats": {}}

    for model_name in sorted(model_results.keys()):
        results = model_results[model_name]
        mean_logprobs = [r["mean_logprob"] for r in results]
        sum_logprobs = [r["sum_logprob"] for r in results]
        num_tokens = [r["num_tokens"] for r in results]

        stats = {
            "mean_logprob": compute_statistics(mean_logprobs),
            "sum_logprob": compute_statistics(sum_logprobs),
            "num_tokens": compute_statistics(num_tokens),
        }
        output["per_model_stats"][model_name] = stats

        print(f"\n{model_name} (n={len(results)}):")
        print(f"  mean_logprob:  mean={stats['mean_logprob']['mean']:.4f}  std={stats['mean_logprob']['std']:.4f}  "
              f"median={stats['mean_logprob']['median']:.4f}  [{stats['mean_logprob']['min']:.4f}, {stats['mean_logprob']['max']:.4f}]")
        print(f"  sum_logprob:   mean={stats['sum_logprob']['mean']:.2f}  std={stats['sum_logprob']['std']:.2f}  "
              f"median={stats['sum_logprob']['median']:.2f}")
        print(f"  num_tokens:    mean={stats['num_tokens']['mean']:.1f}  std={stats['num_tokens']['std']:.1f}")

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
