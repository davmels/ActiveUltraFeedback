#!/usr/bin/env python3
"""
Generate AlpacaEval completions using vLLM (offline batch inference).

This script generates model completions for the AlpacaEval benchmark without
running annotation/judging. The outputs are saved in the format expected by
`alpaca_eval evaluate`.

Usage:
    python generate_alpaca_completions.py \
        --model_path /path/to/model \
        --output_path /path/to/model_outputs.json \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization 0.9 \
        --max_model_len 4096 \
        --max_new_tokens 2048
"""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AlpacaEval completions using vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save model_outputs.json")
    parser.add_argument("--generator_name", type=str, default=None,
                        help="Generator name for outputs (default: basename of model_path)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--hf_home", type=str, default=None, help="HF_HOME / download dir for datasets")
    return parser.parse_args()


def main():
    args = parse_args()

    generator_name = args.generator_name or args.model_path.rstrip("/").split("/")[-1]
    logger.info(f"Generator name: {generator_name}")
    logger.info(f"Model path: {args.model_path}")

    # ---- Load AlpacaEval dataset ----
    logger.info("Loading AlpacaEval dataset...")
    import datasets
    ds = datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
    )["eval"]
    logger.info(f"Loaded {len(ds)} instructions")

    # ---- Build prompts using the model's chat template ----
    logger.info("Building prompts with tokenizer chat template...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    prompts = []
    for ex in ds:
        messages = [{"role": "user", "content": ex["instruction"]}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)
    logger.info(f"Built {len(prompts)} prompts (first 200 chars of prompt[0]: {prompts[0][:200]})")

    # ---- Generate with vLLM ----
    logger.info("Loading vLLM model...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    logger.info("Generating completions...")
    outputs = llm.generate(prompts, sampling_params)
    logger.info(f"Generated {len(outputs)} completions")

    # ---- Build output JSON ----
    results = []
    for ex, out in zip(ds, outputs):
        results.append({
            "dataset": ex.get("dataset", ""),
            "instruction": ex["instruction"],
            "output": out.outputs[0].text.strip(),
            "generator": generator_name,
        })

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(results)} completions to {args.output_path}")

    # ---- Explicitly delete the LLM to free GPU memory ----
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Done. GPU memory released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
