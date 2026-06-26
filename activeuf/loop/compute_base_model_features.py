from argparse import ArgumentParser
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from functools import partial
import glob
import os
import re
import yaml

import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from rewarduq.models.reward_head_ensemble import (
    RewardHeadEnsembleModel,
    RewardHeadEnsembleModelConfig,
)

from activeuf.utils import set_seed

# accelerate launch --config_file=configs/accelerate/single_node.yaml -m activeuf.loop.compute_base_model_features --config_path configs/compute_base_model_features.yaml


def find_completed_checkpoints(out_dir: str, process_index: int) -> set[int]:
    """
    Scan the output directory for existing checkpoint files and return
    the set of batch indices that have been completed for this process.

    File naming: {process_index}-{batch_idx}.pt
    """
    pattern = os.path.join(out_dir, f"{process_index}-*.pt")
    completed = set()
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        match = re.match(rf"{process_index}-(\d+)\.pt", basename)
        if match:
            completed.add(int(match.group(1)))
    return completed


def collate_fn(batch: list[dict], tokenizer):
    """
    Collate function for dynamic padding per batch.
    Pads all sequences in the batch to the length of the longest sequence.
    """
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    temp_ids = [x["temp_id"] for x in batch]

    # Use tokenizer.pad to dynamically pad batch
    batch_inputs = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding="longest",  # pad to the longest sequence in this batch
        return_tensors="pt",
    )

    return temp_ids, batch_inputs


if __name__ == "__main__":
    accelerator = Accelerator()

    cli_parser = ArgumentParser()
    cli_parser.add_argument(
        "--config_path", required=True, help="Path to the YAML config"
    )
    config_path = cli_parser.parse_args().config_path
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    out_dir = f"{args['inputs_path'].rstrip('/')}-feature_partials"
    if accelerator.is_main_process:
        args_path = os.path.join(args["inputs_path"], "main.args")
        with open(args_path, "w") as f_out:
            print(yaml.dump(args), file=f_out)

        os.makedirs(out_dir, exist_ok=True)

    # Wait for main process to create directory before others check for existing files
    accelerator.wait_for_everyone()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args["seed"]:
        set_seed(args["seed"])

    model = RewardHeadEnsembleModel(
        RewardHeadEnsembleModelConfig(
            **{k: v for k, v in args["enn"]["model"].items() if not k.startswith("__")}
        )
    )
    model.eval()
    tokenizer = model.tokenizer
    feature_dim = model.base_model_.config.hidden_size
    model = accelerator.prepare(model)

    dataset = load_from_disk(args["inputs_path"])
    if args["debug"]:
        dataset = dataset.select(range(1000))
    n_response_texts_per_prompt = len(dataset[0]["completions"])

    n = len(dataset)
    dataset = dataset.add_column("prompt_idx", list(range(n)))

    per_proc = (n + accelerator.num_processes - 1) // accelerator.num_processes
    start_idx = accelerator.process_index * per_proc
    end_idx = min(start_idx + per_proc, n)
    _dataset = dataset.select(range(start_idx, end_idx))

    print("Pretokenizing everything")
    # rust tokenizer parallelism would conflict with the process pool below
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # all ranks run this at once, so split the CPUs fairly to avoid oversubscription
    n_tokenize_proc = max(
        1, len(os.sched_getaffinity(0)) // accelerator.num_processes
    )

    def tokenize_prompts(batch):
        temp_ids, all_input_ids, all_attention_mask, lengths = [], [], [], []
        for prompt_idx, prompt, completions in zip(
            batch["prompt_idx"], batch["prompt"], batch["completions"]
        ):
            messages = [
                prompt + [{"role": "assistant", "content": c["response_text"]}]
                for c in completions
            ]
            messages_str = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(
                messages_str,
                padding="do_not_pad",
                truncation=True,
                max_length=args["max_length"],
                return_tensors=None,
            )
            for completion_idx in range(len(completions)):
                ids = inputs["input_ids"][completion_idx]
                temp_ids.append((prompt_idx, completion_idx))
                all_input_ids.append(ids)
                all_attention_mask.append(inputs["attention_mask"][completion_idx])
                lengths.append(len(ids))
        return {
            "temp_id": temp_ids,
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "length": lengths,
        }

    _flattened_inputs = _dataset.map(
        tokenize_prompts,
        batched=True,
        num_proc=n_tokenize_proc,
        remove_columns=_dataset.column_names,
        desc="Pretokenizing",
    )
    # longest sequences first -> efficient dynamic padding per batch
    _flattened_inputs = _flattened_inputs.sort("length", reverse=True)
    _flattened_inputs = _flattened_inputs.remove_columns("length")

    dataloader = DataLoader(
        _flattened_inputs,
        batch_size=args["batch_size"],
        num_workers=0,
        drop_last=False,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )
    n_batches = len(dataloader)

    # Resume support: find already-completed checkpoints and skip those batches
    completed_checkpoints = find_completed_checkpoints(
        out_dir, accelerator.process_index
    )
    if completed_checkpoints:
        # Find the highest completed checkpoint batch index
        resume_from_batch = max(completed_checkpoints)
        print(
            f"[Rank {accelerator.process_index}] "
            f"Found {len(completed_checkpoints)} existing checkpoints, "
            f"resuming from batch {resume_from_batch + 1}",
            flush=True,
        )
    else:
        resume_from_batch = 0
        print(
            f"[Rank {accelerator.process_index}] "
            f"No existing checkpoints found, starting from scratch",
            flush=True,
        )

    temp_ids_local = []
    features_buffer_local = torch.empty(
        (args["n_batches_per_checkpoint"] * args["batch_size"], feature_dim),
        device=accelerator.device,
    )
    offset = 0

    start = time.time()
    n_batches_processed = 0
    # the loop below still iterates (and skips) already-done batches, so let the
    # bar tick through them from 0 rather than passing initial= (would double-count)
    progress = tqdm(
        dataloader,
        total=n_batches,
        desc=f"Featurizing (rank {accelerator.process_index})",
        disable=not accelerator.is_main_process,
    )
    for batch_idx, (temp_ids, inputs) in enumerate(progress, 1):
        # Skip already-completed batches
        if batch_idx <= resume_from_batch:
            continue

        n_batches_processed += 1
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model(output_only_features=True, **inputs)

        temp_ids_local.extend(temp_ids)
        features_buffer_local[offset : offset + len(features)] = features
        offset += len(features)

        if batch_idx % args["n_batches_per_checkpoint"] == 0 or batch_idx == n_batches:
            elapsed = time.time() - start
            avg_time_per_batch = (
                elapsed / n_batches_processed if n_batches_processed > 0 else 0
            )
            remaining_batches = n_batches - batch_idx
            eta = remaining_batches * avg_time_per_batch
            print(
                f"[Rank {accelerator.process_index}] "
                f"{batch_idx}/{n_batches} ({batch_idx / n_batches:.2%}) batches "
                f"done in {elapsed / 60:.1f} min, ETA ≈ {eta / 60:.1f} min",
                flush=True,
            )

            temp_path = os.path.join(
                out_dir, f"{accelerator.process_index}-{batch_idx}.pt"
            )
            torch.save(
                {
                    "temp_ids": temp_ids_local,
                    "features": features_buffer_local[:offset].cpu(),
                },
                temp_path,
            )
            temp_ids_local = []
            offset = 0
