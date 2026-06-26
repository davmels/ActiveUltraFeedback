import os

import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset, load_from_disk

from trl import (
    ModelConfig,
    TrlParser,
    get_peft_config,
)
from activeuf.utils import setup
from activeuf.dpo.trainer import NormedDPOTrainer, NormedDPOConfig

# On these CSCS nodes wandb's viewer query can return None, making
# query_with_timeout crash with a TypeError; tolerate a missing/non-str flags
# payload so wandb.init survives (mirrors activeuf/loop/run_dpo.py).
import json
import wandb
import wandb.sdk.lib.server

_orig_query_with_timeout = wandb.sdk.lib.server.Server.query_with_timeout


def _patched_query_with_timeout(self):
    try:
        _orig_query_with_timeout(self)
    except TypeError:
        flags = self._viewer.get("flags") if getattr(self, "_viewer", None) else None
        self._flags = json.loads(flags) if isinstance(flags, str) else {}


wandb.sdk.lib.server.Server.query_with_timeout = _patched_query_with_timeout


'''
example run command:
accelerate launch --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml --num_processes=4 -m activeuf.dpo.training --dataset_path=allenai/ultrafeedback_binarized_cleaned --config_path=$SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml
'''


@dataclass
class ScriptArguments:
    """
    Arguments for dataset paths and custom processing logic.
    """
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset (local path or HF Hub)."}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration name (if using HF Hub)."}
    )
    debug_mode: bool = field(
        default=False, metadata={"help": "Whether to run in debug mode with a smaller dataset."}
    )
    config_path: Optional[str] = field(
        default=None, metadata={"help": "Path to a DPO specific YAML config to overwrite training args."}
    )


def process_dataset_split(dataset):
    """Normalization to handle different dataset formats (train vs train_prefs)."""
    if isinstance(dataset, dict):
        if "train_prefs" in dataset:
            return dataset["train_prefs"]
        elif "train" in dataset:
            return dataset["train"]
        else:
            raise ValueError("Could not find 'train' or 'train_prefs' in dataset splits.")
    return dataset


def prepare_dataset_for_dpo(dataset, tokenizer, max_length, num_proc=os.cpu_count()):
    """
    Optimized filtering logic:
    1. Formats and tokenizes in a single map pass to calculate lengths.
    2. Filters based on calculated lengths.
    """
    def check_length_fn(examples):
        # Helper to apply template to a specific column key (chosen/rejected)
        def get_len(key):
            return [
                len(tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=False))
                for msg in examples[key]
            ]
        
        try:
            chosen_lens = get_len("chosen")
            rejected_lens = get_len("rejected")
        except Exception:
            # Fallback if columns are not standard list-of-dicts
            return [True] * len(examples["chosen"])

        # Create boolean mask: Keep if BOTH chosen and rejected fit in max_length
        return [
            (c <= max_length) and (r <= max_length) 
            for c, r in zip(chosen_lens, rejected_lens)
        ]

    print(f"Filtering dataset by max_length={max_length}...")
    original_len = len(dataset)
    
    dataset = dataset.filter(
        check_length_fn,
        batched=True,
        num_proc=num_proc
    )
    
    print(f"Filtered dataset: {original_len} -> {len(dataset)} samples.")
    return dataset


def main(script_args, training_args, model_args):
    # 1. Setup (ActiveUF)
    setup()
    os.environ["WANDB_ENTITY"] = "ActiveUF_Plus"
    os.environ["WANDB_PROJECT"] = "DPO"
    
    # 2. Set Seed (TRL/Transformers handles this via training_args, but we enforce explicit expectation)
    set_seed(training_args.seed)

    # 3. Load Model & Tokenizer
    # ModelConfig handles standard args like attn_implementation, trust_remote_code, etc.
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # Fix invalid generation config (temperature/top_p set with do_sample=False)
    if hasattr(model, "generation_config"):
        if not model.generation_config.do_sample:
            model.generation_config.temperature = None
            model.generation_config.top_p = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load & Process Dataset
    # We use main_process_first to ensure map/filter happens once and is cached correctly
    with training_args.main_process_first(desc="dataset loading and processing"):
        try:
            dataset = load_dataset(script_args.dataset_path, name=script_args.dataset_config)
        except Exception:
            dataset = load_from_disk(script_args.dataset_path)

        dataset = process_dataset_split(dataset)

        # Clean columns
        cols_to_keep = {"chosen", "rejected"}
        cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)

        # Debug trimming
        if script_args.debug_mode:
            print("Debug mode: trimming dataset to 100 samples.")
            dataset = dataset.select(range(min(100, len(dataset))))

        # Length filtering
        if training_args.max_length:
            dataset = prepare_dataset_for_dpo(
                dataset, 
                tokenizer, 
                max_length=training_args.max_length
            )
        
        # Shuffle
        dataset = dataset.shuffle(seed=training_args.seed)

    # 5. Initialize Trainer
    # get_peft_config automatically extracts LoRA params from ModelConfig/YAML if present
    peft_config = get_peft_config(model_args)

    trainer = NormedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train & Save
    trainer.train()
    
    print("Saving model to:", training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Model saved.")
    
    final_config = {
        "script_args": asdict(script_args),
        "model_args": asdict(model_args),
        "training_args": training_args.to_dict(), # training_args is a HF object, so we use .to_dict()
    }

    config_path = os.path.join(training_args.output_dir, "final_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False)


if __name__ == "__main__":
    # TrlParser parses arguments into the specified dataclasses
    parser = TrlParser((ScriptArguments, NormedDPOConfig, ModelConfig))
    
    script_args, training_args, model_args = parser.parse_args_and_config()

    # --- ADDED LOGIC FOR YAML OVERWRITE ---
    if script_args.config_path is not None:
        print(f"Loading/Overwriting configuration from: {script_args.config_path}")
        with open(script_args.config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        # Check if there is a 'training' key in the yaml (based on your dpo_training.yaml structure)
        # If your yaml is flat, use yaml_config directly.
        # Based on your attachment: 'training' key exists.
        training_overrides = yaml_config.get("training", {})
        
        for key, value in training_overrides.items():
            if hasattr(training_args, key):
                old_value = getattr(training_args, key)
                setattr(training_args, key, value)
                print(f"  Overwriting training_args.{key}: {old_value} -> {value}")
            else:
                print(f"  Warning: Key '{key}' in YAML not found in NormedDPOConfig, skipping.")
                
        # Also optional: Overwrite model_name_or_path if it exists in the root of yaml
        if "model_path" in yaml_config:
            old_value = model_args.model_name_or_path
            model_args.model_name_or_path = yaml_config["model_path"]
            print(f"  Overwriting model_args.model_name_or_path: {old_value} -> {model_args.model_name_or_path}")
    # --------------------------------------

    # Print args for verification (similar to the SFT script)
    print("Script Arguments:", script_args)
    print("Training Arguments:", training_args)
    print("Model Arguments:", model_args)

    main(script_args, training_args, model_args)