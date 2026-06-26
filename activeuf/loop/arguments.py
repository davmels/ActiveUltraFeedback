import argparse
from dataclasses import dataclass, field
import os
import os.path as path
from transformers import HfArgumentParser
import yaml

from activeuf.acquisition_function.arguments import (
    RandomConfig,
    UltraFeedbackConfig,
    DTSConfig,
    IDSConfig,
    RUCBConfig,
    InfoGainConfig,
    MaxMinLCBConfig,
    DRTSConfig,
    DeltaUCBConfig,
    DeltaQuantileConfig,
    OracleMaxMinConfig,
)
from activeuf.utils import ensure_dataclass


@dataclass
class AcquisitionFunctionConfig:
    beta: float = field(
        default=1.0,
        metadata={
            "help": "Global Beta parameter applied to all relevant acquisition functions."
        },
    )

    # Specific Configs with default_factory
    random: RandomConfig = field(default_factory=RandomConfig)
    ultrafeedback: UltraFeedbackConfig = field(default_factory=UltraFeedbackConfig)

    dts: DTSConfig = field(default_factory=DTSConfig)
    infogain: InfoGainConfig = field(default_factory=InfoGainConfig)
    ids: IDSConfig = field(default_factory=IDSConfig)
    rucb: RUCBConfig = field(default_factory=RUCBConfig)
    maxminlcb: MaxMinLCBConfig = field(default_factory=MaxMinLCBConfig)
    drts: DRTSConfig = field(default_factory=DRTSConfig)
    deltaucb: DeltaUCBConfig = field(default_factory=DeltaUCBConfig)
    deltaquantile: DeltaQuantileConfig = field(default_factory=DeltaQuantileConfig)
    oracle_maxmin: OracleMaxMinConfig = field(default_factory=OracleMaxMinConfig)


@dataclass
class ENNModelConfig:
    base_model_name_or_path: str = field(
        metadata={"help": "Name or path of the base model."}
    )
    num_heads: int = field(
        metadata={"help": "The number of MLPs in the ensemble head."}
    )
    head_num_layers: int = field(
        metadata={"help": "The number of layers in each MLP of the ensemble head."}
    )
    head_hidden_dim: int = field(
        metadata={
            "help": "The dimension of the hidden layers in each MLP of the ensemble head."
        }
    )
    freeze_base_model: bool = field(
        metadata={"help": "Whether to freeze the base model during training."}
    )
    feature_extraction_layer: str = field(
        metadata={"help": "Which layer to use for feature extraction."}
    )
    head_initialization_xavier_gain: float = field(
        metadata={"help": "Xavier gain for weight initialization."}
    )


@dataclass
class ENNTrainerConfig:
    warmup_ratio: float = field()
    lr_scheduler_type: str = field(
        metadata={
            "help": "Type of learning rate scheduler.",
            "choices": ["constant", "linear", "cosine"],
        }
    )
    learning_rate: float = field(metadata={"help": "Initial learning rate."})
    num_train_epochs: int = field(
        metadata={"help": "Number of training epochs per outer batch added."}
    )
    regularization_towards_initial_weights: float = field(
        metadata={"help": "Initial regularization strength"}
    )
    max_length: int = field(metadata={"help": "Maximum sequence length."})
    center_rewards_coefficient: float | None = field(
        metadata={
            "help": "Coefficient to incentivize the reward model to output mean-zero rewards"
        }
    )
    precompute_features: bool = field()
    bf16: bool = field(metadata={"help": "Whether to use bfloat16 precision."})
    disable_tqdm: bool = field(metadata={"help": "Disable tqdm progress bars."})
    report_to: str = field(metadata={"help": "Reporting tool for the trainer."})
    save_strategy: str = field(metadata={"help": "Strategy for saving checkpoints."})
    save_steps: int = field(metadata={"help": "Number of steps between each save"})
    logging_strategy: str = field(metadata={"help": "Strategy for logging."})
    logging_steps: int = field(metadata={"help": "Number of steps between each log"})
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Where to save checkpoints."},
    )


@dataclass
class ENNRegularizationConfig:
    initial_value: float = field(
        metadata={"help": "Strength of regularization towards initial weights."}
    )
    decay_type: str = field(
        metadata={
            "help": "Type of decay for regularization.",
            "choices": ["linear", "exponential"],
        }
    )
    exponential_decay_base: float = field(
        metadata={"help": "Base for exponential decay regularization."}
    )
    exponential_decay_scaler: float = field(
        metadata={"help": "Scaler for exponent in exponential decay regularization."}
    )


@dataclass
class ENNConfig:
    previous_checkpoint_path: str | None = field(
        metadata={
            "help": "Path to a previous checkpoint if resuming training.",
        }
    )
    effective_batch_size: int = field(
        metadata={"help": "Effective batch size for training ENN."}
    )
    inference_batch_size: int = field(
        metadata={"help": "Number of completions per reward forward pass."}
    )

    model: ENNModelConfig = field(metadata={"help": "Configuration for the ENN model."})
    trainer: ENNTrainerConfig = field(
        metadata={"help": "Trainer configuration for ENN."}
    )
    regularization: ENNRegularizationConfig = field(
        metadata={"help": "Regularization settings for ENN."}
    )
    max_steps: int = field(
        metadata={"help": "Maximum number of training steps per outer batch added."}
    )


@dataclass
class BLHModelConfig:
    base_model_name_or_path: str = field(
        metadata={"help": "Name or path of the base model."}
    )
    lambda_reg: float = field(
        metadata={"help": "L2 regularization / Hessian prior precision."}
    )
    std_beta: float = field(
        metadata={"help": "Multiplier for std in uncertainty bounds."}
    )
    feature_extraction_layer: str = field(
        metadata={"help": "Which layer to use for feature extraction."}
    )
    feature_extraction_pooling_strategy: str = field(
        default="last",
        metadata={"help": "Pooling strategy for feature extraction."},
    )


@dataclass
class BLHTrainerConfig:
    warmup_ratio: float = field()
    lr_scheduler_type: str = field(
        metadata={
            "help": "Type of learning rate scheduler.",
            "choices": ["constant", "linear", "cosine"],
        }
    )
    learning_rate: float = field(metadata={"help": "Initial learning rate."})
    num_train_epochs: int = field(
        metadata={"help": "Number of training epochs per outer batch added."}
    )
    l2_reg: float = field(
        metadata={"help": "L2 regularization strength for the head weights."}
    )
    max_length: int = field(metadata={"help": "Maximum sequence length."})
    center_rewards_coefficient: float | None = field(
        metadata={
            "help": "Coefficient to incentivize the reward model to output mean-zero rewards"
        }
    )
    precompute_features: bool = field()
    bf16: bool = field(metadata={"help": "Whether to use bfloat16 precision."})
    disable_tqdm: bool = field(metadata={"help": "Disable tqdm progress bars."})
    report_to: str = field(metadata={"help": "Reporting tool for the trainer."})
    save_strategy: str = field(metadata={"help": "Strategy for saving checkpoints."})
    save_steps: int = field(metadata={"help": "Number of steps between each save"})
    logging_strategy: str = field(metadata={"help": "Strategy for logging."})
    logging_steps: int = field(metadata={"help": "Number of steps between each log"})
    final_hessian_mode: str = field(
        default="unweighted",
        metadata={
            "help": "How to compute the post-training Hessian.",
            "choices": ["unweighted", "weighted"],
        },
    )
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Where to save checkpoints."},
    )


@dataclass
class BLHRegularizationConfig:
    initial_value: float = field(
        metadata={"help": "Initial L2 regularization / lambda_reg value."}
    )
    decay_type: str = field(
        metadata={
            "help": "Type of decay for regularization.",
            "choices": ["linear", "exponential"],
        }
    )
    exponential_decay_base: float = field(
        metadata={"help": "Base for exponential decay regularization."}
    )
    exponential_decay_scaler: float = field(
        metadata={"help": "Scaler for exponent in exponential decay regularization."}
    )


@dataclass
class BLHConfig:
    previous_checkpoint_path: str | None = field(
        metadata={
            "help": "Path to a previous checkpoint if resuming training.",
        }
    )
    effective_batch_size: int = field(
        metadata={"help": "Effective batch size for training BLH."}
    )
    inference_batch_size: int = field(
        metadata={"help": "Number of completions per reward forward pass."}
    )

    model: BLHModelConfig = field(metadata={"help": "Configuration for the BLH model."})
    trainer: BLHTrainerConfig = field(
        metadata={"help": "Trainer configuration for BLH."}
    )
    max_steps: int = field(
        metadata={"help": "Maximum number of training steps per outer batch added."}
    )
    regularization: BLHRegularizationConfig | None = field(
        default=None,
        metadata={"help": "Regularization settings for BLH (unused, kept for backward compat)."},
    )


@dataclass
class LoopConfig:
    # dataset-related configs
    inputs_path: str = field(
        metadata={"help": "Path to the dataset with prompts and response texts."}
    )
    oracle_name: str = field(
        metadata={
            "help": "Oracle scorer for response texts.",
            "choices": ["random", "ultrafeedback"],
        }
    )
    acquisition_function_type: str = field(
        metadata={
            "help": "Acquisition function type",
            "choices": [
                "random",
                "ultrafeedback",
                "dts",
                "ids",
                "rucb",
                "maxminlcb",
                "infogain",
                "infomax",
                "oracle_maxmin",
            ],
        }
    )
    reward_model_type: str = field(
        metadata={
            "help": "Reward model to train.",
            "choices": ["none", "enn", "blh", "static"],
        }
    )

    # global configs
    seed: int = field(metadata={"help": "Random seed for reproducibility."})
    max_length: int = field(metadata={"help": "Max length for all sequences."})
    debug: bool = field(
        metadata={"help": "Set True when debugging the script for speed."}
    )
    outer_loop_batch_size: int = field(
        metadata={"help": "Number of prompts per outer loop batch."}
    )
    save_every_n_outer_batches: int = field(
        metadata={"help": "Save dataset every N outer loop batches."}
    )
    replay_buffer_factor: int = field(
        metadata={
            "help": "Replay buffer for reward model training will contain up to (replay_buffer_factor * outer_loop_batch_size) samples."
        }
    )

    # active learning-related configs
    acquisition_function: AcquisitionFunctionConfig = field(
        metadata={"help": "Configs for acquisition functions."}
    )
    enn: ENNConfig | None = field(
        metadata={"help": "All configs related to ENN reward model and training."}
    )
    blh: BLHConfig | None = field(
        default=None,
        metadata={"help": "All configs related to BLH reward model and training."},
    )
    
    # dataset filtering
    min_non_truncated: int = field(
        default=2,
        metadata={
            "help": "Drop prompts with fewer than this many non-truncated completions "
                    "(needs the `truncated` field). Must be >=2 to form a chosen/rejected pair."
        },
    )

    # direct max/min baseline
    direct_maxmin: bool = field(
        default=False,
        metadata={
            "help": "Direct max/min baseline. When set, the reward used per completion is just "
                    "the oracle overall_score (no reward-model forward passes, no RM training), "
                    "and each prompt's chosen/rejected pair is the max/min non-truncated completion "
                    "by score gap. Implemented by forcing reward_model_type=static and "
                    "acquisition_function_type=oracle_maxmin, so it rides on the existing loop. "
                    "Toggle off to restore the configured reward model / acquisition function."
        },
    )

    # prompt selection
    prompt_selection_K: int | None = field(
        default=None,
        metadata={
            "help": "Number of prompts to select per iteration from the batch of outer_loop_batch_size (L). "
                    "When set, acquisition functions score and select the top K prompts before choosing completion pairs. "
                    "Unselected prompts return to the pool. None means select all (original behavior)."
        },
    )
    domain_quota_selection: bool = field(
        default=False,
        metadata={
            "help": "When selecting K prompts from L, cap each Olmo3 domain at floor(K * pool_fraction) "
                    "on a first ranked (top->bottom) pass, then random-fill any shortfall from the skipped "
                    "(over-quota) prompts. pool_fraction is the domain's share of the filtered selection pool. "
                    "Requires prompt_selection_K and the deltaucb acquisition function."
        },
    )

    # checkpointing
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to a directory containing a checkpoint to resume from (e.g., outputs/run_id/checkpoint-50)."}
    )
    run_tag: str | None = field(
        default=None,
        metadata={"help": "Custom tag to append to run ID (e.g. 'q_0.05')."}
    )
    run_id: str = field(
        default="",
        metadata={"help": "Custom run ID (overrides automatic generation)."},
    )

    # derived fields
    env_local_path: str = ""
    timestamp: str = ""
    output_path: str = ""
    args_path: str = ""
    logs_path: str = ""
    wandb_project: str = ""
    wandb_dir: str = ""


def extract_annotator_name(dataset_path: str) -> str:
    for key in ["llama", "qwen"]:
        if key in path.basename(dataset_path):
            return key
    return "unknown"


def recursive_update(base_dict, new_dict):
    """
    Recursively update a dictionary, handling nested dictionaries and dot-notation keys.
    Values in new_dict will overwrite values in base_dict.
    """
    for key, value in new_dict.items():
        if value is None and key not in base_dict:
            continue

        if "." in key:
            keys = key.split(".")

            leaf_dict = base_dict
            for k in keys[:-1]:
                leaf_dict = leaf_dict[k]

            leaf_dict[keys[-1]] = value
        else:
            base_dict[key] = value

    return base_dict


def parse_overwrites(remaining_args) -> dict:
    overwrite_dict = {}

    for arg in remaining_args:
        if not arg.startswith("--"):
            continue
        arg = arg.lstrip("-")

        key_value_pair = arg.split("=", 1)
        if len(key_value_pair) != 2:
            raise ValueError(f"Invalid argument format: {arg}")

        key, value = key_value_pair
        value = yaml.safe_load(value)

        overwrite_dict[key] = value

        # Special case certain keys as they involve variables in the config
        if key in [
            "enn.regularization.initial_value",
            "enn.trainer.regularization_towards_initial_weights",
        ]:
            overwrite_dict["enn.regularization.initial_value"] = value
            overwrite_dict["enn.trainer.regularization_towards_initial_weights"] = value
        elif key in ["enn.max_steps", "enn.trainer.max_length"]:
            overwrite_dict["enn.max_steps"] = value
            overwrite_dict["enn.trainer.max_length"] = value
        elif key in [
            "blh.trainer.l2_reg",
            "blh.model.lambda_reg",
        ]:
            overwrite_dict["blh.trainer.l2_reg"] = value
            overwrite_dict["blh.model.lambda_reg"] = value
        elif key in ["blh.max_steps", "blh.trainer.max_length"]:
            overwrite_dict["blh.max_steps"] = value
            overwrite_dict["blh.trainer.max_length"] = value

    return overwrite_dict


def get_loop_args(timestamp) -> argparse.Namespace:
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument(
        "--config_path", required=True, help="Path to the YAML config"
    )
    config_args, remaining_args = cli_parser.parse_known_args()
    config_path = config_args.config_path
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if remaining_args:
        sweep_dict = parse_overwrites(remaining_args)
        config_dict = recursive_update(config_dict, sweep_dict)

    acq_config = config_dict.get("acquisition_function", {})
    global_beta = acq_config.get("beta", 1.0)

    beta_dependent_keys = [
        "dts",
        "drts",
        "infogain",
        "rucb",
        "maxminlcb",
        "deltaucb",
        "deltaquantile",
    ]

    # Inject beta into sub-dictionaries
    for key in beta_dependent_keys:
        # Ensure the sub-dict exists
        if key not in acq_config or acq_config[key] is None:
            acq_config[key] = {}

        # Overwrite/Set beta
        acq_config[key]["beta"] = global_beta

    # Update the main config dict
    config_dict["acquisition_function"] = acq_config

    # define timestamp, then use it to create a run id (env var takes precedence over config)
    config_dict["timestamp"] = timestamp
    if not os.getenv("WANDB_RUN_ID"):
        obs = config_dict.get("outer_loop_batch_size", "")
        pk = config_dict.get("prompt_selection_K")
        size_tag = f"L{obs}_K{pk}" if pk else f"L{obs}"
        if pk and config_dict.get("domain_quota_selection"):
            size_tag += "_domquota"
        config_dict["run_id"] = "_".join(
            [
                config_dict["acquisition_function_type"],
                config_dict["reward_model_type"],
                extract_annotator_name(config_dict["inputs_path"]),
                config_dict["oracle_name"],
                size_tag,
                config_dict["timestamp"],
            ]
        )
    else:
        config_dict["run_id"] = os.getenv("WANDB_RUN_ID")

    # setup paths
    if os.getenv("WANDB_SWEEP_ID"):
        config_dict["base_output_dir"] = path.join(
            config_dict["base_output_dir"], os.getenv("WANDB_SWEEP_ID")
        )
    config_dict["output_path"] = path.join(
        config_dict["base_output_dir"], config_dict["run_id"]
    )

    # If base_logs_dir is empty, use base_output_dir instead
    if not config_dict.get("base_logs_dir", "").strip():
        config_dict["base_logs_dir"] = config_dict["output_path"]
    config_dict["args_path"] = path.join(
        config_dict["base_logs_dir"], f"{config_dict['run_id']}.args"
    )
    config_dict["logs_path"] = path.join(
        config_dict["base_logs_dir"], f"{config_dict['run_id']}.log"
    )

    if config_dict["reward_model_type"] != "none":
        config_dict["wandb_project"] = config_dict["base_wandb_project"]
        config_dict["wandb_dir"] = path.join(
            config_dict["base_wandb_dir"], config_dict["run_id"]
        )

        if config_dict["reward_model_type"] != "static":
            trainer_args = config_dict[config_dict["reward_model_type"]]["trainer"]
            if not trainer_args.get("output_dir"):
                trainer_args["output_dir"] = (
                    f"{config_dict['base_trainer_dir']}/{config_dict['run_id']}"
                )

    parser = HfArgumentParser(LoopConfig)
    args = parser.parse_dict(config_dict, allow_extra_keys=True)[0]
    args = ensure_dataclass(LoopConfig, vars(args))

    return args
