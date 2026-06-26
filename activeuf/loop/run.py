from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from collections import Counter, deque
from dataclasses import asdict
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
import json
import os
import random
import numpy as np
import time
import torch
import gc
from torch.utils.data import DataLoader
import wandb
import wandb.sdk.lib.server

# On these nodes wandb's viewer query can return None, making query_with_timeout
# crash with a TypeError; tolerate a missing/non-str flags payload so init survives.
_orig_query_with_timeout = wandb.sdk.lib.server.Server.query_with_timeout


def _patched_query_with_timeout(self):
    try:
        _orig_query_with_timeout(self)
    except TypeError:
        flags = self._viewer.get("flags") if getattr(self, "_viewer", None) else None
        self._flags = json.loads(flags) if isinstance(flags, str) else {}


wandb.sdk.lib.server.Server.query_with_timeout = _patched_query_with_timeout

from activeuf.loop.utils import save_loop_checkpoint, load_loop_checkpoint

from activeuf.acquisition_function import init_acquisition_function
from activeuf.domains import get_category
from activeuf.loop.arguments import get_loop_args
from activeuf.loop import utils as loop_utils
from activeuf.oracle.oracles import init_oracle
from activeuf.utils import (
    get_logger,
    get_timestamp,
    set_seed,
    convert_dataclass_instance_to_yaml_str,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


FEATURES_MEMMAP = None
SHUFFLE_PERM = None


def build_truncation_mask(samples):
    """Build a boolean mask (True = truncated) from the truncated field in completions."""
    n = len(samples)
    n_comp = len(samples[0]["completions"])
    mask = torch.zeros(n, n_comp, dtype=torch.bool)
    for i, s in enumerate(samples):
        for j, c in enumerate(s["completions"]):
            if c.get("truncated", 0):
                mask[i, j] = True
    return mask


def mask_rewards_for_truncated(rewards, trunc_mask):
    """Set rewards to -inf/+inf for truncated completions so acquisition functions ignore them.

    rewards: (n_prompts, n_completions, 3) where channels are (mean, lower, upper)
    trunc_mask: (n_prompts, n_completions) boolean, True = truncated
    """
    rewards = rewards.clone()
    # (mean, lower, upper) = (-inf, +inf, -inf) for truncated completions so they're
    # never picked. Assign the whole channel triple via the 2D mask in one shot --
    # mixing a 2D bool mask with an int index (rewards[trunc_mask, 0]) misparses on
    # some torch versions and raises a shape-mismatch error.
    fill = torch.tensor(
        [float("-inf"), float("inf"), float("-inf")],
        dtype=rewards.dtype, device=rewards.device,
    )
    rewards[trunc_mask] = fill
    return rewards


def dataset_select(dataset, indices):
    """Select rows from HF Dataset by indices, attaching memmap features if available."""
    batch_dict = dataset[indices]
    cols = list(batch_dict.keys())
    samples = [{col: batch_dict[col][i] for col in cols} for i in range(len(indices))]
    for sample, idx in zip(samples, indices):
        sample["original_index"] = SHUFFLE_PERM[idx]
        if FEATURES_MEMMAP is not None:
            sample["features"] = FEATURES_MEMMAP[SHUFFLE_PERM[idx]]
    return samples


# RUN
# accelerate launch --config_file=configs/accelerate/single_node.yaml -m activeuf.loop.run --config_path configs/loop.yaml

if __name__ == "__main__":
    accelerator = Accelerator()
    n_processes = accelerator.num_processes
    print("number of processes:", n_processes)

    # prepare (and export) args
    if accelerator.is_main_process:
        timestamp = get_timestamp(more_detailed=True)
    else:
        timestamp = ""
    timestamp = broadcast_object_list([timestamp])[0]
    args = get_loop_args(timestamp)
    if args.direct_maxmin:
        # Direct max/min baseline: reward == oracle overall_score (no RM forward
        # passes, no RM training), pick the max/min non-truncated pair per prompt.
        # Reuse the existing static-reward + oracle_maxmin code paths so this is a
        # single, easily-cancellable switch on top of the normal loop.
        args.reward_model_type = "static"
        args.acquisition_function_type = "oracle_maxmin"
    try:
        acquisition_function_args = asdict(
            getattr(args.acquisition_function, args.acquisition_function_type)
        )
    except Exception:
        acquisition_function_args = {}
    if hasattr(args, args.reward_model_type):
        reward_args = getattr(args, args.reward_model_type)
    else:
        reward_args = None
    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
        os.makedirs(os.path.dirname(args.args_path), exist_ok=True)
        with open(args.args_path, "w") as f_out:
            print(convert_dataclass_instance_to_yaml_str(args), file=f_out)

    # env setup
    load_dotenv(args.env_local_path)
    logger = get_logger(__name__, args.logs_path, accelerator)
    logger.info = loop_utils.main_process_only(logger.info, accelerator)

    if args.direct_maxmin:
        logger.info(
            "direct_maxmin=True: forcing reward_model_type=static and "
            "acquisition_function_type=oracle_maxmin (no RM forward passes / training)."
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.seed:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("Logging configuration: ")
        logger.info(convert_dataclass_instance_to_yaml_str(args))

    # --- STATE INITIALIZATION (Start Fresh or Resume) ---
    start_outer_batch_idx = 0
    output = []
    _effective_bs = args.prompt_selection_K if args.prompt_selection_K else args.outer_loop_batch_size
    replay_buffer = deque(maxlen=_effective_bs * args.replay_buffer_factor)
    resumed_wandb_id = None

    loaded_remaining_indices = None

    if args.resume_from_checkpoint is not None:
        logger.info(f"Attempting to resume execution from {args.resume_from_checkpoint}")

        if accelerator.is_main_process:
            loaded_state, loaded_buffer, loaded_output, loaded_rng_states, loaded_remaining_indices = load_loop_checkpoint(args.resume_from_checkpoint)

            start_outer_batch_idx = loaded_state["next_outer_batch_idx"]
            resumed_wandb_id = loaded_state.get("wandb_run_id")
            output = loaded_output
            replay_buffer = loaded_buffer

            logger.info(f"Resumed state: Starting at Batch {start_outer_batch_idx}")
        else:
            loaded_rng_states = None
            loaded_remaining_indices = None

        sync_list = [start_outer_batch_idx, replay_buffer, loaded_rng_states, loaded_remaining_indices]
        sync_list = broadcast_object_list(sync_list)

        start_outer_batch_idx = sync_list[0]
        replay_buffer = sync_list[1]
        rng_states = sync_list[2]
        loaded_remaining_indices = sync_list[3]

        if reward_args:
            reward_args.previous_checkpoint_path = args.resume_from_checkpoint

    if accelerator.is_main_process:
        os.environ.setdefault("WANDB_DIR", args.wandb_dir)
        
        # Use resumed ID if we have one, otherwise use the fresh run_id from args
        current_run_id = resumed_wandb_id if resumed_wandb_id else args.run_id
        resume_mode = "allow" if resumed_wandb_id else None

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=os.environ.get("WANDB_ENTITY", "ActiveUF_Plus"),
            id=current_run_id,
            config=vars(args),
            resume=resume_mode,
        )

        # Store environment variables for use in later scripts
        try:
            path = f"./.tmp/loop_vars_{os.getenv('SLURM_JOB_ID', '')}.sh"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"export LOOP_WANDB_RUN_ID='{current_run_id}'\n")
                f.write(f"export LOOP_DATASET_PATH='{args.output_path}'\n")
            logger.info(f"Successfully wrote env vars to {path}")
        except Exception as e:
            logger.error(f"Failed to write env vars to {path}: {e}")

    logger.info(f"Preparing acquisition function ({args.acquisition_function_type})")
    acquisition_function = init_acquisition_function(
        args.acquisition_function_type, **acquisition_function_args
    )

    logger.info(f"Preparing oracle ({args.oracle_name})")
    oracle = init_oracle(args.oracle_name)

    start = time.time()
    logger.info(f"Loading prompts from {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    # True on-disk row count: features.npy is aligned to this (before any debug truncation).
    full_num_rows = len(dataset)
    if args.debug:
        dataset = dataset.select(range(1001))

    # Filter rows that don't have at least 2 non-truncated completions
    original_num = len(dataset)
    has_truncated_field = "truncated" in dataset[0]["completions"][0]
    original_keep_indices = None
    if has_truncated_field:
        min_keep = args.min_non_truncated
        keep_mask = [
            sum(1 for c in row if not c.get("truncated", 0)) >= min_keep
            for row in dataset["completions"]
        ]
        original_keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
        dataset = dataset.select(original_keep_indices)
        logger.info(
            f"Filtered {original_num - len(dataset)} rows with <{min_keep} non-truncated completions "
            f"({len(dataset)}/{original_num} remaining)"
        )

    # Generate shuffle permutation so we can apply it to both dataset and memmap.
    # direct_maxmin keeps the original (post-filter) order so the output dataset
    # matches a plain map-based maxmin build (e.g. construct_maxmin_from_completions.py)
    # row-for-row, instead of the shuffled active-learning order.
    num_samples = len(dataset)
    if args.direct_maxmin:
        shuffle_perm = list(range(num_samples))
    else:
        rng = np.random.RandomState(args.seed)
        shuffle_perm = rng.permutation(num_samples).tolist()
    dataset = dataset.select(shuffle_perm)
    logger.info(f"# Prompts: {num_samples}")

    # Load features memmap if available
    # Compose mapping: shuffled index -> original (pre-filter) index for memmap lookup
    if original_keep_indices is not None:
        SHUFFLE_PERM = [original_keep_indices[shuffle_perm[i]] for i in range(num_samples)]
    else:
        SHUFFLE_PERM = shuffle_perm
    features_path = os.path.join(args.inputs_path, "features.npy")
    if os.path.exists(features_path):
        n_comp = len(dataset[0]["completions"])
        total_floats = os.path.getsize(features_path) // 4
        # memmap rows = full on-disk count (NOT original_num, which shrinks under --debug)
        memmap_num_rows = full_num_rows
        feat_dim = total_floats // (memmap_num_rows * n_comp)
        FEATURES_MEMMAP = np.memmap(
            features_path, dtype=np.float32, mode="r",
            shape=(memmap_num_rows, n_comp, feat_dim),
        )
        logger.info(f"Loaded features memmap: {FEATURES_MEMMAP.shape} from {features_path}")
    else:
        logger.info("No features memmap found, will use inline features from dataset")

    # Drop features column from HF Dataset (features come from memmap)
    if "features" in dataset.column_names and FEATURES_MEMMAP is not None:
        dataset = dataset.remove_columns(["features"])
    dataset_len = len(dataset)
    logger.info(f"Dataset loading took {time.time() - start:.2f}s")

    logger.info(
        f"Preparing reward model, tokenizer, and trainer ({args.reward_model_type})"
    )
    model, trainer = loop_utils.init_model_trainer(
        args.reward_model_type, reward_args, n_processes
    )
    if accelerator.is_main_process and trainer is not None:
        trainer.add_callback(loop_utils.WandbStepLoggerCallback(accelerator))
    tokenizer = model.tokenizer if model is not None else None

    if args.resume_from_checkpoint and trainer is not None:
        opt_path = os.path.join(args.resume_from_checkpoint, "optimizer.pt")
        sch_path = os.path.join(args.resume_from_checkpoint, "scheduler.pt")
        
        if os.path.exists(opt_path):
            logger.info(f"Restoring optimizer state from {opt_path}")
            if trainer.optimizer is None:
                trainer.create_optimizer()
            
            trainer.optimizer.load_state_dict(torch.load(opt_path, map_location=accelerator.device))
        
        if os.path.exists(sch_path):
            logger.info(f"Restoring scheduler state from {sch_path}")
            if trainer.lr_scheduler is None:
                _bs = args.prompt_selection_K if args.prompt_selection_K else args.outer_loop_batch_size
                num_outer_batches = (dataset_len + _bs - 1) // _bs
                total_steps = num_outer_batches * reward_args.max_steps
                
                trainer.create_scheduler(num_training_steps=total_steps)
            
            trainer.lr_scheduler.load_state_dict(torch.load(sch_path, map_location=accelerator.device))
        
        if rng_states is not None:
            
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            
            if rng_states["torch_cuda"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
                except Exception as e:
                    logger.warning(f"Could not restore all CUDA RNG states: {e}. Restoring current device only.")
                    if isinstance(rng_states["torch_cuda"], list) and len(rng_states["torch_cuda"]) > 0:
                        torch.cuda.set_rng_state(rng_states["torch_cuda"][0])

    expected_output_size = dataset_len
    use_prompt_selection = args.prompt_selection_K is not None
    inference_bs = reward_args.inference_batch_size if reward_args else 32

    step_times = []
    loop_start_time = time.time()
    logger.info("Starting dataset generation loop")

    if use_prompt_selection:
        # --- PROMPT SELECTION LOOP ---
        if args.resume_from_checkpoint is not None and loaded_remaining_indices is not None:
            remaining_indices = loaded_remaining_indices
            logger.info(f"Restored {len(remaining_indices)} remaining indices from checkpoint")
        else:
            remaining_indices = list(range(dataset_len))

        logger.info(
            f"Prompt selection: pool={len(remaining_indices)}, "
            f"L={args.outer_loop_batch_size}, K={args.prompt_selection_K}"
        )

        # Domain-quota selection: fraction of each Olmo3 domain in the filtered pool.
        domain_fractions = None
        if args.domain_quota_selection:
            pool_counts = Counter(get_category(pid) for pid in dataset["prompt_id"])
            pool_total = sum(pool_counts.values())
            domain_fractions = {d: c / pool_total for d, c in pool_counts.items()}
            logger.info(
                f"Domain-quota selection ON. Pool fractions (K={args.prompt_selection_K} quotas): "
                + ", ".join(
                    f"{d}={frac:.3f}(q{int(args.prompt_selection_K * frac)})"
                    for d, frac in sorted(domain_fractions.items(), key=lambda kv: -kv[1])
                )
            )

        total_iterations = -(-len(remaining_indices) // args.prompt_selection_K)
        outer_batch_idx = start_outer_batch_idx

        while remaining_indices:
            step_start = time.time()

            if model is not None:
                model.eval()

            L = min(args.outer_loop_batch_size, len(remaining_indices))
            effective_K = min(args.prompt_selection_K, L)

            if accelerator.is_main_process:
                logger.info(
                    f"Step {outer_batch_idx + 1} / ~{total_iterations} | Pool: {len(remaining_indices)} | L={L} K={effective_K}"
                )

            if accelerator.is_main_process:
                batch_indices = random.sample(remaining_indices, L)
            else:
                batch_indices = None
            batch_indices = broadcast_object_list([batch_indices])[0]

            start = time.time()
            all_samples = dataset_select(dataset, batch_indices)
            logger.info(f"- Dataset indexing took {time.time() - start:.2f}s")

            start = time.time()
            chunk_size = -(L // -n_processes)
            proc_start = accelerator.process_index * chunk_size
            proc_end = min(proc_start + chunk_size, L)

            if proc_start < L:
                local_samples = all_samples[proc_start:proc_end]
                if args.reward_model_type == "static":
                    local_rewards = loop_utils.compute_static_rewards(local_samples)
                else:
                    local_rewards = loop_utils.compute_rewards_with_uncertainty_bounds(
                        local_samples, model, tokenizer, inference_bs
                    )
            else:
                n_comp = len(all_samples[0]["completions"])
                local_rewards = torch.zeros(0, n_comp, 3)

            pad_size = chunk_size - local_rewards.shape[0]
            if pad_size > 0:
                local_rewards = torch.cat([
                    local_rewards,
                    torch.zeros(pad_size, *local_rewards.shape[1:])
                ])

            all_rewards = accelerator.gather(local_rewards.to(accelerator.device))[:L].cpu()
            logger.info(f"- Reward computation took {time.time() - start:.2f}s")

            start = time.time()
            if accelerator.is_main_process:
                trunc_mask = build_truncation_mask(all_samples)
                masked_rewards = mask_rewards_for_truncated(all_rewards, trunc_mask)
                if domain_fractions is not None:
                    # floor(effective_K * pool_fraction) per domain; random-fill shortfall
                    quotas = {d: int(effective_K * frac) for d, frac in domain_fractions.items()}
                    domain_labels = [get_category(s["prompt_id"]) for s in all_samples]
                    result = acquisition_function(
                        *masked_rewards.unbind(-1), K=effective_K,
                        domain_labels=domain_labels, quotas=quotas,
                    )
                else:
                    result = acquisition_function(*masked_rewards.unbind(-1), K=effective_K)
                if isinstance(result, tuple):
                    selected_local_indices, acquired_idxs_list = result
                else:
                    selected_local_indices = list(range(L))
                    acquired_idxs_list = result
            else:
                selected_local_indices, acquired_idxs_list = None, None
            broadcast_data = broadcast_object_list([selected_local_indices, acquired_idxs_list])
            selected_local_indices, acquired_idxs_list = broadcast_data[0], broadcast_data[1]
            acquired_idxs = torch.tensor(acquired_idxs_list)
            logger.info(f"- Acquisition function took {time.time() - start:.2f}s")

            start = time.time()
            selected_samples = [all_samples[i] for i in selected_local_indices]
            acquired = loop_utils.get_acquired(selected_samples, acquired_idxs)
            logger.info(f"- Preparing acquired batch took {time.time() - start:.2f}s")

            start = time.time()
            annotated_batch = [
                loop_utils.restructure_sample(x) for x in oracle(acquired)
            ]
            oracle_extremes = loop_utils.compute_oracle_extremes(selected_samples)
            for ann, ext, sample in zip(annotated_batch, oracle_extremes, selected_samples):
                ann.update(ext)
                ann["original_index"] = sample["original_index"]
            logger.info(f"- Oracle annotation took {time.time() - start:.2f}s")

            start = time.time()
            selected_rewards = all_rewards[torch.tensor(selected_local_indices)]
            kpis_batch = loop_utils.compute_kpis(selected_rewards, acquired_idxs)
            modelwise_kpis = loop_utils.compute_modelwise_kpis(selected_samples, selected_rewards)
            logger.info(f"- KPI computation took {time.time() - start:.2f}s")

            selected_global = {batch_indices[i] for i in selected_local_indices}
            remaining_indices = [i for i in remaining_indices if i not in selected_global]

            del all_samples, all_rewards

            # === SHARED POST-PROCESSING ===
            for key, val in kpis_batch[len(kpis_batch) - 1].copy().items():
                key2 = key.replace("per_sample", "per_batch")
                if not key2.startswith("mean_"):
                    key2 = f"mean_{key2}"
                kpis_batch[len(kpis_batch) - 1][key2] = sum(
                    kpi2[key] for kpi2 in kpis_batch
                ) / len(kpis_batch)
            kpis_batch[len(kpis_batch) - 1].update(modelwise_kpis)

            for idx in range(len(annotated_batch)):
                kpis_batch[idx]["actual_chosen_score_per_sample"] = annotated_batch[idx][
                    "chosen_score"
                ]
                kpis_batch[idx]["actual_rejected_score_per_sample"] = annotated_batch[idx][
                    "rejected_score"
                ]
                kpis_batch[idx]["actual_score_difference_per_sample"] = (
                    annotated_batch[idx]["chosen_score"]
                    - annotated_batch[idx]["rejected_score"]
                )

            uncertainty_kpis = loop_utils.compute_uncertainty_kpis(
                kpis_batch, annotated_batch, rewards=selected_rewards, acquired_idxs=acquired_idxs,
            )
            kpis_batch[len(kpis_batch) - 1].update(uncertainty_kpis)
            del selected_rewards

            logger.info(
                f"- Number of samples annotated in this batch: {len(annotated_batch)}"
            )
            if accelerator.is_main_process:
                output += [
                    {key: val for key, val in x.items() if "features" not in key}
                    for x in annotated_batch
                ]
                logger.info(f"Current output dataset size: {len(output)}")

            if trainer is None:
                if accelerator.is_main_process:
                    logger.info("Reporting KPIs to WandB")
                    for kpis in kpis_batch:
                        wandb.log(kpis)
                logger.info("Skipping reward model training")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                loop_utils.WANDB_LOGS_CACHE += kpis_batch

                start = time.time()
                trainsize = reward_args.effective_batch_size * reward_args.max_steps
                logger.info(
                    f"Adding fresh batch to replay buffer, then subsampling {trainsize} for training"
                )
                for idx, x in enumerate(annotated_batch):
                    replay_buffer.append(
                        {
                            "prompt_input_ids": [0],
                            "chosen_input_ids": [0],
                            "rejected_input_ids": [0],
                            "chosen_features": x["features_chosen"].cpu() if isinstance(x["features_chosen"], torch.Tensor) else x["features_chosen"],
                            "rejected_features": x["features_rejected"].cpu() if isinstance(x["features_rejected"], torch.Tensor) else x["features_rejected"],
                        }
                    )
                logger.info(f"- Replay buffer update took {time.time() - start:.2f}s")

                start = time.time()
                train_subsample = random.sample(
                    replay_buffer,
                    min(len(replay_buffer), trainsize),
                )

                if args.reward_model_type == "blh":
                    logger.info(f"- Train subsample preparation took {time.time() - start:.2f}s")
                    start = time.time()
                    newton_logs = loop_utils.blh_newton_map(
                        model, train_subsample, accelerator,
                    )
                    if accelerator.is_main_process and newton_logs:
                        for nl in newton_logs:
                            logger.info(
                                f"  Newton iter {nl['blh_newton/iteration']}: "
                                f"loss={nl['blh_newton/loss']:.4f} "
                                f"(base={nl['blh_newton/loss_base']:.4f} reg={nl['blh_newton/loss_reg']:.4f}) "
                                f"grad_norm={nl['blh_newton/grad_norm']:.6f} "
                                f"win_rate={nl['blh_newton/win_rate']:.3f}"
                            )
                        loop_utils.WANDB_LOGS_CACHE[-1].update(newton_logs[-1])
                        for _logs in loop_utils.WANDB_LOGS_CACHE:
                            wandb.log(_logs)
                        loop_utils.WANDB_LOGS_CACHE = []
                    logger.info(f"- BLH Newton MAP + Hessian update took {time.time() - start:.2f}s")
                else:
                    trainer.train_dataset = Dataset.from_list(train_subsample)
                    trainer.train_dataset.set_format(
                        type="torch", columns=trainer.train_dataset.column_names
                    )
                    loop_utils.MAX_TRAINER_LOGS_CACHE_SIZE = len(trainer.get_train_dataloader())
                    logger.info(f"- Train dataset preparation took {time.time() - start:.2f}s")

                    if args.reward_model_type == "enn":
                        n_done = min(expected_output_size, dataset_len - len(remaining_indices))
                        new_regularisation = loop_utils.get_new_regularization(
                            n_done=n_done,
                            n_total=expected_output_size,
                            **asdict(reward_args.regularization),
                        )
                        trainer.args.regularization_towards_initial_weights = new_regularisation

                    start = time.time()
                    trainer.state.log_history.clear()
                    trainer.state.global_step = 0
                    trainer.state.epoch = 0.0
                    model.train()
                    trainer.train()
                    logger.info(f"- Reward model training took {time.time() - start:.2f}s")

                if outer_batch_idx % 20 == 0:
                    start = time.time()
                    gc.collect()
                    accelerator.wait_for_everyone()
                    accelerator.free_memory()
                    torch.cuda.empty_cache()
                    logger.info(f"- Cleanup took {time.time() - start:.2f}s")

                if accelerator.is_main_process:
                    if outer_batch_idx % args.save_every_n_outer_batches == 0:
                        start = time.time()
                        logger.info(f"Writing output dataset to {args.output_path}")
                        Dataset.from_list(output).save_to_disk(args.output_path)

                        if args.run_tag:
                            ckpt_name = f"checkpoint-{args.run_tag}-{outer_batch_idx}"
                        else:
                            ckpt_name = f"checkpoint-{outer_batch_idx}"

                        checkpoint_dir = os.path.join(args.output_path, ckpt_name)
                        logger.info(f"Saving checkpoint to {checkpoint_dir}")

                        loop_state = {
                            "next_outer_batch_idx": outer_batch_idx + 1,
                            "wandb_run_id": wandb.run.id,
                            "timestamp": timestamp,
                        }

                        save_loop_checkpoint(
                            save_dir=checkpoint_dir,
                            args=args,
                            loop_state=loop_state,
                            replay_buffer=replay_buffer,
                            output_data=output,
                            trainer=trainer,
                            model=model if trainer is None else None,
                            remaining_indices=remaining_indices,
                        )
                        logger.info(f"- Checkpointing took {time.time() - start:.2f}s")

            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            avg_step = sum(step_times) / len(step_times)
            steps_left = -(-len(remaining_indices) // args.prompt_selection_K) if remaining_indices else 0
            eta_s = avg_step * steps_left
            eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)
            total_elapsed = time.time() - loop_start_time
            elapsed_h, elapsed_m = int(total_elapsed // 3600), int((total_elapsed % 3600) // 60)
            logger.info(f"=== Step {outer_batch_idx + 1} took {step_elapsed:.2f}s | Elapsed: {elapsed_h}h {elapsed_m}m | ETA: {eta_h}h {eta_m}m ({steps_left} steps left, avg {avg_step:.1f}s/step) ===")
            outer_batch_idx += 1

    else:
        # --- ORIGINAL LOOP (no prompt selection) ---
        all_indices = list(range(dataset_len))
        start_idx = start_outer_batch_idx * args.outer_loop_batch_size
        total_outer_batches = -(-dataset_len // args.outer_loop_batch_size)

        for i in range(start_outer_batch_idx, total_outer_batches):
            step_start = time.time()
            outer_batch_idx = i

            if model is not None:
                model.eval()

            batch_start = i * args.outer_loop_batch_size
            batch_end = min(batch_start + args.outer_loop_batch_size, dataset_len)
            batch_indices = all_indices[batch_start:batch_end]

            start = time.time()
            outer_batch = dataset_select(dataset, batch_indices)
            logger.info(f"- Dataset indexing took {time.time() - start:.2f}s")

            if accelerator.is_main_process:
                logger.info(f"Step {outer_batch_idx + 1} / {total_outer_batches}")

            start = time.time()
            dataloader = DataLoader(
                outer_batch,
                batch_size=max(1, -(len(outer_batch) // -n_processes)),
                collate_fn=loop_utils.custom_collate,
                shuffle=False,
                drop_last=False,
            )
            logger.info(f"- # Minibatches: {len(dataloader)}")
            dataloader = accelerator.prepare(dataloader)
            logger.info(f"- DataLoader creation + prepare took {time.time() - start:.2f}s")

            annotated_batch = []
            kpis_batch = []
            all_minibatch_samples = []
            all_minibatch_rewards = []
            all_minibatch_acquired_idxs = []
            minibatch_idx = 0
            iter_start = time.time()
            for collated_minibatch in dataloader:
                logger.info(f"- Minibatch {minibatch_idx} iteration (data fetch) took {time.time() - iter_start:.2f}s")

                start = time.time()
                samples_local = loop_utils.custom_decollate(collated_minibatch)
                logger.info(f"- Decollate took {time.time() - start:.2f}s")

                start = time.time()
                if args.reward_model_type == "static":
                    rewards_with_uncertainty_bounds_local = loop_utils.compute_static_rewards(samples_local)
                else:
                    rewards_with_uncertainty_bounds_local = (
                        loop_utils.compute_rewards_with_uncertainty_bounds(
                            samples_local, model, tokenizer, inference_bs
                        )
                    )
                logger.info(f"- Reward computation took {time.time() - start:.2f}s")

                start = time.time()
                trunc_mask_local = build_truncation_mask(samples_local)
                masked_rewards_local = mask_rewards_for_truncated(
                    rewards_with_uncertainty_bounds_local, trunc_mask_local
                )
                acquired_idxs_local = torch.tensor(
                    acquisition_function(*masked_rewards_local.unbind(-1))
                )
                logger.info(f"- Acquisition function took {time.time() - start:.2f}s")

                start = time.time()
                acquired_local = loop_utils.get_acquired(samples_local, acquired_idxs_local)
                logger.info(f"- Preparing acquired batch took {time.time() - start:.2f}s")

                start = time.time()
                annotated_local = [
                    loop_utils.restructure_sample(x) for x in oracle(acquired_local)
                ]
                oracle_extremes_local = loop_utils.compute_oracle_extremes(samples_local)
                for ann, ext, sample in zip(annotated_local, oracle_extremes_local, samples_local):
                    ann.update(ext)
                    ann["original_index"] = sample["original_index"]
                logger.info(f"- Oracle annotation took {time.time() - start:.2f}s")

                start = time.time()
                kpis_local = loop_utils.compute_kpis(
                    rewards_with_uncertainty_bounds_local,
                    acquired_idxs_local,
                )
                all_minibatch_samples.extend(samples_local)
                all_minibatch_rewards.append(rewards_with_uncertainty_bounds_local)
                all_minibatch_acquired_idxs.append(acquired_idxs_local)
                logger.info(f"- KPI computation took {time.time() - start:.2f}s")

                start = time.time()
                annotated_batch += accelerator.gather_for_metrics(annotated_local)
                kpis_batch += accelerator.gather_for_metrics(kpis_local)
                logger.info(
                    f"- Gathering data from processes took {time.time() - start:.2f}s"
                )

                minibatch_idx += 1
                iter_start = time.time()

            for key, val in kpis_batch[len(kpis_batch) - 1].copy().items():
                key2 = key.replace("per_sample", "per_batch")
                if not key2.startswith("mean_"):
                    key2 = f"mean_{key2}"
                kpis_batch[len(kpis_batch) - 1][key2] = sum(
                    kpi2[key] for kpi2 in kpis_batch
                ) / len(kpis_batch)
            all_local_rewards = torch.cat(all_minibatch_rewards, dim=0)
            all_local_acquired_idxs = torch.cat(all_minibatch_acquired_idxs, dim=0)
            modelwise_kpis = loop_utils.compute_modelwise_kpis(
                all_minibatch_samples, all_local_rewards
            )
            kpis_batch[len(kpis_batch) - 1].update(modelwise_kpis)
            del all_minibatch_samples, all_minibatch_rewards, all_minibatch_acquired_idxs

            batch_total = batch_end - batch_start
            chunk_size = -(batch_total // -n_processes)
            pad_size = chunk_size - all_local_rewards.shape[0]
            if pad_size > 0:
                all_local_rewards = torch.cat([all_local_rewards, torch.zeros(pad_size, *all_local_rewards.shape[1:])])
                all_local_acquired_idxs = torch.cat([all_local_acquired_idxs, torch.zeros(pad_size, 2, dtype=all_local_acquired_idxs.dtype)])
            all_gathered_rewards = accelerator.gather(all_local_rewards.to(accelerator.device))[:batch_total].cpu()
            all_gathered_acquired_idxs = accelerator.gather(all_local_acquired_idxs.to(accelerator.device))[:batch_total].cpu()

            for idx in range(len(annotated_batch)):
                kpis_batch[idx]["actual_chosen_score_per_sample"] = annotated_batch[idx][
                    "chosen_score"
                ]
                kpis_batch[idx]["actual_rejected_score_per_sample"] = annotated_batch[idx][
                    "rejected_score"
                ]
                kpis_batch[idx]["actual_score_difference_per_sample"] = (
                    annotated_batch[idx]["chosen_score"]
                    - annotated_batch[idx]["rejected_score"]
                )

            uncertainty_kpis = loop_utils.compute_uncertainty_kpis(
                kpis_batch, annotated_batch, rewards=all_gathered_rewards, acquired_idxs=all_gathered_acquired_idxs,
            )
            kpis_batch[len(kpis_batch) - 1].update(uncertainty_kpis)
            del all_local_rewards, all_local_acquired_idxs, all_gathered_rewards, all_gathered_acquired_idxs

            logger.info(
                f"- Number of samples annotated in this batch: {len(annotated_batch)}"
            )
            if accelerator.is_main_process:
                output += [
                    {key: val for key, val in x.items() if "features" not in key}
                    for x in annotated_batch
                ]
                logger.info(f"Current output dataset size: {len(output)}")

            if trainer is None:
                if accelerator.is_main_process:
                    logger.info("Reporting KPIs to WandB")
                    for kpis in kpis_batch:
                        wandb.log(kpis)

                logger.info("Skipping reward model training")
                step_elapsed = time.time() - step_start
                step_times.append(step_elapsed)
                avg_step = sum(step_times) / len(step_times)
                steps_left = total_outer_batches - (i + 1)
                eta_s = avg_step * steps_left
                eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)
                total_elapsed = time.time() - loop_start_time
                elapsed_h, elapsed_m = int(total_elapsed // 3600), int((total_elapsed % 3600) // 60)
                logger.info(f"=== Step {outer_batch_idx + 1} took {step_elapsed:.2f}s | Elapsed: {elapsed_h}h {elapsed_m}m | ETA: {eta_h}h {eta_m}m ({steps_left} steps left, avg {avg_step:.1f}s/step) ===")
                continue
            else:
                loop_utils.WANDB_LOGS_CACHE += kpis_batch

            start = time.time()
            trainsize = reward_args.effective_batch_size * reward_args.max_steps
            logger.info(
                f"Adding fresh batch to replay buffer, then subsampling {trainsize} for training"
            )
            for idx, x in enumerate(annotated_batch):
                replay_buffer.append(
                    {
                        "prompt_input_ids": [0],
                        "chosen_input_ids": [0],
                        "rejected_input_ids": [0],
                        "chosen_features": x["features_chosen"].cpu() if isinstance(x["features_chosen"], torch.Tensor) else x["features_chosen"],
                        "rejected_features": x["features_rejected"].cpu() if isinstance(x["features_rejected"], torch.Tensor) else x["features_rejected"],
                    }
                )
            logger.info(f"- Replay buffer update took {time.time() - start:.2f}s")

            start = time.time()
            train_subsample = random.sample(
                replay_buffer,
                min(len(replay_buffer), trainsize),
            )

            if args.reward_model_type == "blh":
                logger.info(f"- Train subsample preparation took {time.time() - start:.2f}s")
                start = time.time()
                newton_logs = loop_utils.blh_newton_map(
                    model, train_subsample, accelerator,
                )
                if accelerator.is_main_process and newton_logs:
                    for nl in newton_logs:
                        logger.info(
                            f"  Newton iter {nl['blh_newton/iteration']}: "
                            f"loss={nl['blh_newton/loss']:.4f} "
                            f"(base={nl['blh_newton/loss_base']:.4f} reg={nl['blh_newton/loss_reg']:.4f}) "
                            f"grad_norm={nl['blh_newton/grad_norm']:.6f} "
                            f"win_rate={nl['blh_newton/win_rate']:.3f}"
                        )
                    loop_utils.WANDB_LOGS_CACHE[-1].update(newton_logs[-1])
                    for _logs in loop_utils.WANDB_LOGS_CACHE:
                        wandb.log(_logs)
                    loop_utils.WANDB_LOGS_CACHE = []
                logger.info(f"- BLH Newton MAP + Hessian update took {time.time() - start:.2f}s")
            else:
                trainer.train_dataset = Dataset.from_list(train_subsample)
                trainer.train_dataset.set_format(
                    type="torch", columns=trainer.train_dataset.column_names
                )
                loop_utils.MAX_TRAINER_LOGS_CACHE_SIZE = len(trainer.get_train_dataloader())
                logger.info(f"- Train dataset preparation took {time.time() - start:.2f}s")

                if args.reward_model_type == "enn":
                    new_regularisation = loop_utils.get_new_regularization(
                        n_done=min(
                            expected_output_size, (outer_batch_idx + 1) * args.outer_loop_batch_size
                        ),
                        n_total=expected_output_size,
                        **asdict(reward_args.regularization),
                    )
                    trainer.args.regularization_towards_initial_weights = new_regularisation

                start = time.time()
                model.train()
                trainer.train()
                logger.info(f"- Reward model training took {time.time() - start:.2f}s")

            start = time.time()
            gc.collect()
            accelerator.wait_for_everyone()
            accelerator.free_memory()
            torch.cuda.empty_cache()
            logger.info(f"- Cleanup took {time.time() - start:.2f}s")

            if accelerator.is_main_process:
                if outer_batch_idx % args.save_every_n_outer_batches == 0:
                    start = time.time()
                    logger.info(f"Writing output dataset to {args.output_path}")
                    Dataset.from_list(output).save_to_disk(args.output_path)

                    if args.run_tag:
                        ckpt_name = f"checkpoint-{args.run_tag}-{outer_batch_idx}"
                    else:
                        ckpt_name = f"checkpoint-{outer_batch_idx}"

                    checkpoint_dir = os.path.join(args.output_path, ckpt_name)
                    logger.info(f"Saving checkpoint to {checkpoint_dir}")

                    loop_state = {
                        "next_outer_batch_idx": outer_batch_idx + 1,
                        "wandb_run_id": wandb.run.id,
                        "timestamp": timestamp
                    }

                    save_loop_checkpoint(
                        save_dir=checkpoint_dir,
                        args=args,
                        loop_state=loop_state,
                        replay_buffer=replay_buffer,
                        output_data=output,
                        trainer=trainer,
                        model=model if trainer is None else None
                    )
                    logger.info(f"- Checkpointing took {time.time() - start:.2f}s")

            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            avg_step = sum(step_times) / len(step_times)
            steps_left = total_outer_batches - (i + 1)
            eta_s = avg_step * steps_left
            eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)
            total_elapsed = time.time() - loop_start_time
            elapsed_h, elapsed_m = int(total_elapsed // 3600), int((total_elapsed % 3600) // 60)
            logger.info(f"=== Step {outer_batch_idx + 1} took {step_elapsed:.2f}s | Elapsed: {elapsed_h}h {elapsed_m}m | ETA: {eta_h}h {eta_m}m ({steps_left} steps left, avg {avg_step:.1f}s/step) ===")

    if accelerator.is_main_process:
        wandb.finish()

    if accelerator.is_main_process and len(output) > 0:
        logger.info(f"Writing output dataset to {args.output_path}")
        Dataset.from_list(output).save_to_disk(args.output_path)