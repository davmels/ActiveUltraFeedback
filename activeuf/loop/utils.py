from dataclasses import asdict
from datasets import Dataset
import torch
from transformers import TrainerCallback
import wandb
import os
import json
import pickle
import random
import numpy as np

from rewarduq.methods.mlp_head_ensemble import (
    MLPHeadEnsembleModel as ENNRewardModel,
    MLPHeadEnsembleModelConfig as ENNRewardModelConfig,
    MLPHeadEnsembleTrainer as ENNRewardModelTrainer,
    MLPHeadEnsembleTrainerConfig as ENNRewardModelTrainerConfig,
    MLPHeadEnsemblePipeline as ENNRewardModelPipeline,
)
from rewarduq.methods.bayesian_linear_head import (
    BayesianLinearHeadModel as BLHRewardModel,
    BayesianLinearHeadModelConfig as BLHRewardModelConfig,
    BayesianLinearHeadTrainer as BLHRewardModelTrainer,
    BayesianLinearHeadTrainerConfig as BLHRewardModelTrainerConfig,
    BayesianLinearHeadPipeline as BLHRewardModelPipeline,
)

from activeuf.loop.arguments import ENNConfig, BLHConfig


def main_process_only(f, accelerator):
    """Decorator to ensure the wrapped logging function runs only on the main process."""

    def wrapper(*args, **kwargs):
        if accelerator.is_main_process:
            return f(*args, **kwargs)
        return None

    return wrapper


def custom_collate(batch):
    out = {}
    for key in [
        "prompt_id",
        "prompt",
        "source",
        "completions",
        "features",
        "original_index",
    ]:
        if key in batch[0]:
            out[key] = [x[key] for x in batch]
    return out


def custom_decollate(collated_batch):
    out = []
    for i in range(len(collated_batch["prompt_id"])):
        item = {
            "prompt_id": collated_batch["prompt_id"][i],
            "prompt": collated_batch["prompt"][i],
            "completions": collated_batch["completions"][i],
        }
        if "source" in collated_batch:
            item["source"] = collated_batch["source"][i]
        if "features" in collated_batch:
            item["features"] = collated_batch["features"][i]
        if "original_index" in collated_batch:
            item["original_index"] = collated_batch["original_index"][i]
        out.append(item)
    return out


def compute_acquisition_function_KPIs(rewards, chosen_idxs, rejected_idxs):
    mean_rewards_per_sample = rewards.mean(dim=1)  # (n_samples, 2)

    chosen_rewards = rewards.gather(
        1, chosen_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))
    ).squeeze(1)
    rejected_rewards = rewards.gather(
        1, rejected_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))
    ).squeeze(1)

    # Add to KPIs
    kpis = {
        "mean_rewards_per_sample": mean_rewards_per_sample[:, 0].tolist(),
        "mean_uncertainty_per_sample": mean_rewards_per_sample[:, 1].tolist(),
        "chosen_rewards_per_sample": chosen_rewards[:, 0].tolist(),
        "chosen_uncertainty_per_sample": chosen_rewards[:, 1].tolist(),
        "rejected_rewards_per_sample": rejected_rewards[:, 0].tolist(),
        "rejected_uncertainty_per_sample": rejected_rewards[:, 1].tolist(),
    }
    return kpis


WANDB_LOGS_CACHE = []
TRAINER_LOGS_CACHE = []
MAX_TRAINER_LOGS_CACHE_SIZE = None


class WandbStepLoggerCallback(TrainerCallback):
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Custom callback for wandb logging with caches. This callback
        waits until the trainer's log cache hits the specified maximum size.
        Then, it aggregates the cached trainer logs (by taking mean),
        and lets the aggregated logs "piggyback" with the final log in
        the wandb cache. It ignores logs made at the end of each epoch,
        by checking for the key 'train_runtime'.
        """
        global WANDB_LOGS_CACHE, TRAINER_LOGS_CACHE

        if self.accelerator.is_main_process and logs and "train_runtime" not in logs:
            TRAINER_LOGS_CACHE.append(logs)

            if len(TRAINER_LOGS_CACHE) == MAX_TRAINER_LOGS_CACHE_SIZE:
                # aggregate trainer logs by taking mean
                mean_trainer_logs = {}
                keys = {k for x in TRAINER_LOGS_CACHE for k in x}
                for k in keys:
                    values = [x.get(k) for x in TRAINER_LOGS_CACHE]
                    values = [_ for _ in values if _ is not None]
                    mean_trainer_logs[k] = sum(values) / len(values)

                for key in ["regularization_towards_initial_weights", "l2_reg"]:
                    if hasattr(args, key):
                        mean_trainer_logs[key] = getattr(args, key)

                # let current logs piggyback on the last entry in the cache
                if WANDB_LOGS_CACHE:
                    WANDB_LOGS_CACHE[-1].update(mean_trainer_logs)
                else:
                    WANDB_LOGS_CACHE.append(mean_trainer_logs)

                for _logs in WANDB_LOGS_CACHE:
                    wandb.log(_logs)

                # clear caches
                WANDB_LOGS_CACHE = []
                TRAINER_LOGS_CACHE = []


def init_model_trainer(
    reward_model_type: str, reward_args: ENNConfig | BLHConfig | None, n_processes: int
):
    if reward_model_type in ("none", "static"):
        model, trainer = None, None

    elif reward_model_type == "enn":
        trainer_config = ENNRewardModelTrainerConfig(
            per_device_train_batch_size=-(
                reward_args.effective_batch_size // -n_processes
            ),
            **asdict(reward_args.trainer),
        )

        if reward_args.previous_checkpoint_path:
            model = ENNRewardModel.from_pretrained(
                reward_args.previous_checkpoint_path,
            )
            tokenizer = model.tokenizer
        else:
            pipeline = ENNRewardModelPipeline(
                ENNRewardModelConfig(**asdict(reward_args.model)),
                trainer_config,
            )
            model = pipeline.model
            tokenizer = model.tokenizer

        # initialize trainer with a dummy Dataset. So we have access to the uq_pipeline.trainer before entering the loop.
        # The dataset must have raw columns (prompt, chosen, rejected) because the new rewarduq
        # version runs prepare_preference_dataset during __init__.
        trainer = ENNRewardModelTrainer(
            args=trainer_config,
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(
                [
                    {
                        "prompt": "dummy",
                        "chosen": "dummy",
                        "rejected": "dummy",
                    }
                ]
            ),
        )
    elif reward_model_type == "blh":
        trainer_kwargs = asdict(reward_args.trainer)
        trainer_kwargs["l2_reg"] = float(trainer_kwargs["l2_reg"])
        trainer_config = BLHRewardModelTrainerConfig(
            per_device_train_batch_size=-(
                reward_args.effective_batch_size // -n_processes
            ),
            **trainer_kwargs,
        )

        if reward_args.previous_checkpoint_path:
            model = BLHRewardModel.from_pretrained(
                reward_args.previous_checkpoint_path,
            )
            tokenizer = model.tokenizer
        else:
            model_kwargs = asdict(reward_args.model)
            model_kwargs["lambda_reg"] = float(model_kwargs["lambda_reg"])
            model_kwargs["std_beta"] = float(model_kwargs["std_beta"])
            pipeline = BLHRewardModelPipeline(
                BLHRewardModelConfig(**model_kwargs),
                trainer_config,
            )
            model = pipeline.model
            tokenizer = model.tokenizer

        trainer = BLHRewardModelTrainer(
            args=trainer_config,
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(
                [
                    {
                        "prompt": "dummy",
                        "chosen": "dummy",
                        "rejected": "dummy",
                    }
                ]
            ),
        )
    else:
        raise NotImplementedError(f"{reward_model_type=} not implemented.")

    return model, trainer


def _extract_z_vectors(train_subsample):
    """Extract normalized feature-difference vectors from a training subsample."""
    z_vectors = []
    for sample in train_subsample:
        chosen_f = sample["chosen_features"]
        rejected_f = sample["rejected_features"]
        if not isinstance(chosen_f, torch.Tensor):
            chosen_f = torch.tensor(chosen_f)
        if not isinstance(rejected_f, torch.Tensor):
            rejected_f = torch.tensor(rejected_f)
        chosen_f = torch.nn.functional.normalize(chosen_f.float(), dim=-1)
        rejected_f = torch.nn.functional.normalize(rejected_f.float(), dim=-1)
        z_vectors.append(chosen_f - rejected_f)
    return torch.stack(z_vectors) if z_vectors else None


def blh_newton_map(model, train_subsample, accelerator, max_iter=20, tol=1e-4):
    """Find the MAP estimate for the BLH head using Newton's method (IRLS).

    Returns a dict of wandb-loggable metrics from each iteration.
    """
    inner = accelerator.unwrap_model(model)
    dtype = inner.safe_dtype

    Z = _extract_z_vectors(train_subsample)
    if Z is None:
        return []
    Z = Z.to(dtype=dtype, device="cpu")
    n_samples = Z.shape[0]

    H_prior = inner.H.cpu().to(dtype=dtype)
    mu_prior = inner._prior_mean.cpu().to(dtype=dtype)

    w = mu_prior.clone()

    logs = []
    for it in range(max_iter):
        logits = Z @ w
        probs = torch.sigmoid(logits)

        loss_base = -torch.nn.functional.logsigmoid(logits).mean().item()
        delta = w - mu_prior
        loss_reg = (0.5 * delta @ H_prior @ delta).item()
        win_rate = (probs > 0.5).float().mean().item()

        grad_data = -(1.0 - probs) @ Z / n_samples
        grad_reg = H_prior @ delta
        grad = grad_data + grad_reg
        grad_norm = grad.norm().item()

        logs.append({
            "blh_newton/iteration": it,
            "blh_newton/loss": loss_base + loss_reg,
            "blh_newton/loss_base": loss_base,
            "blh_newton/loss_reg": loss_reg,
            "blh_newton/grad_norm": grad_norm,
            "blh_newton/win_rate": win_rate,
        })

        if grad_norm < tol:
            break

        d = probs * (1.0 - probs)
        H_data = (Z.T * d) @ Z / n_samples
        H_total = H_data + H_prior
        step = torch.linalg.solve(H_total, grad)
        w = w - step

    # Set head weights to MAP estimate
    with torch.no_grad():
        inner.head.linear.weight.copy_(
            w.unsqueeze(0).to(device=inner.head.linear.weight.device, dtype=inner.head.linear.weight.dtype)
        )

    # Update prior and Hessian
    inner.set_prior_from_posterior()

    # Incremental Hessian update for uncertainty
    original_device = inner.H.device
    inner.H = inner.H.cpu().to(dtype=dtype)
    inner.H = inner.H + Z.T @ Z
    inner._H_inv = torch.linalg.pinv(inner.H)
    inner._H_inv_computed = True
    inner.H = inner.H.to(original_device)
    inner._H_inv = inner._H_inv.to(original_device)

    return logs


def compute_blh_hessian(model, train_subsample, accelerator):
    """Incrementally update the Hessian: H_new = H_prev + Z^T Z using the training subsample."""
    inner = accelerator.unwrap_model(model)
    Z = _extract_z_vectors(train_subsample)
    if Z is not None:
        Z = Z.to(dtype=inner.safe_dtype, device="cpu")
        original_device = inner.H.device
        inner.H = inner.H.cpu()
        inner.H = inner.H + Z.T @ Z
        inner._H_inv = torch.linalg.pinv(inner.H)
        inner._H_inv_computed = True
        inner.H = inner.H.to(original_device)
        inner._H_inv = inner._H_inv.to(original_device)


def compute_rewards_with_uncertainty_bounds(
    samples, model, tokenizer, inference_batch_size
) -> torch.tensor:
    n_samples = len(samples)
    n_completions_per_sample = len(samples[0]["completions"])

    if model is None:
        return torch.zeros(
            (n_samples, n_completions_per_sample, 3),
            dtype=torch.float32,
        )

    has_precomputed_features = "features" in samples[0]

    if has_precomputed_features:
        # Use precomputed features
        def get_features_yielder():
            for sample in samples:
                for i in range(n_completions_per_sample):
                    yield torch.tensor(sample["features"][i])

        features_yielder = get_features_yielder()
        rewards_batch = []
        while True:
            features_mbatch = []
            for _ in range(inference_batch_size):
                try:
                    features_mbatch.append(next(features_yielder))
                except StopIteration:
                    break
            if not features_mbatch:
                break
            features_mbatch = torch.stack(features_mbatch).to(model.device)

            with torch.no_grad():
                output = model(features=features_mbatch)

            rewards_batch.extend(output["rewards"].cpu())
    else:
        # Tokenize and compute features on-the-fly
        all_input_ids = []
        for sample in samples:
            for completion in sample["completions"]:
                if isinstance(sample["prompt"], list):
                    prompt_messages = sample["prompt"]
                else:
                    prompt_messages = [{"role": "user", "content": sample["prompt"]}]
                conversation = prompt_messages + [{"role": "assistant", "content": completion["response_text"]}]

                tokenized = tokenizer.apply_chat_template(
                    conversation, tokenize=True, return_tensors="pt"
                ).squeeze(0)
                all_input_ids.append(tokenized)

        rewards_batch = []
        features_batch = []
        for start in range(0, len(all_input_ids), inference_batch_size):
            batch_ids = all_input_ids[start : start + inference_batch_size]
            padded = tokenizer.pad(
                {"input_ids": batch_ids},
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                output = model(
                    input_ids=padded["input_ids"],
                    attention_mask=padded["attention_mask"],
                    output_features=True,
                )

            rewards_batch.extend(output["rewards"].cpu())
            features_batch.extend(output["features"].cpu())

        # Store computed features back in samples for downstream use (replay buffer)
        idx = 0
        for sample in samples:
            sample["features"] = []
            for _ in sample["completions"]:
                sample["features"].append(features_batch[idx])
                idx += 1

    torch.cuda.empty_cache()
    rewards_batch = torch.stack(rewards_batch).view(n_samples, -1, 3)

    return rewards_batch


def compute_static_rewards(samples) -> torch.Tensor:
    """Extract LLM judge scores from samples as rewards with zero uncertainty."""
    n_samples = len(samples)
    n_completions = len(samples[0]["completions"])

    rewards = torch.zeros(n_samples, n_completions, 3, dtype=torch.float32)
    for i, sample in enumerate(samples):
        for j, comp in enumerate(sample["completions"]):
            score = float(comp["overall_score"])
            rewards[i, j, 0] = score
            rewards[i, j, 1] = score
            rewards[i, j, 2] = score

    for sample in samples:
        if "features" not in sample:
            sample["features"] = [torch.zeros(1) for _ in sample["completions"]]

    return rewards


def get_acquired(samples, acquired_idxs):
    acquired = []
    for sample, (a, b) in zip(samples, acquired_idxs):
        assert a != b
        completions = sample["completions"]

        item = {
            "prompt_id": sample["prompt_id"],
            "prompt": sample["prompt"],
            "response_text_1": completions[a]["response_text"],
            "features_1": sample["features"][a],
            "model_1": completions[a]["model"],
            "score_1": completions[a]["overall_score"],
            "response_text_2": completions[b]["response_text"],
            "features_2": sample["features"][b],
            "model_2": completions[b]["model"],
            "score_2": completions[b]["overall_score"],
        }
        if "source" in sample:
            item["source"] = sample["source"]
        acquired.append(item)
    return acquired


def compute_kpis(rewards, acquired_idxs) -> list[dict]:
    _rewards, _lower_bounds, _upper_bounds = rewards.unbind(-1)
    _uncertainty = (_upper_bounds - _lower_bounds) / 2  # half-width of the confidence interval
    _chosen_idxs, _rejected_idxs = acquired_idxs.unbind(-1)

    index = torch.arange(_rewards.size(0))
    mean_rewards_per_sample = _rewards.mean(dim=1)
    mean_uncertainty_per_sample = _uncertainty.mean(dim=1)

    kpis = []
    for i in range(_rewards.size(0)):
        kpi = {
            "mean_rewards_per_sample": mean_rewards_per_sample[i].item(),
            "mean_uncertainty_per_sample": mean_uncertainty_per_sample[i].item(),
            "chosen_rewards_per_sample": _rewards[index, _chosen_idxs][i].item(),
            "rejected_rewards_per_sample": _rewards[index, _rejected_idxs][i].item(),
            "chosen_uncertainty_per_sample": _uncertainty[index, _chosen_idxs][
                i
            ].item(),
            "rejected_uncertainty_per_sample": _uncertainty[index, _rejected_idxs][
                i
            ].item(),
        }
        kpi["reward_differences_per_sample"] = (
            kpi["chosen_rewards_per_sample"] - kpi["rejected_rewards_per_sample"]
        )
        kpis.append(kpi)
    return kpis


def compute_uncertainty_kpis(kpis_batch, annotated_batch, rewards=None, acquired_idxs=None) -> dict:
    """Compute batch-level uncertainty quality metrics.

    Requires kpis_batch (with reward/uncertainty per sample) and annotated_batch
    (with oracle chosen_score/rejected_score) to be aligned.
    When rewards (n, n_comp, 3) and acquired_idxs (n, 2) are provided,
    computes additional deltaUCB-relevant metrics.
    """
    n = len(kpis_batch)
    if n == 0:
        return {}

    confidence_wins = 0
    overconfident_errors = 0
    uncertain_correct = 0
    uncertain_incorrect = 0
    oracle_chosen_agree = 0
    confidence_margin_sum = 0.0

    chosen_ucb_gap_sum = 0.0
    rejected_lcb_gap_sum = 0.0
    chosen_lcb_overlap_sum = 0
    rejected_ucb_overlap_sum = 0
    delta_ucb_score_sum = 0.0
    mean_ci_width_sum = 0.0
    chosen_dominance_ratio_sum = 0.0
    rejected_dominance_ratio_sum = 0.0
    chosen_dominance_count = 0
    rejected_dominance_count = 0

    has_full_rewards = rewards is not None and acquired_idxs is not None
    if has_full_rewards:
        _rewards, _lower_bounds, _upper_bounds = rewards.unbind(-1)
        _chosen_idxs, _rejected_idxs = acquired_idxs.unbind(-1)

    for i in range(n):
        kpi = kpis_batch[i]
        ann = annotated_batch[i]

        chosen_lower = kpi["chosen_rewards_per_sample"] - kpi["chosen_uncertainty_per_sample"]
        rejected_upper = kpi["rejected_rewards_per_sample"] + kpi["rejected_uncertainty_per_sample"]
        confidence_margin_sum += chosen_lower - rejected_upper

        confident = chosen_lower >= rejected_upper
        oracle_agrees = ann["oracle_score_first"] >= ann["oracle_score_second"]

        if oracle_agrees:
            oracle_chosen_agree += 1
        if confident and oracle_agrees:
            confidence_wins += 1
        elif confident and not oracle_agrees:
            overconfident_errors += 1
        elif not confident and oracle_agrees:
            uncertain_correct += 1
        else:
            uncertain_incorrect += 1

        if has_full_rewards:
            ci = _chosen_idxs[i].item()
            ri = _rejected_idxs[i].item()
            ucbs = _upper_bounds[i]
            lcbs = _lower_bounds[i]

            chosen_ucb = ucbs[ci].item()
            chosen_lcb = lcbs[ci].item()
            rejected_ucb = ucbs[ri].item()
            rejected_lcb = lcbs[ri].item()

            delta_ucb_score_sum += chosen_ucb - rejected_lcb
            mean_ci_width_sum += (ucbs - lcbs).mean().item()

            other_ucbs = torch.cat([ucbs[:ci], ucbs[ci+1:]])
            if other_ucbs.numel() > 0:
                second_best_ucb = other_ucbs.max().item()
                chosen_ucb_gap_sum += chosen_ucb - second_best_ucb
                chosen_ci = chosen_ucb - chosen_lcb
                if abs(chosen_ci) > 1e-9:
                    chosen_dominance_ratio_sum += (second_best_ucb - chosen_lcb) / chosen_ci
                    chosen_dominance_count += 1
            chosen_lcb_overlap_sum += int((other_ucbs >= chosen_lcb).sum().item())

            other_lcbs = torch.cat([lcbs[:ri], lcbs[ri+1:]])
            if other_lcbs.numel() > 0:
                second_worst_lcb = other_lcbs.min().item()
                rejected_lcb_gap_sum += rejected_lcb - second_worst_lcb
                rejected_ci = rejected_ucb - rejected_lcb
                if abs(rejected_ci) > 1e-9:
                    rejected_dominance_ratio_sum += (rejected_ucb - second_worst_lcb) / rejected_ci
                    rejected_dominance_count += 1
            rejected_ucb_overlap_sum += int((other_lcbs <= rejected_ucb).sum().item())

    result = {
        "mean_confidence_win_rate_per_batch": confidence_wins / n,
        "mean_overconfident_error_rate_per_batch": overconfident_errors / n,
        "mean_uncertain_correct_rate_per_batch": uncertain_correct / n,
        "mean_uncertain_incorrect_rate_per_batch": uncertain_incorrect / n,
        "mean_oracle_agreement_rate_per_batch": (oracle_chosen_agree) / n,
        "mean_actual_score_difference_per_batch": sum(
            a["chosen_score"] - a["rejected_score"] for a in annotated_batch
        ) / n,
        "mean_confidence_margin_per_batch": confidence_margin_sum / n,
    }

    if has_full_rewards:
        result.update({
            "mean_chosen_ucb_gap_per_batch": chosen_ucb_gap_sum / n,
            "mean_rejected_lcb_gap_per_batch": rejected_lcb_gap_sum / n,
            "mean_chosen_lcb_overlap_count_per_batch": chosen_lcb_overlap_sum / n,
            "mean_rejected_ucb_overlap_count_per_batch": rejected_ucb_overlap_sum / n,
            "mean_delta_ucb_score_per_batch": delta_ucb_score_sum / n,
            "mean_ci_width_per_batch": mean_ci_width_sum / n,
        })
        if chosen_dominance_count > 0:
            result["mean_chosen_dominance_ratio_per_batch"] = chosen_dominance_ratio_sum / chosen_dominance_count
        if rejected_dominance_count > 0:
            result["mean_rejected_dominance_ratio_per_batch"] = rejected_dominance_ratio_sum / rejected_dominance_count

    return result


def compute_oracle_extremes(samples):
    """For each sample, find the oracle's best and worst non-truncated completion indices."""
    results = []
    for sample in samples:
        completions = sample["completions"]
        valid = []
        for i, c in enumerate(completions):
            if c.get("truncated", 0):
                continue
            valid.append((i, c["overall_score"]))

        if len(valid) < 2:
            results.append({"oracle_best_idx": -1, "oracle_worst_idx": -1,
                            "oracle_best_score": -1.0, "oracle_worst_score": -1.0,
                            "oracle_best_response": "", "oracle_worst_response": ""})
        else:
            best_idx, best_score = max(valid, key=lambda x: x[1])
            worst_idx, worst_score = min(valid, key=lambda x: x[1])
            results.append({"oracle_best_idx": best_idx, "oracle_worst_idx": worst_idx,
                            "oracle_best_score": best_score, "oracle_worst_score": worst_score,
                            "oracle_best_response": completions[best_idx]["response_text"],
                            "oracle_worst_response": completions[worst_idx]["response_text"]})
    return results


def compute_modelwise_kpis(samples, rewards) -> dict:
    """Average reward and uncertainty per model across all completions in the batch."""
    _rewards, _lower_bounds, _upper_bounds = rewards.unbind(-1)
    _uncertainty = (_upper_bounds - _lower_bounds) / 2

    from collections import defaultdict
    model_rewards = defaultdict(list)
    model_uncertainties = defaultdict(list)

    for i, sample in enumerate(samples):
        for j, comp in enumerate(sample["completions"]):
            model_name = comp["model"]
            model_rewards[model_name].append(_rewards[i, j].item())
            model_uncertainties[model_name].append(_uncertainty[i, j].item())

    kpis = {}
    for model_name in sorted(model_rewards):
        r = model_rewards[model_name]
        u = model_uncertainties[model_name]
        kpis[f"modelwise_mean_reward/{model_name}"] = sum(r) / len(r)
        kpis[f"modelwise_mean_uncertainty/{model_name}"] = sum(u) / len(u)
    return kpis


def restructure_sample(x: dict) -> dict:
    if isinstance(x["prompt"], list):
        prompt_messages = x["prompt"]
    else:
        prompt_messages = [{"role": "user", "content": x["prompt"]}]
    for key in ["chosen", "rejected"]:
        x[key] = prompt_messages + [{"role": "assistant", "content": x[key]}]
    return x


def get_new_regularization(
    n_done: int,
    n_total: int,
    decay_type: str,
    initial_value: float,
    exponential_decay_base: float = None,
    exponential_decay_scaler: float = None,
) -> float:
    initial_value = float(initial_value)
    if decay_type == "linear":
        return initial_value * (1.0 - n_done / n_total)
    elif decay_type == "exponential":
        frac_done = n_done / n_total
        exponent = float(exponential_decay_scaler) * frac_done
        return initial_value * (float(exponential_decay_base) ** exponent)
    else:
        raise ValueError(f"{decay_type=} not supported")


def save_loop_checkpoint(
    save_dir: str,
    args,
    loop_state: dict,
    replay_buffer,
    output_data: list,
    trainer=None,
    model=None,
    remaining_indices=None,
):
    """Saves loop state, buffer, data, AND RNG states."""
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)

    with open(os.path.join(save_dir, "loop_state.json"), "w") as f:
        json.dump(loop_state, f, indent=2)

    with open(os.path.join(save_dir, "replay_buffer.pkl"), "wb") as f:
        pickle.dump(replay_buffer, f)

    with open(os.path.join(save_dir, "output_list.pkl"), "wb") as f:
        pickle.dump(output_data, f)

    if trainer is not None:
        trainer.save_model(save_dir)
        trainer.save_state()
        if trainer.optimizer is not None:
            torch.save(trainer.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        if trainer.lr_scheduler is not None:
            torch.save(trainer.lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    elif model is not None:
        model.save_pretrained(save_dir)

    if remaining_indices is not None:
        with open(os.path.join(save_dir, "remaining_indices.json"), "w") as f:
            json.dump(remaining_indices, f)

    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    with open(os.path.join(save_dir, "rng_states.pkl"), "wb") as f:
        pickle.dump(rng_states, f)


def load_loop_checkpoint(checkpoint_dir: str):
    """Loads loop state and returns data + RNG states dict."""
    
    # 1. Load Data
    with open(os.path.join(checkpoint_dir, "loop_state.json"), "r") as f:
        loop_state = json.load(f)

    with open(os.path.join(checkpoint_dir, "replay_buffer.pkl"), "rb") as f:
        replay_buffer = pickle.load(f)

    with open(os.path.join(checkpoint_dir, "output_list.pkl"), "rb") as f:
        output_data = pickle.load(f)

    # 2. Load RNG States (but do NOT apply them yet)
    rng_states = None
    rng_path = os.path.join(checkpoint_dir, "rng_states.pkl")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng_states = pickle.load(f)

    remaining_indices = None
    ri_path = os.path.join(checkpoint_dir, "remaining_indices.json")
    if os.path.exists(ri_path):
        with open(ri_path, "r") as f:
            remaining_indices = json.load(f)

    return loop_state, replay_buffer, output_data, rng_states, remaining_indices