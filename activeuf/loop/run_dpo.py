"""Reward-Margin-Guided Active DPO loop.

For each batch of B prompts, per prompt we form all unordered completion pairs,
sort by actual reward gap, and pick the highest-gap pair whose implicit reward
margin under the current policy is below threshold C (MaxMin fallback otherwise).
The B selected pairs are then used for continual DPO gradient steps -- the
optimizer + LR scheduler are created once for the whole run, so each batch's
steps are "in the middle" of one long DPO schedule.

Launch (DeepSpeed ZeRO-2, full fine-tuning of a 7B policy):

    accelerate launch \
        --config_file=configs/accelerate/deepspeed2.yaml \
        --num_processes <N> \
        -m activeuf.loop.run_dpo --config_path configs/loop_dpo.yaml
"""

import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler, broadcast_object_list
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

import wandb
import wandb.sdk.lib.server

# On these CSCS nodes wandb's viewer query can return None, making
# query_with_timeout crash with a TypeError; tolerate a missing/non-str flags
# payload so wandb.init survives (mirrors activeuf/loop/run.py).
_orig_query_with_timeout = wandb.sdk.lib.server.Server.query_with_timeout


def _patched_query_with_timeout(self):
    try:
        _orig_query_with_timeout(self)
    except TypeError:
        flags = self._viewer.get("flags") if getattr(self, "_viewer", None) else None
        self._flags = json.loads(flags) if isinstance(flags, str) else {}


wandb.sdk.lib.server.Server.query_with_timeout = _patched_query_with_timeout

from accelerate import PartialState
from trl import DPOTrainer
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt

from activeuf.dpo.trainer import NormedDPOTrainer, NormedDPOConfig
from activeuf.loop.dpo_utils import (
    compute_logps_no_grad,
    load_policy_and_reference,
    select_pair_for_prompt,
)
from activeuf.utils import get_logger, get_timestamp, set_seed


def get_args() -> SimpleNamespace:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    cli, extras = parser.parse_known_args()
    with open(cli.config_path) as f:
        cfg = yaml.safe_load(f)
    # apply --key=value / --key value overrides (e.g. --threshold=2.5) onto the config
    i = 0
    while i < len(extras):
        tok = extras[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if "=" in key:
            key, val = key.split("=", 1)
            i += 1
        else:
            val = extras[i + 1] if i + 1 < len(extras) else "true"
            i += 2
        cfg[key] = yaml.safe_load(val)   # parses numbers/bools/strings
    return SimpleNamespace(**cfg)


def _pcts(values, qs=(10, 50, 90, 99)):
    """Percentiles of a list of floats, as {q: value}. Empty -> {}."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {q: float(np.percentile(arr, q)) for q in qs}


# Metric keys returned by dpo_loss_for_batch; fixed so the cross-rank reduce
# tensor has the same shape on every process even if a rank gets no micro-batch.
DPO_METRIC_KEYS = [
    "dpo/loss", "dpo/rewards_chosen", "dpo/rewards_rejected",
    "dpo/reward_margin", "dpo/reward_accuracy",
    "dpo/logps_chosen", "dpo/logps_rejected",
    "dpo/logits_chosen", "dpo/logits_rejected",
    "dpo/kl_chosen", "dpo/kl_rejected",
]


def save_pretrained_dir(accelerator, model, tokenizer, save_dir):
    """Write config.json / generation_config.json / tokenizer next to the weights
    that accelerator.save_model() already wrote, so the dir is a self-contained,
    HF/vLLM-loadable model (save_model writes weights only)."""
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.config.save_pretrained(save_dir)
    if getattr(unwrapped, "generation_config", None) is not None:
        unwrapped.generation_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def as_messages(prompt):
    """Normalize a prompt (str or message list) to a chat message list."""
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]


def valid_completions(sample):
    """Return (texts, rewards, indices, models) for non-truncated completions of a sample."""
    texts, rewards, idxs, models = [], [], [], []
    for j, c in enumerate(sample["completions"]):
        if c.get("truncated", 0):
            continue
        texts.append(c["response_text"])
        rewards.append(float(c["overall_score"]))
        idxs.append(j)
        models.append(c.get("model"))
    return texts, rewards, idxs, models


def main():
    args = get_args()

    # Fix the effective (global) batch size; derive grad-accum from the GPU count
    # (WORLD_SIZE, known before Accelerator init) so ANY node count yields the same
    # effective batch = per_device * grad_accum * world_size ~= effective_batch_size.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    target_batch = int(getattr(args, "effective_batch_size", 128))
    # Always use gradient accumulation (gas >= 2): DeepSpeed's no-accumulation path
    # (gas=1) mis-scales the gradient ~2x. Cap per_device (the config value is an upper
    # bound, for memory) so that gas >= 2 at ANY node count -> robust to varying nodes.
    per_device = min(args.per_device_train_batch_size, max(1, target_batch // (2 * world_size)))
    grad_accum = max(2, round(target_batch / (per_device * world_size)))

    # Init distributed state for setup/acquisition collectives. The DeepSpeed-enabled
    # Accelerator is created and owned by the NormedDPOTrainer below (single Accelerator).
    state = PartialState()
    n_proc = state.num_processes
    device = state.device

    load_dotenv(getattr(args, "env_local_path", ".env.local"))

    if state.is_main_process:
        timestamp = get_timestamp(more_detailed=True)
    else:
        timestamp = ""
    timestamp = broadcast_object_list([timestamp])[0]
    run_id = getattr(args, "run_id", "") or f"activedpo_C{args.threshold}_{timestamp}"
    output_path = os.path.join(args.base_output_dir, run_id)
    logs_path = os.path.join(getattr(args, "base_logs_dir", args.base_output_dir),
                             f"{run_id}.log")

    if state.is_main_process:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)

    logger = get_logger(__name__, logs_path, state)
    if args.seed:
        set_seed(args.seed)

    # --- data --------------------------------------------------------------- #
    logger.info(f"Loading prompts from {args.inputs_path}")
    dataset = load_from_disk(args.inputs_path)
    if getattr(args, "debug", False):
        debug_n = int(getattr(args, "debug_n", 1001))
        dataset = dataset.select(range(min(debug_n, len(dataset))))
        logger.info(f"DEBUG: limiting to {len(dataset)} prompts")

    # keep only prompts with >= min_non_truncated usable completions
    min_keep = getattr(args, "min_non_truncated", 2)
    keep = [i for i, comps in enumerate(dataset["completions"])
            if sum(1 for c in comps if not c.get("truncated", 0)) >= min_keep]
    dataset = dataset.select(keep)
    dataset = dataset.shuffle(seed=args.seed)
    num_prompts = len(dataset)
    if state.is_main_process:
        logger.info(f"# usable prompts: {num_prompts}")

    # --- models ------------------------------------------------------------- #
    logger.info(f"Loading policy + frozen reference from {args.model_path}")
    policy, reference, tokenizer = load_policy_and_reference(
        args.model_path,
        torch_dtype=torch.bfloat16,
        gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
    )

    # global batch size and total optimizer steps for ONE pass over the dataset
    global_batch = per_device * grad_accum * n_proc
    steps_per_full_batch = max(1, args.outer_loop_batch_size // global_batch)
    total_outer_batches = math.ceil(num_prompts / args.outer_loop_batch_size)
    total_steps = math.ceil(num_prompts / global_batch) * int(getattr(args, "num_train_epochs", 1))
    if state.is_main_process:
        logger.info(
            f"effective_batch={global_batch} (target {target_batch}: per_device {per_device} "
            f"x grad_accum {grad_accum} x world {n_proc}) | steps/outer_batch~{steps_per_full_batch} | "
            f"total_outer_batches={total_outer_batches} | total_optimizer_steps={total_steps}"
        )

    # --- TRL trainer as the training engine -------------------------------- #
    # Reuse NormedDPOTrainer/DPOTrainer for the DPO loss (+ normalize_logps), the exact
    # TRL tokenization (tokenize_row + DataCollatorForPreference), and a real torch
    # AdamW + linear scheduler -- identical to the TRL baseline -- instead of the old
    # hand-rolled loss + DeepSpeed-native WarmupDecayLR/DummyOptim path (which produced
    # an over-verbose policy). The per-outer-batch acquisition/selection and step
    # accounting are unchanged. We drive the optimizer manually in the outer loop, so
    # we never call trainer.train().
    dpo_cfg = NormedDPOConfig(
        output_dir=output_path,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(args.learning_rate),
        lr_scheduler_type=getattr(args, "lr_scheduler_type", "linear"),
        warmup_ratio=float(args.warmup_ratio),
        num_train_epochs=1,
        max_steps=total_steps,
        bf16=True,
        beta=float(args.beta),
        max_length=args.max_length,
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
        max_grad_norm=float(getattr(args, "max_grad_norm", 1.0)),
        normalize_logps=bool(getattr(args, "length_normalize", False)),
        loss_type="sigmoid",
        precompute_ref_log_probs=False,
        remove_unused_columns=False,
        report_to=[],
        logging_strategy="no",
        dataset_num_proc=1,
    )
    _dummy = Dataset.from_list([{
        "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "no"}],
    }])
    trainer = NormedDPOTrainer(
        model=policy, ref_model=reference, args=dpo_cfg,
        train_dataset=_dummy, processing_class=tokenizer,
    )
    accelerator = trainer.accelerator           # the (DeepSpeed) accelerator owns training
    device = accelerator.device
    reference = trainer.ref_model               # DeepSpeed-inference-prepared frozen reference

    # We drive training manually (no trainer.train()), so HF Trainer never resolves the
    # DeepSpeed "auto" batch fields -- set them here. The optimizer AND LR schedule live
    # in the DeepSpeed config so the ENGINE owns and steps them natively: a client-side
    # torch scheduler (create_scheduler + accelerator.prepare) gets disconnected from the
    # engine under accelerate+DeepSpeed and the applied LR stays 0 (the lr=0 bug). We use
    # DummyOptim/DummyScheduler + a linear WarmupDecayLR (== the TRL linear LR schedule).
    warmup_steps = int(args.warmup_ratio * total_steps)
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is not None:
        ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = per_device
        ds_plugin.deepspeed_config["gradient_accumulation_steps"] = grad_accum
        ds_plugin.deepspeed_config["train_batch_size"] = global_batch
        ds_plugin.deepspeed_config["gradient_clipping"] = float(getattr(args, "max_grad_norm", 1.0))
        ds_plugin.deepspeed_config["optimizer"] = {
            "type": "AdamW",
            "params": {"lr": float(args.learning_rate), "betas": [0.9, 0.999],
                       "eps": 1e-8, "weight_decay": 0.0},
        }
        ds_plugin.deepspeed_config["scheduler"] = {
            "type": "WarmupDecayLR",
            "params": {"warmup_min_lr": 0.0, "warmup_max_lr": float(args.learning_rate),
                       "warmup_num_steps": warmup_steps, "total_num_steps": total_steps,
                       "warmup_type": "linear"},
        }
        optimizer = DummyOptim(trainer.model.parameters(), lr=float(args.learning_rate))
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=total_steps, warmup_num_steps=warmup_steps)
    else:  # non-DeepSpeed fallback (single-GPU debugging)
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=float(args.learning_rate))
        lr_scheduler = get_scheduler(
            name=getattr(args, "lr_scheduler_type", "linear"), optimizer=optimizer,
            num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    policy, optimizer, lr_scheduler = accelerator.prepare(trainer.model, optimizer, lr_scheduler)

    # Sanity: what grad-accum did the DeepSpeed ENGINE actually end up with? accelerate's
    # gas and the deepspeed config's gas can silently desync (-> wrong 1/gas loss scaling).
    if accelerator.is_main_process:
        ds_gas = None
        try:
            ds_gas = policy.gradient_accumulation_steps()
        except Exception:
            ds_cfg = getattr(getattr(accelerator.state, "deepspeed_plugin", None), "deepspeed_config", {})
            ds_gas = ds_cfg.get("gradient_accumulation_steps")
        logger.info(f"DeepSpeed engine gradient_accumulation_steps={ds_gas} (intended grad_accum={grad_accum})")

    if accelerator.is_main_process:
        os.environ.setdefault("WANDB_ENTITY", getattr(args, "wandb_entity", "ActiveUF_Plus"))
        wandb.init(project=getattr(args, "wandb_project", "active_dpo"),
                   id=run_id, config=vars(args))

    acq_micro_bs = getattr(args, "acquisition_micro_batch_size", 64)        # max seqs/batch
    acq_token_budget = getattr(args, "acquisition_token_budget", 16384)     # count*len cap/batch
    length_normalize = bool(getattr(args, "length_normalize", False))
    beta = float(args.beta)
    threshold = float(args.threshold)

    output = []
    global_step = 0                                              # cumulative optimizer steps
    last_ckpt_step = 0
    save_every_n_steps = int(getattr(args, "save_every_n_steps", 25))
    step_times = []                                             # per-outer-batch wall times (for ETA)
    loop_start = time.time()

    # =============================== OUTER LOOP =============================== #
    for outer_idx in range(total_outer_batches):
        t0 = time.time()
        start = outer_idx * args.outer_loop_batch_size
        end = min(start + args.outer_loop_batch_size, num_prompts)
        batch = dataset.select(range(start, end))
        samples = [batch[i] for i in range(len(batch))]

        # ---- flatten all valid (prompt, completion) for sharded logp scoring -- #
        per_prompt = [valid_completions(s) for s in samples]    # list of (texts, rewards, idxs)
        flat = []  # (prompt_pos, prompt, completion_text)
        for pi, (texts, _, _, _) in enumerate(per_prompt):
            for txt in texts:
                flat.append((pi, samples[pi]["prompt"], txt))
        total = len(flat)

        # shard across processes
        chunk = math.ceil(total / n_proc)
        lo = accelerator.process_index * chunk
        hi = min(lo + chunk, total)
        local = flat[lo:hi]
        local_items = [(p, t) for (_, p, t) in local]

        policy.eval()
        show = accelerator.is_main_process
        pol_logp = compute_logps_no_grad(
            policy, tokenizer, local_items, args.max_length, length_normalize, device,
            acq_micro_bs, acq_token_budget, desc=f"score policy (b{outer_idx + 1})", show_progress=show)
        ref_logp = compute_logps_no_grad(
            reference, tokenizer, local_items, args.max_length, length_normalize, device,
            acq_micro_bs, acq_token_budget, desc=f"score ref (b{outer_idx + 1})", show_progress=show)

        # pad to `chunk` and gather (2 channels: policy, ref)
        local_pair = torch.zeros(chunk, 2)
        local_pair[: hi - lo, 0] = pol_logp
        local_pair[: hi - lo, 1] = ref_logp
        gathered = accelerator.gather(local_pair.to(device)).cpu()[:total]

        # ---- acquisition (main process) ----------------------------------- #
        hist_logs = {}   # wandb.Histogram objects, main-only (not broadcast)
        if accelerator.is_main_process:
            # scatter gathered logps back to per-prompt lists
            pol_by_prompt, ref_by_prompt = {}, {}
            for k, (pi, _, _) in enumerate(flat):
                pol_by_prompt.setdefault(pi, []).append(gathered[k, 0].item())
                ref_by_prompt.setdefault(pi, []).append(gathered[k, 1].item())

            selected = []
            all_margins_batch = []   # every candidate pair's margin (uncensored)
            top_gap_margins = []     # per-prompt rank-order pick margin (uncensored)
            deviations = []          # 1.0 if selection deviated from rank order
            n_cand = []
            rank_diffs = []          # rank(rejected) - rank(chosen) per selected pair
            chosen_ranks = []        # 1 = best (from top)
            rejected_ranks = []      # 1 = worst (from bottom)
            for pi, (texts, rewards, _, models) in enumerate(per_prompt):
                if len(texts) < 2:
                    continue
                # beta * (logp_policy - logp_ref): same scale as the DPO reward margin
                implicit = [beta * (p - r) for p, r in zip(pol_by_prompt[pi], ref_by_prompt[pi])]
                c, r, fallback, diag = select_pair_for_prompt(rewards, implicit, threshold)
                selected.append({
                    "prompt_id": samples[pi].get("prompt_id"),
                    "prompt": samples[pi]["prompt"],
                    "chosen": texts[c],
                    "rejected": texts[r],
                    "chosen_model": models[c],
                    "rejected_model": models[r],
                    "chosen_score": rewards[c],
                    "rejected_score": rewards[r],
                    "reward_gap": rewards[c] - rewards[r],
                    "implicit_margin": diag["selected_margin"],
                    "used_fallback": fallback,
                    # reference logps reused by training so it skips the ref forward
                    "ref_chosen_logp": ref_by_prompt[pi][c],
                    "ref_rejected_logp": ref_by_prompt[pi][r],
                })
                all_margins_batch.extend(diag["all_margins"])
                top_gap_margins.append(diag["top_gap_margin"])
                deviations.append(0.0 if diag["selected_is_top_gap"] else 1.0)
                n_cand.append(diag["n_candidate_pairs"])
                rank_diffs.append(diag["rank_difference"])
                chosen_ranks.append(diag["chosen_rank"])
                rejected_ranks.append(diag["rejected_rank"])

            ns = max(1, len(selected))
            sel_margins = [s["implicit_margin"] for s in selected]
            sel_gaps = [s["reward_gap"] for s in selected]
            stats = {
                "select/n_pairs": len(selected),
                "select/fallback_fraction": sum(s["used_fallback"] for s in selected) / ns,
                "select/deviation_from_rankorder_fraction": sum(deviations) / ns,
                "select/mean_n_candidate_pairs": sum(n_cand) / ns,
                "select/mean_reward_gap": sum(sel_gaps) / ns,
                "select/mean_implicit_margin": sum(sel_margins) / ns,
                "select/mean_rank_difference": sum(rank_diffs) / ns,
                "select/mean_chosen_rank": sum(chosen_ranks) / ns,
                "select/mean_rejected_rank": sum(rejected_ranks) / ns,
                "select/mean_chosen_score": sum(s["chosen_score"] for s in selected) / ns,
                "select/mean_rejected_score": sum(s["rejected_score"] for s in selected) / ns,
            }
            # selected-pair (censored at C) distributions
            for q, v in _pcts(sel_margins).items():
                stats[f"select/implicit_margin_p{q}"] = v
            for q, v in _pcts(sel_gaps).items():
                stats[f"select/reward_gap_p{q}"] = v
            # uncensored margin distributions -- the threshold-calibration signal
            for q, v in _pcts(top_gap_margins).items():
                stats[f"margin_uncensored/top_gap_p{q}"] = v
            for q, v in _pcts(all_margins_batch).items():
                stats[f"margin_uncensored/all_pairs_p{q}"] = v
            if top_gap_margins:
                stats["margin_uncensored/top_gap_mean"] = sum(top_gap_margins) / len(top_gap_margins)
                stats["margin_uncensored/top_gap_max"] = max(top_gap_margins)
                stats["margin_uncensored/top_gap_min"] = min(top_gap_margins)
                hist_logs["margin_uncensored/top_gap_hist"] = wandb.Histogram(top_gap_margins)
            if all_margins_batch:
                hist_logs["margin_uncensored/all_pairs_hist"] = wandb.Histogram(all_margins_batch)
        else:
            selected, stats = None, None

        selected, stats = broadcast_object_list([selected, stats])
        acq_time = time.time() - t0

        # ---- DPO training on the selected pairs (TRL machinery) ----------- #
        t_train0 = time.time()
        policy.train()

        # TRL-exact preprocessing: conversational pair -> extract prompt -> apply chat
        # template -> tokenize_row. Replaces the old joint-tokenize (kills the off-by-one)
        # and uses TRL's prompt/completion truncation (max_prompt_length / max_length).
        rows = []
        for s in selected:
            pm = as_messages(s["prompt"])
            ex = {
                "chosen": pm + [{"role": "assistant", "content": s["chosen"]}],
                "rejected": pm + [{"role": "assistant", "content": s["rejected"]}],
            }
            ex = maybe_extract_prompt(ex)
            ex = maybe_apply_chat_template(ex, tokenizer)
            rows.append(DPOTrainer.tokenize_row(
                ex, tokenizer, dpo_cfg.max_prompt_length,
                dpo_cfg.max_completion_length, add_special_tokens=False,
            ))
        loader = DataLoader(
            rows, batch_size=per_device, shuffle=True, drop_last=False,
            collate_fn=trainer.data_collator,
        )
        loader = accelerator.prepare(loader)

        metric_sums, n_micro, n_opt_steps, grad_norm_sum = {}, 0, 0, 0.0
        train_iter = tqdm(loader, desc=f"train (b{outer_idx + 1})", leave=False) \
            if accelerator.is_main_process else loader
        for micro in train_iter:
            # accumulate() gates the optimizer/scheduler step to the gas-th micro-batch
            # (one optimizer + one scheduler step per grad_accum micro-batches), preserving
            # the intended effective batch and the LR schedule. The loss + all metrics come
            # from TRL's NormedDPOTrainer (get_batch_loss_metrics already gathers across ranks).
            with accelerator.accumulate(policy):
                loss, m = trainer.get_batch_loss_metrics(policy, micro, train_eval="train")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            n_micro += 1
            m = dict(m)
            m["loss"] = accelerator.gather_for_metrics(loss.detach()).mean().item()
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            if n_micro % grad_accum == 0:         # the gas-th micro: a real optimizer update
                gn = policy.get_global_grad_norm() if hasattr(policy, "get_global_grad_norm") else None
                if gn is not None:
                    grad_norm_sum += float(gn)    # already a global (cross-rank) norm
                n_opt_steps += 1

        # get_batch_loss_metrics already returns global (gathered) means; average over this
        # outer batch's micro-batches. Keys already match TRL's (loss, rewards/*, logps/*, ...).
        train_metrics = {k: metric_sums[k] / max(1, n_micro) for k in metric_sums}
        train_time = time.time() - t_train0
        global_step += n_opt_steps

        # ---- logging / checkpoint ----------------------------------------- #
        if accelerator.is_main_process:
            batch_elapsed = time.time() - t0
            step_times.append(batch_elapsed)
            avg_batch = sum(step_times) / len(step_times)
            remaining = total_outer_batches - (outer_idx + 1)
            eta_s = avg_batch * remaining
            eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)

            output.extend(selected)
            log = dict(stats)
            log.update(train_metrics)
            log["learning_rate"] = optimizer.param_groups[0]["lr"]   # ground-truth LR DeepSpeed applies
            log["grad_norm"] = grad_norm_sum / max(1, n_opt_steps)
            log["epoch"] = end / num_prompts                         # fraction of the (single) pass done
            log["train/n_optimizer_steps"] = n_opt_steps
            log["timing/acq_seconds"] = acq_time
            log["timing/train_seconds"] = train_time
            log["timing/eta_hours"] = eta_s / 3600
            log["progress/outer_batch"] = outer_idx
            wandb.log({**log, **hist_logs}, step=global_step)   # x-axis = cumulative optimizer steps (~2k)
            logger.info(
                f"Batch {outer_idx + 1}/{total_outer_batches} | "
                f"pairs={stats['select/n_pairs']} "
                f"fb={stats['select/fallback_fraction']:.2f} "
                f"dev={stats['select/deviation_from_rankorder_fraction']:.2f} "
                f"gap={stats['select/mean_reward_gap']:.3f} "
                f"sel_margin={stats['select/mean_implicit_margin']:.3f} "
                f"top_gap_p90={stats.get('margin_uncensored/top_gap_p90', 0.0):.3f} "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['rewards/accuracies']:.3f} "
                f"lr={log['learning_rate']:.2e} gnorm={log['grad_norm']:.2f} steps={n_opt_steps} "
                f"| acq={acq_time:.0f}s tr={train_time:.0f}s tot={batch_elapsed:.0f}s "
                f"eta={eta_h}h{eta_m}m"
            )

        # ---- checkpoint every save_every_n_steps optimizer steps ----------- #
        if global_step - last_ckpt_step >= save_every_n_steps:
            last_ckpt_step = global_step
            ckpt_dir = os.path.join(output_path, f"checkpoint-step{global_step}")
            accelerator.wait_for_everyone()
            accelerator.save_model(policy, ckpt_dir)            # collective (ZeRO consolidate)
            if accelerator.is_main_process:
                save_pretrained_dir(accelerator, policy, tokenizer, ckpt_dir)
                Dataset.from_list(output).save_to_disk(os.path.join(output_path, "selected_pairs"))
                logger.info(f"Saved checkpoint at step {global_step} -> {ckpt_dir}")

    # ----------------------------- finish ---------------------------------- #
    accelerator.wait_for_everyone()
    accelerator.save_model(policy, os.path.join(output_path, "policy"))
    if accelerator.is_main_process:
        save_pretrained_dir(accelerator, policy, tokenizer, os.path.join(output_path, "policy"))
        Dataset.from_list(output).save_to_disk(os.path.join(output_path, "selected_pairs"))
        logger.info(f"Done in {time.time() - loop_start:.0f}s. Saved to {output_path}")
        wandb.finish()


if __name__ == "__main__":
    main()
