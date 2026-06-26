"""Helpers for the reward-margin-guided active DPO loop (activeuf.loop.run_dpo).

Everything the loop needs that is DPO-specific lives here:
  * model loading (policy + frozen reference),
  * per-sequence completion log-probabilities (used both for the acquisition
    implicit-margin score and for the DPO loss),
  * the DPO sigmoid loss (numerically identical to NormedDPOTrainer),
  * the per-prompt pair-selection rule (Algorithm: Reward-Margin-Guided Active DPO).

The logp definition matches NormedDPOTrainer: a completion's score is the sum of
its token log-probs, optionally divided by the number of completion tokens when
`length_normalize=True`. The same flag is used for the acquisition Delta_pi and
the loss, so selection and optimization use the same logp definition.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_policy_and_reference(model_path: str, torch_dtype=torch.bfloat16,
                              gradient_checkpointing: bool = True):
    """Load the trainable policy and a separate frozen reference (full fine-tuning).

    Returns (policy, reference, tokenizer). The reference is put in eval mode with
    grads disabled; the caller is responsible for moving it to the right device.
    """
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if gradient_checkpointing else True,
    )
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()

    reference = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=True,
    )
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # Some Olmo/SFT checkpoints ship a generation_config that is invalid with
    # do_sample=False (temperature/top_p set); mirror dpo/training.py's fix.
    for m in (policy, reference):
        if hasattr(m, "generation_config") and not m.generation_config.do_sample:
            m.generation_config.temperature = None
            m.generation_config.top_p = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return policy, reference, tokenizer


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def _as_messages(prompt) -> list:
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": prompt}]


def tokenize_prompt_completion(tokenizer, prompt, completion: str, max_length: int
                               ) -> Tuple[List[int], List[int]]:
    """Tokenize prompt+completion and return (input_ids, completion_mask).

    completion_mask[i] == 1 iff input_ids[i] is a completion (assistant) token,
    i.e. the prompt tokens are masked out. Truncated to max_length from the right.
    """
    messages = _as_messages(prompt)
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        tokenize=True, add_generation_prompt=False,
    )
    prompt_len = len(prompt_ids)
    full_ids = full_ids[:max_length]
    completion_mask = [0] * min(prompt_len, len(full_ids)) \
        + [1] * max(0, len(full_ids) - prompt_len)
    return full_ids, completion_mask


def _pad_batch(seqs: List[List[int]], masks: List[List[int]], pad_id: int, device):
    """Right-pad a batch of token id / completion-mask lists to a common length."""
    max_len = max(len(s) for s in seqs)
    input_ids, attention_mask, comp_mask = [], [], []
    for s, m in zip(seqs, masks):
        pad = max_len - len(s)
        input_ids.append(s + [pad_id] * pad)
        attention_mask.append([1] * len(s) + [0] * pad)
        comp_mask.append(m + [0] * pad)
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(attention_mask, device=device),
        torch.tensor(comp_mask, device=device, dtype=torch.float32),
    )


# --------------------------------------------------------------------------- #
# Sequence log-probabilities
# --------------------------------------------------------------------------- #
def _sequence_logps(model, input_ids, attention_mask, comp_mask, length_normalize,
                    return_logits=False):
    """Sum (or mean over completion tokens) of token log-probs for each sequence.

    Shapes: input_ids/attention_mask/comp_mask are (B, L). Returns (B,).
    Differentiable when called under grad; caller controls the context.

    Memory note: -cross_entropy == log-prob of the actual next token. Its fused
    kernel accumulates in fp32 internally, so we avoid materializing a full
    (B, L, V) float32 log-softmax (which OOMs for large vocab x long sequences).

    With return_logits=True also returns, per sequence, the summed mean-over-vocab
    raw logit at the completion positions and the completion-token count, so the
    caller can pool them into TRL's logits/chosen, logits/rejected diagnostics.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # token at position t is predicted from logits at t-1
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = comp_mask[:, 1:]
    per_token_logp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
    ).view(labels.shape).float()
    seq_logp = (per_token_logp * mask).sum(dim=-1)
    if length_normalize:
        seq_logp = seq_logp / mask.sum(dim=-1).clamp(min=1.0)
    if not return_logits:
        return seq_logp
    # per-token mean logit over vocab (reduce in bf16 to avoid a (B, L, V) fp32
    # temp), masked to completion positions; the caller pools across the batch.
    tok_logit_mean = logits.mean(dim=-1).float()          # (B, L-1)
    seq_logit_sum = (tok_logit_mean * mask).sum(dim=-1)   # (B,)
    seq_tok_count = mask.sum(dim=-1)                       # (B,)
    return seq_logp, seq_logit_sum, seq_tok_count


@torch.no_grad()
def compute_logps_no_grad(model, tokenizer, items, max_length, length_normalize,
                          device, micro_bs, token_budget, desc=None, show_progress=False) -> torch.Tensor:
    """Per-completion logps for acquisition. `items` is a list of (prompt, completion).

    Returns a CPU float tensor of shape (len(items),), in the original item order.

    Throughput: each prompt is tokenized once (its prompt-only length is cached and
    reused for all its completions), and completions are scored in LENGTH-SORTED
    micro-batches so short sequences are not padded up to the longest in the batch.
    """
    n = len(items)
    if n == 0:
        return torch.zeros(0)

    # tokenize once per item; cache the (otherwise redundant) prompt-only length per prompt
    prompt_len_cache = {}
    toks = []
    for prompt, completion in items:
        key = id(prompt)
        if key not in prompt_len_cache:
            prompt_len_cache[key] = len(tokenizer.apply_chat_template(
                _as_messages(prompt), tokenize=True, add_generation_prompt=True))
        prompt_len = prompt_len_cache[key]
        full = tokenizer.apply_chat_template(
            _as_messages(prompt) + [{"role": "assistant", "content": completion}],
            tokenize=True, add_generation_prompt=False,
        )[:max_length]
        mask = [0] * min(prompt_len, len(full)) + [1] * max(0, len(full) - prompt_len)
        toks.append((full, mask))

    # length-sort, then pack into batches bounded by a token budget (count * the
    # batch's max length): short completions batch in large groups, long ones in
    # small groups. Keeps the GPU busy on short seqs without OOM-ing on long ones,
    # and caps the (count x len x vocab) logits tensor regardless of length.
    # micro_bs is an upper cap on the per-batch count.
    order = sorted(range(n), key=lambda i: len(toks[i][0]))
    batches, i = [], 0
    while i < n:
        j = i + 1
        while (j < n and (j - i + 1) <= micro_bs
               and (j - i + 1) * len(toks[order[j]][0]) <= token_budget):
            j += 1
        batches.append(order[i:j])
        i = j

    out = torch.zeros(n)
    it = tqdm(batches, desc=desc, leave=False) if show_progress else batches
    for idx in it:
        seqs = [toks[k][0] for k in idx]
        masks = [toks[k][1] for k in idx]
        input_ids, attn, cmask = _pad_batch(seqs, masks, tokenizer.pad_token_id, device)
        logp = _sequence_logps(model, input_ids, attn, cmask, length_normalize)
        for jj, k in enumerate(idx):
            out[k] = logp[jj].item()
    return out


# --------------------------------------------------------------------------- #
# DPO loss (matches NormedDPOTrainer, loss_type="sigmoid")
# --------------------------------------------------------------------------- #
def dpo_loss_for_batch(policy, tokenizer, pairs, beta, max_length,
                       length_normalize, device):
    """Compute the mean DPO loss + metrics for a micro-batch of preference pairs.

    `pairs` is a list of dicts with keys: prompt, chosen, rejected (text), plus the
    precomputed reference logps ref_chosen_logp / ref_rejected_logp. The frozen
    reference was already scored during acquisition, so it is NOT re-run here.

    Chosen and rejected are scored in ONE concatenated policy forward (chosen
    rows first, then rejected) so every parameter is used exactly once per
    backward. This is required under ZeRO-2, which asserts if a parameter's
    gradient is reduced twice (which happens with two separate forwards).
    """
    n = len(pairs)
    seqs, masks = [], []
    for p in pairs:                                   # chosen rows [0:n]
        ids, m = tokenize_prompt_completion(tokenizer, p["prompt"], p["chosen"], max_length)
        seqs.append(ids)
        masks.append(m)
    for p in pairs:                                   # rejected rows [n:2n]
        ids, m = tokenize_prompt_completion(tokenizer, p["prompt"], p["rejected"], max_length)
        seqs.append(ids)
        masks.append(m)
    input_ids, attn, cmask = _pad_batch(seqs, masks, tokenizer.pad_token_id, device)

    pol_logps, logit_sum, tok_count = _sequence_logps(
        policy, input_ids, attn, cmask, length_normalize, return_logits=True)
    pol_chosen, pol_rejected = pol_logps[:n], pol_logps[n:]

    # reference logps were already computed during acquisition (frozen model)
    ref_chosen = torch.tensor([p["ref_chosen_logp"] for p in pairs],
                              device=device, dtype=pol_chosen.dtype)
    ref_rejected = torch.tensor([p["ref_rejected_logp"] for p in pairs],
                                device=device, dtype=pol_rejected.dtype)

    chosen_rewards = beta * (pol_chosen - ref_chosen)
    rejected_rewards = beta * (pol_rejected - ref_rejected)
    logits = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(logits)

    # TRL logits/chosen, logits/rejected: mean raw logit over completion tokens,
    # pooled across the micro-batch (sum of per-seq sums / total completion tokens).
    logits_chosen = logit_sum[:n].sum() / tok_count[:n].sum().clamp(min=1)
    logits_rejected = logit_sum[n:].sum() / tok_count[n:].sum().clamp(min=1)
    # TRL kl/chosen, kl/rejected: policy-minus-reference log-ratio (== reward / beta)
    kl_chosen = (pol_chosen - ref_chosen).mean()
    kl_rejected = (pol_rejected - ref_rejected).mean()

    metrics = {
        "dpo/loss": losses.mean().detach(),
        "dpo/rewards_chosen": chosen_rewards.mean().detach(),
        "dpo/rewards_rejected": rejected_rewards.mean().detach(),
        "dpo/reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        "dpo/reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().detach(),
        # raw (policy) summed logps -- compare magnitude vs the TRL baseline's logps/*
        "dpo/logps_chosen": pol_chosen.mean().detach(),
        "dpo/logps_rejected": pol_rejected.mean().detach(),
        "dpo/logits_chosen": logits_chosen.detach(),
        "dpo/logits_rejected": logits_rejected.detach(),
        "dpo/kl_chosen": kl_chosen.detach(),
        "dpo/kl_rejected": kl_rejected.detach(),
    }
    return losses.mean(), metrics


# --------------------------------------------------------------------------- #
# Acquisition: per-prompt pair selection
# --------------------------------------------------------------------------- #
def select_pair_for_prompt(rewards: List[float], implicit: List[float], threshold: float
                           ) -> Tuple[int, int, bool]:
    """Reward-Margin-Guided pair selection for ONE prompt.

    Args:
        rewards:  actual annotator reward per (valid) completion.
        implicit: implicit reward per completion, i.e. beta * (logp_policy - logp_ref),
                  with logps length-normalized iff that flag is on. The margin
                  implicit[chosen] - implicit[rejected] is compared directly to C,
                  so C lives on the same (beta-scaled) scale as the DPO reward margin.
        threshold: C.

    Returns (chosen_idx, rejected_idx, used_fallback, diag). The first three index
    into the `rewards`/`implicit` lists (chosen is the higher-actual-reward member);
    `diag` is a dict of per-prompt diagnostics for logging:
        all_margins:        margin (implicit[chosen]-implicit[rejected]) of EVERY
                            candidate pair -- the uncensored distribution.
        top_gap_margin:     margin of the highest-reward-gap pair (what rank-order
                            DPO would pick); the key quantity for calibrating C.
        selected_margin:    margin of the pair actually selected.
        selected_is_top_gap: True iff the selected pair is the highest-gap pair
                            (i.e. selection did NOT deviate from rank order).
        n_candidate_pairs:  number of unordered pairs considered.

    Rule: enumerate unordered pairs, sort by |r_a - r_b| descending, take the
    first pair whose implicit margin Delta = implicit[chosen] - implicit[rejected]
    is < C. If none qualifies, fall back to the widest-gap pair: chosen = the
    highest-reward completion, rejected = the lowest-reward completion.
    """
    n = len(rewards)
    assert n >= 2, "need at least 2 valid completions to form a pair"

    pairs = []  # (gap, chosen_idx, rejected_idx)
    for a in range(n):
        for b in range(a + 1, n):
            if rewards[a] >= rewards[b]:
                c, r = a, b
            else:
                c, r = b, a
            pairs.append((abs(rewards[a] - rewards[b]), c, r))
    pairs.sort(key=lambda x: x[0], reverse=True)   # highest reward gap first

    # uncensored diagnostics over ALL candidate pairs
    all_margins = [implicit[c] - implicit[r] for _, c, r in pairs]
    top_gap_margin = all_margins[0]                # margin of the rank-order (max-gap) pick

    # threshold pass
    chosen = rejected = None
    for _, c, r in pairs:
        if (implicit[c] - implicit[r]) < threshold:
            chosen, rejected = c, r
            break

    used_fallback = chosen is None
    if used_fallback:
        # MaxMin fallback: chosen = highest-reward completion, rejected = lowest-reward.
        chosen = max(range(n), key=lambda i: rewards[i])
        rejected = min(range(n), key=lambda i: rewards[i])
        if chosen == rejected:                     # all rewards equal -> distinct indices
            rejected = (chosen + 1) % n

    # rank positions by descending reward (0 = best); chosen outranks rejected, so
    # rank_difference >= 1 (N-1 for the best-vs-worst pair, smaller for adjacent pairs)
    rank = {idx: pos for pos, idx in
            enumerate(sorted(range(n), key=lambda i: rewards[i], reverse=True))}

    diag = {
        "all_margins": all_margins,
        "top_gap_margin": top_gap_margin,
        "selected_margin": implicit[chosen] - implicit[rejected],
        "selected_is_top_gap": (chosen, rejected) == (pairs[0][1], pairs[0][2]),
        "n_candidate_pairs": len(pairs),
        "rank_difference": rank[rejected] - rank[chosen],
        "chosen_rank": rank[chosen] + 1,       # 1 = best (highest reward), counting from top
        "rejected_rank": n - rank[rejected],   # 1 = worst (lowest reward), counting from bottom
    }
    return chosen, rejected, used_fallback, diag
