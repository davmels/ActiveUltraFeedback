"""Numeric column-diff: loop's hand-rolled DPO tokenization/logp vs TRL DPOTrainer.

Feeds the SAME preference pairs through both paths using the SAME single model, so
any divergence in prompt/completion token boundaries or per-sequence logps localizes
the bug (chat-template prefix assumption, EOS handling, prompt truncation, masking).

Run (single GPU, no DeepSpeed):
    python diff_dpo_mechanics.py <PREF_DATASET_PATH> [N]

PREF_DATASET_PATH = a dataset whose `chosen`/`rejected` are message lists
                    [user..., assistant] (e.g. the direct_maxmin / reference build).
"""
import sys
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from activeuf.loop.dpo_utils import tokenize_prompt_completion, _pad_batch, _sequence_logps
from activeuf.dpo.trainer import NormedDPOTrainer, NormedDPOConfig

MODEL = "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--allenai--Olmo-3-7B-Instruct-SFT/snapshots/e1452fc572d51966ff4aaeb25118b891eb93e549"
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 512   # TRL DPOConfig default; dpo_training.yaml does NOT override it
LENGTH_NORMALIZE = False
BETA = 0.05


def split_pair(row):
    """From {chosen:[...,assistant], rejected:[...,assistant]} -> (prompt_msgs, chosen_txt, rejected_txt)."""
    prompt_msgs = row["chosen"][:-1]
    return prompt_msgs, row["chosen"][-1]["content"], row["rejected"][-1]["content"]


def loop_logps(model, tok, rows, device):
    """Loop path: tokenize_prompt_completion + _sequence_logps. Returns per-row dicts."""
    out = []
    for r in rows:
        prompt_msgs, ctext, rtext = split_pair(r)
        cids, cmask = tokenize_prompt_completion(tok, prompt_msgs, ctext, MAX_LENGTH)
        rids, rmask = tokenize_prompt_completion(tok, prompt_msgs, rtext, MAX_LENGTH)
        prompt_len = len(tok.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True))
        ci, ca, cm = _pad_batch([cids], [cmask], tok.pad_token_id, device)
        ri, ra, rm = _pad_batch([rids], [rmask], tok.pad_token_id, device)
        with torch.no_grad():
            clogp = _sequence_logps(model, ci, ca, cm, LENGTH_NORMALIZE)[0].item()
            rlogp = _sequence_logps(model, ri, ra, rm, LENGTH_NORMALIZE)[0].item()
        out.append({
            "prompt_tok": prompt_len,
            "chosen_comp_tok": int(sum(cmask)), "rejected_comp_tok": int(sum(rmask)),
            "chosen_total_tok": len(cids), "rejected_total_tok": len(rids),
            "chosen_logp": clogp, "rejected_logp": rlogp,
        })
    return out


def trl_logps(model, tok, rows, device):
    """TRL path: build a DPOTrainer, use its tokenized dataset + collator + concatenated_forward."""
    from datasets import Dataset
    ds = Dataset.from_list([{"chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    args = NormedDPOConfig(
        output_dir="/tmp/_trl_diff", max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH,
        beta=BETA, normalize_logps=LENGTH_NORMALIZE, per_device_train_batch_size=len(rows),
        report_to=[], bf16=True,
    )
    trainer = NormedDPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    tds = trainer.train_dataset  # tokenized rows

    # per-row token boundaries straight from TRL's tokenization
    boundaries = []
    for ex in tds:
        boundaries.append({
            "prompt_tok": len(ex["prompt_input_ids"]),
            "chosen_comp_tok": len(ex["chosen_input_ids"]),
            "rejected_comp_tok": len(ex["rejected_input_ids"]),
        })

    # logps via concatenated_forward on a collated batch
    batch = trainer.data_collator([tds[i] for i in range(len(tds))])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        fwd = trainer.concatenated_forward(trainer.model, batch)
    clogps = fwd["chosen_logps"].float().cpu().tolist()
    rlogps = fwd["rejected_logps"].float().cpu().tolist()
    for i, b in enumerate(boundaries):
        b["chosen_logp"] = clogps[i]
        b["rejected_logp"] = rlogps[i]
    return boundaries


def main(path, n):
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(device).eval()

    ds = load_from_disk(path)
    rows = [ds[i] for i in range(min(n, len(ds)))]

    L = loop_logps(model, tok, rows, device)
    T = trl_logps(model, tok, rows, device)

    hdr = f"{'i':>2} | {'pTok L/T':>12} | {'cComp L/T':>12} | {'rComp L/T':>12} | {'cLogp L':>9} {'cLogp T':>9} {'dC':>7} | {'rLogp L':>9} {'rLogp T':>9} {'dR':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i, (l, t) in enumerate(zip(L, T)):
        dC = l["chosen_logp"] - t["chosen_logp"]
        dR = l["rejected_logp"] - t["rejected_logp"]
        print(f"{i:>2} | {l['prompt_tok']:>5}/{t['prompt_tok']:<6} | "
              f"{l['chosen_comp_tok']:>5}/{t['chosen_comp_tok']:<6} | "
              f"{l['rejected_comp_tok']:>5}/{t['rejected_comp_tok']:<6} | "
              f"{l['chosen_logp']:>9.2f} {t['chosen_logp']:>9.2f} {dC:>7.2f} | "
              f"{l['rejected_logp']:>9.2f} {t['rejected_logp']:>9.2f} {dR:>7.2f}")

    import numpy as np
    dC = np.array([l["chosen_logp"] - t["chosen_logp"] for l, t in zip(L, T)])
    dR = np.array([l["rejected_logp"] - t["rejected_logp"] for l, t in zip(L, T)])
    print(f"\nmean|dChosen|={np.abs(dC).mean():.3f}  mean|dRejected|={np.abs(dR).mean():.3f}")
    print("If token counts match but logps differ -> label-shift/EOS/masking bug.")
    print("If prompt_tok differs (TRL capped at 512) -> max_prompt_length divergence.")
    print("If chosen_comp_tok differs -> chat-template prefix / completion-boundary bug.")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
