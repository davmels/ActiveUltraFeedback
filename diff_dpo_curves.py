"""Overlay the loop's run_dpo (1e9) and the TRL baseline wandb curves on identical data.

Both paths log the same keys (run_dpo mirrors TRL's names): loss, grad_norm,
rewards/margins, rewards/accuracies, logps/chosen, logps/rejected. Same data + same
effective batch + 1 epoch => same #optimizer steps, so the curves align step-for-step.
If the loop over-optimizes (margins grow faster, logps/chosen drops faster, higher
grad_norm) that points at a gradient/LR-scaling bug, not the per-step DPO math.

Run:
    python diff_dpo_curves.py <LOOP_run_id> <TRL_run_id>
    # optional: --loop_project active_dpo --trl_project DPO --entity ActiveUF_Plus
"""
import argparse
import json
import numpy as np
import wandb
import wandb.sdk.lib.server

# On these CSCS nodes wandb's viewer query returns None, making query_with_timeout
# crash with a TypeError; tolerate a missing/non-str flags payload (mirrors run.py).
_orig_query_with_timeout = wandb.sdk.lib.server.Server.query_with_timeout


def _patched_query_with_timeout(self):
    try:
        _orig_query_with_timeout(self)
    except TypeError:
        flags = self._viewer.get("flags") if getattr(self, "_viewer", None) else None
        self._flags = json.loads(flags) if isinstance(flags, str) else {}


wandb.sdk.lib.server.Server.query_with_timeout = _patched_query_with_timeout

KEYS = ["loss", "grad_norm", "rewards/margins", "rewards/accuracies",
        "logps/chosen", "logps/rejected"]


def history(entity, project, run_id):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    # full sampled history with ALL columns (key names differ: run_dpo logs bare keys,
    # HF Trainer/TRL prefixes with "train/")
    df = run.history(samples=100000, pandas=True)
    return df


def _resolve(df, key):
    """Find the column for `key` regardless of a 'train/' (or other) prefix."""
    if key in df.columns:
        return key
    if f"train/{key}" in df.columns:
        return f"train/{key}"
    cands = [c for c in df.columns if c == key or c.endswith("/" + key)]
    return cands[0] if cands else None


def deciles(df, key):
    col = _resolve(df, key)
    if col is None:
        return None
    s = df[col].dropna().to_numpy()
    if len(s) == 0:
        return None
    idx = np.linspace(0, len(s) - 1, 11).astype(int)   # 0%,10%,...,100% of training
    return s[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("loop_run")
    p.add_argument("trl_run")
    p.add_argument("--loop_project", default="active_dpo")
    p.add_argument("--trl_project", default="DPO")
    p.add_argument("--entity", default="ActiveUF_Plus")
    args = p.parse_args()

    loop = history(args.entity, args.loop_project, args.loop_run)
    trl = history(args.entity, args.trl_project, args.trl_run)
    print(f"loop steps logged: {len(loop)} | trl steps logged: {len(trl)}\n")

    pcts = [f"{q}%" for q in range(0, 101, 10)]
    for key in KEYS:
        l, t = deciles(loop, key), deciles(trl, key)
        if l is None or t is None:
            print(f"[{key}] missing in one run (loop={l is not None}, trl={t is not None})")
            continue
        print(f"=== {key} (rows: progress 0..100%) ===")
        print("        " + " ".join(f"{p:>9}" for p in pcts))
        print("loop  " + " ".join(f"{v:>9.3f}" for v in l))
        print("trl   " + " ".join(f"{v:>9.3f}" for v in t))
        print("ratio " + " ".join(f"{(lv/tv) if abs(tv) > 1e-9 else float('nan'):>9.2f}"
                                   for lv, tv in zip(l, t)))
        print()

    print("Read: loop/trl ratio >> 1 on |rewards/margins| or grad_norm, or logps/chosen "
          "dropping faster on the loop side => over-optimization (gradient/LR scaling).")


if __name__ == "__main__":
    main()
