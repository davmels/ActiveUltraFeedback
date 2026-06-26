"""Diff two saved DPO checkpoints (TRL vs loop) to find a post-training / eval-path
discrepancy: config, generation_config, tokenizer (chat_template), special tokens,
and weight tensor shapes/dtypes (catches a broken DeepSpeed consolidation).

Run (no GPU):
    python diff_ckpt_artifacts.py <TRL_dir> <LOOP_dir>

Each dir may be a final model dir, a run dir containing `policy/`, or a run dir with
`checkpoint-step*/` (newest is used).
"""
import glob
import json
import os
import sys


def resolve(d):
    """Find the dir that actually holds config.json."""
    if os.path.exists(os.path.join(d, "config.json")):
        return d
    if os.path.exists(os.path.join(d, "policy", "config.json")):
        return os.path.join(d, "policy")
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-step*")),
                 key=lambda p: int(p.split("step")[-1]) if p.split("step")[-1].isdigit() else -1)
    if cks:
        return cks[-1]
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-*")))
    if cks:
        return cks[-1]
    return d


def load_json(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def diff_json(name, a, b):
    print(f"\n=== {name} ===")
    if a is None or b is None:
        print(f"  PRESENT? trl={a is not None}  loop={b is not None}  <-- mismatch" if (a is None) != (b is None)
              else "  (absent in both)")
        if a is None or b is None:
            return
    keys = sorted(set(a) | set(b))
    any_diff = False
    for k in keys:
        va, vb = a.get(k, "<MISSING>"), b.get(k, "<MISSING>")
        # chat_template can be huge; compare by equality only
        if isinstance(va, str) and len(str(va)) > 120:
            same = va == vb
            if not same:
                any_diff = True
                print(f"  [DIFF] {k}: trl(len {len(str(va))}) != loop(len {len(str(vb))})")
            continue
        if va != vb:
            any_diff = True
            print(f"  [DIFF] {k}: trl={va!r}  loop={vb!r}")
    if not any_diff:
        print("  (identical)")


def tensor_index(d):
    """Map tensor_name -> (shape, dtype) across all safetensors shards, no full load."""
    from safetensors import safe_open
    idx = {}
    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not shards:
        print(f"  no .safetensors in {d} (bin checkpoint?) -- skipping weight diff")
        return None
    for sh in shards:
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                idx[k] = (tuple(sl.get_shape()), sl.get_dtype())
    return idx


def diff_weights(a_dir, b_dir):
    print("\n=== weights (safetensors headers) ===")
    A, B = tensor_index(a_dir), tensor_index(b_dir)
    if A is None or B is None:
        return
    ka, kb = set(A), set(B)
    print(f"  tensors: trl={len(ka)}  loop={len(kb)}")
    only_a, only_b = ka - kb, kb - ka
    if only_a:
        print(f"  [DIFF] {len(only_a)} tensors only in trl, e.g. {sorted(only_a)[:3]}")
    if only_b:
        print(f"  [DIFF] {len(only_b)} tensors only in loop, e.g. {sorted(only_b)[:3]}")
    shape_diff = [k for k in (ka & kb) if A[k][0] != B[k][0]]
    dtype_diff = [k for k in (ka & kb) if A[k][1] != B[k][1]]
    if shape_diff:
        print(f"  [DIFF] {len(shape_diff)} shape mismatches, e.g. {shape_diff[0]}: {A[shape_diff[0]]} vs {B[shape_diff[0]]}")
    if dtype_diff:
        print(f"  [DIFF] dtypes differ ({len(dtype_diff)}), e.g. {dtype_diff[0]}: {A[dtype_diff[0]][1]} vs {B[dtype_diff[0]][1]}")
    if not (only_a or only_b or shape_diff):
        na = sum(_numel(A[k][0]) for k in ka)
        nb = sum(_numel(B[k][0]) for k in kb)
        print(f"  same tensor set & shapes. total params: trl={na:,}  loop={nb:,}"
              + ("  (dtypes differ)" if dtype_diff else ""))


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def main(trl, loop):
    trl, loop = resolve(trl), resolve(loop)
    print(f"TRL  ckpt: {trl}")
    print(f"LOOP ckpt: {loop}")
    print("\nfiles:")
    print("  trl :", sorted(os.listdir(trl)))
    print("  loop:", sorted(os.listdir(loop)))

    for name in ["config.json", "generation_config.json", "tokenizer_config.json",
                 "special_tokens_map.json"]:
        diff_json(name, load_json(trl, name), load_json(loop, name))

    diff_weights(trl, loop)
    print("\nWatch especially: generation_config eos_token_id / max_new_tokens / max_length, "
          "config.json eos_token_id, and tokenizer_config chat_template equality.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
