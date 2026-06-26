"""Greedy-generate many prompts from two checkpoints and COUNT failures per category.

A "failure" = generation ran to max_new_tokens without emitting EOS (the model didn't
stop -> at eval the final answer is truncated / never produced). We split prompts into
gsm8k-style, bbh-style, and MULTI-TURN to see whether the loop model's degradation is
about hard reasoning, multi-turn context, or both.

Run (1 GPU):
    python diff_ckpt_generate.py <TRL_dir> <LOOP_dir> [max_new_tokens]
"""
import glob
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

U = lambda c: {"role": "user", "content": c}
A = lambda c: {"role": "assistant", "content": c}

PROMPTS = [
    # ---- gsm8k-style (need a clean final numeric answer at the end) ----
    ("gsm8k", [U("Natalia sold clips to 48 friends in April, then half as many in May. "
                 "How many clips altogether? Show steps and give the final number.")]),
    ("gsm8k", [U("A robe takes 2 bolts of blue fiber and half that much white fiber. "
                 "How many bolts total? Think step by step.")]),
    ("gsm8k", [U("Compute 17 * 24 by long multiplication, showing each partial product, then give the result.")]),
    ("gsm8k", [U("Compute 23 * 47 by long multiplication, show the partial products, then give the result.")]),
    ("gsm8k", [U("Weng earns $12 an hour babysitting. Yesterday she babysat for 50 minutes. "
                 "How much did she earn? Show steps.")]),
    ("gsm8k", [U("Betty wants a $100 wallet and has half saved. Her parents give $15 and her "
                 "grandparents twice as much as her parents. How much more does she need? Show steps.")]),
    ("gsm8k", [U("A class has 30 students. One third got an A, and one quarter of the remaining "
                 "students got a B. How many students got neither? Show steps.")]),
    ("gsm8k", [U("A train goes 60 mph for 2.5 hours, then 40 mph for 1.5 hours. "
                 "What total distance does it cover? Show steps.")]),
    ("gsm8k", [U("James writes a 3-page letter to 2 different friends twice a week. "
                 "How many pages does he write in a year? Show steps.")]),
    ("gsm8k", [U("What is 144 divided by 16, then multiplied by 7? Show each step.")]),
    # ---- bbh-style (multi-step logic, terse final answer) ----
    ("bbh", [U("Do you return to start? Take 3 steps north, then 3 steps south. "
               "(A) Yes (B) No. Reason step by step, then answer with the letter.")]),
    ("bbh", [U("I have 3 apples, eat 1, buy 4 more, then give away 2. How many apples do I have? "
               "Reason, then give the number.")]),
    ("bbh", [U("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? "
               "(A) Yes (B) No. Reason, then answer with the letter.")]),
    ("bbh", [U("Count the words in this sentence: 'The quick brown fox jumps high'. "
               "Reason, then give the number.")]),
    ("bbh", [U("Order from smallest to largest: elephant, mouse, dog. Reason briefly, then list them.")]),
    # ---- multi-turn (does the model handle prior turns / carried context?) ----
    ("multiturn", [U("Let's solve a puzzle. I'm thinking of a number."),
                   A("Sure! What's the clue?"),
                   U("If I multiply it by 3 and add 6, I get 21. What's my number? Show steps.")]),
    ("multiturn", [U("Define a function f(x) = 2x + 1."),
                   A("Okay, f(x) = 2x + 1."),
                   U("Now compute f(f(3)) step by step.")]),
    ("multiturn", [U("Remember this list: [4, 8, 15, 16]."),
                   A("Got it: 4, 8, 15, 16."),
                   U("What is the sum of the two largest numbers? Show your work.")]),
]


def resolve(d):
    if os.path.exists(os.path.join(d, "config.json")):
        return d
    if os.path.exists(os.path.join(d, "policy", "config.json")):
        return os.path.join(d, "policy")
    cks = sorted(glob.glob(os.path.join(d, "checkpoint-step*")),
                 key=lambda p: int(p.split("step")[-1]) if p.split("step")[-1].isdigit() else -1)
    return cks[-1] if cks else d


def gen_all(ckpt, tok, max_new):
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda().eval()
    eos = model.generation_config.eos_token_id or tok.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    res = []
    for cat, msgs in PROMPTS:
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt").cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        res.append({"cat": cat, "n_new": int(new.shape[0]),
                    "stopped": int(new[-1].item()) in eos_set,
                    "text": tok.decode(new, skip_special_tokens=True)})
    del model
    torch.cuda.empty_cache()
    return res


def summarize(name, res, max_new):
    cats = {}
    for r in res:
        c = cats.setdefault(r["cat"], {"n": 0, "fail": 0, "toks": 0})
        c["n"] += 1
        c["fail"] += 0 if r["stopped"] else 1
        c["toks"] += r["n_new"]
    print(f"\n[{name}] failures = ran to cap({max_new}) without EOS:")
    tot_n = tot_f = 0
    for cat, c in cats.items():
        tot_n += c["n"]; tot_f += c["fail"]
        print(f"  {cat:>10}: {c['fail']}/{c['n']} no-stop | mean_new_tok={c['toks'] / c['n']:.0f}")
    print(f"  {'TOTAL':>10}: {tot_f}/{tot_n} no-stop")
    return tot_f, tot_n


def main(trl, loop, max_new=1024):
    trl, loop = resolve(trl), resolve(loop)
    print(f"TRL : {trl}\nLOOP: {loop}\nmax_new_tokens={max_new}, n_prompts={len(PROMPTS)}")
    tok = AutoTokenizer.from_pretrained(trl)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\n>>> generating TRL ...");  T = gen_all(trl, tok, max_new)
    print(">>> generating LOOP ...");   L = gen_all(loop, tok, max_new)

    # per-prompt comparison
    print("\n" + "=" * 90)
    print(f"{'#':>2} {'cat':>10} | {'TRL n/stop':>14} | {'LOOP n/stop':>14}")
    for i, (t, l) in enumerate(zip(T, L)):
        ts = "stop" if t["stopped"] else "CAP!"
        ls = "stop" if l["stopped"] else "CAP!"
        mark = "   <-- diverge" if t["stopped"] != l["stopped"] else ""
        print(f"{i:>2} {t['cat']:>10} | {t['n_new']:>6} {ts:>6} | {l['n_new']:>6} {ls:>6}{mark}")

    tf, tn = summarize("TRL", T, max_new)
    lf, ln = summarize("LOOP", L, max_new)
    print(f"\n==> TRL no-stop {tf}/{tn}  vs  LOOP no-stop {lf}/{ln}")
    print("If LOOP no-stop >> TRL (esp. in gsm8k/bbh), the degradation is systematic "
          "non-termination on hard reasoning. If it clusters in 'multiturn', it's a "
          "multi-turn-context problem instead.")

    # dump the diverging cases in full for inspection
    for i, (t, l) in enumerate(zip(T, L)):
        if t["stopped"] != l["stopped"]:
            print("\n" + "#" * 90)
            print(f"DIVERGENCE #{i} [{t['cat']}] — prompt: {PROMPTS[i][1][-1]['content'][:80]}...")
            print(f"--- TRL  (n={t['n_new']}, {'stop' if t['stopped'] else 'CAP'}):\n{t['text'][-400:]}")
            print(f"--- LOOP (n={l['n_new']}, {'stop' if l['stopped'] else 'CAP'}):\n{l['text'][-400:]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1024)
