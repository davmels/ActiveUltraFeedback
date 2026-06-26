"""Does the loop's DPO training actually include & train on the EOS / end-of-turn token?

CPU-only. For a few real pairs, replays the loop's tokenize_prompt_completion and prints
the completion's tail tokens + whether the EOS/end-of-turn id falls inside the completion
mask (mask=1 => it IS part of the DPO loss). Also shows TRL-style rendering for contrast.

Run:
    python inspect_eos_in_mask.py <PREF_DATASET_PATH> [N]
"""
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer

from activeuf.loop.dpo_utils import tokenize_prompt_completion

MODEL = "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--allenai--Olmo-3-7B-Instruct-SFT/snapshots/e1452fc572d51966ff4aaeb25118b891eb93e549"
MAX_LENGTH = 4096


def main(path, n=4):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eos_ids = tok.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    print(f"tokenizer.eos_token={tok.eos_token!r} id={tok.eos_token_id}")
    # the template's end-of-assistant-turn marker (what actually closes a turn)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        tokenize=False)
    print(f"sample rendered assistant turn tail: ...{rendered[-40:]!r}\n")

    ds = load_from_disk(path)
    for i in range(min(n, len(ds))):
        row = ds[i]
        prompt_msgs = row["chosen"][:-1]
        ctext = row["chosen"][-1]["content"]
        full_ids, cmask = tokenize_prompt_completion(tok, prompt_msgs, ctext, MAX_LENGTH)
        prompt_len = len(tok.apply_chat_template(prompt_msgs, tokenize=True,
                                                 add_generation_prompt=True))
        comp_ids = full_ids[prompt_len:]
        comp_mask = cmask[prompt_len:]

        last_id = comp_ids[-1] if comp_ids else None
        eos_in_comp = any(t in eos_set for t in comp_ids)
        # is the EOS position actually masked-in (=1, trained) given the loss shift?
        # _sequence_logps trains token j via mask[j] (comp_mask aligned to comp tokens)
        eos_trained = any((t in eos_set) and (m == 1) for t, m in zip(comp_ids, comp_mask))

        print(f"--- pair {i} ---")
        print(f"  completion #tokens={len(comp_ids)}  mask_ones={sum(comp_mask)}")
        print(f"  first 5 comp tokens: {[(t, tok.decode([t])) for t in comp_ids[:5]]}")
        print(f"  last  6 comp tokens: {[(t, tok.decode([t])) for t in comp_ids[-6:]]}")
        print(f"  last token id={last_id} ({tok.decode([last_id])!r})  is_eos={last_id in eos_set}")
        print(f"  EOS id present in completion: {eos_in_comp} | EOS trained (mask=1): {eos_trained}")
        print(f"  full_ids tail decoded: ...{tok.decode(full_ids[-25:])!r}\n")

    print("If 'EOS trained (mask=1)' is True, the loop DOES penalize/learn the stop token "
          "(consistent with 17/18 clean stops). If False, the loop never trains EOS -> "
          "that would be your bug.")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
