from collections import Counter
import re
import sys

from datasets import load_from_disk

# Two prompt_id formats:
#   <source>-request-<n>-<m>          -> prefix is everything before "-request-"
#   multiturn_<type>_<model>_<number> -> prefix is everything before the trailing "_<number>"
def get_prefix(pid):
    if "-request-" in pid:
        return pid.split("-request-")[0]
    return re.sub(r"_\d+$", "", pid)


# Olmo3-report categories -> the prompt_id prefixes that belong to each.
# Mapping verified by matching each prefix's row count against the report's DPO column.
CATEGORIES = {
    "Chat & WildChat": [
        "filtered_wc_sample_500k",
        "Wildchat-1M-gpt-4.1-regenerated-english",
        "Wildchat-1m-gpt-4.1-regeneration-not-english",
    ],
    "Precise IF": [
        "valpy_if_qwq_reasoning_verified_no_reasoning",  # Dolci Instruct Precise IF (shard)
        "IF_sft_data_verified_permissive",               # Dolci Instruct Precise IF (shard)
        "tulu-3-sft-personas-instruction-following-o3",  # Dolci Instruct Persona Precise IF
        "oasst1_converted",                              # OpenAssistant
    ],
    "Math": [
        "tulu-3-sft-personas-math",        # Tulu 3 Persona MATH
        "tulu-3-sft-personas-algebra",     # Tulu 3 Persona Algebra
        "tulu-3-sft-personas-math-grade",  # Tulu 3 Persona GSM
        "tulu_v3.9_open_math_2_gsm8k_50k", # OpenMathInstruct 2
    ],
    "Coding": [
        "correct-python-sft-187k",             # Dolci Instruct Python Algorithms
        "personahub_code_v2_34999",            # Tulu 3 Persona Python
        "evol_codealpaca_heval_decontaminated",# Evol CodeAlpaca
    ],
    "Safety": [
        "tulu-3-sft-coconot-regenerated",                                              # CoCoNot
        "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",          # WildGuardMix
        "tulu_v3.9_wildjailbreak_decontaminated_50k",                                  # WildJailbreak
    ],
    "Science": [
        "tulu_v3.9_sciriff_10k",                       # SciRiff
        "OpenThoughts3-full-filtered-science-no-cot",  # Dolci Instruct OpenThought3+ Science
    ],
    "Multilingual": [
        "tulu_v3.9_aya_100k",  # Aya
    ],
    "Other": [
        "tulu_v3.9_table_gpt_5k",  # TableGPT
        "flan_v2_converted",       # FLAN
    ],
    "Multiturn": [
        "multiturn_self_talk_gpt",
        "multiturn_self_talk_qwen",
        "multiturn_related_query_gpt",
        "multiturn_related_query_qwen",
        "multiturn_paraphrase_gpt",
        "multiturn_paraphrase_qwen",
        "multiturn_repeat_gpt",
        "multiturn_repeat_qwen",
    ],
    "Not used in SFT": [
        "ultrafeedback_cleaned_olmo2_7b",  # UltraFeedback
        "DaringAnteater-prefs_olmo2_7b",   # DaringAnteater
    ],
}

# invert to prefix -> category
PREFIX_TO_CATEGORY = {p: cat for cat, prefixes in CATEGORIES.items() for p in prefixes}

# canonical display order (report order)
CATEGORY_ORDER = list(CATEGORIES.keys())


def get_category(pid):
    return PREFIX_TO_CATEGORY.get(get_prefix(pid), "UNKNOWN")


def category_counts(path):
    """Return (Counter of category -> count, total) over Olmo3 categories for `path`."""
    dataset = load_from_disk(path)
    counts = Counter(get_category(pid) for pid in dataset["prompt_id"])
    total = sum(counts.values())
    if "UNKNOWN" in counts:
        unmapped = sorted({get_prefix(pid) for pid in dataset["prompt_id"]
                           if get_category(pid) == "UNKNOWN"})
        print(f"  WARNING: {counts['UNKNOWN']} rows in unmapped prefixes: {unmapped}")
    return counts, total


def ordered_categories(present):
    """Categories in report order, with any extras (e.g. UNKNOWN) appended."""
    return [c for c in CATEGORY_ORDER if c in present] + sorted(present - set(CATEGORY_ORDER))


def print_stats(path):
    """Print the category distribution of a single dataset (no comparison)."""
    counts, total = category_counts(path)
    print(f"{path}  ({total} rows)\n")
    print(f"{'count':>8} {'%':>8}  category")
    for c in ordered_categories(set(counts)):
        print(f"{counts[c]:>8} {100 * counts[c] / total:7.2f}%  {c}")


def print_comparison(path_a, path_b):
    """Print category-level TVD and per-category percentage-point differences (B - A)."""
    counts_a, total_a = category_counts(path_a)
    counts_b, total_b = category_counts(path_b)
    dist_a = {c: counts_a[c] / total_a for c in counts_a}
    dist_b = {c: counts_b[c] / total_b for c in counts_b}

    all_cats = set(dist_a) | set(dist_b)
    tvd = 0.5 * sum(abs(dist_a.get(c, 0.0) - dist_b.get(c, 0.0)) for c in all_cats)

    print(f"A: {path_a}  ({total_a} rows)")
    print(f"B: {path_b}  ({total_b} rows)")
    print(f"\nTotal Variation Distance (category-level): {tvd:.4f}\n")

    # Per-category percentage-point difference (B - A), most divergent first
    print(f"{'A %':>8} {'B %':>8} {'Δpp (B-A)':>10}  category")
    for c in sorted(ordered_categories(all_cats),
                    key=lambda c: abs(dist_b.get(c, 0.0) - dist_a.get(c, 0.0)), reverse=True):
        a, b = 100 * dist_a.get(c, 0.0), 100 * dist_b.get(c, 0.0)
        print(f"{a:8.2f} {b:8.2f} {b - a:+10.2f}  {c}")


PATH_A = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci_annotated_new_21models_min8"
PATH_B = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/meeting_08_09_2026/partitions2/deltaucb_enn_unknown_ultrafeedback_L1024_K256_domquota_20260609-031737-758_20000"

# "stats" -> distribution of PATH_A only;  "compare" -> PATH_A vs PATH_B
mode = sys.argv[1] if len(sys.argv) > 1 else "stats"
if mode == "stats":
    print_stats(PATH_A)
elif mode == "compare":
    print_comparison(PATH_A, PATH_B)
else:
    raise SystemExit(f"unknown mode {mode!r}; use 'stats' or 'compare'")
