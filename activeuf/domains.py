"""Olmo3-report prompt domains, derived from a row's prompt_id.

Single source of truth for mapping a prompt_id to one of the Olmo3 categories.
The grouping was verified by matching each prefix's row count against the report's
DPO column (see playground3.py).
"""
import re


# Two prompt_id formats:
#   <source>-request-<n>-<m>          -> prefix is everything before "-request-"
#   multiturn_<type>_<model>_<number> -> prefix is everything before the trailing "_<number>"
def get_prefix(prompt_id: str) -> str:
    if "-request-" in prompt_id:
        return prompt_id.split("-request-")[0]
    return re.sub(r"_\d+$", "", prompt_id)


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


def get_category(prompt_id: str) -> str:
    """Map a prompt_id to its Olmo3 category, or 'UNKNOWN' if the prefix is unmapped."""
    return PREFIX_TO_CATEGORY.get(get_prefix(prompt_id), "UNKNOWN")
