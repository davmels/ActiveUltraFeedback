"""Verify domain-quota selection: realized domain counts of the first selected batch
should respect floor(K * pool_fraction) caps (plus a few random-fill extras to reach K).

Output rows are appended in selection order, so output[0:K] is the first step's pick.
"""
import sys
from collections import Counter

from datasets import load_from_disk

from activeuf.domains import get_category

OUT = "/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci/active_orig/dpo/deltaucb_enn_unknown_ultrafeedback_L1024_K256_domquota_20260609-030612-568"
K = 256

path = sys.argv[1] if len(sys.argv) > 1 else OUT
ds = load_from_disk(path)

# Pool fractions == fractions over the whole consumed output (debug drains the pool).
pool = Counter(get_category(p) for p in ds["prompt_id"])
total = sum(pool.values())
quota = {d: int(K * c / total) for d, c in pool.items()}

# Realized domains in the first selected batch.
first = Counter(get_category(p) for p in ds["prompt_id"][:K])

print(f"{path}  ({total} rows)\n")
print(f"first batch size: {sum(first.values())} (expected {K})")
print(f"sum of quotas: {sum(quota.values())}  (random-fill tops up the rest to {K})\n")
print(f"{'quota':>6} {'first-K':>8} {'delta':>6}  domain")
for d, _ in pool.most_common():
    q, r = quota[d], first.get(d, 0)
    print(f"{q:>6} {r:>8} {r - q:>+6}  {d}")
over = sum(max(0, first.get(d, 0) - quota[d]) for d in pool)
print(f"\ncounts above quota caps (= random-fill landing on capped domains): {over}")
print("PASS if first-K tracks the quotas (not deltaucb's usual Science/Safety-heavy skew).")
