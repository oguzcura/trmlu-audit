"""Analyze scaled n=200 matched 3-arm run: accuracy, Wilson CI, McNemar.

Usage: python analyze_scale3arm.py [path-to-scale3arm.jsonl]
Default input: results/scale3arm.jsonl (repo-relative, as produced by
repro/scale3arm.py). Accepts records in either key style:
  - A_answer / B_answer / C_answer + truth_idx (index), or
  - A_ans / B_ans / C_ans + truth (answer letter).
"""
import json, re, sys
from math import sqrt
from scipy import stats

IN = sys.argv[1] if len(sys.argv) > 1 else "results/scale3arm.jsonl"

def parse_letter(raw):
    if not raw: return None
    raw = " ".join(str(raw).split())
    m = re.search(r"\b(?:Cevap|Answer|Cevapla)[:]?\s*\(?([A-Ea-e])\)?", raw)
    if m: return m.group(1).upper()
    m = re.search(r"\b([A-Ea-e])\s*\)\s*\S", raw)   # "A) text"
    if m: return m.group(1).upper()
    m = re.search(r"\b([A-Ea-e])\b", raw)
    return m.group(1).upper() if m else None

def idx_letter(idx):
    try: return chr(65 + int(idx))
    except: return None

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    denom = 1 + z*z/n
    c = (p + z*z/(2*n)) / denom
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (c - h, c + h)

def mcnemar_p(a, b):
    """McNemar exact two-sided p for discordant pair counts (a, b)."""
    n = a + b
    if n == 0: return 1.0
    return stats.binomtest(min(a, b), n, p=0.5).pvalue  # exact, already two-sided

def norm(rec):
    """Normalize either record style to {'A','B','C' answer letters + truth letter}."""
    ans = {k: (rec.get(k + "_answer") or rec.get(k + "_ans")) for k in "ABC"}
    if rec.get("truth_idx") is not None:
        return ans, idx_letter(rec["truth_idx"])
    return ans, rec.get("truth")

recs = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
print(f"records: {len(recs)}\n")

normed = [norm(r) for r in recs]

# fully matched rows (all 3 arms parse)
matched = [(ans, t) for ans, t in normed if all(parse_letter(ans[k]) for k in "ABC")]
print(f"fully-matched rows (all 3 arms parsed): {len(matched)}")

print("\n=== PER-ARM ACCURACY (fully matched denominator) ===")
for arm in "ABC":
    corr = sum(1 for ans, t in matched if parse_letter(ans[arm]) == t)
    n = len(matched)
    lo, hi = wilson(corr, n)
    print(f"  {arm}: {corr}/{n} = {corr/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== McNEMAR (paired, A=baseline) ===")
for arm in "BC":
    pos = sum(1 for ans, t in matched if parse_letter(ans["A"]) == t and parse_letter(ans[arm]) != t)
    neg = sum(1 for ans, t in matched if parse_letter(ans["A"]) != t and parse_letter(ans[arm]) == t)
    print(f"  {arm}: A-corr->{arm}-wrong = {pos}, {arm}-corr->A-wrong = {neg}, "
          f"McNemar two-sided p = {mcnemar_p(pos, neg):.4f}")