"""Analyze scaled n=200 matched 3-arm run: accuracy, Wilson CI, McNemar."""
import json, re
from math import sqrt
from scipy import stats

IN = r"C:\Users\oguzc\ai-team\research\harness\results\scale3arm.jsonl"

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

recs = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
print(f"records: {len(recs)}\n")

# fully matched rows (all 3 arms parse)
arms = {"A": "A_answer", "B": "B_answer", "C": "C_answer"}
letter_arm = {"A": 0, "B": 0, "C": 0}
truth_letter = [idx_letter(r["truth_idx"]) for r in recs]

matched = [r for r in recs if all(parse_letter(r[k]) for k in ("A_answer","B_answer","C_answer"))]
print(f"fully-matched rows (all 3 arms parsed): {len(matched)}")

print("\n=== PER-ARM ACCURACY (fully matched denominator) ===")
for arm, key in arms.items():
    corr = sum(1 for r in matched if parse_letter(r[key]) == idx_letter(r["truth_idx"]))
    n = len(matched)
    lo, hi = wilson(corr, n)
    print(f"  {arm}: {corr}/{n} = {corr/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== McNEMAR (paired, A=baseline) ===")
aok = lambda r: parse_letter(r["A_answer"]) == idx_letter(r["truth_idx"])
aokArm = lambda r, k: parse_letter(r[k+"_answer"]) == idx_letter(r["truth_idx"])
for arm in ("B", "C"):
    pos = sum(1 for r in matched if aok(r) and not aokArm(r, arm))
    neg = sum(1 for r in matched if (not aok(r)) and aokArm(r, arm))
    print(f"  {arm}: A-corr->{arm}-wrong = {pos}, {arm}-corr->A-wrong = {neg}, "
          f"McNemar two-sided p = {mcnemar_p(pos, neg):.4f}")