"""Analyze the Day-5 pilot run (multi-benchmark / multi-model).

Reads results/pilot_audit_2026-08-16.jsonl and computes, per benchmark x model
cell (design paper1_contamination_audit.md §4):

  - M1:  verbatim-hit rate + Wilson 95% CI; mean 8-gram overlap (+/- sd)
  - M2a: flip-follow rate + Wilson 95% CI; paired McNemar exact (two-sided)
         vs the M2b-A (baseline) arm on discordant pairs
  - M2b: per-arm accuracy (A / B / C) + Wilson 95% CI on the fully-matched
         denominator (all 3 arms parsed); McNemar exact for B-vs-A, C-vs-A,
         B-vs-C (the fragility contrast: B collapses while C stays stable)
  - Bonferroni within benchmark x probe family (raw + corrected p)
  - Pre-registered §0 consensus rule per cell: flagged only if >=2 of 3 probe
    modalities indicate AND >=1 dynamic arm (M2a/M2b) p < 0.05 after
    Bonferroni. Everything else = "no significant contamination detected".

Statistical functions (wilson, mcnemar_p) reuse analyze_scale3arm.py logic.
Every cell is reported; no cell is hidden; honest-negative wording is used.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from math import sqrt

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results", "pilot_audit_2026-08-16.jsonl")
OUT_JSON = os.path.join(HERE, "results", "pilot_stats_2026-08-16.json")

BENCHMARKS = ["tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"]
MODELS = ["gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"]
MC = ("tr_mmlu", "tumlu_tr")


# ---------------------------------------------------------------------------
# Stats primitives (logic reused from analyze_scale3arm.py)
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(c - h, 0.0), 4), round(min(c + h, 1.0), 4))


def mcnemar_p(a: int, b: int) -> float:
    """McNemar exact two-sided p for discordant pair counts (a, b)."""
    n = a + b
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(a, b), n, p=0.5).pvalue)


def mean_std(vals):
    if not vals:
        return (None, None)
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return (round(m, 4), round(sd, 4))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load():
    recs = []
    n_skipped = 0
    with open(IN, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                n_skipped += 1          # tolerate a truncated trailing line
    if n_skipped:
        print(f"[warn] skipped {n_skipped} unparseable line(s) (truncated tail?)")
    return recs


def group(recs):
    """item-key -> rec per (benchmark, model, probe)."""
    g = defaultdict(dict)                 # (bench, model, probe) -> {item: rec}
    for r in recs:
        g[(r["benchmark"], r["model"], r["probe"])][r["item_idx"]] = r
    return g


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------
def cell_m1(grp, bench, model):
    recs = grp.get((bench, model, "M1"), {})
    ok = [r for r in recs.values() if r.get("ok")]
    n = len(ok)
    hits = sum(1 for r in ok if r.get("verbatim_hit"))
    ovs = [r.get("overlap8", 0.0) for r in ok]
    m, sd = mean_std(ovs)
    lo, hi = wilson(hits, n)
    return {"n": n, "verbatim_hits": hits, "verbatim_rate": round(hits / n, 4) if n else None,
            "verbatim_ci95": [lo, hi], "mean_overlap8": m, "overlap8_sd": sd}


def cell_m2a(grp, bench, model):
    """flip-follow rate + McNemar vs baseline (M2b_A arm), discordant pairs."""
    m2a = grp.get((bench, model, "M2a"), {})
    base = grp.get((bench, model, "M2b_A"), {})
    ok = [r for r in m2a.values() if r.get("ok")]
    flips = [r for r in ok if r.get("flip_follow")]
    n_flip = len(ok)
    lo, hi = wilson(len(flips), n_flip)
    # paired McNemar vs baseline: base correct? / M2a picked the TRUE answer?
    pos = neg = 0                          # pos: base ok -> M2a wrong; neg: base wrong -> M2a ok
    for item, r in m2a.items():
        b = base.get(item)
        if b is None or not b.get("ok") or not r.get("ok"):
            continue
        base_ok = bool(b.get("answer_correct"))
        test_ok = bool(r.get("answer_correct"))
        if base_ok and not test_ok:
            pos += 1
        elif (not base_ok) and test_ok:
            neg += 1
    n_pair = pos + neg
    return {"n_flip": n_flip, "flip_follows": len(flips),
            "flip_follow_rate": round(len(flips) / n_flip, 4) if n_flip else None,
            "flip_follow_ci95": lo and [lo, hi],
            "mcnemar_discordant": [pos, neg], "mcnemar_n_pairs": n_pair,
            "mcnemar_p_raw": round(mcnemar_p(pos, neg), 4)}


def cell_m2b(grp, bench, model):
    """Per-arm accuracy on the fully-matched denominator (A, B, C all parsed)."""
    arms = {"A": "M2b_A", "B": "M2b_B", "C": "M2b_C"}
    recs = {a: grp.get((bench, model, p), {}) for a, p in arms.items()}

    def parsed(r):
        if not r or not r.get("ok"):
            return False
        # real records carry the model output under 'raw' (content truncated)
        if not (r.get("raw") or r.get("content") or "").strip():
            return False
        if bench in MC:
            return r.get("parsed_letter") is not None
        return True                         # QA: any non-empty output counts as an answer

    items = sorted(set.intersection(*[set(recs[a].keys()) for a in arms]))
    matched = [i for i in items if all(parsed(recs[a].get(i)) for a in arms)]

    def acc(a):
        k = sum(1 for i in matched if recs[a][i].get("answer_correct"))
        lo, hi = wilson(k, len(matched))
        return {"correct": k, "n": len(matched),
                "acc": round(k / len(matched), 4) if matched else None,
                "ci95": [lo, hi]}

    out = {"matched_n": len(matched),
           "arms": {a: acc(a) for a in arms},
           "mcnemar": {}}
    a_ok = lambda i, a: bool(recs[a][i].get("answer_correct"))
    for test in ("B", "C"):
        pos = sum(1 for i in matched if a_ok(i, "A") and not a_ok(i, test))
        neg = sum(1 for i in matched if not a_ok(i, "A") and a_ok(i, test))
        out["mcnemar"][f"{test}_vs_A"] = {"discordant": [pos, neg],
                                          "p_raw": round(mcnemar_p(pos, neg), 4)}
    pos = sum(1 for i in matched if a_ok(i, "B") and not a_ok(i, "C"))
    neg = sum(1 for i in matched if not a_ok(i, "B") and a_ok(i, "C"))
    out["mcnemar"]["C_vs_B"] = {"discordant": [pos, neg],
                                "p_raw": round(mcnemar_p(pos, neg), 4)}
    return out


# ---------------------------------------------------------------------------
# Bonferroni within benchmark x probe family + §0 consensus rule
# ---------------------------------------------------------------------------
def collect_pvals(cells):
    """cell -> list of raw p-values (dynamic-arm contrasts), by family."""
    fam = defaultdict(list)
    for (bench, model), c in cells.items():
        for probe, info in c.items():
            if probe == "M2a":
                fam[(bench, "M2a")].append((model, "M2a_mcnemar", info["mcnemar_p_raw"]))
            elif probe == "M2b":
                for k, v in info["mcnemar"].items():
                    fam[(bench, "M2b")].append((model, k, v["p_raw"]))
    return fam


def consensus(c, bench):
    """§0 pre-registered rule, operationalized transparently (documented in
    the report): modalities indicate ->
      M1 : mean_overlap8 > 0.5 AND verbatim_rate > 0.5 (prompt asks for
           verbatim reproduction, so high M1 is partly instruction-following;
           documented confound)
      M2a: flip-follow rate > 0.5
      M2b: B-acc < C-acc - 0.10  (B collapses while C stays stable)
    flagged = >=2 modalities indicate AND min dynamic-arm corrected p < 0.05.
    """
    m1 = c.get("M1", {})
    m2a = c.get("M2a", {})
    m2b = c.get("M2b", {})
    ind = {}
    ind["M1"] = bool(m1.get("mean_overlap8") is not None and m1["mean_overlap8"] > 0.5
                     and (m1.get("verbatim_rate") or 0) > 0.5)
    ind["M2a"] = bool(bench != "ragturk_formal5k" and (m2a.get("flip_follow_rate") or 0) > 0.5)
    b_a, c_a = (m2b.get("arms", {}).get("B", {}).get("acc"),
                m2b.get("arms", {}).get("C", {}).get("acc"))
    ind["M2b"] = bool(b_a is not None and c_a is not None and b_a < c_a - 0.10)
    return ind


def main():
    recs = load()
    grp = group(recs)
    cells = {}
    for bench in BENCHMARKS:
        for model in MODELS:
            c = {"M1": cell_m1(grp, bench, model)}
            if bench != "ragturk_formal5k":
                c["M2a"] = cell_m2a(grp, bench, model)
            c["M2b"] = cell_m2b(grp, bench, model)
            cells[(bench, model)] = c

    # Bonferroni per benchmark x probe family (raw p -> corrected)
    fam = collect_pvals(cells)
    adj = {}
    for (bench, probe), tests in fam.items():
        k = len(tests)
        for model, name, p in tests:
            adj[(bench, model, probe, name)] = round(min(1.0, p * k), 4)

    # assemble output with per-cell corrected p + consensus
    out = {"phase": "pilot_2026-08-16", "n_per_benchmark": 50, "seed": 42,
           "rule": "contaminated iff >=2/3 modalities indicate AND >=1 dynamic arm "
                   "(M2a/M2b) p<0.05 after Bonferroni (design §0)",
           "bonferroni_family_sizes": {f"{b}|{p}": len(v) for (b, p), v in fam.items()},
           "cells": {}, "spend": {}}
    for (bench, model), c in cells.items():
        c_out = {}
        for probe in ("M1", "M2a", "M2b"):
            if probe not in c:
                continue
            c_out[probe] = dict(c[probe])
        # attach corrected p to M2a + M2b contrasts
        if "M2a" in c_out:
            key = (bench, model, "M2a", "M2a_mcnemar")
            c_out["M2a"]["mcnemar_p_bonf"] = adj.get(key, 1.0)
        if "M2b" in c_out:
            for name, v in c_out["M2b"]["mcnemar"].items():
                v["p_bonf"] = adj.get((bench, model, "M2b", name), 1.0)
        ind = consensus(c, bench)
        dynamic = []
        if bench != "ragturk_formal5k":
            dynamic.append(c_out.get("M2a", {}).get("mcnemar_p_bonf", 1.0))
        dynamic += [v["p_bonf"] for v in c_out.get("M2b", {}).get("mcnemar", {}).values()]
        dyn_min = min(dynamic) if dynamic else 1.0
        n_ind = sum(1 for v in ind.values() if v)
        flagged = n_ind >= 2 and dyn_min < 0.05
        call = ("CONTAMINATED (pre-registered rule)" if flagged
                else "no significant contamination detected")
        c_out["modality_indications"] = ind
        c_out["n_modalities_indicate"] = n_ind
        c_out["min_dynamic_arm_p_bonf"] = dyn_min
        c_out["consensus_call"] = call
        out["cells"][f"{bench}|{model}"] = c_out

    # spend summary from every logged call (incl. translation records)
    sp = defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "retries": 0,
                              "prompt_tokens": 0, "cached_tokens": 0,
                              "completion_tokens": 0, "reasoning_tokens": 0,
                              "cost_est_usd": 0.0, "cost_est_n": 0})
    for r in recs:
        m = r["model"]
        s = sp[m]
        s["calls"] += 1
        s["ok"] += int(r.get("ok"))
        s["fail"] += int(not r.get("ok"))
        s["retries"] += int(r.get("retries") or 0)
        s["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
        s["cached_tokens"] += int(r.get("cached_tokens") or 0)
        s["completion_tokens"] += int(r.get("completion_tokens") or 0)
        s["reasoning_tokens"] += int(r.get("reasoning_tokens") or 0)
        if r.get("cost_est_usd") is not None:
            s["cost_est_usd"] += r["cost_est_usd"]
            s["cost_est_n"] += 1
    out["spend"] = {k: {**v, "cost_est_usd": round(v["cost_est_usd"], 6),
                        "has_rate": v["cost_est_n"] > 0} for k, v in sp.items()}
    out["spend"]["TOTAL_calls"] = len(recs)
    out["spend"]["TOTAL_tokens"] = sum(r.get("total_tokens") or 0 for r in recs)
    total_cost = sum(v["cost_est_usd"] for k, v in out["spend"].items()
                     if k.startswith(("gpt", "deepseek", "mimo")) and v["has_rate"])
    out["spend"]["TOTAL_cost_est_usd_rated"] = round(total_cost, 6)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_JSON}")
    # quick console digest
    for (bench, model), c in sorted(cells.items()):
        cell = out["cells"][f"{bench}|{model}"]
        print(f"{bench:16s} {model:18s} {cell['consensus_call']} "
              f"(modalities={cell['n_modalities_indicate']}, "
              f"min_dyn_p_bonf={cell['min_dynamic_arm_p_bonf']:.4f})")


if __name__ == "__main__":
    main()
