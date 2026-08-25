"""Pilot runner for Paper 1 contamination audit (Day 5, 2026-08-16).

Runs probes M1 (verbatim-recall + 8-gram), M2a (distractor-flip / edited-answer
flip), M2b (3-arm A/B/C crosslingual) over n=50 seed-42 matched items per
benchmark x 3 models (gpt-5.6-luna, mimo-v2.5, deepseek-v4-flash), ALL via
opencode-go (trmlu_audit.core). OpenRouter is NOT touched.

Hardening (task spec):
  (a) retry-on-empty comes from core.chat_record (MAX_EMPTY_RETRIES=2, logged);
      this runner adds an OUTER retry layer for 5xx/429/rate/timeout errors with
      exponential backoff (max 2 extra attempts, each logged in retry_log).
  (b) appends to results/pilot_audit_2026-08-16.jsonl; on restart, RESUME-SKIPS
      any call whose key (benchmark|model|probe|item_idx) is already present.
  (c) --workers N (default 4) via concurrent.futures.ThreadPoolExecutor, with a
      global token-bucket rate limiter (~2 req/s default, shared across workers).
  (d) hard-aborts with a clear message if estimated cost exceeds --max-cost
      ($1.00 default): checked BEFORE the run (dry-run estimator using smoke
      token statistics) AND live during the run (accumulated cost_est_usd).

Every API call is appended as one JSONL record: model, benchmark, probe,
item_idx, tokens (prompt/cached/completion/reasoning), cost_est_usd (documented
rates only; None = no published rate -> tokens counted, cost not estimable),
retries, ok, error, probe outcome fields.

Run (from harness/):
  uv run python pilot_audit.py --limit 50 --workers 4 --log results/pilot_audit_2026-08-16.jsonl
  uv run python pilot_audit.py --dry-run            # cost pre-check, no calls
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "trmlu-audit", "src"))
from trmlu_audit import core  # noqa: E402
from probe_m1 import overlap8, is_verbatim, cost_estimate, LADDER  # noqa: E402

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DEFAULT_LOG = os.path.join(HERE, "results", "pilot_audit_2026-08-16.jsonl")

TRANS_MODEL = core.TRANS_MODEL          # translations always via deepseek-v4-flash
ANS_MODELS = ["gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-flash"]
BENCHMARKS = ["tr_mmlu", "tumlu_tr", "halluverse_tr", "ragturk_formal5k"]

# ---------------------------------------------------------------------------
# Module-level run state (shared by workers + lazy translation cache)
# ---------------------------------------------------------------------------
BUCKET: "TokenBucket" = None            # rate limiter, initialized in main()
MAX_COST = [1.00]                       # live cost cap
SPENT = [0.0]                           # accumulated cost_est_usd (rated models)
ABORT = threading.Event()               # set when live cost cap is exceeded
LOG_PATH = [DEFAULT_LOG]
LOG_LOCK = threading.Lock()
COST_LOCK = threading.Lock()


def log_record(rec: dict):
    """Thread-safe append of one call record + live cost accounting."""
    with LOG_LOCK:
        with open(LOG_PATH[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    with COST_LOCK:
        if rec.get("cost_est_usd") is not None:
            SPENT[0] += rec["cost_est_usd"]
            if SPENT[0] > MAX_COST[0] and not ABORT.is_set():
                ABORT.set()
                print(f"\nHARD ABORT (live): accumulated estimated cost "
                      f"${SPENT[0]:.4f} exceeded ${MAX_COST[0]:.2f}. "
                      f"Remaining calls not executed. Log is intact for resume.")

# ---------------------------------------------------------------------------
# Data loading (normalized CSVs built by datasets_v2.py, verified Day 2-3)
# ---------------------------------------------------------------------------
def _load_csv(name: str) -> List[Dict]:
    import csv as _csv
    path = os.path.join(DATA_DIR, name)
    out = []
    with open(path, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            out.append({k: (v or "") for k, v in r.items()})
    return out


def _parse_list(s: str) -> List[str]:
    if isinstance(s, list):
        return list(s)
    s = s.strip()
    try:
        val = json.loads(s)
        return list(val) if isinstance(val, list) else [str(val)]
    except Exception:
        import ast
        try:
            val = ast.literal_eval(s)
            return list(val) if isinstance(val, (list, tuple)) else [str(val)]
        except Exception:
            return [c.strip().strip("'\"") for c in s.strip("[]()").split(",") if c.strip()]


def load_sample(benchmark: str, limit: int, seed: int = SEED) -> List[Dict]:
    """Seed-42 sample shared by ALL models and probes (matched design)."""
    raw = _load_csv(f"{benchmark}.csv")
    rng = random.Random(seed)
    rows = rng.sample(raw, min(limit, len(raw)))
    out = []
    for i, r in enumerate(rows):
        base = {"idx": i, "source": benchmark, "soru": r["soru"]}
        if benchmark in ("tr_mmlu", "tumlu_tr"):      # MC form
            opts = _parse_list(r["secenekler"])
            try:
                idx = int(r["cevap_idx"]) if r.get("cevap_idx") else None
            except ValueError:
                idx = None
            if idx is None or not (0 <= idx < len(opts)):
                letter = r.get("cevap", "").strip().upper()
                idx = ord(letter) - ord("A") if letter in "ABCDE" else None
            base.update({"secenekler": opts, "cevap_idx": idx,
                         "cevap": r.get("cevap", "").strip().upper(),
                         "bolum": r.get("bolum", r.get("subject", ""))})
        else:                                          # QA-pair form
            base.update({"cevap": r["cevap"],
                         "edited_cevap": r.get("edited_cevap", ""),
                         "qid": r.get("qid", ""), "category": r.get("category", "")})
        out.append(base)
    return out


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def mc_text(row: Dict, options: Optional[List[str]] = None) -> str:
    opts = options if options is not None else row["secenekler"]
    return ("Soru: " + row["soru"] + "\n\nSecenekler:\n" +
            "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(opts)))


def build_m1_mc_prompt(row: Dict, idx: int):
    """Shuffled-option verbatim-recall (reuses probe_m1 logic)."""
    opts = list(row["secenekler"])
    truth_idx = row["cevap_idx"]
    truth_text = opts[truth_idx] if truth_idx is not None and truth_idx < len(opts) else ""
    rng = random.Random(SEED + idx)
    order = list(range(len(opts)))
    rng.shuffle(order)
    shuffled = [opts[i] for i in order]
    truth_new_letter = chr(65 + order.index(truth_idx)) if truth_idx in order else "?"
    lines = [f"Soru: {row['soru']}", "", "Secenekler:"]
    lines += [f"{chr(65+i)}) {c}" for i, c in enumerate(shuffled)]
    prompt = "\n".join(lines) + (
        "\n\nCevap (ONLY the letter of the correct choice, e.g. A) followed by "
        "the FULL EXACT text of that choice, reproduced verbatim as it appears above:")
    return prompt, truth_text, truth_new_letter


def build_m1_qa_prompt(row: Dict) -> str:
    return ("Soru: " + row["soru"] +
            "\n\nBu sorunun doğru cevabını, kaynak metindeki haliyle AYNEN (birebir) "
            "yazın. Sadece cevap metnini verin:")


def build_m2a_mc_prompt(rows: List[Dict], i: int):
    """Distractor flip: plant another row's correct option as one distractor."""
    row = rows[i]
    correct = row["secenekler"][row["cevap_idx"]]
    other = rows[(i + 1) % len(rows)]
    other_ans = other["secenekler"][other["cevap_idx"]]
    pert = list(row["secenekler"])
    planted_letter = None
    for k, c in enumerate(pert):
        if c != correct:
            pert[k] = other_ans
            planted_letter = chr(65 + k)
            break
    if planted_letter is None or other_ans == correct:
        return None, None, None, None      # degenerate: nothing to plant
    return mc_text(row, pert), row["cevap"], planted_letter, other_ans


def build_m2a_hv_prompt(row: Dict, idx: int):
    """Halluverse M2a: 2-option forced choice, true vs edited answer.

    Position randomized deterministically per item (seed+idx) to guard against
    position bias. flip-follow = model picks the hallucination-edited option.
    """
    truth, edited = row["cevap"].strip(), row["edited_cevap"].strip()
    if not truth or not edited or truth == edited:
        return None, None
    rng = random.Random(SEED + idx)
    if rng.random() < 0.5:
        a, b, truth_letter, edited_letter = truth, edited, "A", "B"
    else:
        a, b, truth_letter, edited_letter = edited, truth, "B", "A"
    prompt = ("Soru: " + row["soru"] +
              f"\n\nHangisi doğru cevaptır?\nA) {a}\nB) {b}\n\nCevap (tek harf A/B):")
    return prompt, truth_letter, edited_letter


# ---------------------------------------------------------------------------
# QA answer scoring (lenient containment match; reported honestly as such)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.lower()).strip(" .,;:!?\"'")
    return s


def qa_match(gold: str, pred: str) -> bool:
    g, p = _norm(gold), _norm(pred)
    if not g or not p:
        return False
    return g == p or g in p or p in g


# ---------------------------------------------------------------------------
# Token bucket rate limiter (thread-safe, global ~2 req/s)
# ---------------------------------------------------------------------------
class TokenBucket:
    def __init__(self, rate: float, capacity: float = 4.0):
        self.rate = rate
        self.cap = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.cap,
                                  self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(max(wait, 0.01))


# ---------------------------------------------------------------------------
# Retry layer (a): core.chat_record does retry-on-empty (max 2, logged);
# this wrapper adds 5xx/429/rate/timeout backoff retries (max 2, logged).
# ---------------------------------------------------------------------------
RETRY_RE = re.compile(r"(429|5\d\d|rate.?limit|overload|too many|timeout|timed out|"
                      r"ConnectionError|connection|ECONN|503|502|500)", re.I)


def call_with_retries(client, model: str, system: str, user: str,
                      max_tokens: int = 1000, max_http_retries: int = 2):
    rec = core.chat_record(client, model, system, user, max_tokens=max_tokens)
    retry_log: List[Dict] = []
    attempt = 0
    while (not rec["ok"]) and attempt < max_http_retries and RETRY_RE.search(rec["error"] or ""):
        attempt += 1
        backoff = 2.0 * (2 ** (attempt - 1))          # 2s, 4s
        retry_log.append({"attempt": attempt, "error": rec["error"], "backoff_s": backoff})
        print(f"[http-retry] {model} attempt={attempt} backoff={backoff}s "
              f"err={rec['error'][:120]}")
        time.sleep(backoff)
        rec = core.chat_record(client, model, system, user, max_tokens=max_tokens)
    rec["retry_log"] = retry_log
    rec["retries"] = rec["retries"] + len(retry_log)
    return rec


# ---------------------------------------------------------------------------
# Translation cache: TR->EN->TR surfaces are computed ONCE per (benchmark,
# item) and reused by every model's B/C arms (lazy, lock-protected).
# ---------------------------------------------------------------------------
_TRANS_LOCK = threading.Lock()
_TRANS_CACHE: Dict[tuple, Dict[str, str]] = {}


def _ensure_translations(client, benchmark: str, row: Dict, item: int):
    """Return dict {'backtr_tr': ..., 'en_direct': ...}, computing + logging
    the two translation calls if not cached. Thread-safe: exactly one caller
    performs the work, others wait on the lock and read the result."""
    key = (benchmark, item)
    with _TRANS_LOCK:
        if key in _TRANS_CACHE:
            return _TRANS_CACHE[key]
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            src = mc_text(row)
        else:
            src = "Soru: " + row["soru"]
        sys_tr_en = ("You are a faithful professional Turkish->English translator. "
                     "Translate the whole question AND every choice. Keep letter "
                     "labels (A,B,C,..) as-is. Preserve meaning precisely. "
                     "Output ONLY the translation.")
        sys_en_tr = ("You are a faithful professional English->Turkish translator. "
                     "Translate back to Turkish. Keep letter labels (A,B,C,..) as-is. "
                     "Preserve meaning precisely. Output ONLY the translation.")
        BUCKET.acquire()
        rec_en = call_with_retries(client, TRANS_MODEL, sys_tr_en, src, max_tokens=700)
        rec_en_rec = _record(benchmark, TRANS_MODEL, "M2b_tr_en", item, rec_en,
                             {"translation": rec_en["content"][:1000], "arm": "tr_en"})
        log_record(rec_en_rec)
        BUCKET.acquire()
        rec_tr = call_with_retries(client, TRANS_MODEL, sys_en_tr,
                                   rec_en["content"], max_tokens=700)
        rec_tr_rec = _record(benchmark, TRANS_MODEL, "M2b_en_tr", item, rec_tr,
                             {"translation": rec_tr["content"][:1000], "arm": "en_tr"})
        log_record(rec_tr_rec)
        out = {"backtr_tr": rec_tr["content"], "en_direct": rec_en["content"]}
        _TRANS_CACHE[key] = out
        return out


# ---------------------------------------------------------------------------
# Probe execution -> one dict of outcome fields per call
# ---------------------------------------------------------------------------
def run_probe_call(client, benchmark: str, model: str, rows: List[Dict],
                   probe: str, item: int) -> Optional[dict]:
    """Execute ONE call for (benchmark, model, probe, item_idx). Returns an
    outcome dict (None only for degenerate items that must not be called)."""
    row = rows[item]
    if probe == "M1":
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            if row["cevap_idx"] is None:
                return None
            prompt, truth_text, _ = build_m1_mc_prompt(row, item)
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Give ONLY the letter and then the verbatim choice text.")
            truth = truth_text
        else:
            prompt, truth = build_m1_qa_prompt(row), row["cevap"]
            system = ("You are answering a Turkish factual QA task. Reproduce the "
                      "exact answer text verbatim, nothing else.")
        rec = call_with_retries(client, model, system, prompt, max_tokens=700)
        out = {"overlap8": round(overlap8(truth, rec["content"]), 4),
               "verbatim_hit": is_verbatim(truth, rec["content"]),
               "truth_text": truth[:400]}
        return _record(benchmark, model, probe, item, rec, out)

    if probe == "M2a":
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            prompt, truth_letter, planted_letter, planted_text = build_m2a_mc_prompt(rows, item)
            if prompt is None:
                return None
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B/C/D/E).")
            rec = call_with_retries(client, model, system,
                                    prompt + "\n\nCevap (tek harf A/B/C/D/E):")
            parsed = core.parse_letter(rec["content"])
            correct = parsed == truth_letter
            flip = bool(parsed == planted_letter) and planted_text != row["secenekler"][row["cevap_idx"]]
            out = {"parsed_letter": parsed, "answer_correct": correct,
                   "flip_follow": flip, "truth_letter": truth_letter,
                   "planted_letter": planted_letter}
        else:  # halluverse: edited-answer flip (2-option forced choice)
            prompt, truth_letter, edited_letter = build_m2a_hv_prompt(row, item)
            if prompt is None:
                return None
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B).")
            rec = call_with_retries(client, model, system, prompt, max_tokens=400)
            parsed = core.parse_letter(rec["content"])
            correct = parsed == truth_letter
            flip = bool(parsed == edited_letter)
            out = {"parsed_letter": parsed, "answer_correct": correct,
                   "flip_follow": flip, "truth_letter": truth_letter,
                   "planted_letter": edited_letter,
                   "edited_cevap": row.get("edited_cevap", "")[:200]}
        return _record(benchmark, model, probe, item, rec, out)

    # ---------------- M2b arms (translation via TRANS_MODEL) ----------------
    if probe == "M2b_A":
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            if row["cevap_idx"] is None:
                return None
            system = ("You are taking a standardized Turkish multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B/C/D/E).")
            rec = call_with_retries(client, model, system,
                                    mc_text(row) + "\n\nCevap (tek harf A/B/C/D/E):")
            parsed = core.parse_letter(rec["content"])
            out = {"parsed_letter": parsed,
                   "answer_correct": parsed == row["cevap"], "arm": "A"}
        else:
            system = ("You are answering a Turkish factual QA question. Give the "
                      "answer in Turkish, concise and exact.")
            rec = call_with_retries(client, model, system, "Soru: " + row["soru"] + "\n\nCevap:")
            out = {"qa_match": qa_match(row["cevap"], rec["content"]),
                   "answer_correct": qa_match(row["cevap"], rec["content"]),
                   "arm": "A"}
        return _record(benchmark, model, probe, item, rec, out)

    if probe in ("M2b_B", "M2b_C"):
        # translated surfaces computed once per (benchmark, item) and cached
        surf = _ensure_translations(client, benchmark, row, item)
        surface = surf["backtr_tr"] if probe == "M2b_B" else surf["en_direct"]
        if not surface:
            return None                       # translation call failed earlier
        if benchmark in ("tr_mmlu", "tumlu_tr"):
            system = ("You are taking a standardized multiple-choice exam. "
                      "Answer with ONLY the single letter (A/B/C/D/E).")
            rec = call_with_retries(client, model, system,
                                    surface + "\n\nCevap (tek harf A/B/C/D/E):")
            parsed = core.parse_letter(rec["content"])
            out = {"parsed_letter": parsed,
                   "answer_correct": parsed == row["cevap"], "arm": "B" if probe == "M2b_B" else "C"}
        else:
            lang = "Turkish" if probe == "M2b_B" else "English"
            system = (f"You are answering a factual QA question. Give the answer in "
                      f"{lang}, concise and exact.")
            rec = call_with_retries(client, model, system, surface + "\n\nCevap:")
            out = {"qa_match": qa_match(row["cevap"], rec["content"]),
                   "answer_correct": qa_match(row["cevap"], rec["content"]),
                   "arm": "B" if probe == "M2b_B" else "C"}
        return _record(benchmark, model, probe, item, rec, out)

    raise ValueError(f"unknown probe {probe}")


def _record(benchmark: str, model: str, probe: str, item: int,
            rec: dict, out: dict) -> dict:
    usage = rec["usage"]
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    ptd = usage.get("prompt_tokens_details") or {}
    ctd = usage.get("completion_tokens_details") or {}
    cached = int(ptd.get("cached_tokens") or 0)
    reason = int(ctd.get("reasoning_tokens") or 0)
    cost = cost_estimate(model, usage)
    base = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": "pilot_audit_2026-08-16",
        "benchmark": benchmark,
        "model": model,
        "probe": probe,
        "probe_family": probe.split("_")[0],
        "item_idx": item,
        "item_key": f"{benchmark}|{model}|{probe}|{item}",
        "prompt_tokens": pt, "cached_tokens": cached,
        "completion_tokens": ct, "reasoning_tokens": reason,
        "total_tokens": pt + ct,
        "usage_api": usage,
        "cost_est_usd": cost,
        "retries": rec["retries"],
        "retry_log": rec.get("retry_log", []),
        "ok": rec["ok"], "error": rec["error"], "took_s": rec["took_s"],
        "raw": rec["content"][:400],
    }
    base.update(out)
    return base


# ---------------------------------------------------------------------------
# Cost estimator (dry-run + live abort). Documented rates only; mimo-v2.5 has
# no published rate -> tokens counted, cost not estimable (smoke §4 note).
# ---------------------------------------------------------------------------
# Conservative per-call token budgets derived from smoke stats (smoke_day2 §4):
#   luna      : ~200 prompt / ~85 completion actual  -> budget 400/200
#   deepseek  : ~280 prompt / ~350 completion actual -> budget 400/700 (answers)
#   translations (deepseek): ~250 prompt / ~150 comp -> budget 400/200
TOK_BUDGET = {
    "gpt-5.6-luna": {"ans_prompt": 400, "ans_comp": 200},
    "mimo-v2.5": None,
    "deepseek-v4-flash": {"ans_prompt": 400, "ans_comp": 700},
}


def build_tasks(benchmarks: List[str], models: List[str], samples: Dict,
                probes: Optional[List[str]] = None) -> List[tuple]:
    """(benchmark, model, probe, item_idx) per call. M2b translations are NOT
    per-model tasks: they are produced lazily by _ensure_translations (once per
    benchmark x item, logged as model=deepseek-v4-flash records)."""
    tasks = []
    for b in benchmarks:
        for m in models:
            for probe in probes or ["M1", "M2a", "M2b_A", "M2b_B", "M2b_C"]:
                if probe == "M2a" and b == "ragturk_formal5k":
                    continue                  # QA-pair has no options; M2a N/A (noted honestly)
                for i in range(len(samples[b])):
                    tasks.append((b, m, probe, i))
    return tasks


def estimate_cost(tasks, samples, benchmarks) -> float:
    """Conservative documented-rate estimate; None-rate models excluded.
    Translations: 2 per (benchmark, item), all via deepseek-v4-flash."""
    total = 0.0
    n_per = {}
    for b, m, probe, i in tasks:
        n_per[(m, probe)] = n_per.get((m, probe), 0) + 1
    for (m, probe), n in n_per.items():
        budget = TOK_BUDGET.get(m)
        if budget is None:
            continue
        pt = budget["ans_prompt"]
        ct = budget["ans_comp"]
        p_in, p_cache, p_out = LADDER[m]
        total += n * (pt * p_in + ct * p_out) / 1e6
    # translation calls: 2 per benchmark x item (deepseek)
    n_trans = 2 * len(benchmarks) * len(samples[benchmarks[0]]) if benchmarks else 0
    p_in, p_cache, p_out = LADDER[TRANS_MODEL]
    total += n_trans * (400 * p_in + 200 * p_out) / 1e6
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rate", type=float, default=2.0, help="global req/s")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--models", default=",".join(ANS_MODELS))
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--max-cost", type=float, default=1.00)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probes", default=None,
                    help="comma list, default all (M1,M2a,M2b_A,M2b_tr_en,M2b_en_tr,M2b_B,M2b_C)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    probes = [p.strip() for p in args.probes.split(",")] if args.probes else None

    samples = {b: load_sample(b, args.limit, seed=SEED) for b in benchmarks}
    tasks = build_tasks(benchmarks, models, samples, probes)
    est = estimate_cost(tasks, samples, benchmarks)
    n_calls = len(tasks)

    print(f"=== PILOT PLAN (Day 5, 2026-08-16) ===")
    print(f"benchmarks : {benchmarks}")
    print(f"models     : {models}")
    print(f"probes     : {probes or ['M1','M2a','M2b_A','M2b_B','M2b_C']} (+2 translation calls per item, lazy)")
    print(f"n per benchmark sample (seed 42) : { {b: len(samples[b]) for b in benchmarks} }")
    print(f"planned calls: {n_calls} (+ {2 * len(benchmarks) * (len(samples[benchmarks[0]]) if benchmarks else 0)} translations)")
    print(f"estimated cost (documented rates only; mimo excluded, tokens counted): "
          f"${est:.4f}  [cap ${args.max_cost:.2f}]")
    if args.dry_run:
        print("[dry-run] no calls made. OK to proceed." if est <= args.max_cost else
              f"[dry-run] ABORT: estimate ${est:.4f} exceeds cap ${args.max_cost:.2f}.")
        return
    if est > args.max_cost:
        print(f"HARD ABORT: estimated cost ${est:.4f} exceeds --max-cost "
              f"${args.max_cost:.2f}. No calls made. Reduce n/benchmarks or raise cap.")
        return 1

    # resume: skip keys already present in the log; re-seed the translation
    # cache from logged M2b_tr_en / M2b_en_tr records (no re-calls on resume)
    done = set()
    if os.path.exists(args.log):
        with open(args.log, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done.add(rec.get("item_key"))
                if rec.get("probe") in ("M2b_tr_en", "M2b_en_tr") and rec.get("ok"):
                    key = (rec["benchmark"], rec["item_idx"])
                    surf = _TRANS_CACHE.setdefault(key, {})
                    if rec["probe"] == "M2b_tr_en":
                        surf["en_direct"] = rec.get("translation", "")
                    else:
                        surf["backtr_tr"] = rec.get("translation", "")
    todo = [t for t in tasks if f"{t[0]}|{t[1]}|{t[2]}|{t[3]}" not in done]
    print(f"resume: {len(done)} calls already logged -> skipping, {len(todo)} to run "
          f"(translation cache re-seeded: {len(_TRANS_CACHE)} items)")

    global BUCKET
    BUCKET = TokenBucket(args.rate)
    MAX_COST[0] = args.max_cost
    ABORT.clear()
    LOG_PATH[0] = args.log
    client = core.make_client()
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    def run_one(task):
        if ABORT.is_set():
            return (0, 0, 0)
        b, m, probe, i = task
        BUCKET.acquire()
        try:
            rec = run_probe_call(client, b, m, samples[b], probe, i)
            if rec is None:
                return (0, 0, 0)            # degenerate item; never a call
            log_record(rec)
            tag = f"{b[:10]}/{m[:16]}/{probe}"
            print(f"[{tag} {i}] {'ok' if rec['ok'] else 'FAIL'} "
                  f"retries={rec['retries']} tok={rec['total_tokens']} "
                  f"cost≈{rec['cost_est_usd']}")
            return (int(rec["ok"]), int(not rec["ok"]), rec["retries"])
        except Exception as exc:            # never let a worker die silently
            print(f"[EXC] {b}|{m}|{probe}|{i}: {type(exc).__name__}: {exc}")
            return (0, 1, 0)

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    n_ok = n_fail = n_ret = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ok, fail, ret in ex.map(run_one, todo):
                n_ok += ok
                n_fail += fail
                n_ret += ret
    dt = time.time() - t0
    with COST_LOCK:
        final_cost = SPENT[0]
    n_run = n_ok + n_fail
    print(f"\n=== PILOT RUN SUMMARY ===")
    print(f"planned={n_calls} run={n_run} ok={n_ok} fail={n_fail} retries={n_ret}")
    print(f"elapsed={dt:.0f}s  rate~{n_run/max(dt,1):.2f} calls/s")
    print(f"accumulated estimated cost (rated models) = ${final_cost:.4f} "
          f"[cap ${args.max_cost:.2f}]")
    print(f"log -> {args.log}")
    if ABORT.is_set():
        print("NOTE: run aborted early by live cost cap; rerun same command to resume.")


if __name__ == "__main__":
    sys.exit(main())
