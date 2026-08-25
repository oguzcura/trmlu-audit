"""
Cross-lingual back-translation contamination probe (paper core contribution).

Three arms over the SAME seed-42 sample:
  A. baseline       : original Turkish (already have results/pilot.jsonl)
  B. back-translated: Turkish -> English -> Turkish (round-trip)   [surface-fragility test]
  C. english-direct : Turkish -> English item, answer in English   [language-transfer test]

Contamination signature: accuracy(B) collapses while accuracy(C) is stable => model is
overfit to the memorized Turkish surface rather than the underlying concept. If all arms
agree within noise => robust / null at this scale.

Run:
  uv run python crosslingual_probe.py --limit 30 --out results/pilot_backtrans_bc.jsonl
"""
from __future__ import annotations
import argparse, os, random, sys, json
from typing import List, Dict

TRANS_MODEL = "deepseek/deepseek-v4-flash-0731"
ANS_MODEL = "deepseek/deepseek-v4-flash-0731"


def make_client():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1",
                  api_key=os.getenv("OPENROUTER_API_KEY"))


def _chat(client, model, system, user, max_tokens=1600, strip_letter=True):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0)
    msg = r.choices[0].message
    content = (msg.content or "").strip()
    # Reasoning models hide their chain; if no visible content, fall back to
    # the reasoning tail and look for the answer letter there.
    if not content:
        reasoning = getattr(msg, "reasoning", None)
        if reasoning:
            content = str(reasoning)[-120:] if strip_letter else str(reasoning)[:120]
    return content


# --- Translation arms ---------------------------------------------------------
def item_text(row: Dict) -> str:
    """Whole-item string (question + ordered choices) as one block."""
    return ("Soru: " + row["soru"] + "\n\nSecenekler:\n" +
            "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["secenekler"])))


def tr_to_en(client, row: Dict) -> str:
    sys_ = ("You are a faithful professional Turkish->English translator. "
            "Translate the whole question AND every choice. Keep the letter labels "
            "(A, B, C, ...) exactly as-is. Preserve meaning precisely. Output ONLY the translation.")
    return _chat(client, TRANS_MODEL, sys_, item_text(row))


def en_to_tr(client, en_text: str) -> str:
    sys_ = ("You are a faithful professional English->Turkish translator. "
            "Translate back to Turkish. Keep the letter labels (A, B, C, ...) exactly as-is. "
            "Preserve meaning precisely. Output ONLY the translation.")
    return _chat(client, TRANS_MODEL, sys_, en_text)


def roundtrip(client, row: Dict) -> str:
    en = tr_to_en(client, row)
    return en_to_tr(client, en)


def answer_item(client, text: str, lang_prompt: str) -> str:
    sys_ = ("You are taking a standardized multiple-choice exam. " + lang_prompt)
    return _chat(client, ANS_MODEL, sys_, text + "\n\nCevap (single letter A/B/C/D/E):")


# --- Sampling (must match baseline seed-42 sample) ----------------------------
def load_sample(limit: int, seed: int = 42):
    from datasets import load_dataset
    ds = list(load_dataset("alibayram/turkish_mmlu", split="mmlu"))
    rng = random.Random(seed)
    return rng.sample(ds, min(limit, len(ds)))


def build_item_from_text(text: str, truth_idx: int) -> None:
    """placeholder no-op; exactness handled by analyzers reading original row"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--arms", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    ap.add_argument("--out", default="results/scale3arm.jsonl")
    args = ap.parse_args()

    print(f"[i] Loading sample (seed 42, limit {args.limit}) ...")
    sample = load_sample(args.limit)
    print(f"[i] {len(sample)} items loaded.")
    client = make_client()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(sample):
            rec = {"i": i, "bolum": row["bolum"], "truth_idx": int(row["cevap"]),
                   "soru": row["soru"], "secenekler": row["secenekler"]}
            if "A" in args.arms:
                # baseline: answer the ORIGINAL Turkish item, no perturbation
                txt = item_text(row)
                anc = answer_item(client, txt, "Answer in Turkish (the question is in Turkish). ")
                rec["A_answer"] = anc
            if "B" in args.arms:
                try:
                    tri = roundtrip(client, row)
                    anc = answer_item(client, tri, "Answer in Turkish (the question is in Turkish). ")
                    rec["B_backtr"] = tri
                    rec["B_answer"] = anc
                except Exception as e:
                    rec["B_error"] = str(e)
            if "C" in args.arms:
                try:
                    en = tr_to_en(client, row)
                    anc = answer_item(client, en, "Answer in English (the item is in English). ")
                    rec["C_en"] = en
                    rec["C_answer"] = anc
                except Exception as e:
                    rec["C_error"] = str(e)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(sample)}] {row['bolum'][:22]:22} done")
    print(f"[✓] done -> {args.out}")