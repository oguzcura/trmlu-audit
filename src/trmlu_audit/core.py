"""Core implementation of the TR-MMLU audit probes.

API-only (black-box) contamination detection for TR-MMLU
(alibayram/turkish_mmlu). No weights, no GPU; works on hosted models.

Probes implemented:
  - Black Box           : forward-pass accuracy (baseline)
  - Choice substitution : Yao et al. (2024) pondered distractor planting
  - Cross-lingual       : Tr->En->Tr back-translation + Tr->En direct arms
"""
from __future__ import annotations
import os
import random
import re
from typing import Dict, List, Optional

TRANS_MODEL = "deepseek/deepseek-v4-flash-0731"
ANS_MODEL = "deepseek/deepseek-v4-flash-0731"
_SEED = 42

# ---------------------------------------------------------------------------
# Client / transport
# ---------------------------------------------------------------------------
def make_client():
    from openai import OpenAI
    return OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def _chat(client, model: str, system: str, user: str,
          max_tokens: int = 1600, fallback_tail: bool = True) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0)
    msg = r.choices[0].message
    content = (msg.content or "").strip()
    if not content and fallback_tail:   # reasoning models hide their chain
        reasoning = getattr(msg, "reasoning", None)
        if reasoning:
            content = str(reasoning)[-120:]
    return content


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_tr_mmlu(split: str = "mmlu", limit: Optional[int] = None,
                 seed: int = _SEED) -> List[Dict]:
    """Load the gated TR-MMLU split. Requires HF_TOKEN for first download."""
    if not os.getenv("HF_TOKEN"):
        import sys
        print("[!] HF_TOKEN not set. TR-MMLU is gated (CC BY-NC-ND 4.0).",
              file=sys.stderr)
    from datasets import load_dataset
    rows = list(load_dataset("alibayram/turkish_mmlu", split=split))
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def idx_to_letter(idx) -> str:
    """TR-MMLU stores 'cevap' as an integer index into secenekler."""
    try:
        return chr(65 + int(idx))
    except (TypeError, ValueError):
        return "?"


def item_text(row: Dict) -> str:
    return ("Soru: " + row["soru"] + "\n\nSecenekler:\n" +
            "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["secenekler"])))


# ---------------------------------------------------------------------------
# Answering + parsing
# ---------------------------------------------------------------------------
def answer_item(client, text: str, lang_prompt: str) -> str:
    return _chat(client, ANS_MODEL,
                 "You are taking a standardized multiple-choice exam. " + lang_prompt,
                 text + "\n\nCevap (single letter A/B/C/D/E):")


def parse_letter(raw: Optional[str]) -> Optional[str]:
    """Extract A-E answer letter from free-form model output (incl. reasoning)."""
    if not raw:
        return None
    raw = raw.strip()
    for pat in (r'\b([A-Ea-e])\b\s*\)', r'\bCevap:\s*([A-Ea-e])\b',
                r'\b([A-Ea-e])\b'):
        m = re.search(pat, raw)
        if m and m.group(1).upper() in "ABCDE":
            return m.group(1).upper()
    m = re.fullmatch(r'\s*([A-Ea-e])\s*', raw)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def run_black_box(client, rows: List[Dict], limit: int) -> List[Dict]:
    out = []
    for i, row in enumerate(rows[:limit]):
        out.append({"i": i, "bolum": row["bolum"], "type": "blackbox",
                    "truth": idx_to_letter(row["cevap"]),
                    "raw": answer_item(client, item_text(row),
                                       "Answer in Turkish (the question is in Turkish). ")})
    return out


def build_perturbed(rows: List[Dict], idx: int) -> List[str]:
    """Replace one distractor with the correct answer of another question."""
    correct = rows[idx]["secenekler"][int(rows[idx]["cevap"])]
    other = rows[(idx + 1) % len(rows)]["secenekler"][int(rows[(idx + 1) % len(rows)]["cevap"])]
    pert = list(rows[idx]["secenekler"])
    for i, c in enumerate(pert):
        if c != correct:
            pert[i] = other
            break
    return pert


def run_choice_substitution(client, rows: List[Dict], limit: int) -> List[Dict]:
    out = []
    item_text_pert = item_text
    for i in range(min(limit, len(rows))):
        base = ("Soru: " + rows[i]["soru"] + "\n\nSecenekler:\n" +
                "\n".join(f"{chr(65+k)}) {c}" for k, c in
                          enumerate(build_perturbed(rows, i))))
        out.append({"i": i, "bolum": rows[i]["bolum"], "type": "choice_sub",
                    "truth": idx_to_letter(rows[i]["cevap"]),
                    "raw": answer_item(client, base,
                                       "Answer in Turkish (the question is in Turkish). ")})
    return out


def tr_to_en(client, row: Dict) -> str:
    sys_ = ("You are a faithful professional Turkish->English translator. Translate the "
            "whole question AND every choice. Keep letter labels (A,B,C,..) as-is. "
            "Preserve meaning precisely. Output ONLY the translation.")
    return _chat(client, TRANS_MODEL, sys_, item_text(row))


def en_to_tr(client, en_text: str) -> str:
    sys_ = ("You are a faithful professional English->Turkish translator. Translate back to "
            "Turkish. Keep letter labels (A,B,C,..) as-is. Preserve meaning precisely. "
            "Output ONLY the translation.")
    return _chat(client, TRANS_MODEL, sys_, en_text)


def roundtrip(client, row: Dict) -> str:
    return en_to_tr(client, tr_to_en(client, row))


def run_crosslingual(client, rows: List[Dict], limit: int,
                     arms: Optional[List[str]] = None) -> List[Dict]:
    arms = arms or ["B", "C"]
    out = []
    for i, row in enumerate(rows[:limit]):
        rec = {"i": i, "bolum": row["bolum"], "truth": idx_to_letter(row["cevap"]),
               "soru": row["soru"], "secenekler": row["secenekler"]}
        if "B" in arms:
            rec["B_backtr"] = roundtrip(client, row)
            rec["B_answer"] = answer_item(
                client, rec["B_backtr"],
                "Answer in Turkish (the question is in Turkish). ")
        if "C" in arms:
            en = tr_to_en(client, row)
            rec["C_en"] = en
            rec["C_answer"] = answer_item(
                client, en, "Answer in English (the item is in English). ")
        out.append(rec)
    return out