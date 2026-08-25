"""
Matched 3-arm run at scale: baseline / back-translated / english-direct over the
SAME n items (seed 42). Ensures perfect per-item pairing for McNemar.

Writes one JSONL with per-item records for all three arms.
"""
import os, json, random, time
from datasets import load_dataset
from openai import OpenAI
from trmlu_audit.core import (translate_tr_en, translate_en_tr, build_prompt,
                              idx_letter, parse_letter)

MODEL = os.getenv("AUDIT_MODEL", "deepseek/deepseek-v4-flash-0731")
BASE = "https://openrouter.ai/api/v1"

def client():
    return OpenAI(base_url=BASE, api_key=os.getenv("OPENROUTER_API_KEY"))

def load_sample(n, seed=42):
    ds = load_dataset("alibayram/turkish_mmlu", split="mmlu")
    rng = random.Random(seed)
    rows = rng.sample(list(ds), min(n, len(ds)))
    return rows

def arm_answer(cli, src: dict, system_extra="", lang="tr") -> str:
    """One answer call for an arm. Returns raw string (or '<NO_ANSWER>')."""
    prompt = build_prompt(src, lang=lang)
    r = cli.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system_extra},
                  {"role":"user","content":prompt}],
        max_tokens=1600, temperature=0)
    msg = r.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        reasoning = getattr(msg,"reasoning",None)
        content = str(reasoning)[-140:] if reasoning else "<NO_ANSWER>"
    return content

def to_arm_SRC_english(orig):
    """English-direct arm: SRC = translated-to-English question/choices."""
    en_q = translate_tr_en(orig["soru"])
    en_c = [translate_tr_en(c) for c in orig["secenekler"]]
    return {**orig, "soru": en_q, "secenekler": en_c}

def to_arm_SRC_backtrans(orig):
    """Back-translated arm: TR->EN->TR round trip."""
    en_q, en_c = translate_tr_en(orig["soru"]), [translate_tr_en(c) for c in orig["secenekler"]]
    tr_q = translate_en_tr(en_q)
    tr_c = [translate_en_tr(c) for c in en_c]
    return {**orig, "soru": tr_q, "secenekler": tr_c}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", default="results/scale3arm.jsonl")
    args = p.parse_args()

    rows = load_sample(args.limit)
    cli = client()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out,"w",encoding="utf-8")
    n = len(rows)
    for i, orig in enumerate(rows, 1):
        truth = idx_letter(orig["cevap"])
        # Arm A: baseline original
        raw_a = arm_answer(cli, orig, lang="tr")
        rec = {"i": i-1, "bolum": orig["bolum"], "truth": truth,
               "A_raw": raw_a, "A_ans": parse_letter(raw_a)}
        # translate once for C, reuse for B's EN hop
        try:
            nb = to_arm_SRC_backtrans(orig)
            raw_b = arm_answer(cli, nb, lang="tr")
            rec["B_raw"] = raw_b; rec["B_ans"] = parse_letter(raw_b)
        except Exception as e:
            rec["B_raw"] = f"ERR:{e}"; rec["B_ans"] = None
        try:
            nc = to_arm_SRC_english(orig)
            raw_c = arm_answer(cli, nc, lang="en")
            rec["C_raw"] = raw_c; rec["C_ans"] = parse_letter(raw_c)
        except Exception as e:
            rec["C_raw"] = f"ERR:{e}"; rec["C_ans"] = None
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        f.flush()
        print(f"[{i}/{n}] {orig['bolum'][:28]:30s} A={rec['A_ans']} B={rec['B_ans']} C={rec['C_ans']}")
    f.close()
    print(f"[✓] done -> {args.out} ({n} items)")

if __name__ == "__main__":
    main()