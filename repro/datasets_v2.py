"""Normalized dataset loaders for Paper 1 (Datasets v2).

Loads TUMLU-tr and RAGTurk and normalizes every row to the harness's
item form used by TR-MMLU:
    MC form:    {soru, secenekler, cevap(letter or index)}
    QA-pair:    {soru, cevap}           (RAGTurk has no MC options)

CRITICAL LOADING RULE (from a prior failed run): never `cd` into / shell-exec
HuggingFace repo names. All access happens inside this module via the Python
`datasets` / `huggingface_hub` libraries. RAGTurk has NO clean parquet split:
loading it via `datasets.load_dataset("metunlp/ragturk")` materializes the whole
repo (~8106 files) and is rejected. Instead we list the per-article JSON files
on the Hub and `hf_hub_download` only the files we actually sample.

Smoke test: `uv run python datasets_v2.py` prints the first n=5 parsed rows of
both TUMLU-tr and RAGTurk (no API spend — disk/Hub only).

Schema notes (verified live 2026-08-16):
  - TUMLU-mini (config "turkish", split dev/test): columns
        subject, question, choices (str repr of a list), answer (letter), CoT
    `answer` is the letter of the correct choice (A..); choices is the 0-based list.
  - RAGTurk: formal_5k/dataset/json/<article>.json each has
        questions.items = [{question, answer, category, related_chunk_ids}]
    Informal_6k exists in the repo tree as a directory but contains NO data
    files on the Hub (only .DS_Store) -> loaders report 0 articles from it.
"""
from __future__ import annotations

import ast
import json
import os
import random
from typing import Dict, List, Optional

import huggingface_hub

TUMLLU = "jafarisbarov/TUMLU-mini"
RAGTURK = "metunlp/ragturk"
HALLUVERSE = "sabdalja/HalluVerse-M3"
TR_MMLU = "alibayram/turkish_mmlu"
_SEED = 42

# ---------------------------------------------------------------------------
# TUMLU-tr  (multiple-choice; native Turkish split)
# ---------------------------------------------------------------------------
def _parse_choices(choices) -> List[str]:
    """TUMLU stores choices as a string repr of a list; robustly parse it."""
    if isinstance(choices, list):
        return list(choices)
    s = str(choices).strip()
    try:
        val = ast.literal_eval(s)
        return list(val) if isinstance(val, (list, tuple)) else [str(val)]
    except Exception:
        # fallback: split on commas, strip brackets
        return [c.strip() for c in s.strip("[]()").split(",") if c.strip()]


def normalize_tumlu(row: Dict) -> Dict:
    """Convert a raw TUMLU row to {soru, secenekler, cevap(letter)}."""
    choices = _parse_choices(row.get("choices"))
    ans = str(row.get("answer", "")).strip().upper()
    # normalise A.. -> 0-based index AND keep the letter for the harness
    try:
        idx = ord(ans) - ord("A") if ans else None
    except Exception:
        idx = None
    if idx is not None and not (0 <= idx < len(choices)):
        idx = None
    return {
        "source": "tumlu",
        "soru": str(row.get("question", "")).strip(),
        "secenekler": choices,
        "cevap": ans,            # letter, e.g. "C"
        "cevap_idx": idx,        # 0-based index into secenekler
        "subject": str(row.get("subject", "")).strip(),
        "CoT": str(row.get("CoT", "")),
    }


def load_tumlu(name: str = "turkish", split: str = "test",
               limit: Optional[int] = None, seed: int = _SEED) -> List[Dict]:
    """Load TUMLU-tr and return normalized MC rows.

    Public repo, no auth required. Tuples of (question, choices, answer).
    """
    from datasets import load_dataset
    ds = load_dataset(TUMLLU, name=name)  # always in Python; never shell
    rows = list(ds[split])                # iterable of dict-like rows
    out = [normalize_tumlu(r) for r in rows]
    if limit is not None:
        rng = random.Random(seed)
        out = rng.sample(out, min(limit, len(out)))
    return out


# ---------------------------------------------------------------------------
# RAGTurk  (QA-pair form)
# ---------------------------------------------------------------------------
def _iter_ragturk_json_paths(pool: str = "formal_5k"):
    """Yield full Hub paths of per-article JSON files under a pool dir."""
    base = f"{pool}/dataset/json"
    try:
        for item in huggingface_hub.list_repo_tree(
            RAGTURK, repo_type="dataset", path_in_repo=base, recursive=False
        ):
            if getattr(item, "is_directory", False):
                continue
            name = item.path.rsplit("/", 1)[-1]
            if name.endswith(".json"):
                yield item.path
    except Exception as exc:  # informal_6k / missing dir handled gracefully
        print(f"[warn] could not list {base}: {exc}")


def _parse_ragturk_json(blob: Dict) -> List[Dict]:
    """Turn one RAGTurk article JSON into normalized QA-pair rows."""
    rendered = blob.get("questions", {})
    if isinstance(rendered, str):  # stored as a repr string
        try:
            rendered = ast.literal_eval(rendered)
        except Exception:
            rendered = {}
    items = rendered.get("items", []) if isinstance(rendered, dict) else []
    out = []
    for q in items:
        out.append({
            "source": "ragturk",
            "soru": str(q.get("question", "")).strip(),
            "cevap": str(q.get("answer", "")).strip(),
            "category": str(q.get("category", "")).strip(),
        })
    return out


def load_ragturk(pool: str = "formal_5k", limit: Optional[int] = None,
                 seed: int = _SEED) -> List[Dict]:
    """Load RAGTurk QA pairs by sampling N article JSON files on the Hub.

    Only the sampled files are downloaded (avoiding the huge repo).
    """
    paths = list(_iter_ragturk_json_paths(pool))
    if limit is not None and paths:
        rng = random.Random(seed)
        paths = rng.sample(paths, min(limit, len(paths)))
    out: List[Dict] = []
    for p in paths:
        try:
            local = huggingface_hub.hf_hub_download(
                RAGTURK, p, repo_type="dataset")
            with open(local, encoding="utf-8") as fh:
                blob = json.load(fh)
            out.extend(_parse_ragturk_json(blob))
        except Exception as exc:
            print(f"[warn] skipping {p}: {exc}")
    return out


def load_ragturk_full(pool: str = "formal_5k") -> tuple[List[Dict], int]:
    """Download EVERY article JSON of a pool (snapshot_download, parallel
    workers) and parse all QA pairs. Returns (rows, n_article_files).

    formal_5k on the Hub = 2,790 JSON files / ~44 MB (verified 2026-08-16),
    so a full materialization is cheap. informal_6k contains NO data files
    (only .DS_Store, verified live) -> returns ([], 0).
    """
    import glob
    local = huggingface_hub.snapshot_download(
        RAGTURK, repo_type="dataset",
        allow_patterns=[f"{pool}/dataset/json/*.json"])
    files = sorted(glob.glob(os.path.join(
        local, pool, "dataset", "json", "*.json")))
    out: List[Dict] = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as fh:
                blob = json.load(fh)
            out.extend(_parse_ragturk_json(blob))
        except Exception as exc:
            print(f"[warn] skipping {p}: {exc}")
    return out, len(files)


# ---------------------------------------------------------------------------
# Halluverse-M3 tr  (QA-pair + hallucinated edited_answer flip-control)
# ---------------------------------------------------------------------------
def load_halluverse_tr(limit: Optional[int] = None,
                       seed: int = _SEED) -> List[Dict]:
    """Download QA_tr.xlsx (verified: FINAL_DATASET/QA/QA_tr.xlsx) and
    normalize to {soru, cevap, edited_cevap, label, qid}.

    Schema (verified live 2026-08-16, 780 rows): Question, QuestionId,
    Answer, edited_answer, label. `edited_answer` is the hallucination-edit
    twin of the true answer -> the natural flipped-option control for M2a.
    """
    import pandas as pd
    local = huggingface_hub.hf_hub_download(
        HALLUVERSE, "FINAL_DATASET/QA/QA_tr.xlsx", repo_type="dataset")
    df = pd.read_excel(local, sheet_name=0)
    out: List[Dict] = []
    for _, r in df.iterrows():
        out.append({
            "source": "halluverse",
            "qid": str(r.get("QuestionId", "")).strip(),
            "soru": str(r.get("Question", "")).strip(),
            "cevap": str(r.get("Answer", "")).strip(),
            "edited_cevap": str(r.get("edited_answer", "")).strip(),
            "label": str(r.get("label", "")).strip(),
        })
    if limit is not None:
        rng = random.Random(seed)
        out = rng.sample(out, min(limit, len(out)))
    return out


# ---------------------------------------------------------------------------
# TR-MMLU (primary target; cached locally -> no HF_TOKEN needed)
# ---------------------------------------------------------------------------
def load_tr_mmlu_rows(limit: Optional[int] = None,
                      seed: int = _SEED) -> List[Dict]:
    """Full TR-MMLU split ('mmlu') normalized to {soru, secenekler, cevap_idx,
    cevap(letter), bolum}. Repo is gated but cached on this machine."""
    from datasets import load_dataset
    ds = load_dataset(TR_MMLU, split="mmlu")
    out: List[Dict] = []
    for r in ds:
        opts = _parse_choices(r.get("secenekler"))
        idx = int(r.get("cevap"))
        out.append({
            "source": "tr_mmlu",
            "bolum": str(r.get("bolum", "")).strip(),
            "soru": str(r.get("soru", "")).strip(),
            "secenekler": opts,
            "cevap_idx": idx,
            "cevap": chr(65 + idx) if 0 <= idx < len(opts) else "?",
        })
    if limit is not None:
        rng = random.Random(seed)
        out = rng.sample(out, min(limit, len(out)))
    return out


# ---------------------------------------------------------------------------
# CSV export (harness/data/)
# ---------------------------------------------------------------------------
def _csv_row_mc(row: Dict) -> Dict:
    return {"source": row["source"], "soru": row["soru"],
            "secenekler": json.dumps(row["secenekler"], ensure_ascii=False),
            "cevap": row.get("cevap", ""),
            "cevap_idx": row.get("cevap_idx", ""),
            "bolum": row.get("bolum", row.get("subject", "")),
            "CoT": row.get("CoT", "")}


def _csv_row_qa(row: Dict) -> Dict:
    return {"source": row["source"], "soru": row["soru"], "cevap": row["cevap"],
            "qid": row.get("qid", ""), "edited_cevap": row.get("edited_cevap", ""),
            "label": row.get("label", ""), "category": row.get("category", "")}


def build_csvs(outdir: str = "data") -> Dict:
    """Write one clean CSV per dataset + a manifest.json. Returns manifest."""
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    manifest: Dict = {"date": "2026-08-16", "seed": _SEED, "datasets": {}}

    def save(name, rows, cols):
        df = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows],
                          columns=cols)   # explicit cols -> header-only CSV when 0 rows
        path = os.path.join(outdir, name)
        df.to_csv(path, index=False, encoding="utf-8")
        return path, len(df)

    # 1. TR-MMLU (full)
    trm = load_tr_mmlu_rows()
    p, n = save("tr_mmlu.csv", trm, ["source", "bolum", "soru", "secenekler", "cevap", "cevap_idx"])
    manifest["datasets"]["tr_mmlu"] = {"file": p, "items": n, "schema": "MC", "note": "full split, cached"}

    # 2. TUMLU-tr (dev + test, full)
    from datasets import load_dataset
    ds = load_dataset(TUMLLU, name="turkish")
    tum = []
    for split in ("dev", "test"):
        tum.extend(normalize_tumlu(r) for r in ds[split])
    p, n = save("tumlu_tr.csv", tum, ["source", "soru", "secenekler", "cevap", "cevap_idx", "subject", "CoT"])
    manifest["datasets"]["tumlu_tr"] = {"file": p, "items": n, "schema": "MC",
                                        "note": "dev=45 test=900 (verified live)"}

    # 3. RAGTurk formal_5k (full materialization)
    rt5, n_files = load_ragturk_full("formal_5k")
    p, n = save("ragturk_formal5k.csv", rt5, ["source", "soru", "cevap", "category", "qid", "edited_cevap", "label"])
    manifest["datasets"]["ragturk_formal5k"] = {"file": p, "items": n, "articles": n_files,
                                                "schema": "QA-pair", "note": "full parse of 2,790 Hub JSONs"}

    # 4. RAGTurk informal_6k (VERIFIED EMPTY on the Hub)
    rt6, n_files6 = load_ragturk_full("informal_6k")
    p, n = save("ragturk_informal6k.csv", rt6, ["source", "soru", "cevap", "category", "qid", "edited_cevap", "label"])
    manifest["datasets"]["ragturk_informal6k"] = {"file": p, "items": n, "articles": n_files6,
                                                  "schema": "QA-pair",
                                                  "note": "Hub contains ONLY .DS_Store (verified live 2026-08-16); 0 data files"}

    # 5. Halluverse-M3 tr
    hv = load_halluverse_tr()
    p, n = save("halluverse_tr.csv", hv, ["source", "qid", "soru", "cevap", "edited_cevap", "label", "category"])
    manifest["datasets"]["halluverse_tr"] = {"file": p, "items": n, "schema": "QA-pair+flip-control",
                                             "note": "QA_tr.xlsx full (780 rows)"}

    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# Self-test (no API spend)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    t0 = time.time()
    manifest = build_csvs("data")
    print("\n### MANIFEST (harness/data/manifest.json) ###")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    # schema verification: re-read each CSV and print first row
    import pandas as pd
    print("\n### schema verification (first row per CSV) ###")
    for name, info in manifest["datasets"].items():
        path = info["file"]
        df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
        print(f"\n{name}: rows={len(df)} cols={list(df.columns)}")
        if len(df):
            print("  example:", json.dumps({c: str(df.iloc[0][c])[:90] for c in df.columns}, ensure_ascii=False))
    print(f"\n[✓] dataset build done in {time.time()-t0:.1f}s -> harness/data/")