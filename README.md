# trmlu-audit

**API-only, cross-lingual contamination detection tooling for TR-MMLU — the Turkish
MMLU benchmark for LLM evaluation.**

This package lets you run black-box (API-only) probes against the gated
[`alibayram/turkish_mmlu`](https://huggingface.co/datasets/alibayram/turkish_mmlu)
dataset to look for *benchmark contamination* — especially the translation-mediated
form that hides from English-trained detectors. It powers a workshop paper on the
state of cross-lingual contamination in Turkish LLM evaluation.

**No GPU. No model weights.** Works on any hosted model (DeepSeek, GPT, Claude) via an
OpenAI-compatible endpoint (OpenRouter by default).

## Why this exists

Data contamination — test-set leakage into pre-training data — inflates LLM benchmark
scores. Most detectors are English-centric and use exact text overlap, so contamination
introduced *via translation* (e.g. a Turkish test set back-translated into the training
corpus) can evade them ([Yao et al., 2024](https://arxiv.org/abs/2406.13236);
[Contamination Report for Multilingual Benchmarks, 2024](https://arxiv.org/abs/2410.16186)).
Turkish is largely uncharacterized here. This toolkit provides the probes to change that.

## Probes

| Probe | What it does |
|-------|--------------|
| `black_box`          | Baseline forward-pass accuracy on TR-MMLU |
| `choice_substitution`| Plants the correct answer of a *different* question as a distractor (Yao et al.), to test if the model is memorizing answer choices |
| `crosslingual`       | `Tr→En→Tr` back-translation (arm B: surface-fragility test) and `Tr→En` direct (arm C: language-transfer test) |

## Paper

The accompanying paper, *"Cross-lingual Contamination of Turkish LLM Evaluation:
A Reproducible Black-Box Audit of TR-MMLU"*, lives in [`paper/`](paper/):

| File | Description |
|------|-------------|
| [`paper_preprint.pdf`](paper/paper_preprint.pdf) | **Recommended** — named, camera-ready version (author: *Oğuz Emre Cura*). Ready for arXiv or workshop submission. |
| [`paper.pdf`](paper/paper.pdf) | Anonymous **review** version (no author name — for peer review). |
| [`paper_preprint.tex`](paper/paper_preprint.tex) | LaTeX source for the preprint (compile this). |
| [`paper.tex`](paper/paper.tex) | LaTeX source for the anonymous review version. |
| [`custom.bib`](paper/custom.bib) | Verified references (real author names, live-fetched from arXiv). |

**Compile the recommended preprint PDF** (requires `tectonic` or another XeTeX engine):

```bash
cd paper
tectonic paper_preprint.tex
# -> paper_preprint.pdf
```

To switch to the anonymous review version, compile `paper.tex` instead
(`\usepackage[review]{acl}` is already set).

## Install

```bash
git clone <your-repo-url> trmlu-audit
cd trmlu-audit
uv sync             # or: pip install -e ".[dev]"
```

Set environment variables (e.g. in `.env`, gitignored):

```bash
export OPENROUTER_API_KEY="sk-..."   # model endpoint key
export HF_TOKEN="hf_..."             # HF read token for the gated TR-MMLU dataset
```

## Usage

```python
from trmlu_audit import make_client, load_tr_mmlu, run_black_box

client = make_client()
rows  = load_tr_mmlu(split="mmlu", limit=100)      # seed 42 sample
recs  = run_black_box(client, rows, limit=100)     # forward-pass accuracy
```

CLI:

```bash
trmlu-audit black_box    --limit 100 --out results/bb.jsonl
trmlu-audit choice_sub   --limit 100 --out results/cs.jsonl
trmlu-audit crosslingual --limit 30  --arms B C --out results/xl.jsonl
```

## Reproducing the paper's results

The fully-matched results reported in the paper (`n=175` item-tuples, seed 42,
each answered in all three arms) are:

| Arm | n (matched) | Accuracy | 95% CI (Wilson) |
|-----|------------|----------|-----------------|
| A — baseline (original Turkish) | 175 | **88.0%** | [82.4, 92.0] |
| B — back-translated Turkish  | 175 | 84.6% | [78.5, 89.2] |
| C — English-direct          | 175 | 88.0% | [82.4, 92.0] |

McNemar exact tests vs. baseline: arm B `p = 0.0703` (7→1 directional flips),
arm C `p = 1.0000` (5↔5 symmetric).

The honest headline: **no statistically significant cross-lingual contamination
signal** in DeepSeek v4 Flash on TR-MMLU at this scale, while the round-trip
(arm B) shows a *suggestive, non-significant* fragility direction that motivates
the paper's native-speaker-validation protocol.

Re-run them with `--limit 200` (or the `crosslingual_probe.py --arms A B C`
matched runner used for the paper). Note TR-MMLU is **CC BY-NC-ND 4.0** — that
license governs the dataset, not this tooling.

## License / ethics

- This repository: **MIT**.
- The TR-MMLU dataset is distributed by its authors under **CC BY-NC-ND 4.0**; audit
  results here are for analysis, not redistribution of the dataset.
- Detection probes are statistical audit signals, **not** proof of contamination.
  A null result does not certify a model is clean; it only says no signal was detected
  with these probes at this scale.

## Project status

Research / pilot stage. Single-author project by a high-school researcher. Contributions,
issues, and reproduction reports welcome.