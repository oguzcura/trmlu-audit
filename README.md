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

## Reproducing the pilot results

The results reported in the paper (seed 42) are:

| Probe | n | Accuracy |
|-------|---|----------|
| Black Box baseline | 100 | 81.6% |
| Choice-substitution | 100 | 79.8% |
| Back-translated (matched) | 21 | 90.5% |
| English-direct (matched) | 21 | 95.2% |

Re-run them with the commands above using `--limit` matching each n. Note TR-MMLU is
**CC BY-NC-ND 4.0** — that license governs the dataset, not this tooling.

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