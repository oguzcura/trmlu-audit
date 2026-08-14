"""trmlu-audit: API-only detection & audit tooling for TR-MMLU.

Exposes the three probes used in the cross-lingual contamination audit:
  - black_box            : baseline forward-pass accuracy
  - choice_substitution  : Yao et al. (2024) style distractor-plant perturbation
  - back_translation     : Tr->En->Tr round-trip + Tr->En direct (cross-lingual arms)

All detection is black-box / API-only (works on hosted models like
DeepSeek, GPT, Claude via an OpenAI-compatible endpoint). No model weights,
no GPU required. Requires a Hugging Face read token for the gated TR-MMLU
dataset and an OPENROUTER_API_KEY (or compatible endpoint).
"""
from .core import (
    make_client,
    load_tr_mmlu,
    run_black_box,
    run_choice_substitution,
    run_crosslingual,
    build_perturbed,
    roundtrip,
    tr_to_en,
    answer_item,
    parse_letter,
    idx_to_letter,
)

__all__ = [
    "make_client",
    "load_tr_mmlu",
    "run_black_box",
    "run_choice_substitution",
    "run_crosslingual",
    "build_perturbed",
    "roundtrip",
    "tr_to_en",
    "answer_item",
    "parse_letter",
    "idx_to_letter",
]
__version__ = "0.1.0"