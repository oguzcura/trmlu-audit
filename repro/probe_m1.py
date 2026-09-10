"""Repo-local M1 probe primitives (vendored).

Verbatim-recall (M1) scoring helpers used by ``pilot_audit.py``: 8-gram
surface overlap, verbatim-hit detection, and documented-rate cost
estimation. Vendored from the private audit harness ``probe_m1.py``
(2026-08-16) so that ``repro/`` has no dependency on the harness tree;
the harness copy remains authoritative for the original M1 runner.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

NGRAM = 8

# Documented per-1M-token rates (in, cached-in, out); None = no published rate.
#   deepseek-v4-flash: spec ladder (paper1_endpoints.md)
#   gpt-5.6-luna     : resolved live 2026-08-16 (endpoint notes) $0.10 / $0.60
#   mimo-v2.5        : no published rate -> tokens only (cost_est = None)
LADDER = {
    "deepseek-v4-flash": (0.07, 0.0014, 0.14),
    "gpt-5.6-luna": (0.10, 0.0, 0.60),
    "mimo-v2.5": None,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def grams8(s: str):
    s = _norm(s)
    return {s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1)}


def overlap8(target: str, output: str) -> float:
    """Coverage of target 8-grams by the output (0.0 if target < 8 chars)."""
    tg, og = grams8(target), grams8(output)
    if not tg:
        return 0.0
    return len(tg & og) / len(tg)


def is_verbatim(target: str, output: str) -> bool:
    t, o = _norm(target), _norm(output)
    return t != "" and t in o


def cost_estimate(model: str, usage: Dict) -> Optional[float]:
    if model not in LADDER or LADDER[model] is None:
        return None
    p_in, p_cache, p_out = LADDER[model]
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    ptd = usage.get("prompt_tokens_details") or {}
    cached = int(ptd.get("cached_tokens") or 0)
    return round(pt * p_in / 1e6 + cached * p_cache / 1e6 + ct * p_out / 1e6, 8)
