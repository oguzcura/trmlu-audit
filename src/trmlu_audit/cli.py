"""CLI for trmlu-audit probes."""
from __future__ import annotations
import argparse, json, os


def _save(recs, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser("trmlu-audit")
    sub = ap.add_subparsers(dest="probe", required=True)

    def add_shared(p):
        p.add_argument("--split", default="mmlu")
        p.add_argument("--limit", type=int, default=100)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--out", default="results/probe.jsonl")

    b = sub.add_parser("black_box"); add_shared(b)
    c = sub.add_parser("choice_sub"); add_shared(c)
    x = sub.add_parser("crosslingual"); add_shared(x)
    x.add_argument("--arms", nargs="+", choices=["B", "C"], default=["B", "C"])

    from . import (make_client, load_tr_mmlu, run_black_box,
                   run_choice_substitution, run_crosslingual)
    args = ap.parse_args()

    client = make_client()
    rows = load_tr_mmlu(args.split, limit=args.limit, seed=args.seed)
    print(f"[i] loaded {len(rows)} TR-MMLU questions ({args.split})")

    if args.probe == "black_box":
        _save(run_black_box(client, rows, args.limit), args.out)
    elif args.probe == "choice_sub":
        _save(run_choice_substitution(client, rows, args.limit), args.out)
    elif args.probe == "crosslingual":
        _save(run_crosslingual(client, rows, args.limit, args.arms), args.out)
    print(f"[✓] -> {args.out}")


if __name__ == "__main__":
    main()