"""Blind validity audit: 100 drift mutants (10 per operator) + 50 equivalents, shuffled.

Raters see only A (original) and B (variant) and answer whether B keeps exactly the
same obligations. The answer key (operator, kind) is written to a separate file.
Run:  PYTHONPATH=src python -m snp.audit
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EQUIV_QUOTA = {"E1": 8, "E2": 8, "E3": 9, "E4": 5, "E5": 10, "E6": 10}  # sums to 50


def main() -> None:
    req = {json.loads(l)["rid"]: json.loads(l) for l in open(ROOT / "data/derived/requirements.jsonl")}
    by_op = defaultdict(list)
    for line in open(ROOT / "data/derived/variants.jsonl"):
        v = json.loads(line)
        by_op[v["op"]].append(v)
    rng = random.Random(20261101)
    picked = []
    for op in [f"M{i}" for i in range(1, 11)]:
        picked += rng.sample(by_op[op], 10)
    for op, k in EQUIV_QUOTA.items():
        picked += rng.sample(by_op[op], k)
    rng.shuffle(picked)
    out = ROOT / "data/audit"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "audit_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "A_original", "B_variant",
                    "rater1_same_obligations (same/different/unsure)", "rater1_B_is_garbled (y/n)",
                    "rater2_same_obligations (same/different/unsure)", "rater2_B_is_garbled (y/n)", "notes"])
        for i, v in enumerate(picked, 1):
            w.writerow([i, req[v["rid"]]["text"], v["text"], "", "", "", "", ""])
    with open(out / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "vid", "op", "kind", "expected"])
        for i, v in enumerate(picked, 1):
            w.writerow([i, v["vid"], v["op"], v["kind"], "different" if v["kind"] == "drift" else "same"])
    print(f"wrote {len(picked)} items to {out}")


if __name__ == "__main__":
    main()
