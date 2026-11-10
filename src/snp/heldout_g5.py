"""Score held-out mutants (H1-H4) with the frozen commitment-diff gate G5.

  PYTHONPATH=src python -m snp.heldout_g5   -> results/heldout_g5.csv  (pid, op, subtype, g5_accept, g5_diff)
Run only after tag protocol-freeze-v1.3 (the operators and this plan are frozen first).
"""
import csv
import json
from collections import Counter
from pathlib import Path

from .extract import extract

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rows = [json.loads(l) for l in open(ROOT / "data/derived/heldout_pairs.jsonl")]
    out = ROOT / "results/heldout_g5.csv"
    acc, n = Counter(), Counter()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "op", "subtype", "g5_accept", "g5_diff"])
        for r in rows:
            d = extract(r["original"]).diff(extract(r["revision"]))
            w.writerow([r["pid"], r["op"], r["subtype"], int(not d), json.dumps(d)])
            n[r["op"]] += 1
            acc[r["op"]] += int(not d)
            n[(r["op"], r["subtype"])] += 1
            acc[(r["op"], r["subtype"])] += int(not d)
    for k in sorted(n, key=str):
        print(f"{str(k):40s} G5 silent acceptance {acc[k]}/{n[k]} = {acc[k] / n[k]:.1%}")


if __name__ == "__main__":
    main()
