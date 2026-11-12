"""Blind validity audit: inter-rater agreement and operator validity.

  PYTHONPATH=src python -m snp.audit_stats   -> results/audit_stats.txt
Inputs: data/audit/audit_ratings.csv (two raters, blind to operator) and data/audit/audit_key.csv.
The two raters annotated separate copies of the sheet; this file merges them (a copy-paste error in an
earlier merge, in the text of item 92 only, was corrected; the ratings were unaffected).
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import wilson

ROOT = Path(__file__).resolve().parents[2]
C1, C2 = "rater1_same_obligations (same/different/unsure)", "rater2_same_obligations (same/different/unsure)"
G1, G2 = "rater1_B_is_garbled (y/n)", "rater2_B_is_garbled (y/n)"
EXCLUDE: set[str] = set()


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in set(x) | set(y))
    return (po - pe) / (1 - pe), po


def main() -> None:
    key = {r["item"]: r for r in csv.DictReader(open(ROOT / "data/audit/audit_key.csv", encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(ROOT / "data/audit/audit_ratings.csv", encoding="utf-8-sig")))
    out = []
    for name, a, b in (("obligations", C1, C2), ("garbled", G1, G2)):
        k, po = kappa([r[a].strip().lower() for r in rows], [r[b].strip().lower() for r in rows])
        out.append(f"Cohen's kappa ({name}, n={len(rows)}): {k:.3f}  observed agreement {po:.3f}")
    st = defaultdict(Counter)
    for r in rows:
        if r["item"] in EXCLUDE:
            continue
        k = key[r["item"]]
        a, b = r[C1].strip().lower(), r[C2].strip().lower()
        for s in (k["op"], k["kind"]):
            st[s]["n"] += 1
            st[s]["both_valid"] += a == b == k["expected"]
            st[s]["both_invalid"] += a != k["expected"] and b != k["expected"]
            st[s]["split"] += a != b
            st[s]["garbled_both"] += r[G1].strip() == "y" and r[G2].strip() == "y"
    out.append(f"\n{'op':6s}{'n':>4s}{'both valid':>12s}{'both invalid':>14s}{'split':>7s}{'garbled (both)':>16s}")
    for s in sorted(st, key=lambda o: (len(o) > 3, o[0], int(o[1:]) if o[1:].isdigit() else 0)):
        c = st[s]
        out.append(f"{s:6s}{c['n']:4d}{c['both_valid']:12d}{c['both_invalid']:14d}{c['split']:7d}{c['garbled_both']:16d}")
    for kind in ("drift", "equiv"):
        c = st[kind]
        _, lo, hi = wilson(c["both_valid"], c["n"])
        out.append(f"{kind}: both raters agree with construction {c['both_valid']}/{c['n']} "
                   f"({c['both_valid'] / c['n']:.1%}, Wilson 95% {float(lo):.1%}-{float(hi):.1%})")
    text = "\n".join(out)
    (ROOT / "results/audit_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
