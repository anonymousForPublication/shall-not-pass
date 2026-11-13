"""Held-out audit (protocol v1.4): 40 H1-H4 mutants + 10 equivalents, two blind raters.

  PYTHONPATH=src python -m snp.heldout_audit_stats   -> results/heldout_audit_stats.txt
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import wilson
from .audit_stats import kappa

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "data/audit"
S, G = "same_obligations (same/different/unsure)", "B_is_garbled (y/n)"


def main():
    key = {r["item"]: r for r in csv.DictReader(open(A / "heldout_key.csv", encoding="utf-8-sig"))}
    sheet = {r["item"]: r for r in csv.DictReader(open(A / "heldout_sheet_rater1.csv", encoding="utf-8-sig"))}
    r1 = {r["item"]: r for r in csv.DictReader(open(A / "heldout_ratings_rater1.csv", encoding="utf-8-sig"))}
    r2 = {r["item"]: r for r in csv.DictReader(open(A / "heldout_ratings_rater2.csv", encoding="utf-8-sig"))}
    n = lambda s: " ".join(s.split())
    bad = [i for i in key for r in (r1, r2)
           if (n(r[i]["A_original"]), n(r[i]["B_variant"])) != (n(sheet[i]["A_original"]), n(sheet[i]["B_variant"]))]
    items = sorted(key, key=int)
    out = [f"text mismatches vs generated sheet: {sorted(set(bad), key=int) or 'none'}"]
    for name, col in (("obligations", S), ("garbled", G)):
        x = [r1[i][col].strip().lower() for i in items]
        y = [r2[i][col].strip().lower() for i in items]
        k, po = kappa(x, y)
        out.append(f"Cohen's kappa ({name}, n={len(items)}): {k:.3f}  observed agreement {po:.3f}  PABAK {2 * po - 1:.3f}")
    st = defaultdict(Counter)
    for i in items:
        k = key[i]
        a, b = r1[i][S].strip().lower(), r2[i][S].strip().lower()
        ga, gb = r1[i][G].strip().lower() == "y", r2[i][G].strip().lower() == "y"
        for s in (k["op"], k["kind"]):
            st[s]["n"] += 1
            st[s]["both_valid"] += a == b == k["expected"]
            st[s]["both_invalid"] += a != k["expected"] and b != k["expected"]
            st[s]["split"] += a != b
            st[s]["garbled_both"] += ga and gb
            st[s]["garbled_any"] += ga or gb
    out.append(f"\n{'op':6s}{'n':>4s}{'both valid':>12s}{'both invalid':>14s}{'split':>7s}{'garbled both/any':>18s}")
    for s in ["H1", "H2", "H3", "H4", "E1", "E2", "E3", "E4", "E5", "E6", "drift", "equiv"]:
        if s in st:
            c = st[s]
            out.append(f"{s:6s}{c['n']:4d}{c['both_valid']:12d}{c['both_invalid']:14d}{c['split']:7d}"
                       f"{c['garbled_both']:>12d}/{c['garbled_any']}")
    for kind in ("drift", "equiv"):
        c = st[kind]
        _, lo, hi = wilson(c["both_valid"], c["n"])
        out.append(f"{kind}: both raters agree with construction {c['both_valid']}/{c['n']} "
                   f"({c['both_valid'] / c['n']:.1%}, Wilson 95% {float(lo):.1%}-{float(hi):.1%})")
    text = "\n".join(out)
    (ROOT / "results/heldout_audit_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
