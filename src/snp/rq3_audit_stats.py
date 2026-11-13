"""RQ3 extractor validation (protocol v1.4): 150 (repair, commitment type) items, two blind raters.

  PYTHONPATH=src python -m snp.rq3_audit_stats   -> results/rq3_audit_stats.txt, results/analysis/rq3_audit.json
Precision of the extractor per type = flagged items that BOTH raters answer "yes" / flagged items.
Lenient precision = flagged items that AT LEAST ONE rater answers "yes".
"Unflagged agreement" = unflagged items both raters answer "no" (the extractor's misses are the rest).
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import wilson
from .audit_stats import kappa

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "data/audit"
ANS = "answer (yes/no/unsure)"


def main():
    key = {r["item"]: r for r in csv.DictReader(open(A / "rq3_key.csv", encoding="utf-8-sig"))}
    sheet = {r["item"]: r for r in csv.DictReader(open(A / "rq3_sheet_rater1.csv", encoding="utf-8-sig"))}
    r1 = {r["item"]: r for r in csv.DictReader(open(A / "rq3_ratings_rater1.csv", encoding="utf-8-sig"))}
    r2 = {r["item"]: r for r in csv.DictReader(open(A / "rq3_ratings_rater2.csv", encoding="utf-8-sig"))}
    n = lambda s: " ".join(s.split())
    out = []
    for name, r in (("rater1", r1), ("rater2", r2)):
        bad = [i for i in key if (n(r[i]["A_original"]), n(r[i]["B_repair"]), n(r[i]["question"]))
               != (n(sheet[i]["A_original"]), n(sheet[i]["B_repair"]), n(sheet[i]["question"]))]
        out.append(f"{name}: text differs from generated sheet in items {sorted(bad, key=int) or 'none'}")
    items = sorted(key, key=int)
    x = [r1[i][ANS].strip().lower() for i in items]
    y = [r2[i][ANS].strip().lower() for i in items]
    k, po = kappa(x, y)
    out.append(f"Cohen's kappa (n={len(items)}): {k:.3f}  observed agreement {po:.3f}  PABAK {2 * po - 1:.3f}")
    out.append(f"answers rater1 {dict(Counter(x))}  rater2 {dict(Counter(y))}")
    st = defaultdict(Counter)
    res = {}
    for i in items:
        t, f = key[i]["type"], key[i]["extractor_flag"] == "1"
        a, b = r1[i][ANS].strip().lower(), r2[i][ANS].strip().lower()
        for s in (t, "ALL", key[i]["repair_model"] + "|" + ("flag" if f else "noflag")):
            c = st[s]
            pre = "f_" if f else "u_"
            c[pre + "n"] += 1
            c[pre + "both_yes"] += a == b == "yes"
            c[pre + "any_yes"] += "yes" in (a, b)
            c[pre + "both_no"] += a == b == "no"
            c[pre + "split"] += a != b
    out.append(f"\n{'type':10s}{'flagged':>8s}{'prec (both yes)':>22s}{'lenient':>9s}{'unflagged':>10s}{'both no':>9s}{'split':>7s}")
    for t in ("modality", "values", "negation", "quant", "cond", "ALL"):
        c = st[t]
        p, lo, hi = wilson(c["f_both_yes"], c["f_n"])
        out.append(f"{t:10s}{c['f_n']:8d}{c['f_both_yes']:6d} = {p:5.1%} [{float(lo):.0%}-{float(hi):.0%}]"
                   f"{c['f_any_yes'] / c['f_n']:9.1%}{c['u_n']:10d}{c['u_both_no']:9d}{c['f_split'] + c['u_split']:7d}")
        res[t] = {k2: v for k2, v in c.items()}
    out.append("\nper repair model (flagged items, precision both yes):")
    for s in sorted(st):
        if s.endswith("|flag"):
            c = st[s]
            out.append(f"  {s.split('|')[0]:16s}{c['f_both_yes']}/{c['f_n']} = {c['f_both_yes'] / c['f_n']:.1%}")
    text = "\n".join(out)
    (ROOT / "results/rq3_audit_stats.txt").write_text(text + "\n")
    (ROOT / "results/analysis/rq3_audit.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main()
