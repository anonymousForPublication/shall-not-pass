"""Precision-adjusted RQ3 rates (protocol v1.4, post hoc estimate).

  PYTHONPATH=src python -m snp.rq3_adjust  -> results/analysis/rq3_adjusted.json
A flagged repair is truly changed if at least one of its flagged commitment types is confirmed; with per-type
precision p_t from the 150-item validation and independence across types, P(changed) = 1 - prod(1 - p_t).
Unflagged repairs count as unchanged (43 of 45 unflagged audit items were confirmed unchanged).
Fabricated values are scaled by the precision of fabrication flags among audited value items.
95% intervals: 2,000 bootstrap resamples of the audited items within each type.
"""
import csv
import json
import random
from pathlib import Path

from .extract import extract, find_values

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "data/audit"
TYPES = ["modality", "values", "negation", "quant", "cond"]
ANS = "answer (yes/no/unsure)"


def main():
    key = list(csv.DictReader(open(A / "rq3_key.csv", encoding="utf-8-sig")))
    r1 = {r["item"]: r[ANS].strip().lower() for r in csv.DictReader(open(A / "rq3_ratings_rater1.csv", encoding="utf-8-sig"))}
    r2 = {r["item"]: r[ANS].strip().lower() for r in csv.DictReader(open(A / "rq3_ratings_rater2.csv", encoding="utf-8-sig"))}
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/derived/repair_pairs.jsonl")}
    conf = {t: [] for t in TYPES}
    fabconf = []
    for k in key:
        if k["extractor_flag"] != "1":
            continue
        ok = r1[k["item"]] == r2[k["item"]] == "yes"
        conf[k["type"]].append(ok)
        if k["type"] == "values":
            p = P[k["pid"]]
            src = {(v.num, v.unit) for v in find_values(p["original"])}
            if any((v.num, v.unit) not in src for v in find_values(p["revision"])):
                fabconf.append(ok)
    reps = {}
    for p in P.values():
        if not p["revision"].strip():
            d, fab = set(), False
        else:
            d = set(extract(p["original"]).diff(extract(p["revision"])))
            src = {(v.num, v.unit) for v in find_values(p["original"])}
            fab = any((v.num, v.unit) not in src for v in find_values(p["revision"]))
        reps.setdefault(p["repair_model"], []).append((d, fab))

    def est(pt, pf):
        out = {}
        for m, rs in reps.items():
            n = len(rs)
            anyc = beyond = 0.0
            for d, _ in rs:
                q = 1.0
                for t in d:
                    q *= 1 - pt[t]
                anyc += 1 - q
                q = 1.0
                for t in d - {"modality"}:
                    q *= 1 - pt[t]
                beyond += 1 - q
            out[m] = {"changed": anyc / n, "beyond_modality": beyond / n, "fabricated": pf * sum(f for _, f in rs) / n,
                      "extractor_changed": sum(bool(d) for d, _ in rs) / n, "extractor_fabricated": sum(f for _, f in rs) / n}
        return out

    point = est({t: sum(v) / len(v) for t, v in conf.items()}, sum(fabconf) / len(fabconf))
    rng = random.Random(1)
    boots = []
    for _ in range(2000):
        pt = {t: sum(rng.choices(v, k=len(v))) / len(v) for t, v in conf.items()}
        pf = sum(rng.choices(fabconf, k=len(fabconf))) / len(fabconf)
        boots.append(est(pt, pf))
    res = {"precision": {t: [sum(v), len(v)] for t, v in conf.items()}, "fabrication_precision": [sum(fabconf), len(fabconf)],
           "models": {}}
    for m in point:
        res["models"][m] = {}
        for k2 in point[m]:
            xs = sorted(b[m][k2] for b in boots)
            res["models"][m][k2] = [point[m][k2], xs[50], xs[1949]]
            print(f"{m:16s}{k2:22s}{point[m][k2]:7.1%}  [{xs[50]:.1%}, {xs[1949]:.1%}]")
    print("fabrication precision", res["fabrication_precision"])
    (ROOT / "results/analysis/rq3_adjusted.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
