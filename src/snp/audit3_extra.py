"""Protocol v1.5, analyses added after the ratings (exploratory; logged in notes/FREEZE.md).

  PYTHONPATH=src python -m snp.audit3_extra   -> results/repair_audit_extra.txt

Adds to snp.audit3.analyse: Wilson intervals for the gate rates, the self judge per repair model, G5 (extractor
diff) and G6 (G5 and G3) against the repair-level reference, the extractor's population-weighted recall, the
commitment types named for changed repairs, and the same statistics on agreed items only (sensitivity).
"""
import csv
import json
from collections import Counter, defaultdict
from math import sqrt

from .audit3 import A, MODELS, OTHER, Q1, Q2, RES, SELF, _pairs
from .analysis import calibrate, load


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def labels(use_adj):
    key = {r["item"]: r for r in csv.DictReader(open(A / "repair_key.csv", encoding="utf-8-sig"))}
    r1 = {r["item"]: r for r in csv.DictReader(open(A / "repair_ratings_rater1.csv", encoding="utf-8-sig"))}
    r2 = {r["item"]: r for r in csv.DictReader(open(A / "repair_ratings_rater2.csv", encoding="utf-8-sig"))}
    adj, adj_q2 = {}, {}
    if use_adj:
        for rec in csv.DictReader(open(A / "repair_adjudication.csv", encoding="utf-8-sig")):
            adj[str(rec["item"])] = str(rec.get(Q1) or "").strip().lower()
            adj_q2[str(rec["item"])] = str(rec.get(Q2) or "").strip().lower()
    lab, types = {}, {}
    for i in key:
        a, b = r1[i][Q1].strip().lower(), r2[i][Q1].strip().lower()
        if a == b and a in ("same", "different"):
            lab[i] = a
            types[i] = {t.strip() for q in (r1[i][Q2], r2[i][Q2]) for t in q.lower().replace("/", ",").split(",")
                        if t.strip()}
        elif adj.get(i) in ("same", "different"):
            lab[i] = adj[i]
            types[i] = {t.strip() for t in adj_q2[i].replace("/", ",").split(",") if t.strip()}
        else:
            lab[i] = None
    return key, lab, types


def run(use_adj, out):
    key, lab, types = labels(use_adj)
    _, rows = load()
    th, _ = calibrate(rows)
    gates = {r["pid"]: r for r in csv.DictReader(open(RES / "repair_gates_g1g3.csv"))}
    judge = {f: {json.loads(l)["pid"]: (json.loads(l)["score"] or 0) for l in open(RES / f"{f}.jsonl")}
             for f in set(SELF.values()) | set(OTHER.values())}
    acc = defaultdict(Counter)
    for i, l in lab.items():
        if not l:
            continue
        k = key[i]
        pid, m, flag = k["pid"], k["repair_model"], k["extractor_flag"] == "1"
        g = gates[pid]
        g3 = float(g["g3_min"]) >= th["G3_NLI"]
        dec = {"G1": float(g["g1_bertscore_f1"]) >= th["G1_BERTScore"],
               "G2": float(g["g2_cosine"]) >= th["G2_SBERT"], "G3": g3,
               "G5": not flag, "G6": (not flag) and g3,
               "self judge": judge[SELF[m]].get(pid, 0) >= 4, "other judge": judge[OTHER[m]].get(pid, 0) >= 4,
               f"self judge {m}": judge[SELF[m]].get(pid, 0) >= 4,
               f"other judge {m}": judge[OTHER[m]].get(pid, 0) >= 4}
        for name, d in dec.items():
            c = acc[name]
            if l == "different":
                c["changed"] += 1
                c["silent"] += d
            else:
                c["preserved"] += 1
                c["false_esc"] += not d
    out.append(f"\n== {'with adjudication' if use_adj else 'agreed items only'}: {Counter(lab.values())}")
    out.append(f"{'gate':28s} silent acceptance [95% Wilson]        false escalation [95% Wilson]")
    for name, c in acc.items():
        s, n, f, p = c["silent"], c["changed"], c["false_esc"], c["preserved"]
        ls, us = wilson(s, n)
        lf, uf = wilson(f, p)
        out.append(f"{name:28s} {s:3d}/{n:<3d} {s / n:6.1%} [{ls:5.1%}, {us:5.1%}]   "
                   f"{f:3d}/{p:<3d} {f / p:6.1%} [{lf:5.1%}, {uf:5.1%}]")
    # extractor recall, weighted by the population share of flagged and unflagged repairs
    P = _pairs()
    pop = Counter((p["repair_model"], p["flag"]) for p in P)
    for m in MODELS:
        rate = {}
        for fl in (True, False):
            v = [lab[i] == "different" for i in lab if lab[i] and key[i]["repair_model"] == m
                 and (key[i]["extractor_flag"] == "1") == fl]
            rate[fl] = sum(v) / len(v)
        tp, fn = pop[(m, True)] * rate[True], pop[(m, False)] * rate[False]
        out.append(f"extractor recall {m:16s} {tp / (tp + fn):.1%}  (unflagged share of repairs "
                   f"{pop[(m, False)] / (pop[(m, True)] + pop[(m, False)]):.1%})")
    for fl in ("1", "0"):
        t = Counter(x for i in lab if lab[i] == "different" and key[i]["extractor_flag"] == fl for x in types[i])
        n = sum(1 for i in lab if lab[i] == "different" and key[i]["extractor_flag"] == fl)
        out.append(f"types named for changed repairs, extractor flag {fl} (n={n}): {dict(t.most_common())}")


def main():
    out = []
    run(True, out)
    run(False, out)
    text = "\n".join(out)
    (RES / "repair_audit_extra.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
