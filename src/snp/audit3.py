"""Protocol v1.5: repair-level reference standard for RQ3 (frozen before rating).

  PYTHONPATH=src python -m snp.audit3 sheet     -> data/audit/repair_sheet_rater{1,2}.csv, repair_key.csv
  PYTHONPATH=src python -m snp.audit3 analyse   -> results/repair_audit_stats.txt, results/analysis/repair_audit.json

Sample: for each main repair model (DeepSeek, Qwen 9B, Gemma 12B), 30 repairs the extractor flags as changed and
10 it does not, drawn from repairs NOT used in the 150-item type-level check (seed 20261113). Raters see A and B
only (no model, no extractor verdict, no gate result) and answer
  Q1 does B keep exactly the same obligations as A? (same / different / unsure)
  Q2 if different, which commitments change? (modality, value, polarity, quantifier, condition, actor, scope, other)
Reference: a repair is "changed" if both raters answer "different" and "preserved" if both answer "same";
disagreements are reported, then resolved by discussion (column resolved_label, filled by the raters).
Analysis (fixed now):
  * Cohen's kappa and PABAK on Q1;
  * extractor repair-level precision and recall per model;
  * share of repairs changed per model, weighted by the population share of flagged/unflagged repairs, with a
    bootstrap 95% interval (resampling within strata);
  * for each gate (G1-G3 at the frozen thresholds; self and other judge at score >= 4): share of changed repairs
    accepted (silent acceptance) and share of preserved repairs rejected (false escalation).
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .extract import extract

ROOT = Path(__file__).resolve().parents[2]
D, A, RES = ROOT / "data/derived", ROOT / "data/audit", ROOT / "results"
MODELS = ["deepseek-flash", "qwen3.5:9b", "gemma4:12b"]
Q1 = "Q1 same obligations? (same/different/unsure)"
Q2 = "Q2 if different, which? (modality/value/polarity/quantifier/condition/actor/scope/other)"
SELF = {"deepseek-flash": "rjudge_deepseek-flash", "qwen3.5:9b": "rjudge_qwen3.5-9b", "gemma4:12b": "rjudge_gemma4-12b"}
OTHER = {"deepseek-flash": "rjudge_qwen3.5-9b", "qwen3.5:9b": "rjudge_deepseek-flash", "gemma4:12b": "rjudge_deepseek-flash"}


def _pairs():
    P = [json.loads(l) for l in open(D / "repair_pairs.jsonl")]
    P = [p for p in P if p["repair_model"] in MODELS and p["revision"].strip()]
    for p in P:
        p["flag"] = bool(extract(p["original"]).diff(extract(p["revision"])))
    return P


def sheet():
    used = {r["pid"] for r in csv.DictReader(open(A / "rq3_key.csv", encoding="utf-8-sig"))}
    rng = random.Random(20261113)
    items = []
    for m in MODELS:
        pool = [p for p in _pairs() if p["repair_model"] == m and p["pid"] not in used
                and len(p["original"]) < 400 and len(p["revision"]) < 600]
        items += rng.sample([p for p in pool if p["flag"]], 30) + rng.sample([p for p in pool if not p["flag"]], 10)
    rng.shuffle(items)
    head = ["item", "A_original", "B_repair", Q1, Q2, "notes"]
    for r in ("rater1", "rater2"):
        with open(A / f"repair_sheet_{r}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            for i, p in enumerate(items, 1):
                w.writerow([i, p["original"], p["revision"], "", "", ""])
    with open(A / "repair_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "pid", "repair_model", "extractor_flag"])
        for i, p in enumerate(items, 1):
            w.writerow([i, p["pid"], p["repair_model"], int(p["flag"])])
    print(len(items), "items")


def analyse():
    from .analysis import calibrate, load
    from .audit_stats import kappa
    key = {r["item"]: r for r in csv.DictReader(open(A / "repair_key.csv", encoding="utf-8-sig"))}
    r1 = {r["item"]: r for r in csv.DictReader(open(A / "repair_ratings_rater1.csv", encoding="utf-8-sig"))}
    r2 = {r["item"]: r for r in csv.DictReader(open(A / "repair_ratings_rater2.csv", encoding="utf-8-sig"))}
    items = sorted(key, key=int)
    x = [r1[i][Q1].strip().lower() for i in items]
    y = [r2[i][Q1].strip().lower() for i in items]
    k, po = kappa(x, y)
    out = [f"Cohen's kappa Q1 (n={len(items)}): {k:.3f}  agreement {po:.3f}  PABAK {2 * po - 1:.3f}"]
    lab = {}
    for i, a, b in zip(items, x, y):
        res = (r1[i].get("resolved_label") or r2[i].get("resolved_label") or "").strip().lower()
        lab[i] = a if a == b and a in ("same", "different") else (res if res in ("same", "different") else None)
    out.append(f"labels: {Counter(lab.values())}  (None = unresolved, excluded)")
    pop = Counter((p["repair_model"], p["flag"]) for p in _pairs())
    tot = Counter(p["repair_model"] for p in _pairs())
    _, rows = load()
    th, _ = calibrate(rows)
    gates = {r["pid"]: r for r in csv.DictReader(open(RES / "repair_gates_g1g3.csv"))}
    judge = {}
    for f in set(SELF.values()) | set(OTHER.values()):
        judge[f] = {json.loads(l)["pid"]: (json.loads(l)["score"] or 0) for l in open(RES / f"{f}.jsonl")}
    res = {}
    rng = random.Random(7)
    for m in MODELS:
        strata = {fl: [lab[i] == "different" for i in items if key[i]["repair_model"] == m
                       and key[i]["extractor_flag"] == str(int(fl)) and lab[i]] for fl in (True, False)}
        w = {fl: pop[(m, fl)] / tot[m] for fl in (True, False)}
        est = lambda s: sum(w[fl] * (sum(s[fl]) / len(s[fl])) for fl in s)
        boots = sorted(est({fl: rng.choices(v, k=len(v)) for fl, v in strata.items()}) for _ in range(2000))
        tp, fp = sum(strata[True]), len(strata[True]) - sum(strata[True])
        fn = sum(strata[False])
        res[m] = {"changed_weighted": [est(strata), boots[50], boots[1949]],
                  "extractor_precision": [tp, tp + fp], "unflagged_changed": [fn, len(strata[False])]}
        out.append(f"{m:16s} changed (weighted) {est(strata):.1%} [{boots[50]:.1%}, {boots[1949]:.1%}]  "
                   f"extractor precision {tp}/{tp + fp}  changed among unflagged {fn}/{len(strata[False])}")
    acc = defaultdict(Counter)
    for i in items:
        if not lab[i]:
            continue
        k_ = key[i]
        pid, m = k_["pid"], k_["repair_model"]
        g = gates.get(pid)
        dec = {"G1": g and float(g["g1_bertscore_f1"]) >= th["G1_BERTScore"],
               "G2": g and float(g["g2_cosine"]) >= th["G2_SBERT"],
               "G3": g and float(g["g3_min"]) >= th["G3_NLI"],
               "self judge": judge[SELF[m]].get(pid, 0) >= 4, "other judge": judge[OTHER[m]].get(pid, 0) >= 4}
        for gname, d in dec.items():
            c = acc[gname]
            if lab[i] == "different":
                c["changed"] += 1
                c["silent"] += bool(d)
            else:
                c["preserved"] += 1
                c["false_esc"] += not d
    out.append("\ngate        silent acceptance (changed)   false escalation (preserved)  [unweighted sample]")
    for gname, c in acc.items():
        out.append(f"{gname:12s}{c['silent']:4d}/{c['changed']:<4d}{c['silent'] / max(c['changed'], 1):6.1%}"
                   f"{c['false_esc']:14d}/{c['preserved']:<4d}{c['false_esc'] / max(c['preserved'], 1):6.1%}")
        res[gname] = dict(c)
    text = "\n".join(out)
    (RES / "repair_audit_stats.txt").write_text(text + "\n")
    (RES / "analysis/repair_audit.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    {"sheet": sheet, "analyse": analyse}[sys.argv[1]]()
