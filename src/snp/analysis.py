"""Analysis of RQ1-RQ3 from the frozen result files.

Rules (protocol v1.2.1/v1.2.2, fixed before the full runs):
  * a missing judge score counts as an escalation: it is scored 0 (below every threshold);
  * thresholds: theta_g = lower 5th percentile of g's scores on CALIBRATION-split equivalents;
    a pair is accepted iff s_g >= theta_g (so >= 95% of calibration equivalents are accepted);
  * silent-acceptance rate (SAR) and false-escalation rate (FER) are measured on the EVALUATION split;
  * an empty repair is a failed repair: kept in RQ3 denominators, excluded from gate scoring.
Run:  PYTHONPATH=src python -m snp.analysis   -> results/analysis/*.json and printed tables
"""
from __future__ import annotations

import csv
import difflib
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

from .extract import extract, find_values

ROOT = Path(__file__).resolve().parents[2]
D, RES = ROOT / "data/derived", ROOT / "results"
OUT = RES / "analysis"
DRIFT_OPS = [f"M{i}" for i in range(1, 11)]
EQ_OPS = [f"E{i}" for i in range(1, 7)]


# ----------------------------------------------------------------- helpers
def jl(p):
    return [json.loads(l) for l in open(p)]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Wilson score interval (Wilson 1927)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def auroc(pos_scores, neg_scores) -> float:
    """P(score_equiv > score_mutant), ties counted 1/2 (Mann-Whitney U / (n1 n2))."""
    if not len(pos_scores) or not len(neg_scores):
        return float("nan")
    u = mannwhitneyu(pos_scores, neg_scores, alternative="two-sided").statistic
    return u / (len(pos_scores) * len(neg_scores))


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired accept decisions (McNemar 1947)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    p = binomtest(n01, n, 0.5).pvalue if n else 1.0
    return n01, n10, p


def holm(pvals: dict) -> dict:
    """Holm (1979) step-down adjustment."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def theta(scores, q: float = 0.05) -> float:
    return float(np.quantile(np.asarray(scores, float), q, method="lower"))


# ----------------------------------------------------------------- load
def load():
    req = {r["rid"]: r for r in jl(D / "requirements.jsonl")}
    var = {v["vid"]: v for v in jl(D / "variants.jsonl")}
    pairs = jl(D / "gate_pairs.jsonl")
    g13 = {r["pid"]: r for r in csv.DictReader(open(RES / "gates_g1g3.csv"))}
    judge = {}
    for name, f in (("G4_DeepSeek", "judge_deepseek-flash.jsonl"), ("G4_Qwen", "judge_qwen3.5-9b.jsonl"),
                    ("G4_Argon", "judge_argon_subsample.jsonl")):
        judge[name] = {r["pid"]: (r["score"] if r["score"] is not None else 0) for r in jl(RES / f)}
    rows = []
    for p in pairs:
        r = req[p["rid"]]
        s = {"G1_BERTScore": float(g13[p["pid"]]["g1_bertscore_f1"]),
             "G2_SBERT": float(g13[p["pid"]]["g2_cosine"]),
             "G3_NLI": float(g13[p["pid"]]["g3_min"]),
             "G5_diff": 1.0 if var[p["pid"]]["g5_preserved"] else 0.0}
        for name, d in judge.items():
            if p["pid"] in d:
                s[name] = float(d[p["pid"]])
        rows.append({**p, "corpus": r["corpus"], "label": r["label"], "cluster": r["cluster"], "s": s})
    return req, rows


THRESH_GATES = ["G1_BERTScore", "G2_SBERT", "G3_NLI", "G4_DeepSeek", "G4_Qwen"]


def calibrate(rows, q=0.05, exclude_ops=()):
    cal = [r for r in rows if r["split"] == "calibration" and r["kind"] == "equiv" and r["op"] not in exclude_ops]
    th = {g: theta([r["s"][g] for r in cal], q) for g in THRESH_GATES}
    return th, len(cal)


def decide(r, th):
    s = r["s"]
    d = {g: s[g] >= th[g] for g in THRESH_GATES}
    d["G5_diff"] = s["G5_diff"] == 1.0
    d["G6_hybrid"] = d["G5_diff"] and d["G3_NLI"]
    return d


GATES = THRESH_GATES + ["G5_diff", "G6_hybrid"]


# ----------------------------------------------------------------- RQ1 / RQ2
def rq12(rows, q=0.05, exclude_ops=(), tag="main", verbose=True):
    th, ncal = calibrate(rows, q, exclude_ops)
    ev = [r for r in rows if r["split"] == "evaluation" and r["op"] not in exclude_ops]
    for r in ev:
        r["d"] = decide(r, th)
    drift = [r for r in ev if r["kind"] == "drift"]
    equiv = [r for r in ev if r["kind"] == "equiv"]
    out = {"tag": tag, "q": q, "thresholds": th, "n_cal_equiv": ncal, "n_eval_drift": len(drift),
           "n_eval_equiv": len(equiv), "sar": {}, "sar_pooled": {}, "fer": {}, "auroc": {}}
    for g in GATES:
        for op in DRIFT_OPS:
            xs = [r["d"][g] for r in drift if r["op"] == op]
            out["sar"].setdefault(g, {})[op] = (sum(xs), len(xs), *wilson(sum(xs), len(xs)))
        xs = [r["d"][g] for r in drift]
        out["sar_pooled"][g] = (sum(xs), len(xs), *wilson(sum(xs), len(xs)))
        ys = [not r["d"][g] for r in equiv]
        out["fer"][g] = (sum(ys), len(ys), *wilson(sum(ys), len(ys)))
        if g != "G6_hybrid":
            out["auroc"][g] = auroc([r["s"][g] for r in equiv], [r["s"][g] for r in drift])
    # per corpus and SecReq security label
    out["sar_by_group"] = {}
    for gname, pred in (("QuRE", lambda r: r["corpus"] == "QuRE"), ("SecReq", lambda r: r["corpus"] == "SecReq"),
                        ("SecReq-sec", lambda r: r["corpus"] == "SecReq" and r["label"] == "sec"),
                        ("SecReq-nonsec", lambda r: r["corpus"] == "SecReq" and r["label"] == "nonsec")):
        sub = [r for r in drift if pred(r)]
        out["sar_by_group"][gname] = {g: (sum(r["d"][g] for r in sub), len(sub)) for g in GATES}
    # paired comparisons on the same mutants
    pv, cnt = {}, {}
    for a, b in combinations(GATES, 2):
        n01, n10, p = mcnemar([r["d"][a] for r in drift], [r["d"][b] for r in drift])
        pv[(a, b)], cnt[(a, b)] = p, (n01, n10)
    adj = holm(pv)
    out["mcnemar"] = {f"{a}|{b}": {"a_only": cnt[(a, b)][0], "b_only": cnt[(a, b)][1], "p_holm": adj[(a, b)]}
                      for a, b in pv}
    # cluster bootstrap for pooled SAR (requirement clusters as units)
    rng = random.Random(1)
    by_cl = defaultdict(list)
    for r in drift:
        by_cl[r["cluster"]].append(r)
    cls = list(by_cl)
    boot = {g: [] for g in GATES}
    for _ in range(1000):
        samp = [r for c in (rng.choice(cls) for _ in cls) for r in by_cl[c]]
        for g in GATES:
            boot[g].append(sum(r["d"][g] for r in samp) / len(samp))
    out["sar_pooled_cluster_ci"] = {g: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                                    for g, v in boot.items()}
    if verbose:
        print(f"\n=== RQ1/RQ2 [{tag}] q={q} | calibration equivalents={ncal} | eval mutants={len(drift)}, "
              f"eval equivalents={len(equiv)}")
        print("thresholds:", {g: round(v, 4) for g, v in th.items()})
        hdr = "op    n   " + " ".join(f"{g[:12]:>12}" for g in GATES)
        print(hdr)
        for op in DRIFT_OPS:
            n = out["sar"][GATES[0]][op][1]
            print(f"{op:<4} {n:4d}  " + " ".join(f"{out['sar'][g][op][2]*100:11.1f}%" for g in GATES))
        print("ALL  " + f"{len(drift):4d}  " + " ".join(f"{out['sar_pooled'][g][2]*100:11.1f}%" for g in GATES))
        print("FER  " + f"{len(equiv):4d}  " + " ".join(f"{out['fer'][g][2]*100:11.1f}%" for g in GATES))
        print("AUROC      " + " ".join(f"{out['auroc'].get(g, float('nan')):12.3f}" for g in GATES))
    return out


def argon_check(rows):
    """Strong-judge check on the 1,048-pair subsample (own threshold from its 200 calibration equivalents)."""
    sub = [r for r in rows if "G4_Argon" in r["s"]]
    cal = [r for r in sub if r["split"] == "calibration" and r["kind"] == "equiv"]
    th_main, _ = calibrate(rows)
    th = {**th_main, "G4_Argon": theta([r["s"]["G4_Argon"] for r in cal])}
    ev = [r for r in sub if r["split"] == "evaluation"]
    drift = [r for r in ev if r["kind"] == "drift"]
    equiv = [r for r in ev if r["kind"] == "equiv"]
    gates = ["G1_BERTScore", "G3_NLI", "G4_DeepSeek", "G4_Qwen", "G4_Argon"]
    acc = lambda r, g: r["s"][g] >= th[g]
    res = {"theta_argon": th["G4_Argon"], "n_drift": len(drift), "n_equiv": len(equiv), "sar": {}, "fer": {}}
    for g in gates:
        res["sar"][g] = {op: (sum(acc(r, g) for r in drift if r["op"] == op), sum(1 for r in drift if r["op"] == op))
                         for op in DRIFT_OPS}
        k = sum(acc(r, g) for r in drift)
        res["sar"][g]["ALL"] = (k, len(drift))
        res["fer"][g] = (sum(not acc(r, g) for r in equiv), len(equiv))
    for other in ("G4_DeepSeek", "G4_Qwen"):
        n01, n10, p = mcnemar([acc(r, "G4_Argon") for r in drift], [acc(r, other) for r in drift])
        res[f"mcnemar_Argon_vs_{other}"] = {"argon_only": n01, "other_only": n10, "p": p}
    print(f"\n=== Strong-judge check (Gemini 4 Argon), subsample: {len(drift)} eval mutants, "
          f"{len(equiv)} eval equivalents, theta_Argon={th['G4_Argon']}")
    print("op    " + " ".join(f"{g[:12]:>12}" for g in gates))
    for op in DRIFT_OPS + ["ALL"]:
        print(f"{op:<5} " + " ".join(f"{res['sar'][g][op][0]:5d}/{res['sar'][g][op][1]:<5d}" + " " for g in gates))
    print("FER   " + " ".join(f"{res['fer'][g][0]:5d}/{res['fer'][g][1]:<5d} " for g in gates))
    print({k: v for k, v in res.items() if k.startswith("mcnemar")})
    return res


# ----------------------------------------------------------------- RQ3
_DIGITLESS = re.compile(r"[^a-z0-9]+")


def tokens(t):
    return [w for w in _DIGITLESS.split(t.lower()) if w]


def rq3(req, rows_rq12, fixed4=False):
    """fixed4=False: pre-registered judge thresholds (calibrated; Gemma >= 4).
    fixed4=True (post hoc): every judge accepts iff score >= 4, because the calibrated thresholds of the
    near-binary judges are degenerate (Qwen: theta = 1 accepts everything)."""
    th, _ = calibrate(rows_rq12)
    pairs = jl(D / "repair_pairs.jsonl")
    g13 = {r["pid"]: r for r in csv.DictReader(open(RES / "repair_gates_g1g3.csv"))}
    rj = {}
    for name, f in (("DeepSeek", "rjudge_deepseek-flash.jsonl"), ("Qwen", "rjudge_qwen3.5-9b.jsonl"),
                    ("Gemma", "rjudge_gemma4-12b.jsonl")):
        rj[name] = {r["pid"]: (r["score"] if r["score"] is not None else 0) for r in jl(RES / f)}
    model_name = {"deepseek-flash": "DeepSeek", "qwen3.5:9b": "Qwen", "gemma4:12b": "Gemma"}
    indep = {"DeepSeek": "Qwen", "Qwen": "DeepSeek", "Gemma": "DeepSeek"}
    # judge thresholds: calibrated for DeepSeek/Qwen (RQ1); Gemma never judged equivalents -> fixed rule >= 4
    jth = {"DeepSeek": th["G4_DeepSeek"], "Qwen": th["G4_Qwen"], "Gemma": 4.0}
    if fixed4:
        jth = {"DeepSeek": 4.0, "Qwen": 4.0, "Gemma": 4.0}
    missing = {}
    for name, f in (("DeepSeek", "rjudge_deepseek-flash.jsonl"), ("Qwen", "rjudge_qwen3.5-9b.jsonl"),
                    ("Gemma", "rjudge_gemma4-12b.jsonl")):
        missing[name] = {r["pid"] for r in jl(RES / f) if r["score"] is None}
    out = {"judge_thresholds": jth, "models": {}}
    print(f"\n=== RQ3: real repairs | gate thresholds from RQ1 calibration; judge thresholds {jth}")
    for mkey, m in model_name.items():
        P = [p for p in pairs if p["repair_model"] == mkey]
        st = Counter()
        types = Counter()
        flagged_rows = []
        edit_ok = []
        for p in P:
            r = req[p["rid"]]
            st["n"] += 1
            st[f"n_{r['label']}"] += 1
            rev = p["revision"].strip()
            if not rev:
                st["empty"] += 1
                continue
            # weak-word removal (defects): every defect weak word gone (prefix match: 'adequate' ~ 'adequately')
            if r["label"] == "defect":
                gone = all(not re.search(r"(?i)\b" + re.escape(w), rev) for w in r["defect_words"])
                st["ww_removed"] += gone
            diff = extract(p["original"]).diff(extract(rev))
            src_vals = {(v.num, v.unit) for v in find_values(p["original"])}
            fab = any((v.num, v.unit) not in src_vals for v in find_values(rev))
            for t in diff:
                types[t] += 1
            if "modality" in diff:
                mo, mr = extract(p["original"]).modality, extract(rev).modality
                for cls in ("binding", "nonbinding", "declarative"):
                    if mr[cls] and not mo[cls]:
                        types[f"mod_{cls}_added"] += 1
                    if mo[cls] and not mr[cls]:
                        types[f"mod_{cls}_removed"] += 1
            pcc = bool(diff)
            st["pcc"] += pcc
            st["fab"] += fab
            st[f"pcc_{r['label']}"] += pcc
            st[f"fab_{r['label']}"] += fab
            if r["label"] == "ok":
                a, b = tokens(p["original"]), tokens(rev)
                edit_ok.append(1 - difflib.SequenceMatcher(None, a, b).ratio())
                st["ok_unchanged"] += (a == b)
            flag = pcc or fab
            st["flagged"] += flag
            g = g13.get(p["pid"])
            dec = {}
            if g:
                dec["G1_BERTScore"] = float(g["g1_bertscore_f1"]) >= th["G1_BERTScore"]
                dec["G2_SBERT"] = float(g["g2_cosine"]) >= th["G2_SBERT"]
                dec["G3_NLI"] = float(g["g3_min"]) >= th["G3_NLI"]
            dec["self_judge"] = rj[m][p["pid"]] >= jth[m]
            dec["indep_judge"] = rj[indep[m]][p["pid"]] >= jth[indep[m]]
            dec["G5_diff"] = not pcc
            dec["G6_hybrid"] = dec["G5_diff"] and dec.get("G3_NLI", False)
            dec["self_missing"] = p["pid"] in missing[m]
            dec["indep_missing"] = p["pid"] in missing[indep[m]]
            if flag:
                flagged_rows.append(dec)
            else:
                st["unflag_n"] += 1
                for k, v in dec.items():
                    st[f"unflag_acc_{k}"] += v
        acc = {k: (sum(d[k] for d in flagged_rows if k in d), sum(1 for d in flagged_rows if k in d))
               for k in ("G1_BERTScore", "G2_SBERT", "G3_NLI", "self_judge", "indep_judge")}
        n01, n10, p = mcnemar([d["self_judge"] for d in flagged_rows], [d["indep_judge"] for d in flagged_rows])
        res = {"n": st["n"], "n_defect": st["n_defect"], "n_ok": st["n_ok"], "empty": st["empty"],
               "ww_removed": st["ww_removed"], "pcc": st["pcc"], "fab": st["fab"],
               "pcc_defect": st["pcc_defect"], "pcc_ok": st["pcc_ok"], "fab_defect": st["fab_defect"],
               "fab_ok": st["fab_ok"], "flagged": st["flagged"], "types": dict(types),
               "ok_mean_token_edit": float(np.mean(edit_ok)) if edit_ok else None,
               "ok_unchanged": st["ok_unchanged"], "silent_acceptance_of_flagged": acc,
               "unflagged_n": st["unflag_n"],
               "judge_missing_on_flagged": {"self": sum(d["self_missing"] for d in flagged_rows),
                                            "indep": sum(d["indep_missing"] for d in flagged_rows)},
               "unflagged_acceptance": {k: st[f"unflag_acc_{k}"] for k in
                                        ("G1_BERTScore", "G2_SBERT", "G3_NLI", "self_judge", "indep_judge")},
               "mcnemar_self_vs_indep": {"self_only": n01, "indep_only": n10, "p": p},
               "indep_judge": indep[m]}
        out["models"][m] = res
        nn = res["n"] - res["empty"]
        print(f"\n-- {m} repairs: n={res['n']} (defect {res['n_defect']}, ok {res['n_ok']}), empty={res['empty']}")
        print(f"   weak words removed (defects): {res['ww_removed']}/{res['n_defect']} "
              f"({res['ww_removed']/max(1,res['n_defect'])*100:.1f}%)")
        print(f"   protected-commitment change: {res['pcc']}/{nn} ({res['pcc']/nn*100:.1f}%)  "
              f"[defect {res['pcc_defect']}/{res['n_defect']}, ok {res['pcc_ok']}/{res['n_ok']}]  types={res['types']}")
        print(f"   fabricated specificity: {res['fab']}/{nn} ({res['fab']/nn*100:.1f}%)  "
              f"[defect {res['fab_defect']}, ok {res['fab_ok']}]")
        print(f"   ok controls: unchanged {res['ok_unchanged']}/{res['n_ok']}, mean token edit distance "
              f"{res['ok_mean_token_edit']:.3f}")
        print(f"   flagged (commitment change or fabricated value): {res['flagged']}/{nn}")
        print("   accepted among flagged: " + ", ".join(f"{k} {a}/{n} ({a/max(1,n)*100:.1f}%)" for k, (a, n) in acc.items()))
        un = res["unflagged_n"]
        print("   accepted among unflagged: " + ", ".join(f"{k} {v}/{un} ({v/max(1,un)*100:.1f}%)"
                                                     for k, v in res["unflagged_acceptance"].items()))
        print(f"   self vs independent judge on flagged (McNemar): {res['mcnemar_self_vs_indep']}")
    return out


def heldout(rows):
    """Protocol v1.3: SAR on held-out drift H1-H4 at the frozen main thresholds (q = 0.05) and,
    for judges, at the fixed rule score >= 4. Gates whose result files are missing are skipped."""
    th, _ = calibrate(rows, 0.05)
    pairs = jl(D / "heldout_pairs.jsonl")
    g5 = {r["pid"]: r["g5_accept"] == "1" for r in csv.DictReader(open(RES / "heldout_g5.csv"))}
    g13p = RES / "heldout_gates_g1g3.csv"
    g13 = {r["pid"]: r for r in csv.DictReader(open(g13p))} if g13p.exists() else {}
    judges = {}
    for name, f in (("G4_DeepSeek", "heldout_judge_deepseek-flash.jsonl"), ("G4_Qwen", "heldout_judge_qwen3.5-9b.jsonl"),
                    ("G4_Argon", "heldout_judge_argon.jsonl")):
        if (RES / f).exists():
            judges[name] = {r["pid"]: (r["score"] if r["score"] is not None else 0) for r in jl(RES / f)}
    acc = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        pid, op = p["pid"], p["op"]
        d = {"G5_diff": g5[pid]}
        if pid in g13:
            for g, col in (("G1_BERTScore", "g1_bertscore_f1"), ("G2_SBERT", "g2_cosine"), ("G3_NLI", "g3_min")):
                d[g] = float(g13[pid][col]) >= th[g]
            d["G6_hybrid"] = d["G5_diff"] and d["G3_NLI"]
        for name, sc in judges.items():
            if pid in sc:
                d[name + " (>=4)"] = float(sc[pid]) >= 4
                if name in th:
                    d[name + " (theta)"] = float(sc[pid]) >= th[name]
        for g, v in d.items():
            acc[g][op].append(v)
            acc[g]["ALL"].append(v)
    out = {g: {op: (sum(v), len(v), *wilson(sum(v), len(v))) for op, v in ops.items()} for g, ops in acc.items()}
    print("\n=== Held-out drift (v1.3): silent acceptance at frozen thresholds")
    cols = ["H1", "H2", "H3", "H4", "ALL"]
    print(f"{'gate':22s}" + "".join(f"{c:>14s}" for c in cols))
    for g, ops in out.items():
        print(f"{g:22s}" + "".join(f"{ops[c][0]:>5d}/{ops[c][1]:<4d}{ops[c][0] / ops[c][1]:>4.0%}" if c in ops else f"{'-':>14s}"
                                   for c in cols))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    req, rows = load()
    res_h = heldout(rows)
    res = {"main": rq12(rows)}
    res["q90"] = rq12(rows, q=0.10, tag="90% equivalents accepted", verbose=True)
    res["q99"] = rq12(rows, q=0.01, tag="99% equivalents accepted", verbose=True)
    res["noE5"] = rq12(rows, exclude_ops=("E5",), tag="without E5", verbose=True)
    res["argon"] = argon_check(rows)
    res["rq3"] = rq3(req, rows)
    res["rq3_fixed4"] = rq3(req, rows, fixed4=True)
    res["heldout"] = res_h
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwritten {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
