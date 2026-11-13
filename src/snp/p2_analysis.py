"""Protocol v1.4: compare repair prompts P1 and P2 (P2 allows "return it unchanged").

  PYTHONPATH=src python -m snp.p2_analysis   -> results/analysis/p2.json, printed table
Same deterministic measures as RQ3; "unchanged" = identical token sequence (analysis.tokens).
"""
import json
from collections import Counter
from pathlib import Path

from .analysis import tokens, wilson
from .extract import extract, find_values
from .pairs import clean_repair

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"DeepSeek": "deepseek-flash", "Qwen": "qwen3.5-9b", "Gemma": "gemma4-12b"}


def measure(path, R):
    st = {"defect": Counter(), "ok": Counter()}
    for line in open(path):
        r = json.loads(line)
        q = R[r["rid"]]
        c = st[q["label"]]
        c["n"] += 1
        rev = clean_repair(r["response"] or "")
        if not rev:
            c["empty"] += 1
            continue
        if tokens(q["text"]) == tokens(rev):
            c["unchanged"] += 1
        d = set(extract(q["text"]).diff(extract(rev)))
        c["pcc"] += bool(d)
        c["modality_only"] += d == {"modality"}
        c["beyond_modality"] += bool(d - {"modality"})
        c["beyond_mod_cond"] += bool(d - {"modality", "cond"})
        src = {(v.num, v.unit) for v in find_values(q["text"])}
        c["fabricated"] += any((v.num, v.unit) not in src for v in find_values(rev))
    return st


def main():
    R = {json.loads(l)["rid"]: json.loads(l) for l in open(ROOT / "data/derived/requirements.jsonl")}
    out = {}
    keys = ["unchanged", "pcc", "beyond_modality", "beyond_mod_cond", "fabricated"]
    for name, f in MODELS.items():
        for tag, pre in (("P1", "repair_"), ("P2", "p2_repair_")):
            st = measure(ROOT / "results" / f"{pre}{f}.jsonl", R)
            out[f"{name}|{tag}"] = {g: dict(c) for g, c in st.items()}
            for g in ("ok", "defect"):
                c = st[g]
                print(f"{name:9s}{tag} {g:6s} n={c['n']:5d} empty={c['empty']:3d} " +
                      " ".join(f"{k}={c[k] / c['n']:.1%}" for k in keys))
    (ROOT / "results/analysis/p2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
