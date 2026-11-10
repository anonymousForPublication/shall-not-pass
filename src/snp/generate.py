"""Build the frozen evaluation set: requirements with clusters/splits, mutants, equivalents.

Run:  PYTHONPATH=src python -m snp.generate
Writes data/derived/{requirements,variants}.jsonl and data/derived/summary.json
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import load_all
from .extract import extract
from .operators import DRIFT, EQUIV, apply

OUT = Path(__file__).resolve().parents[2] / "data" / "derived"
JACCARD = 0.8  # near-duplicate threshold (paper Sec. 4.4)


def _tokens(t: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", t.lower()))


def clusters(reqs) -> dict[str, str]:
    """Union-find over token-Jaccard >= JACCARD, within each corpus."""
    parent = {r.rid: r.rid for r in reqs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_corpus = defaultdict(list)
    for r in reqs:
        by_corpus[r.corpus].append((r.rid, _tokens(r.text)))
    for items in by_corpus.values():
        for i in range(len(items)):
            a_id, a = items[i]
            for j in range(i + 1, len(items)):
                b_id, b = items[j]
                if a and b and len(a & b) / len(a | b) >= JACCARD:
                    parent[find(a_id)] = find(b_id)
    return {rid: find(rid) for rid in parent}


def split_of(cluster_id: str) -> str:
    return "calibration" if int(hashlib.sha256(cluster_id.encode()).hexdigest(), 16) % 2 == 0 else "evaluation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reqs = load_all()
    cl = clusters(reqs)
    rows, variants = [], []
    for r in reqs:
        rows.append({"rid": r.rid, "corpus": r.corpus, "source": r.source, "label": r.label,
                     "text": r.text, "weak_words": list(r.weak_words), "defect_words": list(r.defect_words),
                     "cluster": cl[r.rid], "split": split_of(cl[r.rid])})
        src_c = extract(r.text)
        for kind, ops in (("drift", DRIFT), ("equiv", EQUIV)):
            for op in ops:
                m = apply(op, r.rid, r.text)
                if m is None:
                    continue
                diff = src_c.diff(extract(m.text))
                variants.append({"vid": f"{r.rid}|{op}", "rid": r.rid, "op": op, "kind": kind,
                                 "subtype": m.subtype, "text": m.text,
                                 "g5_preserved": not diff, "g5_diff": diff})
    with open(OUT / "requirements.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(OUT / "variants.jsonl", "w") as f:
        for x in variants:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    rid_info = {x["rid"]: x for x in rows}
    summary = {"requirements": Counter(f"{x['corpus']}:{x['label']}" for x in rows),
               "clusters": {c: len({x['cluster'] for x in rows if x['corpus'] == c}) for c in ("QuRE", "SecReq")},
               "split": Counter(f"{x['corpus']}:{x['split']}" for x in rows),
               "variants": {}, "g5_detects_drift": {}, "g5_preserves_equiv": {}}
    for op in list(DRIFT) + list(EQUIV):
        vs = [v for v in variants if v["op"] == op]
        summary["variants"][op] = Counter(rid_info[v["rid"]]["corpus"] for v in vs)
        if vs:
            key = "g5_detects_drift" if op in DRIFT else "g5_preserves_equiv"
            ok = sum((not v["g5_preserved"]) if op in DRIFT else v["g5_preserved"] for v in vs)
            summary[key][op] = f"{ok}/{len(vs)}"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
