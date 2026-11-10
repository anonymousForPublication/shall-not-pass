"""Stratified subsample of gate pairs for the strong-judge check (Gemini 4 Argon).

Added in protocol v1.1 (2026-11-08), before any gate was run.
  * up to 70 evaluation-split drift mutants per operator (M1..M10),
  * 150 evaluation-split equivalents, 200 calibration-split equivalents
    (so Argon gets its own threshold, set exactly like the other judges').
Run:  PYTHONPATH=src python -m snp.subsample   -> data/derived/argon_subsample.jsonl
"""
import json
import random
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data/derived"
PER_DRIFT_OP, EVAL_EQUIV, CAL_EQUIV = 70, 150, 200


def main() -> None:
    pairs = [json.loads(l) for l in open(D / "gate_pairs.jsonl")]
    rng = random.Random(20261108)
    drift = defaultdict(list)
    eval_eq, cal_eq = [], []
    for p in pairs:
        if p["kind"] == "drift":
            drift[p["op"]].append(p)
        elif p["split"] == "evaluation":
            eval_eq.append(p)
        else:
            cal_eq.append(p)
    out = []
    for op in sorted(drift, key=lambda o: int(o[1:])):
        out += rng.sample(drift[op], min(PER_DRIFT_OP, len(drift[op])))
    out += rng.sample(eval_eq, EVAL_EQUIV) + rng.sample(cal_eq, CAL_EQUIV)
    with open(D / "argon_subsample.jsonl", "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for p in out:
        by[p["op"] if p["kind"] == "drift" else f"equiv-{p['split']}"] += 1
    print(len(out), dict(by))


if __name__ == "__main__":
    main()
