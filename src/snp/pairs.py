"""Build (original, revision) pairs for the gates.

  python -m snp.pairs variants                 -> data/derived/gate_pairs.jsonl
      calibration-split equivalents (to set thresholds) + all evaluation-split variants
  python -m snp.pairs repairs results/repair_*.jsonl  -> data/derived/repair_pairs.jsonl
      one pair per (requirement, repair model); the response is cleaned deterministically
  python -m snp.pairs repairs_L results/L_repair_*.jsonl -> data/derived/L_repair_pairs.jsonl  (arm L; never
      overwrites the main repair_pairs.jsonl)
  python -m snp.pairs dry                     -> data/derived/dry_pairs.jsonl
      10 drift mutants + 10 equivalents from the CALIBRATION split (never used for reported
      results), to sanity-check that a judge separates them before the full runs
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/derived"


def _reqs():
    return {json.loads(l)["rid"]: json.loads(l) for l in open(D / "requirements.jsonl")}


def variants() -> None:
    R = _reqs()
    n = 0
    with open(D / "gate_pairs.jsonl", "w") as f:
        for line in open(D / "variants.jsonl"):
            v = json.loads(line)
            split = R[v["rid"]]["split"]
            if split == "calibration" and v["kind"] != "equiv":
                continue  # calibration mutants are never used
            f.write(json.dumps({"pid": v["vid"], "rid": v["rid"], "op": v["op"], "kind": v["kind"],
                                "split": split, "original": R[v["rid"]]["text"], "revision": v["text"]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} gate pairs")


def clean_repair(text: str) -> str:
    """Strip wrappers some models add; never alter the requirement wording itself."""
    t = text.strip()
    t = re.sub(r"(?is)^<think>.*?</think>", "", t).strip()          # reasoning traces (local models)
    t = re.sub(r"(?i)^(revised|improved)\s+requirement\s*:\s*", "", t).strip()
    t = re.sub(r"(?i)^requirement\s*:\s*", "", t).strip()
    if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'`":
        t = t[1:-1].strip()
    return t


def repairs(paths: list[str], out: str = "repair_pairs.jsonl") -> None:
    R = _reqs()
    n = 0
    with open(D / out, "w") as f:
        for p in paths:
            for line in open(p):
                r = json.loads(line)
                rev = clean_repair(r["response"])
                f.write(json.dumps({"pid": f"{r['rid']}|repair|{r['model']}", "rid": r["rid"],
                                    "op": "REPAIR", "kind": "repair", "repair_model": r["model"],
                                    "split": R[r["rid"]]["split"], "label": R[r["rid"]]["label"],
                                    "original": R[r["rid"]]["text"], "revision": rev},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} repair pairs")


def dry() -> None:
    import random
    R = _reqs()
    cal = [json.loads(l) for l in open(D / "variants.jsonl")]
    cal = [v for v in cal if R[v["rid"]]["split"] == "calibration"]
    rng = random.Random(7)
    drift = [v for v in cal if v["kind"] == "drift" and v["op"] in ("M1", "M2", "M3", "M5", "M7")]
    equiv = [v for v in cal if v["kind"] == "equiv" and v["op"] in ("E1", "E3", "E5", "E6")]
    pick = rng.sample(drift, 10) + rng.sample(equiv, 10)
    with open(D / "dry_pairs.jsonl", "w") as f:
        for v in pick:
            f.write(json.dumps({"pid": "dry|" + v["vid"], "rid": v["rid"], "op": v["op"], "kind": v["kind"],
                                "split": "calibration", "original": R[v["rid"]]["text"], "revision": v["text"]},
                               ensure_ascii=False) + "\n")
    print("20 dry pairs (10 drift, 10 equiv, calibration split)")


if __name__ == "__main__":
    {"variants": lambda: variants(), "repairs": lambda: repairs(sys.argv[2:]),
     "repairs_L": lambda: repairs(sys.argv[2:], "L_repair_pairs.jsonl"), "dry": lambda: dry()}[sys.argv[1]]()
