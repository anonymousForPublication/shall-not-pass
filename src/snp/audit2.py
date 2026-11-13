"""Protocol v1.4 audit sheets (blind, two raters).

  PYTHONPATH=src python -m snp.audit2
Writes to data/audit/:
  rq3_sheet_rater1.csv, rq3_sheet_rater2.csv, rq3_key.csv
      150 (repair, commitment type) items: per repair model (3) and commitment type (5), 7 repairs the
      extractor flags as changed for that type and 3 it does not. Raters answer one question per item and
      see neither the model nor the extractor's verdict. Gives extractor precision per type.
  heldout_sheet_rater1.csv, heldout_sheet_rater2.csv, heldout_key.csv
      40 held-out mutants (10 per H1-H4) mixed with 10 equivalent variants not used in the first audit,
      same two questions as the first audit.
"""
import csv
import json
import random
from pathlib import Path

from .extract import extract

ROOT = Path(__file__).resolve().parents[2]
D, A = ROOT / "data/derived", ROOT / "data/audit"
QUESTIONS = {
    "modality": "Does B change the BINDING FORCE of A? (adds, removes or changes shall/must/should/may/will, or turns a description into an obligation or the reverse)",
    "values": "Does B add, remove or change a QUANTITATIVE VALUE or UNIT compared with A?",
    "negation": "Does B change the POLARITY of what A states? (something required/true becomes prohibited/false, or the reverse)",
    "quant": "Does B change the QUANTIFICATION of A? (all/each/every/only/some/any/at least one ...: who or what is covered)",
    "cond": "Does B add, remove or change a CONDITION under which A applies? (if/when/after/during ...; a reworded condition with the same meaning counts as no)",
}


def _write(name, header, rows):
    with open(A / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def rq3(rng):
    pairs = [json.loads(l) for l in open(D / "repair_pairs.jsonl")]
    pairs = [p for p in pairs if p["revision"].strip()]
    for p in pairs:
        p["diff"] = set(extract(p["original"]).diff(extract(p["revision"])))
    items, used = [], set()
    for model in sorted({p["repair_model"] for p in pairs}):
        mine = [p for p in pairs if p["repair_model"] == model]
        for t in QUESTIONS:
            for flag, k in ((True, 7), (False, 3)):
                pool = [p for p in mine if (t in p["diff"]) == flag and p["pid"] not in used
                        and len(p["original"]) < 400 and len(p["revision"]) < 600]
                for p in rng.sample(pool, k):
                    used.add(p["pid"])
                    items.append((p, t, flag))
    rng.shuffle(items)
    head = ["item", "A_original", "B_repair", "question", "answer (yes/no/unsure)", "notes"]
    rows = [[i, p["original"], p["revision"], QUESTIONS[t], "", ""] for i, (p, t, _) in enumerate(items, 1)]
    _write("rq3_sheet_rater1.csv", head, rows)
    _write("rq3_sheet_rater2.csv", head, rows)
    _write("rq3_key.csv", ["item", "pid", "repair_model", "type", "extractor_flag"],
           [[i, p["pid"], p["repair_model"], t, int(f)] for i, (p, t, f) in enumerate(items, 1)])
    return len(items)


def heldout(rng):
    H = [json.loads(l) for l in open(D / "heldout_pairs.jsonl")]
    first = {r["vid"] for r in csv.DictReader(open(A / "audit_key.csv", encoding="utf-8-sig"))}
    E = [p for p in map(json.loads, open(D / "gate_pairs.jsonl"))
         if p["kind"] == "equiv" and p["split"] == "evaluation" and p["pid"] not in first]
    items = []
    for op in ("H1", "H2", "H3", "H4"):
        items += [(p, "different") for p in rng.sample([h for h in H if h["op"] == op], 10)]
    items += [(p, "same") for p in rng.sample(E, 10)]
    rng.shuffle(items)
    head = ["item", "A_original", "B_variant", "same_obligations (same/different/unsure)", "B_is_garbled (y/n)", "notes"]
    rows = [[i, p["original"], p["revision"], "", "", ""] for i, (p, _) in enumerate(items, 1)]
    _write("heldout_sheet_rater1.csv", head, rows)
    _write("heldout_sheet_rater2.csv", head, rows)
    _write("heldout_key.csv", ["item", "pid", "op", "kind", "expected"],
           [[i, p["pid"], p["op"], p["kind"], e] for i, (p, e) in enumerate(items, 1)])
    return len(items)


if __name__ == "__main__":
    rng = random.Random(20261112)
    print("rq3 items:", rq3(rng), " held-out items:", heldout(rng))
