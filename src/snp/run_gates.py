"""Score (original, revision) pairs with gates G1-G3 on a local GPU.

  G1 BERTScore F1, roberta-large, baseline rescaling      (bert-score package)
  G2 cosine similarity, sentence-transformers/all-mpnet-base-v2
  G3 min(P(entail a->b), P(entail b->a)), microsoft/deberta-large-mnli

Usage (on the GPU machine):
  python -m snp.run_gates --pairs data/derived/gate_pairs.jsonl --out results/gates_g1g3.csv
Resumable: rows already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NLI_MODEL = "microsoft/deberta-large-mnli"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pairs (dry run)")
    a = ap.parse_args()

    import torch
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = [json.loads(l) for l in open(a.pairs)][: a.limit]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {r["pid"] for r in csv.DictReader(open(out))}
    todo = [p for p in pairs if p["pid"] not in done]
    print(f"{len(todo)} pairs to score ({len(done)} already done) on {dev}")

    bs = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, device=dev)
    sb = SentenceTransformer(SBERT_MODEL, device=dev)
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev).eval()
    ent = [i for i, l in nli.config.id2label.items() if l.lower().startswith("entail")][0]

    def p_entail(prem, hyp):
        with torch.no_grad():
            enc = tok(prem, hyp, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
            return torch.softmax(nli(**enc).logits, -1)[:, ent].tolist()

    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["pid", "g1_bertscore_f1", "g2_cosine", "g3_fwd", "g3_bwd", "g3_min"])
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            a_txt = [p["original"] for p in chunk]
            b_txt = [p["revision"] for p in chunk]
            _, _, f1 = bs.score(b_txt, a_txt)
            ea = sb.encode(a_txt, convert_to_tensor=True, normalize_embeddings=True)
            eb = sb.encode(b_txt, convert_to_tensor=True, normalize_embeddings=True)
            cos = (ea * eb).sum(-1).tolist()
            fwd, bwd = p_entail(a_txt, b_txt), p_entail(b_txt, a_txt)
            for p, g1, g2, x, y in zip(chunk, f1.tolist(), cos, fwd, bwd):
                w.writerow([p["pid"], f"{g1:.6f}", f"{g2:.6f}", f"{x:.6f}", f"{y:.6f}", f"{min(x, y):.6f}"])
            f.flush()
            print(f"{min(i + BATCH, len(todo))}/{len(todo)}", end="\r")
    print("\ndone")


if __name__ == "__main__":
    main()
