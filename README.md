# Shall Not Pass: replication package

Replication package for the anonymous ENASE 2027 submission
*"Shall Not Pass: Mutation Testing of Preservation Gates for LLM-Based Requirements Repair"*.

The package contains everything needed to re-run or re-analyse the study: the data loaders, the drift
and equivalence operators, the protected-commitment extractor, all generated mutants, the prompts, every
raw model output (scores, repairs, token counts, timestamps), the frozen protocol with its change log,
and the analysis that produces every number in the paper.

## Quick start: reproduce the paper's numbers (no GPU, no API key, about 5 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt
export PYTHONPATH=src
python -m pytest -q tests
python -m snp.analysis                 # writes results/analysis/results.json and prints all tables
```

`results/analysis/analysis_main.txt` is the output of that command for the submitted version.

## Layout

| Path | Content |
|---|---|
| `src/snp/data.py` | Loaders for QuRE and SecReq (deduplication, labels, cleaning rules documented in the docstring) |
| `src/snp/extract.py` | Protected-commitment extractor N(C(r)) and diff used by gate G5 |
| `src/snp/operators.py` | Drift operators M1–M10 and equivalence operators E1–E6 |
| `src/snp/heldout.py` | Held-out drift operators H1–H4 (protocol v1.3); imports nothing from the extractor |
| `src/snp/generate.py`, `pairs.py`, `subsample.py`, `audit.py` | Variant generation, gate pairs, Argon subsample, blind audit sheet |
| `src/snp/run_gates.py` | G1 BERTScore, G2 SBERT cosine, G3 bidirectional NLI (GPU) |
| `src/snp/run_llm.py` | LLM repairs and LLM judges (G4) through any OpenAI-compatible endpoint |
| `src/snp/heldout_g5.py` | G5 on the held-out mutants |
| `src/snp/analysis.py` | Calibration, silent-acceptance and false-escalation rates, AUROC, Wilson and cluster-bootstrap intervals, McNemar/Holm, RQ3 |
| `prompts/` | Repair and judge prompts, used verbatim |
| `data/raw/` | QuRE and SecReq as downloaded from Zenodo |
| `data/derived/` | Requirements, 7,796 drift mutants, 4,517 equivalents, 667 held-out mutants, gate pairs, repair pairs |
| `data/audit/` | Blind audits by two raters (one author, one non-author): mutant/equivalent audit (`audit_*`, `snp.audit_stats`), held-out audit (`heldout_*`, `snp.heldout_audit_stats`), type-level RQ3 check (`rq3_*`, `snp.rq3_audit_stats`, `snp.rq3_adjust`), repair-level RQ3 check (`repair_*`, `snp.audit3`; key added after rating; the 16 disagreements were adjudicated by a third expert blind to both raters, `repair_adjudication.csv`; post-hoc analyses in `snp.audit3_extra`) |
| `results/` | Raw outputs of every gate and model run (one JSON line per call) |
| `notes/FREEZE.md` | Frozen protocol, file hashes, every later revision with date and reason, run environment |
| `RUNBOOK.md` | Exact commands used for the GPU and API runs |

## Re-running the experiments

`RUNBOOK.md` lists every command in the order it was run. In short:

1. `python -m snp.generate` rebuilds the mutants (deterministic; RNG seeded per requirement and operator).
2. `python -m snp.run_gates` scores G1–G3 (install `requirements-gpu.txt` and a CUDA build of PyTorch first).
3. `python -m snp.run_llm repair|judge` runs the LLMs. Local models were served with Ollama; API models
   need `DEEPSEEK_API_KEY` or `GEMINI_API_KEY` in the environment. No key is stored in this package.
4. `python -m snp.heldout` and `python -m snp.heldout_g5` build and score the held-out operators
   (`python -m nltk.downloader wordnet` first).

Model versions used (8–10 November 2026): DeepSeek V4.1 Flash 1017 (API id `deepseek-flash`), Gemini 4 Argon
(reasoning `low`), Qwen 3.5 9B (Ollama `56671c2ab938`) and Gemma 4 12B (Ollama `6114515d63c1`) with reasoning
off; `roberta-large` 722cf37b, `sentence-transformers/all-mpnet-base-v2` e8c3b32e,
`microsoft/deberta-large-mnli` 7296194b. API models change over time, so the archived outputs in
`results/` are the reference.

## Protocol and deviations

The data, operators, extractor, mutants, prompts and decision rules were frozen and hashed before any gate
was run (`notes/FREEZE.md`). Later changes are logged there with their date and reason, and every analysis
added after the runs (judge rule score >= 4, robust-equivalent calibration, held-out operators H1–H4) is
labelled as such in the paper and in the log.

## Data licences

* **QuRE**: Femmer, Houdek, Unterbusch, Vogelsang (2025), *Description and Comparative Analysis of QuRE:
  A New Industrial Requirements Quality Dataset*, REW 2025, doi:10.1109/REW66121.2025.00008.
  Dataset: Zenodo record 16049822, doi:10.5281/zenodo.15656471, CC BY 4.0.
* **SecReq**: Knauss, Houmb, Schneider, Islam, Jürjens (2011), *Supporting Requirements Engineers in
  Recognising Security Issues*, REFSQ 2011. Dataset: Zenodo record 4530183, doi:10.5281/zenodo.4530182,
  CC BY 4.0.

Both are redistributed unchanged in `data/raw/`; derived files are released under the same licence.
Model outputs in `results/` are released under CC BY 4.0. Code is released under the MIT licence (`LICENSE`).
