# Protocol freeze

Frozen before any gate was run. Any later change to these files must be reported in
the paper as a protocol deviation.

## Change log

- v1 (2026-11-01, tag `protocol-freeze-v1`): data, operators, extractor, mutants, prompts, runners.
- v1.1 (2026-11-08, tag `protocol-freeze-v1.1`), still before any gate run: model configuration set to
  DeepSeek V4.1 Flash 1017, Qwen 3.8 27B and Gemma 4 26B, plus Gemini 4 Argon on a 1,048-pair subsample;
  added `subsample.py` and `argon_subsample.jsonl`; `run_llm.py` gained `--api-key-env` and `--repair-model`.
  No change to the data, operators, extractor, mutants or prompts (their hashes are identical to v1).
- v1.2 (2026-11-08, tag `protocol-freeze-v1.2`), still before any gate run:
  * Main local arm: Qwen 3.5 9B and Gemma 4 12B with reasoning off. A speed test on the 12 GiB L40-12Q vGPU
    measured 0.63 tokens/s for Qwen 3.8 27B (18 min per judgment) and 2 tokens/s prompt reading for Gemma 4 26B;
    the 9B/12B models run 100% on the GPU (129 and 33 tokens/s).
  * Optional arm L, pre-registered: Qwen 3.8 27B and Gemma 4 26B on a Windows RTX 4000 SFF Ada (20 GB), run only
    if a machine is free by 2026-12-05, with identical prompts and data and reasoning off, reported as a model-size
    comparison. If it is not run, that is reported. No other condition decides whether it is run or reported.
  * Reasoning: local models use `--reasoning-effort none` (Ollama ignores `think:false`); DeepSeek uses the API
    default; Gemini 4 Argon uses `low`.
  * Code: `run_llm.py` sends temperature 0 when reasoning is `none`; `pairs.py` gained a calibration-only `dry`
    sample (20 pairs, never used for reported results). Data, operators, extractor, mutants, subsample and prompts
    are unchanged.
- v1.2.1 (2026-11-08, after the dry runs on the 20 calibration-only pairs and before any full run): DeepSeek calls
  use `--max-tokens 4000`, because its default reasoning (about 140–260 tokens) truncated one of 20 dry answers at
  the 400-token cap. Analysis rule fixed now: a missing judge score counts as an escalation (rejection), never
  as an acceptance, and missing counts are reported per judge. No change to code, data or prompts.
- v1.2.2 (2026-11-08 23:45, after the DeepSeek judge run and BEFORE any repair output was inspected): observed that
  41 of 8,391 DeepSeek judge calls (30 on equivalents, 11 on mutants) returned an empty answer after exhausting the
  4,000-token budget in reasoning; under the v1.2.1 rule they count as escalations. Rule fixed now for repairs: a repair
  whose cleaned text is empty (e.g., budget exhausted) counts as a failed repair, is reported per model, stays in the
  denominators of RQ3 rates, and is excluded from gate scoring. No rerun of any call.
- Execution note (2026-11-09): 6 of 6,333 repair pairs (all DeepSeek) have an empty revision. `bert_score` raises an
  error on empty candidates with the installed `transformers`, so G1-G3 were run on `repair_pairs_nonempty.jsonl`
  (6,327 pairs), as the v1.2.2 rule already prescribes. Judges received all 6,333 pairs; empty ones are excluded in analysis.
- Analysis log (2026-11-10, after all runs). Pre-registered analysis: `src/snp/analysis.py` (main, 90%/99%, without E5,
  Argon subsample, RQ3). POST HOC additions, reported as such:
  * Gemma never judged mutants, so it has no calibrated threshold; its self-judgments use the fixed rule score >= 4.
  * The 5th-percentile rule is ill-defined for judges whose scores are effectively binary (DeepSeek, Argon: 1 or 5)
    or miscalibrated (Qwen gives 1 to 844/1,283 punctuation-only equivalents): theta jumps to 1. Reported alongside:
    threshold-free AUROC and the fixed rule score >= 4.
  * Strong judges rejected 35% of E3 and 18% of E6 equivalents; inspection shows many of these are NOT meaning-
    preserving (parse-induced reorderings; condition fronting across sentences or changing the scope of 'only' or 'if
    necessary'). Sensitivity analysis: calibration and FER on E1, E2, E4, E5 only. The human spot-check quantifies this.
  * RQ3 extractor precision: condition-marker substitutions (if->when, as soon as->upon) are flagged as changes;
    commitment-change rates are therefore upper estimates; fabricated specificity is unaffected.

## v1.3 addendum (2026-11-10): held-out drift operators, frozen BEFORE any gate scores them

Problem. M1-M7 and the extractor behind G5 share lexicons (modals, negators, quantifiers, condition
markers, number/unit parser; operators.py imports from extract.py) and were debugged together. G5's
100% detection of M1-M7 and 100% preservation of E1-E6 therefore verify the implementation; they are
not evidence that G5 generalises. In the paper, G5's M1-M7 rows are reported as an implementation
check, not as a result.

Fix. Four drift operators whose wording comes from external sources not used to build the extractor
(src/snp/heldout.py; a unit test checks it imports nothing from extract.py or operators.py):
  H1 loophole (ISO/IEC/IEEE 29148:2018 5.2.7: if possible / as appropriate / as applicable)
  H2 effort hedge (attempt to / make reasonable efforts to / endeavour to)
  H3 bound-to-mean ("on average" before a numeric bound)
  H4 WordNet direct antonym of the verb governed by shall/must (two most frequent senses; light verbs excluded)
Applied to evaluation-split requirements only: H1 308, H2 308, H3 26, H4 25 (667 pairs,
data/derived/heldout_pairs.jsonl). No new calibration: every gate keeps its frozen threshold.

Analysis plan. SAR per gate and operator (Wilson 95% CI) at the main thresholds (q = 0.05) and, for the
judges, also at the fixed rule score >= 4; G5 per subtype; G6 = G5 AND G3. Judges: DeepSeek V4.1 Flash,
Qwen 3.5 9B, and Gemini 4 Argon (all 667; Argon's low cost makes the subsample unnecessary here).
Predictions written before scoring:
  P1 G5 SAR >= 90% on H2, H3, H4; on H1 G5 detects only "if possible" (via the condition marker "if").
  P2 G1 and G2 accept H1-H3 at rates at least as high as their pooled SAR on M1-M10.
  P3 DeepSeek and Argon (score >= 4 rule) accept at most 10% of H1-H4.
H4 may be ungrammatical when WordNet's antonym does not fit the verb frame; reported as a validity threat.

Environment: Python 3.13.18, spaCy 3.8.18, en_core_web_sm 3.8.0

| file | sha256 |
|---|---|
| data/derived/argon_subsample.jsonl | 76fd0bc7411de66e99b34e7f8ce57a79c2153358c57bd92e0174fb9a57491a1e |
| data/derived/dry_pairs.jsonl | f6e9fa04b575c9717bbc39b71c4d3e3df832fe676aac84ffa2f31bb6ca07b2c2 |
| data/derived/gate_pairs.jsonl | e4e171ae39be93250fbea8487ac9410c76bbc6a10d76056787171270f5a5296e |
| data/derived/requirements.jsonl | 2ff5eb802afbb831f5bf3fc184d4e0dab2eb6507ae6ae4b44be2722fb1f525a9 |
| data/derived/variants.jsonl | 4c4bd8883fd7022236ee38c678f46cd1c7ebfd4694319112d46ea6baaf3a55ac |
| data/raw/CPN.csv | e3bab0224f5d33e76877eec288170dfa789968bf95012b7cab38749d5c84883b |
| data/raw/GPS.csv | a838f280125ed153c76b75244365ce0b719249983457621c5b3eddc40367c46b |
| data/raw/QuRE.csv | c6a0c872873ba54028b97361c4cf1d5ad488bdea622a922cbee33265814925d6 |
| data/raw/ePurse-selective.csv | fab57d82f2482e3f35455f3dd3d659241822d1e9dccc87cde9adaa9f40fd6b64 |
| prompts/judge.txt | d59c1e4c84a3f229fd5bd40e05f7529a637e3bc283b811e7738ce5579df8fc36 |
| prompts/repair.txt | f280e10d26b7f7979735f0a967217547a65b89abc7568593ad8d6d141005504d |
| src/snp/audit.py | 162d8e6183fcc0ca9a9184cb63cdccf35bc8b2749615366e70a33b6fea4cf880 |
| src/snp/data.py | a5983082892812c6e061a58ca92633261cc23a0ae576a53b619219bacf8a24ca |
| src/snp/extract.py | 71be199f0550bf0a5073c1e4d6df72b4ffdd514a936d1d1b4aef89a8473e5ea1 |
| src/snp/generate.py | b4f79275d70743443ea6a9af80a445588087daf506c18c16bae366d1d80b6351 |
| src/snp/operators.py | bee036406c5e97bcd0ed8bcc2a12c2846da4bcba78e6b00805c42e3427bd6bdc |
| src/snp/pairs.py | 99a23f9602f69ec6205f568bd3465a591e3ad0bb53e0c5698a5730d50e168ece |
| src/snp/run_gates.py | 396c79f448e9afd7b6173d42113d1624dcc57a1b0c952c27c79184436171028d |
| src/snp/run_llm.py | df9149ef2d90a02ca6b1ef1f6c7aa75c00c336e8245636c32428800966c01297 |
| src/snp/subsample.py | 9950f27505a11fa72ab350c0a98729bf905c44c71a25c635b4642c5a8156b147 |
| src/snp/heldout.py | 47e0a68f4357ee2eff8707ae0e056f7e3d57e19fd863c8d40ee80f2d1b11463b |
| src/snp/heldout_g5.py | 32c11408a836b6d709543914fefb91c3c350261dd05674060d4354703c5b89f1 |
| data/derived/heldout_pairs.jsonl | a0419d049c83ac63f3d599403f1f610180a58c2319c081c1dc6f13baf6a8f428 |

## v1.3 outcome (2026-11-10, after RUNBOOK step 7; 1 DeepSeek score missing -> escalation)
P1 held: G5 SAR H1 69.2% (catches only "if possible"), H2 99.7%, H3 100%, H4 100%; all 85.6%.
P2 held: G1 84.6%, G2 94.6% on H (pooled M: 68.3%, 63.3%).
P3 held: DeepSeek 0/667, Argon 1/667 (score >= 4).
Not predicted: G3 accepts 7/667 (1.0%), so G6 = 1.0%; diff and entailment fail on different drift.

## Run environment (recorded 2026-11-10)
Runs: 8-10 Nov 2026 (timestamps in every results/*.jsonl). GPU: NVIDIA L40-12Q vGPU, driver 580.82.07.
Ollama IDs: qwen3.5:9b 56671c2ab938, gemma4:12b 6114515d63c1 (arm L candidates pulled, not run: qwen3.8:27b aaee06c39dcf, gemma4:26b 001e5dafc3c7).
HF snapshots: roberta-large 722cf37b1afa9454edce342e7895e588b6ff1d59,
  sentence-transformers/all-mpnet-base-v2 e8c3b32edf5434bc2275fc9bab85f82640a19130,
  microsoft/deberta-large-mnli 7296194b9009373def4f7c5dad292651e4b5cf4e.
