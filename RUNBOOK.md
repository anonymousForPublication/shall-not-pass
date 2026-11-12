# Runbook: gate scoring and LLM runs (Linux GPU server)

Model configuration (protocol v1.2, 2026-11-08):

| Role | Model | Endpoint |
|---|---|---|
| Repairer + judge (API) | DeepSeek V4.1 Flash 1017, model id `deepseek-flash` | `https://api.deepseek.com` |
| Repairer + judge (local) | Qwen 3.5 9B (`qwen3.5:9b`), reasoning off | Ollama, `http://localhost:11434/v1` |
| Repairer + self-judge (local) | Gemma 4 12B (`gemma4:12b`), reasoning off | Ollama |
| Strong-judge check (1,048-pair subsample only) | Gemini 4 Argon | `https://generativelanguage.googleapis.com/v1beta/openai/` |

Everything up to the frozen mutant set is done (`notes/FREEZE.md`, git tag `protocol-freeze-v1.2`).
Do not edit anything under `src/`, `prompts/` or `data/` before these runs. If something must
change, stop and report it, because it becomes a protocol deviation.

All commands are bash, run from the `shall_not_pass_code` folder. Long runs should go inside
`tmux` (or `screen`) so they survive a dropped SSH connection: `tmux new -s snp`, detach with
Ctrl-b d, reattach with `tmux attach -t snp`.

## Server specs (record these first)

```bash
{ nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv;
  nproc; free -h; df -h .; cat /etc/os-release | head -2; python3 --version;
  nvcc --version 2>/dev/null | tail -1; curl -sI https://huggingface.co | head -1;
  curl -sI https://api.deepseek.com | head -1; } 2>&1 | tee server_specs.txt
```

## 0. One-time setup (about 20 min)

Server: 16 vCPU, 45 GiB RAM, NVIDIA L40-12Q vGPU (**12 GiB**), driver 580.82.07,
PyTorch 2.11.0+cu128 already installed. Reuse that PyTorch instead of reinstalling it:

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -r requirements-gpu.txt
export PYTHONPATH=src
python -m pytest -q tests          # expect: 16 passed
```

Ollama, with sudo: `curl -fsSL https://ollama.com/install.sh | sh`.
Without sudo, install it in your home directory:
```bash
mkdir -p ~/ollama && curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz | tar xz -C ~/ollama
export PATH=~/ollama/bin:$PATH
nohup ollama serve > ~/ollama/serve.log 2>&1 &
```

API keys (never commit them; keep them in `~/.snp_keys` and `source` that file):
```bash
export DEEPSEEK_API_KEY="sk-..."   # platform.deepseek.com
export GEMINI_API_KEY="..."        # aistudio.google.com
```

## 0b. Local models: measured on this VM (2026-11-08)

`qwen3.8:27b` (53% CPU, 0.63 tokens/s, 18 min for a single judgment) and `gemma4:26b` (20% CPU, 2 tokens/s
prompt reading) do not fit the 12 GiB vGPU and are too slow. `qwen3.5:9b` (129 tokens/s) and `gemma4:12b`
(33 tokens/s) run entirely on the GPU and are used instead (protocol v1.2). Thinking is switched off with
`--reasoning-effort none`; `"think":false` is ignored by Ollama's OpenAI-compatible endpoint.

```bash
ollama pull qwen3.5:9b && ollama pull gemma4:12b
ollama rm qwen3.8:27b gemma4:26b      # optional: frees about 36 GB
```

**Order on this VM.** Start the API steps first, because they need no GPU: the Argon subsample (step 4b, before
30 Nov) and the DeepSeek steps. Run them in one `tmux` window, with G1–G3 and then the local models in a
second window.

**Cost tips.**
- DeepSeek charges half price off-peak. Peak hours are weekdays 01:00–04:00 and 06:00–10:00 UTC, which is **05:00–08:00 and 10:00–14:00 Dubai time**, so run outside those windows.
- Argon's introductory price ($2/$10) ends on **30 November 2026**, so run step 4b before then.

## 1. Dry runs first (10 min). Check the outputs before the full runs

```bash
source ~/.snp_keys
python -m snp.pairs dry      # 10 mutants + 10 equivalents from the calibration split (not used in results)
python -m snp.run_gates --pairs data/derived/gate_pairs.jsonl --out results/dry_gates.csv --limit 64
python -m snp.run_llm judge --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --pairs data/derived/dry_pairs.jsonl --out results/dry_judge_deepseek.jsonl
python -m snp.run_llm judge --model qwen3.5:9b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/dry_pairs.jsonl --out results/dry_judge_qwen.jsonl
python -m snp.run_llm judge --model gemma4:12b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/dry_pairs.jsonl --out results/dry_judge_gemma.jsonl
python -m snp.run_llm judge --model gemini-4-argon --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GEMINI_API_KEY --reasoning-effort low --pairs data/derived/dry_pairs.jsonl --out results/dry_judge_argon.jsonl
python -m snp.run_llm repair --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --out results/dry_repair_deepseek.jsonl --limit 3
python -m snp.run_llm repair --model qwen3.5:9b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/dry_repair_qwen.jsonl --limit 3
# summary: mean judge score for mutants vs equivalents, missing scores, tokens per call
python - <<'PY'
import json,glob,statistics as st
kind={json.loads(l)["pid"]:json.loads(l)["kind"] for l in open("data/derived/dry_pairs.jsonl")}
for f in sorted(glob.glob("results/dry_judge_*.jsonl")):
    R=[json.loads(l) for l in open(f)]
    d=[r["score"] for r in R if kind[r["pid"]]=="drift" and r["score"]]; e=[r["score"] for r in R if kind[r["pid"]]=="equiv" and r["score"]]
    tok=[r["usage"].get("completion_tokens",0) for r in R]
    print(f"{f}: drift mean={st.mean(d) if d else None}  equiv mean={st.mean(e) if e else None}  missing={sum(r['score'] is None for r in R)}  completion tokens/call={st.mean(tok):.0f}")
PY
```

Check the summary lines and one repair from each `dry_repair_*` file. The drift mean should be clearly
below the equivalent mean; a judge that fails to separate even these 20 pairs is a finding in itself. If a model id is rejected, list the available ids:
```bash
python -c "from openai import OpenAI;import os;c=OpenAI(base_url='https://generativelanguage.googleapis.com/v1beta/openai/',api_key=os.environ['GEMINI_API_KEY']);print([m.id for m in c.models.list() if 'argon' in m.id])"
```
Then delete the dry-run files: `rm results/dry_*`.

## 2. Gates G1 to G3 on the 8,391 mutant/equivalent pairs (GPU, expected under 1 h)

```bash
python -m snp.run_gates --pairs data/derived/gate_pairs.jsonl --out results/gates_g1g3.csv
```

## 3. Repairs: 2,111 QuRE requirements x 3 models (RQ3)

```bash
python -m snp.run_llm repair --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --out results/repair_deepseek-flash.jsonl
python -m snp.run_llm repair --model qwen3.5:9b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/repair_qwen3.5-9b.jsonl
python -m snp.run_llm repair --model gemma4:12b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/repair_gemma4-12b.jsonl
```

## 4. Judges (G4) on the mutant/equivalent pairs

```bash
# 4a. two judges on all 8,391 pairs
python -m snp.run_llm judge --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --out results/judge_deepseek-flash.jsonl
python -m snp.run_llm judge --model qwen3.5:9b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/judge_qwen3.5-9b.jsonl
# 4b. strong-judge check: Gemini 4 Argon on the 1,048-pair subsample (before 30 Nov)
python -m snp.run_llm judge --model gemini-4-argon --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GEMINI_API_KEY --reasoning-effort low --pairs data/derived/argon_subsample.jsonl --out results/judge_argon_subsample.jsonl
```

## 5. Score the repairs

```bash
python -m snp.pairs repairs results/repair_deepseek-flash.jsonl results/repair_qwen3.5-9b.jsonl results/repair_gemma4-12b.jsonl
python -m snp.run_gates --pairs data/derived/repair_pairs.jsonl --out results/repair_gates_g1g3.csv

# Judge design: each repair gets a self-judge (same model) and an independent judge (another vendor).
#   DeepSeek repairs: self = DeepSeek, independent = Qwen
#   Qwen repairs:     self = Qwen,     independent = DeepSeek
#   Gemma repairs:    self = Gemma,    independent = DeepSeek
python -m snp.run_llm judge --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --pairs data/derived/repair_pairs.jsonl --out results/rjudge_deepseek-flash.jsonl
python -m snp.run_llm judge --model qwen3.5:9b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/repair_pairs.jsonl --repair-model deepseek-flash,qwen3.5:9b --out results/rjudge_qwen3.5-9b.jsonl
python -m snp.run_llm judge --model gemma4:12b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/repair_pairs.jsonl --repair-model gemma4:12b --out results/rjudge_gemma4-12b.jsonl
```

All runs can be resumed: if one stops, run the same command again.

## 5b. Arm L: larger local models on Windows (pre-registered; run only if a machine is free by 5 Dec 2026)

Same prompts, frozen data and decision rules as the 9B/12B arm, reasoning off; reported as a model-size
comparison. Machine: Windows workstation with an RTX 4000 SFF Ada (20 GB). About 12 hours of GPU time in total,
so run it overnight. Everything runs locally; no API key is needed on Windows.

**Setup (PowerShell, about 30 min).** Install Python 3.11 or newer from python.org and Ollama for Windows from
ollama.com. Unzip `snp_armL_windows.zip` into, for example, `C:\snp`, then:
```powershell
cd C:\snp
python -m venv .venv; .\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install openai
$env:PYTHONPATH = "src"
ollama pull qwen3.8:27b; ollama pull gemma4:26b
ollama list        # IDs must match the VM: qwen3.8:27b aaee06c39dcf, gemma4:26b 001e5dafc3c7
```

**Check that each model fits on the GPU (5 min).** The protocol requires 100% GPU; if `ollama ps` shows any CPU
share, stop and report it instead of running.
```powershell
python -m snp.run_llm judge --model qwen3.8:27b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/dry_pairs.jsonl --out results/L_dry_qwen3.8-27b.jsonl
ollama ps
python -m snp.run_llm judge --model gemma4:26b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/dry_pairs.jsonl --out results/L_dry_gemma4-26b.jsonl
ollama ps
```

**Runs (in this order; each command resumes if interrupted).**
```powershell
# L1 judge on the 8,391 mutant/equivalent pairs (about 3 h) and on the 667 held-out pairs (about 15 min)
python -m snp.run_llm judge --model qwen3.8:27b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/L_judge_qwen3.8-27b.jsonl
python -m snp.run_llm judge --model qwen3.8:27b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/heldout_pairs.jsonl --out results/L_heldout_judge_qwen3.8-27b.jsonl
# L2 repairs of the 2,111 QuRE requirements (about 3.5 h per model)
python -m snp.run_llm repair --model qwen3.8:27b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/L_repair_qwen3.8-27b.jsonl
python -m snp.run_llm repair --model gemma4:26b --base-url http://localhost:11434/v1 --reasoning-effort none --out results/L_repair_gemma4-26b.jsonl
# L3 self-judges of the repairs (about 40 min per model); PowerShell does not expand *, so list both files
python -m snp.pairs repairs_L results/L_repair_qwen3.8-27b.jsonl results/L_repair_gemma4-26b.jsonl
python -m snp.run_llm judge --model qwen3.8:27b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/L_repair_pairs.jsonl --repair-model qwen3.8:27b --out results/L_rjudge_qwen3.8-27b.jsonl
python -m snp.run_llm judge --model gemma4:26b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/L_repair_pairs.jsonl --repair-model gemma4:26b --out results/L_rjudge_gemma4-26b.jsonl
```

**Then on the Linux VM** (copy `data/derived/L_repair_pairs.jsonl` and the `results/L_*` files into the code folder):
```bash
python -m snp.run_gates --pairs data/derived/L_repair_pairs.jsonl --out results/L_repair_gates_g1g3.csv
python -m snp.run_llm judge --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --pairs data/derived/L_repair_pairs.jsonl --out results/L_rjudge_deepseek-flash.jsonl
```
The DeepSeek call is the independent judge for both L repairers (about $2).

## 6. Collect the outputs

Pack the results: `tar czf results_$(date +%Y%m%d).tgz results/ data/derived/repair_pairs.jsonl server_specs.txt`.
Also record the output of:

```bash
sha256sum results/*
```

Record the run dates and times, the Ollama tags plus the `ollama show <model>` digest, and the GPU and driver version. All of these go in the paper. `deepseek-flash` is a moving alias, and on these dates it serves V4.1-Flash-1017.

## 7. Protocol v1.3: held-out drift operators (667 pairs, about 30 min in total)

Why: G5 shares its word lists with M1–M7, so its perfect score there is circular. H1–H4 take their wording
from external sources (ISO 29148 loopholes, best-efforts clauses, "on average" bounds, WordNet antonyms).
The pairs are already generated (`data/derived/heldout_pairs.jsonl`), so the VM needs no new packages.
Unzip `snp_v13_update.zip` over the code folder, then, from the code folder with the venv active:

```bash
source ~/.snp_keys
# window 1: GPU gates (a few minutes), then the local judge
python -m snp.run_gates --pairs data/derived/heldout_pairs.jsonl --out results/heldout_gates_g1g3.csv
python -m snp.run_llm judge --model qwen3.5:9b --base-url http://localhost:11434/v1 --reasoning-effort none --pairs data/derived/heldout_pairs.jsonl --out results/heldout_judge_qwen3.5-9b.jsonl
# window 2: API judges (outside DeepSeek peak hours; Argon before 30 Nov)
python -m snp.run_llm judge --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --max-tokens 4000 --pairs data/derived/heldout_pairs.jsonl --out results/heldout_judge_deepseek-flash.jsonl
python -m snp.run_llm judge --model gemini-4-argon --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GEMINI_API_KEY --reasoning-effort low --pairs data/derived/heldout_pairs.jsonl --out results/heldout_judge_argon.jsonl
wc -l results/heldout_*      # expect 668 lines in the csv (header) and 667 in each jsonl
```
Outputs: the four `results/heldout_*` files. API cost: under $1 for DeepSeek and about $1 for Argon.
