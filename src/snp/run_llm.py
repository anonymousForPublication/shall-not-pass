"""LLM repairs (RQ3) and LLM-judge scores (G4), via any OpenAI-compatible endpoint.

DeepSeek (https://api.deepseek.com), the Gemini API
(https://generativelanguage.googleapis.com/v1beta/openai/) and a local Ollama server
(http://localhost:11434/v1) all speak this protocol, so one script covers every model.

  # repairs: one revision per QuRE requirement
  python -m snp.run_llm repair --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --out results/repair_deepseek.jsonl
  python -m snp.run_llm repair --model qwen3.8:27b --base-url http://localhost:11434/v1 --out results/repair_qwen.jsonl

  # judge: score pairs (mutants/equivalents, the Argon subsample, or repairs)
  python -m snp.run_llm judge --model deepseek-flash --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --out results/judge_deepseek.jsonl
  python -m snp.run_llm judge ... --pairs data/derived/repair_pairs.jsonl --repair-model qwen3.8:27b,gemma4:26b

Resumable: ids already in --out are skipped. Every raw response is stored.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def client(base_url: str | None, api_key_env: str | None):
    from openai import OpenAI
    key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not key:
        raise SystemExit(f"environment variable {api_key_env} is not set")
    if base_url:
        return OpenAI(base_url=base_url, api_key=key or "ollama")  # Ollama ignores the key
    return OpenAI(api_key=key) if key else OpenAI()  # default: OPENAI_API_KEY


def call(cli, model: str, prompt: str, effort: str | None, max_tokens: int) -> tuple[str, dict]:
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if effort:
        kw["reasoning_effort"] = effort
    if not effort or effort == "none":
        # non-reasoning call (or reasoning switched off, e.g. Ollama models): deterministic decoding
        kw["temperature"] = 0
        kw["max_tokens"] = max_tokens
    for attempt in range(6):
        try:
            r = cli.chat.completions.create(**kw)
            usage = r.usage.model_dump() if r.usage else {}
            return (r.choices[0].message.content or "").strip(), usage
        except Exception as e:  # rate limits / transient errors: back off and retry
            wait = 2 ** attempt
            print(f"  retry in {wait}s: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)
    raise RuntimeError("giving up after 6 attempts")


def _done(out: Path, key: str) -> set[str]:
    return {json.loads(l)[key] for l in open(out)} if out.exists() else set()


def run(items, key, make_prompt, a, parse=None):
    cli, out = client(a.base_url, a.api_key_env), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out, key)
    todo = [x for x in items if x[key] not in done][: a.limit]
    print(f"{len(todo)} to do, {len(done)} already done, model={a.model}")
    with open(out, "a") as f:
        for i, x in enumerate(todo, 1):
            text, usage = call(cli, a.model, make_prompt(x), a.reasoning_effort, a.max_tokens)
            rec = {key: x[key], "model": a.model, "reasoning_effort": a.reasoning_effort,
                   "response": text, "usage": usage, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if parse:
                rec.update(parse(text))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["repair", "judge"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/derived/gate_pairs.jsonl"))
    ap.add_argument("--base-url", default=None, help="e.g. http://localhost:11434/v1 for Ollama")
    ap.add_argument("--api-key-env", default=None,
                    help="name of the environment variable holding the API key, e.g. DEEPSEEK_API_KEY")
    ap.add_argument("--reasoning-effort", default=None,
                    help="e.g. 'low' for reasoning models (then temperature is not sent); "
                         "mapped to thinking_level by the Gemini API")
    ap.add_argument("--repair-model", default=None,
                    help="judge only repair pairs from these repair models (comma-separated)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="dry run on the first N items")
    a = ap.parse_args()

    if a.task == "repair":
        tmpl = (ROOT / "prompts/repair.txt").read_text()
        reqs = [json.loads(l) for l in open(ROOT / "data/derived/requirements.jsonl")]
        items = [r for r in reqs if r["corpus"] == "QuRE"]
        run(items, "rid", lambda r: tmpl.format(requirement=r["text"]), a)
    else:
        tmpl = (ROOT / "prompts/judge.txt").read_text()
        items = [json.loads(l) for l in open(a.pairs)]
        if a.repair_model:
            keep = set(a.repair_model.split(","))
            items = [p for p in items if p.get("repair_model") in keep]

        def parse(text):
            text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()  # local reasoning traces
            m = re.search(r"\b([1-5])\b", text)
            return {"score": int(m.group(1)) if m else None}
        run(items, "pid", lambda p: tmpl.format(original=p["original"], revision=p["revision"]), a, parse)


if __name__ == "__main__":
    main()
