"""Held-out drift operators H1-H4 (protocol v1.3).

Why this module exists: the drift operators M1-M7 and the commitment extractor (G5) share
their lexicons (modals, negators, quantifiers, condition markers, the number/unit parser) and
were debugged together, so G5's detection of M1-M7 verifies the implementation, not its
generalisation. H1-H4 inject commitment changes whose wording comes from EXTERNAL sources
that played no part in building the extractor:

  H1 loophole        ISO/IEC/IEEE 29148:2018, 5.2.7 (loopholes: "if possible",
                     "as appropriate", "as applicable")      shall X -> shall, as appropriate, X
  H2 effort hedge    best-efforts drafting (Adams, Manual of Style for Contract Drafting)
                     shall X -> shall attempt to X / make reasonable efforts to X / endeavour to X
  H3 bound to mean   a bound on every occurrence becomes a bound on the average
                     within 2 seconds -> on average within 2 seconds; at least 100 Mbit/s -> on average at least ...
  H4 antonym         main verb under a binding modal replaced by its WordNet antonym
                     shall enable X -> shall disable X

Independence rule (checked by a unit test): this module imports nothing from snp.extract or
snp.operators. Same conventions as operators.py: one edit at one site, site chosen by an RNG
seeded with (requirement id, operator id); None when the precondition is not met.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from functools import lru_cache

import spacy

LOOPHOLES = ["if possible", "as appropriate", "as applicable"]               # ISO 29148 5.2.7
EFFORT = ["attempt to", "make reasonable efforts to", "endeavour to"]         # best-efforts clauses

# a bound = comparator + digit number + unit token (letters/%/deg/slash, <= 8 chars)
BOUND = re.compile(
    r"\b(?:within|not (?:more|less) than|in less than|less than|more than|greater than|no (?:more|longer|less) than|at (?:most|least)|"
    r"not (?:to )?exceed(?:ing)?|(?:a )?(?:maximum|minimum|max\.?|min\.?)(?: of)?|up to|below|above)\s+"
    r"\d+(?:[.,]\d+)?\s*(?:%|°C|(?!(?:times|of|and|or|the|to|in|per|for|with|at|on|by|from|if|when|as|is|are|a|an|after|before|during|until|than|seconds?\s+on)\b)[A-Za-zµ/]{1,8}\b)")
TIME_BOUND = BOUND  # name kept for the tests


@dataclass(frozen=True)
class Mutation:
    op: str
    text: str
    subtype: str = ""


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=8192)
def parse(text: str):
    return nlp()(text)


@lru_cache(maxsize=1)
def _wordnet():
    from nltk.corpus import wordnet as wn  # data: nltk_data/corpora/wordnet
    return wn


def rng_for(rid: str, op: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{rid}|{op}".encode()).hexdigest()[:12], 16))


def _binding_sites(doc):
    """Binding modal (shall/must) directly governing a base-form verb, not negated."""
    out = []
    for t in doc:
        if t.lower_ in ("shall", "must") and t.tag_ == "MD":
            v = t.head
            if v.tag_ != "VB" or v.i <= t.i or any(c.dep_ == "neg" for c in v.children):
                continue
            out.append((t, v))
    return out


def _clean(s: str) -> str:
    return re.sub(r"\s{2,}", " ", s).strip()


def h1_loophole(text, rng):
    sites = _binding_sites(parse(text))
    if not sites:
        return None
    m, _ = rng.choice(sites)
    phrase = rng.choice(LOOPHOLES)
    end = m.idx + len(m)
    return Mutation("H1", _clean(f"{text[:end]}, {phrase},{text[end:]}"), phrase)


def h2_effort_hedge(text, rng):
    sites = _binding_sites(parse(text))
    if not sites:
        return None
    m, _ = rng.choice(sites)
    phrase = rng.choice(EFFORT)
    end = m.idx + len(m)
    return Mutation("H2", _clean(f"{text[:end]} {phrase}{text[end:]}"), phrase)


def h3_bound_to_mean(text, rng):
    sites = list(BOUND.finditer(text))
    if not sites:
        return None
    s = rng.choice(sites)
    if text[:s.start()].rstrip().lower().endswith("on average"):
        return None
    return Mutation("H3", _clean(f"{text[:s.start()]}on average {text[s.start():]}"), "on average")


LIGHT_VERBS = {"be", "have", "do", "make", "take", "give", "get", "set", "put", "go", "come", "keep"}


def antonyms(lemma: str) -> list[str]:
    """Direct WordNet antonyms from the lemma's two most frequent verb senses (WordNet orders
    senses by tagged frequency); light/support verbs are excluded (take into account, make available)."""
    if lemma in LIGHT_VERBS:
        return []
    wn = _wordnet()
    senses = [syn for syn in wn.synsets(lemma, "v") if any(l.name() == lemma for l in syn.lemmas())][:2]
    return sorted({a.name() for syn in senses for l in syn.lemmas()
                   if l.name() == lemma for a in l.antonyms() if "_" not in a.name()})


def h4_antonym(text, rng):
    sites = []
    for _, v in _binding_sites(parse(text)):
        if any(c.dep_ in ("xcomp", "ccomp", "prt") for c in v.children):
            continue
        ants = antonyms(v.lemma_.lower())
        if ants:
            sites.append((v, ants))
    if not sites:
        return None
    v, ants = rng.choice(sites)
    new = rng.choice(ants)
    new = new[0].upper() + new[1:] if v.text[:1].isupper() else new
    return Mutation("H4", text[:v.idx] + new + text[v.idx + len(v):], f"{v.lemma_.lower()}->{new.lower()}")


HELDOUT = {"H1": h1_loophole, "H2": h2_effort_hedge, "H3": h3_bound_to_mean, "H4": h4_antonym}


def apply(op: str, rid: str, text: str) -> Mutation | None:
    m = HELDOUT[op](text, rng_for(rid, op))
    return m if m is not None and m.text != text else None


def main() -> None:
    """Write data/derived/heldout_pairs.jsonl: H1-H4 mutants of EVALUATION-split requirements.
    Does not score anything; gates are run afterwards (G5 via snp.heldout_g5, G1-G4 on the VM)."""
    import json
    from collections import Counter
    from pathlib import Path
    d = Path(__file__).resolve().parents[2] / "data/derived"
    reqs = [json.loads(l) for l in open(d / "requirements.jsonl")]
    n = Counter()
    with open(d / "heldout_pairs.jsonl", "w") as f:
        for r in reqs:
            if r["split"] != "evaluation":
                continue
            for op in HELDOUT:
                m = apply(op, r["rid"], r["text"])
                if m is None:
                    continue
                n[op] += 1
                f.write(json.dumps({"pid": f"{r['rid']}|{op}", "rid": r["rid"], "op": op, "kind": "drift",
                                    "subtype": m.subtype, "split": "evaluation",
                                    "original": r["text"], "revision": m.text}, ensure_ascii=False) + "\n")
    print(dict(n), sum(n.values()))


if __name__ == "__main__":
    main()
