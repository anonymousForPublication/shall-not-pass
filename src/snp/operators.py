"""Drift operators M1-M10 (commitment changes) and equivalence operators E1-E6 (surface changes).

Every operator returns either None (precondition not met) or a Mutation with the new text
and the replaced character span. Each makes ONE edit at ONE site; when several sites
qualify, one is chosen with an RNG seeded by (requirement id, operator id), so the whole
mutant set is reproducible.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from functools import lru_cache

import spacy

from .extract import NUMBER_WORDS, UNITS, find_values

_WORD_OF = {v: k for k, v in NUMBER_WORDS.items() if v <= 20}


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=8192)
def parse(text: str):
    return nlp()(text)


@dataclass(frozen=True)
class Mutation:
    op: str
    text: str
    start: int      # span replaced in the ORIGINAL text
    end: int
    replacement: str
    subtype: str = ""


def rng_for(rid: str, op: str) -> random.Random:
    seed = int(hashlib.sha256(f"{rid}|{op}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def _replace(text: str, start: int, end: int, new: str, op: str, subtype: str = "") -> Mutation | None:
    out = text[:start] + new + text[end:]
    out = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", out)).strip()
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"^[,;]\s*", "", out)
    if out and out[0].islower() and start == 0:
        out = out[0].upper() + out[1:]
    if out == text or len(out.split()) < 3:
        return None
    return Mutation(op, out, start, end, new, subtype)


def _case_like(src: str, new: str) -> str:
    return new[0].upper() + new[1:] if src[:1].isupper() else new


def _pick(items, rng):
    return rng.choice(items) if items else None


# ----------------------------------------------------------------- drift operators
def m1_modality_weakening(text, rng):
    doc = parse(text)
    sites = [t for t in doc if t.lower_ in ("shall", "must") and t.tag_ == "MD"]
    t = _pick(sites, rng)
    if t is None:
        return None
    new = rng.choice(["should", "may"])
    return _replace(text, t.idx, t.idx + len(t), _case_like(t.text, new), "M1", new)


_VAGUE = {"time": "a short time", "count": None, "data": "a sufficient amount of data",
          "rate": "a sufficient rate", "ratio": "a high proportion", "length": "a suitable distance",
          "voltage": "a suitable voltage", "current": "a suitable current", "power": "suitable power",
          "frequency": "a suitable frequency", "temperature": "a suitable temperature",
          "mass": "a suitable mass", "speed": "a suitable speed", "level": "a suitable level"}


def m2_vague_threshold(text, rng):
    vals = find_values(text)
    v = _pick(vals, rng)
    if v is None:
        return None
    if v.unit is None or UNITS_DIM.get(v.unit) == "count":
        # "after five failed attempts" -> "after several failed attempts"; keep the noun
        return _replace(text, v.num_start, v.num_end, "several", "M2", "count")
    phrase = _VAGUE[UNITS_DIM[v.unit]]
    return _replace(text, v.num_start, v.end, phrase, "M2", UNITS_DIM[v.unit])


UNITS_DIM = {canon: dim for canon, dim in UNITS.values()}


def _fmt_number(original: str, value: float) -> str:
    if original.strip().lower() in NUMBER_WORDS:
        return _WORD_OF.get(int(value), f"{int(value):,}" if value >= 1000 else str(int(value)))
    if "," in original:
        return f"{int(value):,}"
    return str(int(value)) if value == int(value) else f"{value:g}"


_FLIP = {">=": "at most", "<=": "at least", "<": "more than", ">": "less than"}


def m3_value_change(text, rng):
    vals = find_values(text)
    v = _pick(vals, rng)
    if v is None:
        return None
    if v.cmp and rng.random() < 0.5:   # comparator direction flip
        cmp_text = text[v.start:v.num_start].rstrip()
        return _replace(text, v.start, v.start + len(cmp_text), _case_like(cmp_text, _FLIP[v.cmp]), "M3", "comparator")
    orig = text[v.num_start:v.num_end]
    num = NUMBER_WORDS.get(orig.lower(), None)
    num = float(num) if num is not None else float(orig.replace(",", ""))
    new = num + 1 if (orig.lower() in NUMBER_WORDS and num < 20) else num * 10
    return _replace(text, v.num_start, v.num_end, _fmt_number(orig, new), "M3", "value")


_UNIT_SWAP = {"ms": "s", "s": "min", "min": "h", "h": "day", "day": "week", "week": "month",
              "month": "year", "year": "month", "mV": "V", "V": "mV", "kV": "V", "mA": "A", "A": "mA",
              "W": "kW", "kW": "W", "Hz": "kHz", "kHz": "MHz", "MHz": "kHz", "mm": "cm", "cm": "m",
              "m": "km", "km": "m", "g": "kg", "kg": "g", "byte": "kB", "kB": "MB", "MB": "GB",
              "GB": "MB", "bit": "byte", "kbit/s": "Mbit/s", "Mbit/s": "kbit/s"}


def m4_unit_swap(text, rng):
    vals = [v for v in find_values(text) if v.unit in _UNIT_SWAP]
    v = _pick(vals, rng)
    if v is None:
        return None
    unit_text = text[v.num_end:v.end].strip()
    u_start = text.index(unit_text, v.num_end)
    target = _UNIT_SWAP[v.unit]
    # keep the surface style: abbreviation stays abbreviation, word stays word
    if unit_text in UNIT_ABBR:
        new = target if target in UNIT_ABBR else UNIT_WORD[target] + ("" if v.num == 1 else "s")
    else:
        plural = unit_text.endswith("s") and not UNIT_WORD[target].endswith("s") and " per " not in UNIT_WORD[target]
        new = UNIT_WORD[target] + ("s" if plural else "")
    return _replace(text, u_start, u_start + len(unit_text), new, "M4", v.unit + "->" + target)


UNIT_ABBR = {"ms", "s", "min", "h", "mV", "V", "kV", "mA", "A", "W", "kW", "Hz", "kHz", "MHz", "mm",
             "cm", "m", "km", "g", "kg", "kB", "KB", "MB", "GB", "kbit/s", "Mbit/s", "B", "rpm", "dB"}
UNIT_WORD = {"ms": "millisecond", "s": "second", "min": "minute", "h": "hour", "day": "day",
             "week": "week", "month": "month", "year": "year", "mV": "millivolt", "V": "volt",
             "kV": "kilovolt", "mA": "milliampere", "A": "ampere", "W": "watt", "kW": "kilowatt",
             "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "mm": "millimetre", "cm": "centimetre",
             "m": "metre", "km": "kilometre", "g": "gram", "kg": "kilogram", "byte": "byte",
             "kB": "kilobyte", "MB": "megabyte", "GB": "gigabyte", "bit": "bit",
             "kbit/s": "kilobits per second", "Mbit/s": "megabits per second"}


def m5_polarity_flip(text, rng):
    doc = parse(text)
    removals = [t for t in doc if t.lower_ == "not" and t.dep_ == "neg"]
    inserts = [t for t in doc if t.tag_ == "MD" and t.lower_ in ("shall", "must", "should", "may", "will")
               and not (t.i + 1 < len(doc) and doc[t.i + 1].lower_ == "not")]
    if removals and (not inserts or rng.random() < 0.5):
        t = rng.choice(removals)
        return _replace(text, t.idx, t.idx + len(t), "", "M5", "remove")
    t = _pick(inserts, rng)
    if t is None:
        return None
    end = t.idx + len(t)
    return _replace(text, t.idx, end, t.text + " not", "M5", "insert")


_QSHIFT = {"all": "some", "every": "some", "each": "some", "any": "some", "only": "also",
           "always": "sometimes", "exactly": "approximately", "solely": "also"}


def m6_quantifier_shift(text, rng):
    doc = parse(text)
    sites = [t for t in doc if t.lower_ in _QSHIFT]
    t = _pick(sites, rng)
    if t is None:
        return None
    new = _case_like(t.text, _QSHIFT[t.lower_])
    return _replace(text, t.idx, t.idx + len(t), new, "M6", t.lower_ + "->" + _QSHIFT[t.lower_])


_COND_MARK = {"if", "when", "whenever", "unless", "while", "after", "before", "until", "once"}
_COND_PREP = {"during", "upon", "after", "before", "until", "on"}
_ON_HEADS = {"receipt", "reception", "detection", "transmission", "indication"}


def _span_with_comma(doc, lo, hi):
    """Extend [lo, hi) token span over one adjacent comma so the sentence stays well formed."""
    if hi < len(doc) and doc[hi].text == ",":
        hi += 1
    elif lo > 0 and doc[lo - 1].text == ",":
        lo -= 1
    return lo, hi


def _condition_sites(doc):
    sites = []
    for t in doc:
        if t.dep_ in ("mark", "advmod") and t.lower_ in _COND_MARK and t.head.dep_ == "advcl":
            sub = list(t.head.subtree)
            sites.append((sub[0].i, sub[-1].i + 1, "clause"))
        elif t.dep_ == "prep" and t.lower_ in _COND_PREP:
            if t.lower_ == "on" and (t.i != 0 or not any(c.lower_ in _ON_HEADS for c in t.children)):
                continue
            sub = list(t.subtree)
            sites.append((sub[0].i, sub[-1].i + 1, "prep"))
    return sites


def m7_condition_drop(text, rng):
    doc = parse(text)
    sites = [s for s in _condition_sites(doc) if s[1] - s[0] < len(doc) - 3]
    s = _pick(sites, rng)
    if s is None:
        return None
    lo, hi = _span_with_comma(doc, s[0], s[1])
    start = doc[lo].idx
    end = doc[hi - 1].idx + len(doc[hi - 1])
    return _replace(text, start, end, "", "M7", s[2])


def m8_actor_swap(text, rng):
    doc = parse(text)
    pairs = []
    for t in doc:
        if t.dep_ not in ("nsubj", "nsubjpass") or t.pos_ == "PRON":
            continue
        subj = _chunk(doc, t)
        for c in t.head.subtree:
            if c.dep_ == "pobj" and c.pos_ in ("NOUN", "PROPN") and c.i > t.i:
                if c.head.lower_ not in ("by", "from", "to", "with", "between") or c.head.head.i != t.head.i:
                    continue
                obj = _chunk(doc, c)
                if not (obj and subj and _clean_np(doc, subj) and _clean_np(doc, obj)):
                    continue
                if obj[0] > subj[1] and doc[subj[0]:subj[1]].text.lower() != doc[obj[0]:obj[1]].text.lower():
                    pairs.append((subj, obj))
    p = _pick(pairs, rng)
    if p is None:
        return None
    (a0, a1), (b0, b1) = p
    a, b = doc[a0:a1], doc[b0:b1]
    a_txt, b_txt = a.text, b.text
    new_a = _case_like(a_txt, b_txt[0].lower() + b_txt[1:] if not b_txt[:2].isupper() else b_txt)
    new_b = a_txt[0].lower() + a_txt[1:] if a0 == 0 and not a_txt[:2].isupper() else a_txt
    out = text[:a.start_char] + new_a + text[a.end_char:b.start_char] + new_b + text[b.end_char:]
    if out == text:
        return None
    return Mutation("M8", out, a.start_char, b.end_char, f"{a_txt}<->{b_txt}", "swap")


def _clean_np(doc, span) -> bool:
    toks = doc[span[0]:span[1]]
    return len(toks) <= 5 and all(re.fullmatch(r"[A-Za-z][A-Za-z-]*", t.text) for t in toks)


def _chunk(doc, head):
    """Noun-phrase token span (determiner + modifiers + head), without clauses or PPs."""
    for nc in doc.noun_chunks:
        if nc.root.i == head.i:
            return (nc.start, nc.end)
    return None


def m9_scope_reduction(text, rng):
    doc = parse(text)
    sites = []
    for t in doc:
        if t.dep_ != "conj" or t.head.pos_ not in ("NOUN", "PROPN", "VERB", "ADJ"):
            continue
        sub = list(t.subtree)
        lo, hi = sub[0].i, sub[-1].i + 1
        # remove the coordinator/comma immediately before the conjunct
        while lo > 0 and (doc[lo - 1].dep_ == "cc" or doc[lo - 1].text == ",") and doc[lo - 1].head.i in (t.head.i, t.i):
            lo -= 1
        if lo > 0 and doc[lo - 1].text == ",":
            lo -= 1
        if hi - lo >= len(doc) - 3:
            continue
        if hi < len(doc) and doc[hi].pos_ in ("NOUN", "PROPN") and doc[hi].head.i != t.head.i:
            # conjunct may be a modifier of a longer phrase ("integrity and source authenticity");
            # only allow it inside an explicit comma list ("load, unload, and currency exchange ...")
            if not any(doc[k].text == "," for k in range(t.head.i, t.i)):
                continue
        sites.append((lo, hi))
    s = _pick(sites, rng)
    if s is None:
        return None
    start = doc[s[0]].idx
    end = doc[s[1] - 1].idx + len(doc[s[1] - 1])
    return _replace(text, start, end, "", "M9", "drop-conjunct")


_ADDITIONS = [("technology", "using a relational database"), ("technology", "via a REST interface"),
              ("bound", "within 5 seconds"), ("bound", "at least 3 times")]


def m10_unsupported_addition(text, rng):
    sub, phrase = rng.choice(_ADDITIONS)
    m = re.search(r"[.;:]?\s*$", text)
    pos = m.start()
    return _replace(text, pos, pos, " " + phrase, "M10", sub)


# ----------------------------------------------------------------- equivalence operators
def e1_numeral_form(text, rng):
    vals = [v for v in find_values(text)]
    cands = []
    for v in vals:
        s = text[v.num_start:v.num_end]
        if v.unit and v.num_end < len(text) and not text[v.num_end].isspace():
            continue
        if s.lower() in NUMBER_WORDS and NUMBER_WORDS[s.lower()] <= 20:
            cands.append((v, str(NUMBER_WORDS[s.lower()])))
        elif s.isdigit() and int(s) in _WORD_OF:
            cands.append((v, _WORD_OF[int(s)]))
    c = _pick(cands, rng)
    if c is None:
        return None
    v, new = c
    return _replace(text, v.num_start, v.num_end, new, "E1")


def e2_unit_spelling(text, rng):
    cands = []
    for v in find_values(text):
        if v.unit is None or UNITS_DIM.get(v.unit) == "count" or v.unit not in UNIT_WORD:
            continue
        unit_text = text[v.num_end:v.end].strip()
        if not unit_text:
            continue
        u_start = text.index(unit_text, v.num_end)
        if unit_text in UNIT_ABBR:
            w = UNIT_WORD[v.unit]
            word = w if (v.num == 1 or w.endswith("s") or " per " in w) else w + "s"
            cands.append((u_start, unit_text, word))
        elif v.unit in UNIT_ABBR:
            cands.append((u_start, unit_text, v.unit))
    c = _pick(cands, rng)
    if c is None:
        return None
    u_start, old, new = c
    return _replace(text, u_start, u_start + len(old), new, "E2")


def e3_conjunct_reorder(text, rng):
    doc = parse(text)
    sites = []
    for t in doc:
        if t.dep_ != "conj" or t.pos_ not in ("NOUN", "PROPN") or t.head.pos_ not in ("NOUN", "PROPN"):
            continue
        a, b = _chunk(doc, t.head), _chunk(doc, t)
        cc = [c for c in t.head.children if c.dep_ == "cc" and c.lower_ in ("and", "or")]
        if not (a and b and cc) or any(w.lower_ in ("then", "followed") for w in doc[a[0]:b[1]]):
            continue
        if not (a[1] == cc[0].i and cc[0].i + 1 == b[0]):
            continue  # only the contiguous pattern "A and B" (no material between the conjuncts)
        sites.append((a, b))
    s = _pick(sites, rng)
    if s is None:
        return None
    (a0, a1), (b0, b1) = s
    A, B = doc[a0:a1], doc[b0:b1]
    if A.text.lower() == B.text.lower():
        return None
    a_new = _case_like(A.text, B.text if B.text[:2].isupper() else B.text[0].lower() + B.text[1:])
    b_new = A.text if (a0 > 0 or A.text[:2].isupper()) else A.text[0].lower() + A.text[1:]
    out = text[:A.start_char] + a_new + text[A.end_char:B.start_char] + b_new + text[B.end_char:]
    return Mutation("E3", out, A.start_char, B.end_char, f"{A.text}<->{B.text}") if out != text else None


def e4_punctuation(text, rng):
    if text.rstrip().endswith("."):
        return Mutation("E4", text.rstrip()[:-1], len(text.rstrip()) - 1, len(text.rstrip()), "", "drop-period")
    return Mutation("E4", text.rstrip() + ".", len(text.rstrip()), len(text.rstrip()), ".", "add-period")


def e5_binding_synonym(text, rng):
    doc = parse(text)
    sites = [t for t in doc if t.lower_ in ("shall", "must") and t.tag_ == "MD"]
    t = _pick(sites, rng)
    if t is None:
        return None
    new = "must" if t.lower_ == "shall" else "shall"
    return _replace(text, t.idx, t.idx + len(t), _case_like(t.text, new), "E5", t.lower_ + "->" + new)


def e6_condition_fronting(text, rng):
    doc = parse(text)
    sites = []
    for t in doc:
        if t.dep_ in ("mark", "advmod") and t.lower_ in ("if", "when", "whenever", "unless") and t.head.dep_ == "advcl":
            sub = list(t.head.subtree)
            lo, hi = sub[0].i, sub[-1].i + 1
            root = t.head.head
            if lo > root.i and hi >= len(doc) - 1:          # clause at the end -> move to front
                sites.append(("front", lo, hi))
            elif lo == 0 and hi < len(doc) and doc[hi].text == ",":  # clause at front -> move to end
                sites.append(("back", lo, hi))
    s = _pick(sites, rng)
    if s is None:
        return None
    kind, lo, hi = s
    clause = doc[lo:hi].text
    end_punct = text.rstrip()[-1] if text.rstrip()[-1:] in ".;" else ""
    if kind == "front":
        main = text[:doc[lo].idx].rstrip().rstrip(",")
        first = doc[0]
        main = main if (first.pos_ == "PROPN" or first.text[:2].isupper()) else main[0].lower() + main[1:]
        out = f"{clause[0].upper()}{clause[1:]}, {main}{end_punct}"
    else:
        rest = text[doc[hi].idx + 1:].strip().rstrip(".;")
        out = f"{rest[0].upper()}{rest[1:]} {clause[0].lower()}{clause[1:]}{end_punct}"
    out = re.sub(r"\s{2,}", " ", out)
    return Mutation("E6", out, doc[lo].idx, doc[hi - 1].idx + len(doc[hi - 1]), "", kind) if out != text else None


DRIFT = {"M1": m1_modality_weakening, "M2": m2_vague_threshold, "M3": m3_value_change,
         "M4": m4_unit_swap, "M5": m5_polarity_flip, "M6": m6_quantifier_shift,
         "M7": m7_condition_drop, "M8": m8_actor_swap, "M9": m9_scope_reduction,
         "M10": m10_unsupported_addition}
EQUIV = {"E1": e1_numeral_form, "E2": e2_unit_spelling, "E3": e3_conjunct_reorder,
         "E4": e4_punctuation, "E5": e5_binding_synonym, "E6": e6_condition_fronting}


def apply(op: str, rid: str, text: str) -> Mutation | None:
    fn = DRIFT.get(op) or EQUIV[op]
    return fn(text, rng_for(rid, op))
