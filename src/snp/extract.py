"""Protected-commitment extractor N(C(r)) used by gate G5 and by the RQ3 measures.

Deterministic and lexical by design. It covers five commitment types:
  modality   binding (shall, must, is/are required to, has/have to) | nonbinding
             (should, may, can, could, might) | declarative (will)   [ISO/IEC/IEEE 29148]
  values     (comparator, number, unit) for quantities with a unit, comparator or count noun
  negation   number of negators
  quant      quantifier classes (universal, existential, restrictive, frequency)
  cond       condition markers (if, when, unless, after, on receipt of, ...)
It deliberately does NOT cover actors, scope items, or free-form additions;
these are the expected blind spots of G5 (operators M8, M9, M10-technology).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
    "thousand": 1000,
}

# surface form -> (canonical unit, dimension). Longest forms are matched first.
UNITS: dict[str, tuple[str, str]] = {}
def _u(canon: str, dim: str, *forms: str) -> None:
    for f in forms:
        UNITS[f] = (canon, dim)
_u("ms", "time", "ms", "msec", "millisecond", "milliseconds")
_u("s", "time", "s", "sec", "secs", "second", "seconds")
_u("min", "time", "min", "mins", "minute", "minutes")
_u("h", "time", "h", "hr", "hrs", "hour", "hours")
_u("day", "time", "day", "days")
_u("week", "time", "week", "weeks")
_u("month", "time", "month", "months")
_u("year", "time", "year", "years")
_u("mV", "voltage", "mV", "millivolt", "millivolts")
_u("V", "voltage", "V", "volt", "volts")
_u("kV", "voltage", "kV", "kilovolt", "kilovolts")
_u("mA", "current", "mA", "milliampere", "milliamperes")
_u("A", "current", "A", "ampere", "amperes", "amp", "amps")
_u("W", "power", "W", "watt", "watts")
_u("kW", "power", "kW", "kilowatt", "kilowatts")
_u("Hz", "frequency", "Hz", "hertz")
_u("kHz", "frequency", "kHz", "kilohertz")
_u("MHz", "frequency", "MHz", "megahertz")
_u("degC", "temperature", "°C", "degrees Celsius", "degree Celsius")
_u("percent", "ratio", "%", "percent", "per cent")
_u("mm", "length", "mm", "millimetre", "millimetres", "millimeter", "millimeters")
_u("cm", "length", "cm", "centimetre", "centimetres", "centimeter", "centimeters")
_u("m", "length", "m", "metre", "metres", "meter", "meters")
_u("km", "length", "km", "kilometre", "kilometres", "kilometer", "kilometers")
_u("km/h", "speed", "km/h", "kph", "kilometres per hour", "kilometers per hour")
_u("g", "mass", "g", "gram", "grams")
_u("kg", "mass", "kg", "kilogram", "kilograms")
_u("byte", "data", "byte", "bytes", "B")
_u("kB", "data", "kB", "KB", "kilobyte", "kilobytes")
_u("MB", "data", "MB", "megabyte", "megabytes")
_u("GB", "data", "GB", "gigabyte", "gigabytes")
_u("bit", "data", "bit", "bits")
_u("kbit/s", "rate", "kbit/s", "kbps", "kilobits per second")
_u("Mbit/s", "rate", "Mbit/s", "Mbps", "megabits per second")
_u("dB", "level", "dB", "decibel", "decibels")
_u("rpm", "rate", "rpm")
# count nouns: a number before these is a threshold, not an identifier
for _c in ("time", "attempt", "user", "character", "digit", "cycle", "retry", "repetition",
           "method", "slot", "entry", "icon", "socket", "interface", "channel", "key"):
    _u(_c, "count", _c, _c + "s", _c[:-1] + "ies" if _c.endswith("y") else _c + "s")

COMPARATORS = {
    ">=": ["at least", "no less than", "not less than", "minimum of", "minimum", "min.", ">=", "≥"],
    "<=": ["at most", "no more than", "not more than", "maximum of", "maximum", "max.", "up to",
           "within", "not exceeding", "not exceed", "<=", "≤"],
    "<": ["less than", "lower than", "below", "<"],
    ">": ["more than", "greater than", "higher than", "above", "exceeding", "exceeds", ">"],
}
_CMP_CANON = {s: k for k, v in COMPARATORS.items() for s in v}

_num_words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_cmp = "|".join(re.escape(s) for s in sorted(_CMP_CANON, key=len, reverse=True))
_units = "|".join(re.escape(s) for s in sorted(UNITS, key=len, reverse=True))
VALUE_RE = re.compile(
    rf"(?:(?P<cmp>(?i:{_cmp}))\s*)?"
    rf"(?<![\w.,])(?<!\d\s)(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|(?i:\b(?:{_num_words})\b))"
    rf"(?:\s?(?P<unit>{_units}))?(?![\w/])")
_count_forms = "|".join(sorted((f for f, (c, d) in UNITS.items() if d == "count"), key=len, reverse=True))
COUNT_GAP_RE = re.compile(rf"\s+(?:[A-Za-z-]+\s+){{1,2}}({_count_forms})\b", re.I)
# a number directly preceded by these words is an identifier, not a quantity
IDENT_BEFORE = re.compile(
    r"(?i:\b(?:spec|iso|iec|ts|es|en|din|part|section|clause|chapter|table|figure|fig\.|rfc|"
    r"release|version|draft|annex|appendix|step|lv|vda|cat|type|class|level|phase|layer))\s*[-.]?\s*$")

MODALS = {
    "binding": [r"\bshall\b", r"\bmust\b", r"\b(?:is|are) required to\b", r"\b(?:has|have) to\b"],
    "nonbinding": [r"\bshould\b", r"\bmay\b", r"\bcan(?:not)?\b", r"\bcould\b", r"\bmight\b"],
    "declarative": [r"\bwill\b"],
}
NEGATION = re.compile(r"(?i)\bnot\b|n't\b|\bcannot\b|\bnever\b|\bno\b|\bnone\b|\bnor\b|\bwithout\b")
QUANTIFIERS = {
    "universal": ["all", "every", "each", "any", "always"],
    "existential": ["some", "several", "a few", "many", "sometimes", "occasionally"],
    "restrictive": ["only", "exactly", "solely"],
}
CONDITIONS = ["as soon as", "in case of", "in case", "in the event of", "in the event that",
              "provided that", "on receipt of", "on reception of", "on reception", "on detection of",
              "on transmission of", "on indication", "whenever", "unless", "if", "when", "while",
              "during", "after", "before", "until", "upon", "once"]


@dataclass(frozen=True)
class Value:
    cmp: str | None
    num: float
    unit: str | None
    start: int
    end: int
    num_start: int
    num_end: int


def parse_number(s: str) -> float:
    s = s.strip().lower()
    if s in NUMBER_WORDS:
        return float(NUMBER_WORDS[s])
    return float(s.replace(",", ""))


def find_values(text: str) -> list[Value]:
    """Quantities that carry a comparator, a unit, or a count noun; identifiers excluded."""
    out = []
    for m in VALUE_RE.finditer(text):
        cmp_s, unit_s = m.group("cmp"), m.group("unit")
        gap_unit = None
        if not unit_s:
            # count noun up to two words later: "five failed (login) attempts"
            g = COUNT_GAP_RE.match(text, m.end("num"))
            if g:
                gap_unit = UNITS[g.group(1).lower()][0]
        if not cmp_s and not unit_s and not gap_unit:
            continue
        ns, ne = m.span("num")
        before, after = text[:ns], text[ne:ne + 2]
        if IDENT_BEFORE.search(before) or re.match(r"[-./]\d", after) or re.search(r"[-./\[]$", before):
            continue
        if re.fullmatch(r"(19|20)\d\d", m.group("num")) and not unit_s:
            continue  # a year, not a threshold
        unit = UNITS[unit_s][0] if unit_s else gap_unit
        cmp = _CMP_CANON.get(cmp_s.lower() if cmp_s and cmp_s.lower() in _CMP_CANON else cmp_s) if cmp_s else None
        start = m.start("cmp") if cmp_s else ns
        out.append(Value(cmp, parse_number(m.group("num")), unit, start, m.end(), ns, ne))
    return out


def _count_patterns(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


@dataclass
class Commitments:
    modality: Counter = field(default_factory=Counter)
    values: Counter = field(default_factory=Counter)
    negation: int = 0
    quant: Counter = field(default_factory=Counter)
    cond: Counter = field(default_factory=Counter)

    def diff(self, other: "Commitments") -> list[str]:
        """Names of commitment types that differ (empty list = preserved)."""
        return [k for k in ("modality", "values", "negation", "quant", "cond")
                if getattr(self, k) != getattr(other, k)]


def extract(text: str) -> Commitments:
    c = Commitments()
    for cls, pats in MODALS.items():
        n = _count_patterns(text, pats)
        if n:
            c.modality[cls] = n
    for v in find_values(text):
        c.values[(v.cmp, v.num, v.unit)] += 1
    # "cannot" is both a nonbinding modal and a negator; count negation once per token
    c.negation = len(NEGATION.findall(text))
    for cls, words in QUANTIFIERS.items():
        n = sum(len(re.findall(rf"(?i)\b{re.escape(w)}\b", text)) for w in words)
        if n:
            c.quant[cls] = n
    rest = text
    for marker in CONDITIONS:  # longest multi-word markers first, consumed once
        pat = rf"(?i)\b{re.escape(marker)}\b"
        n = len(re.findall(pat, rest))
        if n:
            c.cond[marker] = n
            rest = re.sub(pat, " ", rest)
    return c


def preserved(src: str, rev: str) -> bool:
    """Gate G5: accept iff normalised protected commitments are equal."""
    return not extract(src).diff(extract(rev))
