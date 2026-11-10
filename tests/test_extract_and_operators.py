"""Unit tests. Each expected value names its source: a standard, a definition in
the paper (Section 4, Table 1), or the dataset line the sentence was taken from."""
import difflib

import pytest

from snp.extract import extract, find_values, preserved
from snp.operators import apply

# Real requirements (SecReq, Zenodo 4530183, CC BY 4.0)
CPN5 = "The CNG shall implement an authentication failure handling policy."          # CPN.csv line 5
CPN8 = ("When the CNG detects replayed user and/or device credentials, the CNG shall stop "
        "the relevant processes.")                                                      # CPN.csv line 8
CPN33 = ("LAN interfaces. The CNG shall be equipped with at least one Ethernet (minimum 100 Mbit/s) "
         "is interface.")                                                              # CPN.csv line 33 (prefix)
EP2 = ("On-line authentication must take place between the card issuer and the CEP card for load, "
       "unload, and currency exchange transactions")                                   # ePurse line 2
CPN40 = ("The CNG should support redirection of IP Multimedia sessions, as defined by TS 181 005 [1], "
         "independently from the type of device used (NGN terminal or other CND).")    # CPN.csv line 40


# ------------------------------------------------------------------ extractor
def test_shall_vs_should_not_preserved():
    # ISO/IEC/IEEE 29148: "shall" = binding requirement, "should" = non-binding goal.
    assert not preserved("The system shall log every access.", "The system should log every access.")


def test_shall_vs_must_preserved():
    # RFC 2119 Sec. 1: MUST, REQUIRED and SHALL are synonyms for an absolute requirement.
    assert preserved("The system shall log every access.", "The system must log every access.")


def test_numeral_word_and_unit_spelling_preserved():
    # Definition of E1/E2 (paper Table 1): surface form of a number or unit is not a commitment.
    assert preserved("The page shall load within 2 s.", "The page shall load within two seconds.")
    assert preserved(CPN33, CPN33.replace("100 Mbit/s", "100 megabits per second"))


def test_values_with_comparators_and_units():
    # CPN line 33 contains exactly two quantitative commitments: ">= 1" and ">= 100 Mbit/s".
    got = sorted((v.cmp, v.num, v.unit) for v in find_values(CPN33))
    assert got == [(">=", 1.0, None), (">=", 100.0, "Mbit/s")]


def test_identifiers_are_not_values():
    # "TS 181 005 [1]" is a document identifier (CPN line 40), not a threshold.
    assert find_values(CPN40) == []
    assert find_values("in accordance with SPEC 10305-1 Draft 1 and ISO/IEC 7816-4") == []


def test_negation_and_condition_counts():
    # Definitions in extract.py: negators counted per token; condition markers per occurrence.
    c = extract("When the CNG detects a replay, the CNG shall not accept the message.")
    assert c.negation == 1 and c.cond["when"] == 1 and c.modality["binding"] == 1


# ------------------------------------------------------------------ drift operators (paper Table 1)
def _single_site(a, b):
    ops = [o for o in difflib.SequenceMatcher(None, a.split(), b.split()).get_opcodes() if o[0] != "equal"]
    return len(ops) == 1


def test_m1_weakens_modality():
    m = apply("M1", "t-cpn5", CPN5)
    assert m and ("should" in m.text or "may" in m.text) and "shall" not in m.text
    assert _single_site(CPN5, m.text) and not preserved(CPN5, m.text)


def test_m5_inserts_negation():
    m = apply("M5", "t-cpn5", CPN5)
    assert m.text == "The CNG shall not implement an authentication failure handling policy."


def test_m7_drops_condition_clause():
    # Removing the 'When ...' clause leaves the unconditional obligation.
    m = apply("M7", "t-cpn8", CPN8)
    assert m.text == "The CNG shall stop the relevant processes."
    assert not preserved(CPN8, m.text)


def test_m9_drops_one_conjunct():
    m = apply("M9", "t-ep2", EP2)
    kept = [w for w in ("load", "unload", "currency exchange") if w in m.text.split("for ", 1)[1]]
    assert m and len(kept) == 2, m.text


def test_m3_and_m4_change_value_or_unit():
    src = "The page shall load within 100 ms."
    m3, m4 = apply("M3", "t1", src), apply("M4", "t1", src)
    assert m3.text in ("The page shall load within 1000 ms.", "The page shall load at least 100 ms.")
    assert m4.text == "The page shall load within 100 s."
    assert not preserved(src, m3.text) and not preserved(src, m4.text)


def test_m2_makes_threshold_vague():
    src = "The system shall lock the account after five failed attempts."
    m = apply("M2", "t1", src)
    assert m.text == "The system shall lock the account after several failed attempts."
    assert not preserved(src, m.text)


# ------------------------------------------------------------------ equivalence operators
def test_e5_binding_synonym_preserved():
    m = apply("E5", "t-cpn5", CPN5)
    assert m.text == "The CNG must implement an authentication failure handling policy."
    assert preserved(CPN5, m.text)


def test_e6_condition_fronting_round_trip():
    m = apply("E6", "t-cpn8", CPN8)
    assert m.text == "The CNG shall stop the relevant processes when the CNG detects replayed user and/or device credentials."
    assert preserved(CPN8, m.text)


def test_operators_are_deterministic():
    for op in ("M1", "M5", "M7", "M9", "M10", "E4", "E6"):
        a, b = apply(op, "rid-x", CPN8), apply(op, "rid-x", CPN8)
        assert (a.text if a else None) == (b.text if b else None)


def test_e2_rate_units_not_pluralised_wrongly():
    # Regression (found in corpus inspection): "megabits per seconds" must not be produced.
    m = apply("E2", "t-cpn33", CPN33)
    assert "megabits per second" in m.text and "per seconds" not in m.text
    assert preserved(CPN33, m.text)
