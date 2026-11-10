"""Held-out operators H1-H4 (protocol v1.3). Sentences: SecReq CPN.csv (Zenodo 4530183, CC BY 4.0)."""
import ast
from pathlib import Path

from snp.heldout import TIME_BOUND, antonyms, apply

CPN5 = "The CNG shall implement an authentication failure handling policy."   # CPN.csv line 5


def test_module_is_independent_of_extractor():
    src = Path(__file__).resolve().parents[1] / "src/snp/heldout.py"
    mods = {n.module for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.ImportFrom)}
    assert not any(m and ("extract" in m or "operators" in m) for m in mods)


def test_h1_inserts_iso29148_loophole_after_modal():
    m = apply("H1", "t1", CPN5)
    assert m.subtype in ("if possible", "as appropriate", "as applicable")
    assert m.text == f"The CNG shall, {m.subtype}, implement an authentication failure handling policy."


def test_h2_effort_hedge():
    m = apply("H2", "t1", CPN5)
    assert m.text == f"The CNG shall {m.subtype} implement an authentication failure handling policy."


def test_h3_bound_becomes_average():
    m = apply("H3", "t1", "The ECU shall send the status message within 100 ms after ignition.")
    assert m.text == "The ECU shall send the status message on average within 100 ms after ignition."
    assert TIME_BOUND.search("no more than 5 seconds")
    assert apply("H3", "t1", CPN5) is None


def test_h4_wordnet_antonym():
    # WordNet 3.0: enable <-> disable (direct antonym)
    assert "disable" in antonyms("enable")
    m = apply("H4", "t1", "The system shall enable the diagnostic interface.")
    assert m.text == "The system shall disable the diagnostic interface."


def test_preconditions():
    assert apply("H1", "t", "The CNG should implement a policy.") is None      # non-binding
    assert apply("H2", "t", "The CNG shall not store passwords.") is None      # negated
    assert apply("H4", "t", "The system shall allow the user to log in.") is None  # xcomp: skip
