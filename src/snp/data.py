"""Loaders for QuRE (Zenodo 16049822) and SecReq (Zenodo 4530183), both CC BY 4.0.

Each loader returns a list of Requirement records with a stable id.
Cleaning rules (documented in the paper):
  * exact-duplicate texts (after whitespace normalisation) are merged;
  * QuRE: a requirement is 'defect' if ANY of its weak-word rows is labelled defect;
  * SecReq: rows whose label is not 'sec'/'nonsec' are dropped (one GPS row is 'xyz');
  * SecReq: fragments are dropped (text starting with '(' or with < 5 tokens),
    e.g. CPN "(e.g. VoIP phones, Video phones, STBs, etc)." -- these are
    continuation lines of the preceding requirement, not requirements.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Requirement:
    rid: str          # e.g. "qure-0017", "secreq-cpn-0042"
    corpus: str       # "QuRE" | "SecReq"
    source: str       # "QuRE" | "ePurse" | "CPN" | "GPS"
    label: str        # QuRE: "defect"|"ok"; SecReq: "sec"|"nonsec"
    text: str
    weak_words: tuple[str, ...] = ()   # QuRE only: flagged weak words (all rows)
    defect_words: tuple[str, ...] = () # QuRE only: weak words labelled defect


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def load_qure(path: Path = RAW / "QuRE.csv") -> list[Requirement]:
    groups: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = _norm(row["requirement"])
            g = groups.setdefault(t, {"ww": [], "dw": [], "first": int(row["id"])})
            g["ww"].append(row["weak_word"])
            if row["defect"].strip() == "defect":
                g["dw"].append(row["weak_word"])
    out = []
    for t, g in sorted(groups.items(), key=lambda kv: kv[1]["first"]):
        out.append(Requirement(
            rid=f"qure-{g['first']:04d}", corpus="QuRE", source="QuRE",
            label="defect" if g["dw"] else "ok", text=t,
            weak_words=tuple(dict.fromkeys(g["ww"])), defect_words=tuple(dict.fromkeys(g["dw"]))))
    return out


def _secreq_lines(path: Path) -> list[str]:
    raw = path.read_bytes().decode("cp1252", errors="replace")
    # Split on LF only: CPN contains byte 0x85 (cp1252 ellipsis), which
    # str.splitlines() would wrongly treat as NEL line break.
    return raw.split("\n")


def load_secreq(drop_fragments: bool = True) -> list[Requirement]:
    out: list[Requirement] = []
    seen: set[str] = set()   # dedupe across the whole corpus
    for source, fname in (("ePurse", "ePurse-selective.csv"), ("CPN", "CPN.csv"), ("GPS", "GPS.csv")):
        for i, line in enumerate(_secreq_lines(RAW / fname)):
            line = line.rstrip("\r")
            if ";" not in line:
                continue
            text, label = line.rsplit(";", 1)
            label = label.strip().lower()
            if label not in {"sec", "nonsec"}:
                continue
            text = _norm(text.strip().strip('"').replace('""', '"').strip('"'))
            if not text or text in seen:
                continue
            if drop_fragments and (text.startswith("(") or len(text.split()) < 5):
                continue
            seen.add(text)
            out.append(Requirement(rid=f"secreq-{source.lower()}-{i:04d}", corpus="SecReq",
                                   source=source, label=label, text=text))
    return out


def load_all() -> list[Requirement]:
    return load_qure() + load_secreq()
