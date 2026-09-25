from typing import NamedTuple, Optional
import re


class Expansion(NamedTuple):
    expansion : str
    iri: Optional[str] = None

CARDINAL_DIRECTIONS = r"(North|South|East|West|S[uü]d|Ost|Nord|S[uü]d|West|Osten|Norden|S[uü]den|Westen)"

def _optional_suffix(pattern: str | list[str]) -> str:
    if isinstance(pattern, str):
        pattern = [pattern]
    sb = []
    for c in pattern:
        sb.append(f"(?:{c}")
    for c in pattern:
        sb.append(")?")
    return "".join(sb)

def _single_letter_name_suffix(letter: str) -> str:
    return r"\b(?<=\w\w\w\w+\W*)[-/]" + letter + r"\b"

ABBREV_EXPANSIONS = [
    (
        re.compile(r"\bWestpr" + _optional_suffix(["e", "u", "(?:ss?|ß)", "e", "n"]) + "\.?\b", re.IGNORECASE), 
        Expansion("westpreußen", iri="https://sws.geonames.org/2847618/")
        ),
    (
        re.compile(r"\bOstpr" + _optional_suffix(["e", "u", "(?:ss?|ß)", "e", "n"]) + "\.?\b", re.IGNORECASE), 
        Expansion("ostpreußen")
        ),
    (
        re.compile(r"\bF\.?F\.?M\.?\b", re.IGNORECASE), 
        Expansion("frankfurt am main")
        ),
    (
        re.compile(r"\bMfr\.?\b", re.IGNORECASE), 
        Expansion("mittelfranken")
        ),
    (
        re.compile(r"\bUnterfr\.?\b", re.IGNORECASE), 
        Expansion("unterfranken")
        ),
    (
        re.compile(r"\b[JYI]ugs?l?\.?", re.IGNORECASE),
        Expansion("yugoslavia")
        ),
    (
        re.compile(r"\bC\.?S\.?R\.?\b", re.IGNORECASE),
        Expansion("czechoslovakia", iri="https://sws.geonames.org/8505031")
        ),
    (
        re.compile(r"\bO\.?S\.\b", re.IGNORECASE),
        Expansion("oberschlesien")
        ),
    (
        re.compile(r"\bOpf\.?\b", re.IGNORECASE),
        Expansion("oberschlesien")
        ),

    (
        re.compile(_single_letter_name_suffix("M"), re.IGNORECASE),
        Expansion("Main")
        ),
    (
        re.compile(_single_letter_name_suffix("N"), re.IGNORECASE),
        Expansion("Neckar")
        ),
    (
        re.compile(r"\bProv" + _optional_suffix("ince") + r"\.?\b", re.IGNORECASE),
        Expansion("Province")
    ),
]