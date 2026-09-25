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
    return r"(?<=[-/])" + letter + _ABBREV_END

# End of an abbreviation: consumes the trailing dot if there is one
_ABBREV_END = r"(?:\.|\b)"

ABBREV_EXPANSIONS = [
    (
        re.compile(r"\bWestpr" + _optional_suffix(["e", "u", "(?:ss?|ß)", "e", "n"]) + _ABBREV_END, re.IGNORECASE), 
        Expansion("westpreußen", iri="https://sws.geonames.org/2847618/")
        ),
    (
        re.compile(r"\bOstpr" + _optional_suffix(["e", "u", "(?:ss?|ß)", "e", "n"]) + _ABBREV_END, re.IGNORECASE), 
        Expansion("ostpreußen")
        ),
    (
        re.compile(r"\bF\.?F\.?M" + _ABBREV_END, re.IGNORECASE), 
        Expansion("frankfurt am main")
        ),
    (
        re.compile(r"\bMfr" + _ABBREV_END, re.IGNORECASE), 
        Expansion("mittelfranken")
        ),
    (
        re.compile(r"\bUnterfr" + _ABBREV_END, re.IGNORECASE), 
        Expansion("unterfranken")
        ),
    (
        re.compile(r"\b[JYI]ugs?l?\.?", re.IGNORECASE),
        Expansion("yugoslavia")
        ),
    (
        re.compile(r"\bC\.?S\.?R" + _ABBREV_END, re.IGNORECASE),
        Expansion("czechoslovakia", iri="https://sws.geonames.org/8505031")
        ),
    (
        re.compile(r"\bO\.?S\.", re.IGNORECASE),
        Expansion("oberschlesien")
        ),
    (
        re.compile(r"\bOpf" + _ABBREV_END, re.IGNORECASE),
        Expansion("oberpfalz")
        ),
    
    # Meyers gazetteer abbreviations ( https://www.familysearch.org/en/wiki/Abbreviation_Table_for_Meyers_Orts_und_Verkehrs_Lexikon_Des_Deutschen_Reichs ).
    # Compound names come before their parts
    # (e.g. Els.-Loth. before Els. and Loth., Sa.-A. before Sa.), and Hessen-N.
    # comes before the single letter suffix rules so it is not read as Neckar.
    (
        re.compile(r"\bEls(?:\.|a(?:ss|ß))?\s*-\s*Lothr?" + _ABBREV_END, re.IGNORECASE),
        Expansion("elsaß-lothringen")
        ),
    (
        re.compile(r"\bB\.\s*Lothr?" + _ABBREV_END, re.IGNORECASE),
        Expansion("bezirk lothringen")
        ),
    (
        re.compile(r"\bB\.\s*O\.?\s*Els" + _ABBREV_END, re.IGNORECASE),
        Expansion("bezirk oberelsaß")
        ),
    (
        re.compile(r"\bB\.\s*U\.?\s*Els" + _ABBREV_END, re.IGNORECASE),
        Expansion("bezirk unterelsaß")
        ),
    (
        re.compile(r"\bEls" + _ABBREV_END, re.IGNORECASE),
        Expansion("elsaß")
        ),
    (
        re.compile(r"\bLothr?" + _ABBREV_END, re.IGNORECASE),
        Expansion("lothringen")
        ),
    (
        re.compile(r"\bHessen\s*-\s*N" + _ABBREV_END, re.IGNORECASE),
        Expansion("hessen-nassau")
        ),
    (
        re.compile(r"\bMeckl(?:enb(?:urg)?)?\.?\s*-\s*Schw(?:er)?" + _ABBREV_END, re.IGNORECASE),
        Expansion("mecklenburg-schwerin")
        ),
    (
        re.compile(r"\bMeckl(?:enb(?:urg)?)?\.?\s*-\s*Str(?:el)?" + _ABBREV_END, re.IGNORECASE),
        Expansion("mecklenburg-strelitz")
        ),
    (
        re.compile(r"\bSchaumb(?:urg)?\.?\s*-\s*L" + _ABBREV_END, re.IGNORECASE),
        Expansion("schaumburg-lippe")
        ),
    (
        re.compile(r"\bSchlesw(?:ig)?\.?\s*-\s*Holst" + _ABBREV_END, re.IGNORECASE),
        Expansion("schleswig-holstein")
        ),
    (
        re.compile(r"\bSchwarzb(?:urg)?\.?\s*-\s*Rud" + _ABBREV_END, re.IGNORECASE),
        Expansion("schwarzburg-rudolstadt")
        ),
    (
        re.compile(r"\bSchwarzb(?:urg)?\.?\s*-\s*Sond" + _ABBREV_END, re.IGNORECASE),
        Expansion("schwarzburg-sondershausen")
        ),
    (
        re.compile(r"\bSa\.?\s*-\s*A" + _ABBREV_END, re.IGNORECASE),
        Expansion("sachsen-altenburg")
        ),
    (
        re.compile(r"\bSa\.?\s*-\s*C\.?\s*-\s*G" + _ABBREV_END, re.IGNORECASE),
        Expansion("sachsen-coburg-gotha")
        ),
    (
        re.compile(r"\bSa\.?\s*-\s*M" + _ABBREV_END, re.IGNORECASE),
        Expansion("sachsen-meiningen")
        ),
    (
        re.compile(r"\bSa\.?\s*-\s*W\.?\s*-\s*E" + _ABBREV_END, re.IGNORECASE),
        Expansion("sachsen-weimar-eisenach")
        ),
    (
        re.compile(r"\bSa\."),
        Expansion("sachsen")
        ),
    (
        re.compile(r"\bReu(?:ss|ß)\s+[aä]\.?\s*L" + _ABBREV_END, re.IGNORECASE),
        Expansion("reuß ältere linie")
        ),
    (
        re.compile(r"\bReu(?:ss|ß)\s+j\.?\s*L" + _ABBREV_END, re.IGNORECASE),
        Expansion("reuß jüngere linie")
        ),
    (
        re.compile(r"\bM\.?\s*-?\s*Franken\b", re.IGNORECASE),
        Expansion("mittelfranken")
        ),
    (
        re.compile(r"\bO\.?\s*-?\s*Franken\b", re.IGNORECASE),
        Expansion("oberfranken")
        ),
    (
        re.compile(r"\bU\.?\s*-?\s*Franken\b", re.IGNORECASE),
        Expansion("unterfranken")
        ),
    (
        re.compile(r"\bO\.?\s*-?\s*Pfalz\b", re.IGNORECASE),
        Expansion("oberpfalz")
        ),
    (
        re.compile(r"\bO\.?\s*-?\s*Bay" + _ABBREV_END, re.IGNORECASE),
        Expansion("oberbayern")
        ),
    (
        re.compile(r"\bN(?:ie)?d\.?\s*-?\s*Bay" + _ABBREV_END, re.IGNORECASE),
        Expansion("niederbayern")
        ),
    (
        re.compile(r"\bBay" + _ABBREV_END, re.IGNORECASE),
        Expansion("bayern")
        ),
    (
        re.compile(r"\bAnh" + _ABBREV_END, re.IGNORECASE),
        Expansion("anhalt")
        ),
    (
        re.compile(r"\bBrand(?:en)?b(?:g)?" + _ABBREV_END, re.IGNORECASE),
        Expansion("brandenburg")
        ),
    (
        re.compile(r"\bBraunschw" + _ABBREV_END, re.IGNORECASE),
        Expansion("braunschweig")
        ),
    (
        re.compile(r"\bHann" + _ABBREV_END, re.IGNORECASE),
        Expansion("hannover")
        ),
    (
        re.compile(r"\bNeum\.", re.IGNORECASE),
        Expansion("neumark")
        ),
    (
        re.compile(r"\bOldenb" + _ABBREV_END, re.IGNORECASE),
        Expansion("oldenburg")
        ),
    (
        re.compile(r"\bPomm" + _ABBREV_END, re.IGNORECASE),
        Expansion("pommern")
        ),
    (
        re.compile(r"\bPr\.", re.IGNORECASE),
        Expansion("preußen")
        ),
    (
        re.compile(r"\bRheinl" + _ABBREV_END, re.IGNORECASE),
        Expansion("rheinland")
        ),
    (
        re.compile(r"\bRu(?:ss|ß)(?:l(?:and)?)?" + _ABBREV_END, re.IGNORECASE),
        Expansion("russland")
        ),
    (
        re.compile(r"\bSchles" + _ABBREV_END, re.IGNORECASE),
        Expansion("schlesien")
        ),
    (
        re.compile(r"\bTh[uü]r\.", re.IGNORECASE),
        Expansion("thüringen")
        ),
    (
        re.compile(r"\bWestf" + _ABBREV_END, re.IGNORECASE),
        Expansion("westfalen")
        ),
    (
        re.compile(r"\bW[uü]rtt?" + _ABBREV_END, re.IGNORECASE),
        Expansion("württemberg")
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
        re.compile(r"\bProv" + _optional_suffix("ince") + _ABBREV_END, re.IGNORECASE),
        Expansion("Province")
    ),
]

def expand_abbreviations(text: str) -> tuple[str, Optional[str]]:
    """Replaces every abbreviation in `text` with its expansion.

    Rules are applied in the order of ABBREV_EXPANSIONS. If a rule matches the
    whole (stripped) text and has an IRI, that IRI is returned with the result,
    otherwise the returned IRI is None.
    """
    iri = None
    stripped = text.strip()
    for pattern, expansion in ABBREV_EXPANSIONS:
        if iri is None and expansion.iri is not None and pattern.fullmatch(stripped):
            iri = expansion.iri
        text = pattern.sub(expansion.expansion, text)
    return text, iri
