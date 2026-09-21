"""
Tags an address field with the thematic categories it mentions, instead of (or
alongside) a real place name: concentration camps, ghettos, displaced persons
camps, deportation, emigration, and "the location is missing/unknown" markers.

Tags are not mutually exclusive, and matching one does not necessarily mean
the field fails to name a location: "DP-Lager Foehrenwald" is tagged
"displaced_persons_camp", but "Foehrenwald" is still a place to be resolved
normally. "location_unspecified" is only added when, once every recognized
non-location word is stripped out, nothing recognizable as a place name is
left.

"concentration_camp" and "ghetto" are only tagged here for generic, unnamed
mentions (e.g. bare "KZ"); when a specific camp/ghetto is named (e.g. "KZ
Auschwitz"), modules.camp_search.CampReferenceMatcher resolves it and tags it
instead, using wikidata's own classification of the matched place.
"""

import re
from typing import Optional

# A leading "in"/"nach"/... or a trailing "gestorben"/... carries no location
# information on its own; stripped out before deciding whether anything
# resembling a place name remains.
_FILLER_WORD_PATTERN = re.compile(
    r"\b("
    r"in|im|an|am|bei|nach|zur?|zum|"
    r"der|die|das|dem|den|des|ein|eine|einem|einen|einer|"
    r"und|oder|or|of|the|to|at|"
    r"gestorben|verstorben|starb|tot|died"
    r")\b",
    re.IGNORECASE,
)
_PUNCTUATION_PATTERN = re.compile(r"[?!\-.,;:()\[\]\"']")

# Blank, or only punctuation/symbols (e.g. "?", "-"); or an explicit "unknown"
# word/abbreviation, in German or English.
UNKNOWN_PATTERN = re.compile(r"^[\s?!\-.,;:]*$|\b(unbekannt|unbek\.?|unknown)\b", re.IGNORECASE)
DEPORTATION_PATTERN = re.compile(r"\b(deportiert|deportation|deported)\b", re.IGNORECASE)
MISSING_PATTERN = re.compile(r"\b(vermisst|verschollen|missing)\b", re.IGNORECASE)
EMIGRATION_PATTERN = re.compile(r"\b(emigr(?:iert|ation|ated|ate)|ausgewandert)\b", re.IGNORECASE)
# Requires a "Lager"/"Camp" suffix so the "DP" abbreviation alone (which could
# be almost anything) is never enough to match on its own.
DISPLACED_PERSONS_CAMP_PATTERN = re.compile(
    r"\bD\.?P\.?[\s-]*(?:Lager|Camp)\b|\bDisplaced[\s-]?Persons?[\s-]?(?:Lager|Camp)\b",
    re.IGNORECASE,
)
CONCENTRATION_CAMP_TERM_PATTERN = re.compile(r"\bK\.?Z\.?\b|\bKonzentrationslager\b|\bConcentration\s*Camp\b", re.IGNORECASE)
GHETTO_TERM_PATTERN = re.compile(r"\bGh?etto\b", re.IGNORECASE)

# Tags whose presence is decided purely by pattern search, independent of
# whether a place name is also present in the same value.
_THEMATIC_PATTERNS: dict[str, re.Pattern] = {
    "unknown": UNKNOWN_PATTERN,
    "deportation": DEPORTATION_PATTERN,
    "missing": MISSING_PATTERN,
    "emigration": EMIGRATION_PATTERN,
    "displaced_persons_camp": DISPLACED_PERSONS_CAMP_PATTERN,
}

# Tags whose presence additionally requires that no place name is named
# alongside the generic term; see CONCENTRATION_CAMP_TERM_PATTERN docstring.
_GENERIC_ONLY_PATTERNS: dict[str, re.Pattern] = {
    "concentration_camp": CONCENTRATION_CAMP_TERM_PATTERN,
    "ghetto": GHETTO_TERM_PATTERN,
}


def tag_address(text: Optional[str]) -> frozenset[str]:
    """
    Return the set of thematic tags that apply to `text` (see module
    docstring for the full list and their semantics).
    """
    if not isinstance(text, str):
        return frozenset()

    tags = set()
    remaining = text
    # Every term is stripped out of `remaining` before punctuation is, since
    # some terms are themselves made of punctuation-joined letters (e.g.
    # "K.Z."): collapsing punctuation first would split them apart and make
    # the term's own pattern fail to match what is left over.
    for tag, pattern in _THEMATIC_PATTERNS.items():
        if pattern.search(text):
            tags.add(tag)
        remaining = pattern.sub(" ", remaining)

    present_generic_tags = [tag for tag, pattern in _GENERIC_ONLY_PATTERNS.items() if pattern.search(text)]
    for pattern in _GENERIC_ONLY_PATTERNS.values():
        remaining = pattern.sub(" ", remaining)

    remaining = _FILLER_WORD_PATTERN.sub(" ", remaining)
    remaining = _PUNCTUATION_PATTERN.sub(" ", remaining)

    if remaining.strip() == "":
        tags.add("location_unspecified")
        # Only tagged as generic once every other term/filler/punctuation is
        # accounted for and nothing resembling a place name is left.
        tags.update(present_generic_tags)
    return frozenset(tags)
