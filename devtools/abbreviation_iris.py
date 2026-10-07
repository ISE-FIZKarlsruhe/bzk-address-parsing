"""
Sets the IRIs of the abbreviation expansions of modules/abbrev_list_expander.py.

For each expansion, runs an exact search (the "exact" phase of
TantivySearchIndex.search) for the expanded term. The abbreviations name
regions (states, provinces, historical regions and countries, mostly of the
German Reich, see the Meyers gazetteer), so only region-like candidates
(those that cannot be a City) are considered: the IRI is set when exactly one
remains. Expansions for which that is wrong or ambiguous are decided in
DECISIONS instead.

Usage (from the repository root):
    uv run python devtools/abbreviation_iris.py          # report only
    uv run python devtools/abbreviation_iris.py --write  # also rewrite abbrev_list_expander.py
"""
import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.abbrev_list_expander import ABBREV_EXPANSIONS
from modules.geo_db_search import TantivySearchIndex, _remove_stop_words, normalized_search_strings
from modules.pipeline.geographical_entity import GeographicalEntityType, GeographicalName
from modules.pipeline.storage.encoding_util import decode_from_dict

SEARCH_INDEX_PATH = ".geo_db_search_index/tantivy_index"
EXPANDER_PATH = Path("modules/abbrev_list_expander.py")
SEARCH_LIMIT = 200


class Decision(NamedTuple):
    iri: Optional[str]
    reason: str


# Expansions with no single obvious region-like candidate, or whose single
# region-like candidate is not the place the abbreviation stands for
DECISIONS = {
    "frankfurt am main": Decision(
        "https://sws.geonames.org/2925533",
        "F.F.M. names the city; the ADM3/ADM4 candidates are its municipality, with the same extent"),
    "mittelfranken": Decision(
        "https://sws.geonames.org/2870736",
        "Regierungsbezirk (A.ADM2) over the L.RGN of the same name, like the other Franconian/Bavarian regions"),
    "elsaß": Decision(
        "https://sws.geonames.org/3038033",
        "Alsace (A.ADM1H) over the Alsace L.RGN, and over Bas-Rhin/Haut-Rhin, its two departments"),
    "lothringen": Decision(
        "https://sws.geonames.org/2997551",
        "Lorraine (A.ADM1H) over the Lorraine L.RGN"),
    "mecklenburg-strelitz": Decision(
        None,
        "only candidate is Landkreis Mecklenburg-Strelitz (1994-2011), not the Grand Duchy the abbreviation names"),
    "sachsen": Decision(
        "https://sws.geonames.org/2842566",
        "Saxony (A.ADM1) over the Saxony L.RGN and the villages named Sachsen"),
    "bayern": Decision(
        "https://sws.geonames.org/2951839",
        "Bavaria (A.ADM1) over the village and the Ortsteil named Bayern"),
    "brandenburg": Decision(
        "https://sws.geonames.org/2945356",
        "Brandenburg (A.ADM1) over the Brandenburgia L.RGN and the villages named Brandenburg"),
    "braunschweig": Decision(
        "https://sws.geonames.org/2945023",
        "Regierungsbezirk Braunschweig (A.ADM2H), the closest to the former state, over the city and its "
        "municipality"),
    "hannover": Decision(
        "https://sws.geonames.org/2910829",
        "Regierungsbezirk Hannover (A.ADMD), the closest to the former province, over the city, its "
        "municipality and Region Hannover"),
    "neumark": Decision(
        None,
        "the historical region is not in the database; candidates are only villages and municipalities "
        "named Neumark"),
    "oldenburg": Decision(
        None,
        "the former state is not in the database; candidates are the city, its municipality and Landkreis "
        "Oldenburg, none of which is obvious"),
    "pommern": Decision(
        None,
        "Pomerania (L.RGN 3088388) has no country, so it is not searchable nor resolvable; the only "
        "region-like candidate is the municipality Pommern (Mosel)"),
    "schlesien": Decision(
        "https://sws.geonames.org/3066138",
        "Silesia (L.RGN) over the Silesian Voivodeship (A.ADM1), only a part of it"),
    "thüringen": Decision(
        "https://sws.geonames.org/2822542",
        "Thuringia (A.ADM1) over the Thuringia L.RGN and Thüringen in Austria"),
    "Main": Decision(
        None, "not a place but part of a name (e.g. Frankfurt/M.); only El Main (Algeria) is region-like"),
    "Neckar": Decision(None, "not a place but part of a name (e.g. Heilbronn/N.)"),
    "Province": Decision(None, "not a place"),
    "Saint": Decision(None, "not a place"),
}


class Candidate(NamedTuple):
    iri: str
    name: str
    classification: Optional[str]
    country: Optional[str]
    population: Optional[int]
    entity_types: tuple[str, ...]

    @property
    def is_region_like(self) -> bool:
        return "City" not in self.entity_types

    def __str__(self) -> str:
        return (
            f"{self.iri} {self.name!r} ({self.classification}, {self.country}, population {self.population}, "
            f"types {'/'.join(self.entity_types)})")


def exact_candidates(search_index : TantivySearchIndex, term : str) -> list[Candidate]:
    query_strings = normalized_search_strings(_remove_stop_words(term))
    candidates = {}
    for match in search_index._search_inner(limit=SEARCH_LIMIT, query_strings=query_strings, distance_threshold=0):
        entity = decode_from_dict(match.retrieved_data, GeographicalName).entity
        candidates.setdefault(entity.iri, Candidate(
            iri=entity.iri,
            name=entity.name,
            classification=entity.classification,
            country=entity.country.iso_code if entity.country else None,
            population=entity.population,
            entity_types=tuple(t.name for t in entity.possible_entity_types),
        ))
    return sorted(candidates.values(), key=lambda c: -(c.population or 0))


def decide(term : str, candidates : list[Candidate]) -> tuple[Optional[str], str]:
    """(iri, how it was decided) for an expansion."""
    if term in DECISIONS:
        decision = DECISIONS[term]
        return decision.iri, f"decided: {decision.reason}"
    region_like = [c for c in candidates if c.is_region_like]
    if len(region_like) == 1:
        return region_like[0].iri, "single region-like candidate"
    if len(region_like) == 0:
        return None, "no region-like candidate"
    return None, "UNDECIDED: several region-like candidates, add a decision to DECISIONS"


_EXPANSION_CALL_REGEX = re.compile(r'Expansion\("(?P<term>[^"]*)"(?:,\s*iri="[^"]*")?\)')


def rewrite_expander(iris : dict[str, Optional[str]]) -> None:
    def _replace(match : re.Match) -> str:
        term = match.group("term")
        iri = iris[term]
        return f'Expansion("{term}", iri="{iri}")' if iri else f'Expansion("{term}")'
    source = EXPANDER_PATH.read_text(encoding="utf-8")
    EXPANDER_PATH.write_text(_EXPANSION_CALL_REGEX.sub(_replace, source), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"rewrite {EXPANDER_PATH} with the IRIs found")
    args = parser.parse_args()

    search_index = TantivySearchIndex(SEARCH_INDEX_PATH)
    iris : dict[str, Optional[str]] = {}
    for _, expansion in ABBREV_EXPANSIONS:
        term = expansion.expansion
        if term in iris:
            continue
        candidates = exact_candidates(search_index, term)
        iris[term], how = decide(term, candidates)
        changed = "" if iris[term] == expansion.iri else f" (was {expansion.iri})"
        print(f"{term!r}: {iris[term]}{changed} -- {how}")
        for candidate in candidates:
            marker = "*" if candidate.iri == iris[term] else " "
            print(f"  {marker} {candidate}")
    if args.write:
        rewrite_expander(iris)
        print(f"Rewrote {EXPANDER_PATH}")


if __name__ == "__main__":
    main()
