"""
Matches full addresses against a reference list of concentration camps and
ghettos retrieved from wikidata (reference_data/wikidata_camps_and_ghettos.csv).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import pandas as pd

from modules.geo_db_search import ascii_normalize, german_normalize

DEFAULT_CAMPS_REFERENCE_PATH = Path("reference_data/wikidata_camps_and_ghettos.csv")

GEONAMES_INVALID_IDS = {
    "11862441", # Geonames no longer recognizes this id although it is on wikidata
}

KZ_REGEX = re.compile(
    r"\b(K\.?\s*Z\.?|Konzentrationslager|KonzLager|KonzLgr)\b",
    re.IGNORECASE
)

NAME_EXCLUDE_LIST = {
    "warschau" # This name alone is too ambiguous to match as a camp
}
                      
# wikidata's "higherClass" for every row of a given place is consistently
# either the concentration camp branch or the ghetto branch (a place can have
# rows in both, e.g. Theresienstadt was administered as both).
_HIGHER_CLASS_TAGS = {
    "http://www.wikidata.org/entity/Q328468": "concentration_camp",
    "http://www.wikidata.org/entity/Q2583015": "ghetto",
}


@dataclass(frozen=True)
class CampMatch:
    iri: str
    label: str
    wikidata_iri: str
    # "concentration_camp" and/or "ghetto", per _HIGHER_CLASS_TAGS.
    tags: frozenset[str] = frozenset()


class CampReferenceMatcher:
    """
    Matches full addresses against a reference list of concentration camps and
    ghettos retrieved from wikidata, preferring a linked geonames id over the
    bare wikidata IRI when one is available.
    """

    def __init__(self, csv_path: Path | str = DEFAULT_CAMPS_REFERENCE_PATH):
        self.csv_path = Path(csv_path)
        self._index: dict[str, CampMatch] = {}

    # Preferred languages for the human-readable label kept in CampMatch.label,
    # when a normalized key is shared by several label translations of the
    # same camp/ghetto.
    _LABEL_LANG_PRIORITY = {"de": 0, "en": 1}

    def initialize(self):
        df = pd.read_csv(self.csv_path, dtype=str)
        grouped = df.groupby("place")
        places = set(df["place"].unique())

        # "partOf" links a place to its parent (e.g. a sub-camp to its main
        # camp); only links to another place actually in this reference list
        # are usable, since there is no data for anything else.
        parent_of: dict[str, set[str]] = {}
        for place_iri, group in grouped:
            parents = {p for p in group["partOf"].dropna().unique() if p in places}
            if parents:
                parent_of[place_iri] = parents

        own_geoname: dict[str, str] = {}
        for place_iri, group in grouped:
            geoname_ids = group["geoname"].dropna()
            if len(geoname_ids) > 0:
                geoname_id = geoname_ids.iloc[0]
                if geoname_id in GEONAMES_INVALID_IDS:
                    print(f"Warning: ignoring invalid geonames id {geoname_id} for {place_iri}")
                else:
                    own_geoname[place_iri] = geoname_id
        resolved_geoname = self._propagate_geoname_ids(parent_of, own_geoname)

        ambiguous_keys = set()
        for place_iri, group in grouped:
            preferred_iri = self._preferred_iri(place_iri, resolved_geoname)
            tags = frozenset(
                _HIGHER_CLASS_TAGS[higher_class]
                for higher_class in group["higherClass"].dropna().unique()
                if higher_class in _HIGHER_CLASS_TAGS
            )
            labels = group[["label", "labelLang"]].dropna(subset=["label"])
            labels = labels.iloc[labels["labelLang"].map(lambda lang: self._LABEL_LANG_PRIORITY.get(lang, 99)).argsort(kind="stable")]
            for label in labels["label"]:
                if label.strip() == "":
                    continue
                match = CampMatch(iri=preferred_iri, label=label, wikidata_iri=place_iri, tags=tags)
                for key in self._normalized_keys(label):
                    if key in ambiguous_keys:
                        continue
                    existing = self._index.get(key)
                    if existing is not None:
                        if existing.wikidata_iri != place_iri:
                            if self._is_ancestor(place_iri, existing.wikidata_iri, parent_of):
                                # This place is the parent of the already-indexed
                                # one; the parent's entry takes precedence.
                                self._index[key] = match
                            elif self._is_ancestor(existing.wikidata_iri, place_iri, parent_of):
                                # The already-indexed place is the parent of this
                                # one; keep the parent's entry.
                                pass
                            else:
                                # Same label used by more than one unrelated
                                # camp/ghetto; too ambiguous to use for matching
                                # full addresses.
                                del self._index[key]
                                ambiguous_keys.add(key)
                        # else: keep the already-indexed, higher-priority label
                        continue
                    self._index[key] = match

    @staticmethod
    def _propagate_geoname_ids(parent_of: dict[str, set[str]], own_geoname: dict[str, str]) -> dict[str, str]:
        """
        Fill in missing geoname ids using the "partOf" hierarchy: a place
        with no geoname id of its own inherits its parent's, and a place
        whose own children unanimously agree on one geoname id inherits it
        too. A place left without a value (no own id, and either no
        relatives with one or relatives that disagree) stays unresolved
        rather than being guessed at.
        """
        children_of: dict[str, set[str]] = {}
        for child, parents in parent_of.items():
            for parent in parents:
                children_of.setdefault(parent, set()).add(child)

        resolved = dict(own_geoname)
        changed = True
        while changed:
            changed = False
            for parent, children in children_of.items():
                if resolved.get(parent) is not None:
                    continue
                child_values = {resolved[child] for child in children if resolved.get(child) is not None}
                if len(child_values) == 1:
                    resolved[parent] = next(iter(child_values))
                    changed = True
                if len(child_values) > 1:
                    print(f"Warning: conflicting geonames ids among children of {parent}: {child_values}")
                # else: no, or conflicting, values among children; leave unresolved.
            for child, parents in parent_of.items():
                if resolved.get(child) is not None:
                    continue
                parent_values = {resolved[parent] for parent in parents if resolved.get(parent) is not None}
                if len(parent_values) == 1:
                    resolved[child] = next(iter(parent_values))
                    changed = True
        return resolved

    @staticmethod
    def _is_ancestor(candidate_ancestor: str, place: str, parent_of: dict[str, set[str]]) -> bool:
        """Whether `candidate_ancestor` is `place`'s parent, grandparent, ..."""
        visited = set()
        to_visit = list(parent_of.get(place, ()))
        while to_visit:
            current = to_visit.pop()
            if current == candidate_ancestor:
                return True
            if current in visited:
                continue
            visited.add(current)
            to_visit.extend(parent_of.get(current, ()))
        return False

    @staticmethod
    def _preferred_iri(place_iri: str, resolved_geoname: dict[str, str]) -> str:
        geoname_id = resolved_geoname.get(place_iri)
        if geoname_id is not None:
            return f"https://sws.geonames.org/{geoname_id}"
        # TODO let's use the wikidata id as the fallback
        # Maybe the gnd iri when available should be included as an extra field
        #
        # gnd_ids = group["gnd"].dropna()
        # if len(gnd_ids) > 0:
        #     return f"https://d-nb.info/gnd/{gnd_ids.iloc[0]}"
        return place_iri

    @staticmethod
    def _normalized_keys(label: str) -> set[str]:
        keys = set((
            label,
            KZ_REGEX.sub("", label)
        ))
        keys = {
            new_key 
            for key in keys
            for new_key in (ascii_normalize(key), german_normalize(key))
            if new_key
        }
        keys = {key for key in keys if key not in NAME_EXCLUDE_LIST}
        return keys

    def match(self, full_address: Optional[str]) -> Optional[CampMatch]:
        if not full_address:
            return None
        for key in self._normalized_keys(full_address):
            match = self._index.get(key)
            if match is not None:
                return match
        return None
