"""
Matches full addresses against a reference list of concentration camps and
ghettos retrieved from wikidata (reference_data/wikidata_camps_and_ghettos.csv).
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import pandas as pd

from modules.address_tagging import GHETTO_TERM_PATTERN
from modules.geo_db_search import ascii_normalize, german_normalize, similarity_and_distance
from modules.pipeline.linked_data import AddressProcessingData, BZKFieldName

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
    # GND (Gemeinsame Normdatei) iri, when wikidata links one for this place.
    # Kept as an extra field alongside `iri` rather than as a fallback for it,
    # since `iri` already falls back to the wikidata iri when no geonames id
    # is available.
    gnd_iri: Optional[str] = None


class CampReferenceMatcher:
    """
    Matches full addresses against a reference list of concentration camps and
    ghettos retrieved from wikidata, preferring a linked geonames id over the
    bare wikidata IRI when one is available.
    """

    def __init__(self, csv_path: Path | str = DEFAULT_CAMPS_REFERENCE_PATH):
        self.csv_path = Path(csv_path)
        self._index: dict[str, CampMatch] = {}
        self._keys_by_length: dict[int, list[str]] = defaultdict(list)

    # Preferred languages for the human-readable label kept in CampMatch.label,
    # when a normalized key is shared by several label translations of the
    # same camp/ghetto.
    _LABEL_LANG_PRIORITY = {"de": 0, "en": 1}

    # Bounds for the fuzzy fallback in match(): only tolerate a single-character
    # edit (typo/OCR error), and only on keys long enough that a one-character
    # difference is unlikely to turn one place into another unrelated one.
    # Kept deliberately conservative since a wrong camp/ghetto match is worse
    # than a missed one.
    _FUZZY_MAX_DISTANCE = 1
    _FUZZY_MIN_KEY_LENGTH = 6

    # Bound for the substring fallback in match(): only keys at least this
    # long are considered, so a short/generic reference name (e.g. an
    # abbreviation) doesn't spuriously match as a substring of an unrelated
    # address.
    _SUBSTRING_MIN_KEY_LENGTH = 6

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
            gnd_iri = self._gnd_iri(group)
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
                match = CampMatch(iri=preferred_iri, label=label, wikidata_iri=place_iri, tags=tags, gnd_iri=gnd_iri)
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

        for key in self._index:
            self._keys_by_length[len(key)].append(key)

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
        return place_iri

    @staticmethod
    def _gnd_iri(group: pd.DataFrame) -> Optional[str]:
        gnd_ids = group["gnd"].dropna()
        if len(gnd_ids) > 0:
            return f"https://d-nb.info/gnd/{gnd_ids.iloc[0]}"
        return None

    @staticmethod
    def _normalized_keys(label: str) -> set[str]:
        keys = set((
            label,
            KZ_REGEX.sub("", label),
            GHETTO_TERM_PATTERN.sub("", label),
        ))
        keys = {
            new_key
            for key in keys
            for new_key in (ascii_normalize(key), german_normalize(key))
            if new_key
        }
        keys = {key for key in keys if key not in NAME_EXCLUDE_LIST}
        return keys

    @staticmethod
    def _is_acceptable(match: CampMatch, full_address: str) -> bool:
        """
        A ghetto is usually a specific district of an otherwise ordinary
        town, so its reference-list entry is often keyed on that town's bare
        name (e.g. "Litzmannstadt"/"Lodz"); matching that name alone is not
        enough evidence that the address is about the ghetto rather than an
        unrelated mention of the same town. Requiring the address to also use
        the word "Ghetto"/"Getto" itself corroborates the match. The word's
        position is ignored (and stripped from the indexed key, see
        `_normalized_keys`) since it appears inconsistently before or after
        the place name, both in the reference data and in input addresses.

        A place administered as both a concentration camp and a ghetto (e.g.
        Theresienstadt) is keyed under both tags; the concentration camp
        reading always takes precedence, since concentration camp names are
        their own and not shared with an ordinary place with unrelated
        mentions, unlike ghetto names.
        """
        if "concentration_camp" in match.tags:
            return True
        if "ghetto" not in match.tags:
            return True
        return GHETTO_TERM_PATTERN.search(full_address) is not None

    def match(self, address: AddressProcessingData, tags : list[str]) -> Optional[CampMatch]:
        if address.bzk_field_name.is_current_address():
            return None # People cannot currently reside in a concentration camp or ghetto
        
        # However there were people that were born in concentraion camps and ghettos and therefore that 
        # hypothesis should be considered as well. The current address is not relevant for this case.
        queries = [address.full_address]
        for entity in address.entities:
            if entity.entity_type == "City":
                queries.append(entity.raw_text)
                break
        for query in queries:
            if not query:
                continue
            keys = self._normalized_keys(query)
            for key in keys:
                match = self._index.get(key)
                if match is not None and self._is_acceptable(match, query):
                    return match
            # Exact match misses small typos/OCR errors, so fall back to a bounded
            # fuzzy match. The reference list is small enough (a few thousand
            # unique keys) that a length-bucketed scan is cheap, so this doesn't
            # need a dedicated search index the way GeoDBSearch's much larger
            # geonames data does.
            match = self._fuzzy_match(keys, query)
            if match is not None:
                return match
        
        if len(address.entities) == 0 or "concentration_camp" in tags or "ghetto" in tags:
            # substring match is expensive and therefore should only be used 
            # if there are no parsed entities that can be matched or if the address
            # is already tagged as a concentration camp or ghetto.
            
            # Full addresses often carry more than just the place name (e.g.
            # a street or a surrounding region), so also accept a reference
            # name appearing as a whole-word substring of the query.
            match = self._substring_match(keys, address.full_address)
            if match is not None:
                return match
        return None

    def _fuzzy_match(self, keys: set[str], full_address: str) -> Optional[CampMatch]:
        best_match = None
        best_similarity = -1.0
        for key in keys:
            if len(key) < self._FUZZY_MIN_KEY_LENGTH:
                continue
            for length in range(len(key) - self._FUZZY_MAX_DISTANCE, len(key) + self._FUZZY_MAX_DISTANCE + 1):
                for candidate_key in self._keys_by_length.get(length, ()):
                    candidate_match = self._index[candidate_key]
                    if not self._is_acceptable(candidate_match, full_address):
                        continue
                    edit_distance, similarity = similarity_and_distance(
                        key, candidate_key, self._FUZZY_MAX_DISTANCE)
                    if edit_distance <= self._FUZZY_MAX_DISTANCE and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = candidate_match
        return best_match

    def _substring_match(self, keys: set[str], full_address: str) -> Optional[CampMatch]:
        best_match = None
        best_key_length = -1
        for normalized_query in keys:
            for index_key, candidate_match in self._index.items():
                if len(index_key) < self._SUBSTRING_MIN_KEY_LENGTH:
                    continue
                if len(index_key) <= best_key_length:
                    # Already found an equally-specific or more specific match.
                    continue
                if not self._is_acceptable(candidate_match, full_address):
                    continue
                if re.search(rf"\b{re.escape(index_key)}\b", normalized_query):
                    best_match = candidate_match
                    best_key_length = len(index_key)
        return best_match
