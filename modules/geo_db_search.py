"""
Classes related to matching extracted place names to place names on the database.
"""

from pathlib import Path
import pandas as pd
import modules.utils as utils
import duckdb
import modules.build_geonames_db as build_geonames_db
from typing import Collection, Iterable, NamedTuple, Optional, Literal, Callable, TYPE_CHECKING
import contextlib
import functools
import threading
import enum
import dataclasses
import textwrap
import warnings
import textwrap
import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from modules.pipeline.geographical_entity import Coordinates, CountryData, RegionGeometry, GeographicalBranch, GeographicalEntity, GeographicalEntityType, GeographicalName, GeonamesAdminCodes
from modules.pipeline.linked_data import MatchedEntity, MatchedName, RawEntity
from modules.pipeline.linking_steps import LinkingStep
import tantivy
from tqdm.auto import tqdm
import unicodedata
import re
import string
import os
import editdistpy
import sys
import dataclasses
import json
import hashlib
import logging
import pyarrow
import unidecode
import math
import cologne_phonetics
import uuid
import numpy as np
from sklearn.cluster import DBSCAN
from modules import abbrev_list_expander, phonetics_fuzzy_scoring
from abc import ABC, abstractmethod
from modules.pipeline.storage.encoding_util import decode_from_dict, encode_as_dict

# Common parent of the entity linking loggers (GeoDBSearch, TantivySearchIndex,
# CampReferenceMatcher, Disambiguator), which leave their own level unset so
# that setting this logger's level adjusts all of them at once. Only defaulted
# to INFO when no level was configured before this module was imported.
ENTITY_LINKING_LOGGER = logging.getLogger("entity_linking")
if ENTITY_LINKING_LOGGER.level == logging.NOTSET:
    ENTITY_LINKING_LOGGER.setLevel(logging.INFO)

_CASE_TRANSPOSE_COST = 0.1

def levenshtein_for_scoring(a : str, b : str, max_distance : int) -> int:
    """
    Computes the Levenshtein distance between two strings, with a custom cost for case transposition.
    If max_distance is provided, the computation will stop if the distance exceeds max_distance.
    """
    # wrapper method to correct logic
    cased_dist = editdistpy.levenshtein.distance(a, b, max_distance=max_distance)
    uncased_dist = editdistpy.levenshtein.distance(a.lower(), b.lower(), max_distance=max_distance)
    case_cost = (cased_dist - uncased_dist) * _CASE_TRANSPOSE_COST
    dist = int(uncased_dist + case_cost)
    if dist < 0:
        return max(len(a), len(b))
    return dist

def levenshtein(a : str, b : str, max_distance : int) -> int:
    """
    Computes the Levenshtein distance between two strings.
    If max_distance is provided, the computation will stop if the distance exceeds max_distance.
    """
    # wrapper method to correct logic
    dist = editdistpy.levenshtein.distance(a, b, max_distance=max_distance)
    if dist < 0:
        return max(len(a), len(b))
    return dist

def similarity_and_distance(a : str, b : str, max_distance : int, distance_function : Callable[[str, str, int], int] = levenshtein):
    if len(a) == 0 and len(b) == 0:
        return 0, 1.0
    edit_distance = distance_function(a, b, max_distance)
    similarity = 1 - (edit_distance / max(len(a), len(b)))
    return edit_distance, similarity

def similarity_and_distance_with_onset_penalty(
        a : str, b : str, max_distance : int,
        distance_function : Callable[[str, str, int], int] = levenshtein, onset_penalty : int = 1
    ):
    """
    Like similarity_and_distance, but adding onset_penalty to the edit
    distance when the strings differ in their first character, to prefer,
    among equally distant strings, those starting the same. Meant for
    phonetic keys, where the first character stands for the first sound
    (e.g. "Eechfeld" sounds closer to "Eschfeld" than to "Lechfeld").
    """
    if len(a) == 0 and len(b) == 0:
        return 0, 1.0
    edit_distance = distance_function(a, b, max_distance)
    if a[:1] != b[:1]:
        edit_distance += onset_penalty
    similarity = max(0.0, 1 - (edit_distance / max(len(a), len(b))))
    return edit_distance, similarity


# Regional and minority languages of Germany whose names are indexed for every
# entity, like german ones, since the addresses may use their exonyms (e.g. the
# bavarian "Beer Scheva" for Beersheba)
_GERMAN_REGIONAL_LANGUAGES = (
    "nds", # Low German
    "frs", # East Frisian Low Saxon
    "bar", # Bavarian
    "gsw", # Alemannic
    "als", # Alemannic (wikipedia code)
    "ksh", # Ripuarian (Kölsch)
    "pfl", # Palatine German
    "vmf", # Main-Franconian
    "sxu", # Upper Saxon
    "sli", # Lower Silesian
    "frr", # North Frisian
    "stq", # Saterland Frisian
    "hsb", # Upper Sorbian
    "dsb", # Lower Sorbian
    "wen", # Sorbian
)

_NAME_LANGUAGE_FILTER = f"""(
    isolanguage == '' OR
    isolanguage LIKE 'en%' OR
    isolanguage LIKE 'de%' OR
    isolanguage == 'ger' OR
    isolanguage IN ({", ".join(f"'{language}'" for language in _GERMAN_REGIONAL_LANGUAGES)}) OR
    isolanguage == 'abbr' OR
    list_bool_or([(isolanguage IN lang) FOR lang IN entity.country.iso_languages])
)"""

# Entities left out of the search index:
# - rivers and lakes (geonames stream, lake and reservoir features, typed
#   Region in the geo db), which addresses name as a qualifier of a place
#   (e.g. "Ulm/Donau", "Tutzing am Starnberger See") rather than as the place
#   itself, and are better located by the places named after them (see
#   build_regional_terms)
# - wikidata entities with a number in any of their names, which are
#   monuments misclassified as settlements (e.g. "Bodendenkmal in Hausen bei
#   Würzburg, #D-6-6026-0013", also named "Siedlung in ..."), cluttering the
#   search and the word statistics of the place names they mention. Geonames
#   entities are not filtered this way, since geonames lists postal codes
#   among the names of actual places.
_INDEX_EXCLUDED_ENTITIES_FILTER = """(
    coalesce(entity.classification, '') NOT LIKE 'H.STM%' AND
    coalesce(entity.classification, '') NOT LIKE 'H.LK%' AND
    coalesce(entity.classification, '') != 'H.RSV' AND
    NOT (
        entity.iri LIKE 'http://www.wikidata.org/entity/%' AND
        entity.iri IN (
            SELECT entity.iri FROM geo_db.geographical_names_with_entities
            WHERE entity.iri LIKE 'http://www.wikidata.org/entity/%' AND regexp_matches(name, '[0-9]')
        )
    )
)"""

POP_LANGUAGE_FILTERED_NAMES_SELECT = f"""
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE {_NAME_LANGUAGE_FILTER} AND len(entity.possible_entity_types) > 0 AND {_INDEX_EXCLUDED_ENTITIES_FILTER}
"""

OTHER_LANGUAGE_FILTERED_NAMES_SELECT = f"""
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE {_NAME_LANGUAGE_FILTER} AND len(entity.possible_entity_types) = 0
"""

def preferred_name_by_entity_iri_select(schema : str = "") -> str:
    """
    Query selecting, as from geographical_names_with_entities, the preferred
    name of the entity with the iri given as parameter (the first name, if
    none is preferred), from the tables of `schema` (e.g. "geo_db." for an
    attached geo duckdb). Selecting from the view itself with a filter on
    entity.iri would scan the whole of its join, duckdb not pushing the filter
    down into it; filtering both sides of the join by iri is about ten times
    faster, the names table holding no index on iri alone.
    """
    return f"""
        SELECT n.* EXCLUDE (iri), e AS entity
        FROM (SELECT * FROM {schema}geographical_names WHERE iri = $1) n
        INNER JOIN (SELECT * FROM {schema}geographical_entities_with_countries WHERE iri = $1) e USING (iri)
        ORDER BY is_preferred_name DESC NULLS LAST, name_id LIMIT 1
    """

# match periods follwoing an isolated letter
_STRIP_PERIODS_ABBREV_REGEX = re.compile(r'((?<=\W\w)|(?<=^\w))\.')
_STRIP_PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
_DEDUPE_WHITESPACE_REGEX = re.compile(r'\s+')
# Punctuation that still separates words in phonetic keys, left for
# ascii_normalize to replace by spaces (e.g. "Beer-Sheva", "Frankfurt/Main",
# "St.Gallen"); any other punctuation is removed before phonetic encoding
_PHONETIC_WORD_SEPARATORS = frozenset("./-")

class _PhoneticPunctuationRemovalTable(dict):
    r"""
    str.translate table removing punctuation other than _PHONETIC_WORD_SEPARATORS,
    so that e.g. apostrophes marking glottal stops in transliterations do not split
    a word in two ("Be'er Sheva`" -> "Beer Sheva" rather than "Be er Sheva").
    A character counts as punctuation when its ascii transliteration consists only
    of ascii punctuation. This covers non ascii punctuation (e.g. "’", or "–" which
    is kept as a separator), as well as modifier letters that \w considers letters
    but unidecode turns into apostrophes (e.g. "ʾ", "ʿ" in "Biʾr as-Sabʿ").
    Each character is classified on its first lookup and cached.
    """
    def __missing__(self, codepoint : int) -> Optional[int]:
        transliteration = unidecode.unidecode(chr(codepoint))
        is_punctuation = transliteration != "" and all(c in string.punctuation for c in transliteration)
        if is_punctuation and not all(c in _PHONETIC_WORD_SEPARATORS for c in transliteration):
            value = None
        else:
            value = codepoint
        self[codepoint] = value
        return value

_PHONETIC_PUNCTUATION_REMOVAL_TABLE = _PhoneticPunctuationRemovalTable()

# Most addresses are in german, so german stop words take priority, but a few
# common spanish and english ones are included too since some addresses use
# those languages instead.
_STOP_WORDS = frozenset((
    "der", "die", "das", "des", "dem", "den",
    "und", "in", "im", "am", "an", "auf", r"[ia]\.", "bei",
    "zu", "zum", "zur", "von", "vom", "nach", "fuer", "fur",
    "the", "and", "of", "at",
    "el", "la", "los", "las", "de", "del", "y", "en",
    "kreis", r"kr\.?", r"krs\.?", "prov", r"provinz\.?", "province", 
    r"regier[uü]ngsbezirk", "region", r"reg\W*bez\.?"
))

# A stop word is delimited by the start/end of the string or by non word
# characters other than periods (e.g. spaces, dashes as in "Frankfurt-am-Main"),
# which are consumed along with it. Periods are excluded so as not to break up
# abbreviations (e.g. "y" in "N.Y."). The boundaries are checked with
# lookarounds, which see the original string, so that adjacent stop words
# (e.g. "von der") are all matched even though the separator between them
# is consumed by the first.
_compiled_stop_words = [
    re.compile(rf"[^\w.]*(?<![\w.])(?:{w})(?![\w.])[^\w.]*", re.IGNORECASE) for w in _STOP_WORDS
]

def _remove_stop_words(normalized_string : str) -> str:
    """
    Removes stop words from a string, along with the non word characters around
    them, which are replaced by a single space. If every word is a stop word, the
    string is returned unchanged rather than reduced to an empty search key.
    """
    result = normalized_string
    for regex in _compiled_stop_words:
        result = regex.sub(" ", result)
    result = _DEDUPE_WHITESPACE_REGEX.sub(" ", result).strip()
    if not result:
        return normalized_string
    return result

def normalize_for_phonetics(nfc_string : str):
    result = _remove_stop_words(nfc_string)
    result = result.translate(_PHONETIC_PUNCTUATION_REMOVAL_TABLE)
    result = ascii_normalize(result, preserve_german_diacritics=True)
    return result

def ascii_normalize(nfc_string : str, preserve_german_diacritics : bool = False) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_string.lower()
    if not (preserve_german_diacritics and all(c in _GERMAN_DIACRITICS or c.isascii() for c in result)):
        # Converts non ascii characters to ascii equivalents, e.g. "é" -> "e"
        result = unidecode.unidecode(result)
    # strip periods to normalize abbreviations, e.g. "U.S.A." -> "USA"
    result = _STRIP_PERIODS_ABBREV_REGEX.sub('', result)
    # replace other punctuations with spaces eg. 
    result = _STRIP_PUNCTUATION_REGEX.sub(' ', result)
    result = _DEDUPE_WHITESPACE_REGEX.sub(' ', result)
    result = result.strip()
    return result

_GERMAN_NORMALIZATION_REPLACEMENTS = [
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("ß", "ss")
]

_GERMAN_DIACRITICS = set(unicodedata.normalize('NFC', "äöüßÄÖÜ"))

def normalize_for_scoring(name : str) -> str:
    """
    Normalize a name for calculating edit score for the purpose of disambiguation.
    This means that if there are only german diacritics (ä, ö, ü, ß), 
    the string is preserved. Otherwise, all diacritics are
    removed.

    Casing is also preserved.
    """
    nfc = unicodedata.normalize('NFC', name)
    nfc = _remove_stop_words(nfc)
    if all(c in _GERMAN_DIACRITICS or c.isascii() for c in nfc):
        return nfc
    return unidecode.unidecode_expect_nonascii(name)

# ensure the replacement keys are NFC normalized themselves
for i, (key, value) in enumerate(_GERMAN_NORMALIZATION_REPLACEMENTS):
    _GERMAN_NORMALIZATION_REPLACEMENTS[i] = (unicodedata.normalize("NFC", key), value)

def cologne_phonetic_normalize(nfc_string : str) -> str:
    normalized = nfc_string.lower()
    normalized = _remove_stop_words(nfc_string)
    encoded = cologne_phonetics.encode(normalized)
    return " ".join(b for a, b in encoded)

def phonetics_for_scoring(nfc_string : str) -> str:
    """
    Returns a phonetic key for a name, using the phonetics_fuzzy_scoring algorithm.
    This is used for matching against similarly normalized names in the database.
    """
    normalized = nfc_string.lower()
    encoded = phonetics_fuzzy_scoring.encode(normalized)
    return " ".join(b for a, b in encoded)

def german_normalize(nfc_string : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    Accent stripping takes into account german rules for conversion of umlauts and ß.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_string.lower()
    for old, new in _GERMAN_NORMALIZATION_REPLACEMENTS:
        result = result.replace(old, new)
    return ascii_normalize(result)

def normalized_search_strings(nfc_string : str) -> list[str]:
    """
    Produces (if differing) two ascii normalized versions of the input string,
    with stop words removed.
    One will use the german ascii normalization rules for umlauts and ß,
    the other will use the basic ascii normalization rules.
    """
    basic_normalization = ascii_normalize(nfc_string)
    german_normalization = german_normalize(nfc_string)
    if basic_normalization == german_normalization:
        return [basic_normalization]
    else:
        return [basic_normalization, german_normalization]
    
def abbreviation_pattern_to_regexes(part : str) -> str:
    """
    Converts an abbreviation pattern to a regex pattern.
    """
    prefixes = part.split(".")
    if len(prefixes) == 1:
        return None
    if all(len(p.strip()) <= 1 for p in prefixes):
        # Likely a standard abbreviation for exact match: e.g. "U.S.A.", "N.Y."
        return None
    ascii_regex_pattern = "[a-z]* ".join([ascii_normalize(x) for x in prefixes]).strip()
    german_regex_pattern = "[a-z]* ".join([german_normalize(x) for x in prefixes]).strip()
    if ascii_regex_pattern == german_regex_pattern:
        return ascii_regex_pattern
    else:
        return f"({ascii_regex_pattern})|({german_regex_pattern})"

# Min length of a word abbreviated in a query for it to pair, in a partial
# word match, with the words it is a prefix of (see abbreviated_words): a
# shorter one (e.g. "b." for "bei", "St." for "Sankt") is the prefix of too
# many unrelated words
MIN_ABBREVIATED_WORD_LENGTH = 3

def abbreviated_words(part : str) -> frozenset[str]:
    """
    The words of `part` written abbreviated, i.e. directly followed by a
    period (e.g. "bergstr" in "Weinheim/Bergstr."), normalized both ways as
    in normalized_search_strings: the words abbreviation_pattern_to_regexes
    lets any letters follow. Standard abbreviations made up only of initials
    (e.g. "U.S.A.") and words shorter than MIN_ABBREVIATED_WORD_LENGTH are
    left out.
    """
    prefixes = part.split(".")
    if len(prefixes) == 1 or all(len(p.strip()) <= 1 for p in prefixes):
        return frozenset()
    words = set()
    for prefix in prefixes[:-1]:
        for normalize in (ascii_normalize, german_normalize):
            normalized_words = normalize(prefix).split(" ")
            if len(normalized_words[-1]) >= MIN_ABBREVIATED_WORD_LENGTH:
                words.add(normalized_words[-1])
    return frozenset(words)

def is_entity_type_ruled_out(
        entity_type : GeographicalEntityType, match_entity_types : Collection[GeographicalEntityType]
    ) -> bool:
    """
    Whether a match whose entity may only be of match_entity_types can be no
    entity of entity_type: a City or Neighborhood is only matched by a City
    or Neighborhood, never by a coarser entity (e.g. "Donau" the river for
    the city "Ulm/Donau"). Such a match is still kept by GeoDBSearch, where
    it only may not settle the search, so that the Disambiguator can still
    make use of it (see Disambiguator._split_entity) before pruning it.
    """
    return (
        entity_type in (GeographicalEntityType.City, GeographicalEntityType.Neighborhood) and
        not any(t in match_entity_types for t in (GeographicalEntityType.City, GeographicalEntityType.Neighborhood))
    )

# Mean earth radius (IUGG), for the great-circle distances below
EARTH_RADIUS_KM = 6371.0088

def geodesic_distance_km(a : Coordinates, b : Coordinates) -> float:
    """
    Great-circle distance in km between two points, by the haversine formula
    on a sphere of the earth's mean radius (within 0.5% of the distance on
    the WGS84 ellipsoid, plenty for scoring regional proximity).
    """
    lat1, lon1, lat2, lon2 = map(math.radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))

def cluster_coordinates(
        points : Collection[Coordinates], max_distance_km : float
    ) -> tuple[tuple[Coordinates, ...], ...]:
    """
    Clusters the points by merging any two within max_distance_km of each
    other (single linkage, so a chain of nearby points ends up in one
    cluster, e.g. the places along a river), and returns the points of every
    cluster, largest cluster first.
    """
    points = list(points)
    if len(points) == 0:
        return ()
    radians = np.radians([[p.latitude, p.longitude] for p in points])
    labels = DBSCAN(
        eps=max_distance_km / EARTH_RADIUS_KM, min_samples=1, metric="haversine", algorithm="ball_tree"
    ).fit_predict(radians)
    clusters = defaultdict(list)
    for point, label in zip(points, labels):
        clusters[label].append(point)
    return tuple(tuple(cluster) for cluster in sorted(clusters.values(), key=len, reverse=True))

def distance_to_geometry_km(point : Coordinates, geometry : RegionGeometry) -> float:
    """Great-circle distance in km from the point to the closest point of the geometry."""
    lat, lon = geometry.points_radians.T
    lat1, lon1 = math.radians(point.latitude), math.radians(point.longitude)
    h = np.sin((lat - lat1) / 2) ** 2 + math.cos(lat1) * np.cos(lat) * np.sin((lon - lon1) / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, h))).min())

def geometries_distance_km(a : RegionGeometry, b : RegionGeometry) -> float:
    """Great-circle distance in km between the closest points of the two geometries."""
    if len(a.points_radians) > len(b.points_radians):
        a, b = b, a
    return min(
        distance_to_geometry_km(Coordinates(math.degrees(lat), math.degrees(lon)), b)
        for lat, lon in a.points_radians)

_NAME_SCORING_LOGGER = ENTITY_LINKING_LOGGER.getChild("NameScoring")

class NameSimilarity(NamedTuple):
    edit_distance : int
    fuzzy_score : float
    phonetic_score : float

# Letters an abbreviated word must share with the start of a word for an
# abbreviation match to get a full fuzzy score (see abbreviation_similarity)
ABBREVIATION_FULL_SCORE_LETTERS = 5

# Queries with fewer letters than this are scored with the strict edit
# distance of normalize_for_scoring, which keeps punctuation and casing (see
# score_name_similarity): for them, a single character carries a lot of the
# name's identity
STRICT_SCORING_MAX_QUERY_LETTERS = 5

# A period directly followed by a word character (e.g. in "a.Lech")
_ABBREVIATION_PERIOD_REGEX = re.compile(r"\.(?=\w)")

def _common_prefix_length(a : str, b : str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length

def _abbreviated_word_similarity(query_word : str, alt_name_word : str) -> float:
    if query_word == alt_name_word:
        return 1.0
    shared_letters = _common_prefix_length(query_word, alt_name_word)
    return min(shared_letters, ABBREVIATION_FULL_SCORE_LETTERS) / ABBREVIATION_FULL_SCORE_LETTERS

def abbreviation_similarity(nfc_query : str, nfc_alt_name : str) -> float:
    """
    Fuzzy score of an abbreviation match (e.g. "Rum." for "Rumänien"): the
    number of letters each query word shares with the start of the name's
    word at the same position, out of ABBREVIATION_FULL_SCORE_LETTERS (more
    count as all of them), and 1 for a word written in full. Averaged over
    the words, or taken over the whole names when they differ in word count.
    The names are normalized as for search (see normalized_search_strings),
    keeping the best of the basic and german ascii normalizations, after
    separating the words joined by a period (e.g. "a.Lech"), which the
    abbreviation pattern splits too (see abbreviation_pattern_to_regexes).
    """
    nfc_query = _ABBREVIATION_PERIOD_REGEX.sub(". ", nfc_query)
    nfc_alt_name = _ABBREVIATION_PERIOD_REGEX.sub(". ", nfc_alt_name)
    best_similarity = 0.0
    for normalize in (ascii_normalize, german_normalize):
        query_words = normalize(_remove_stop_words(nfc_query)).split(" ")
        alt_name_words = normalize(_remove_stop_words(nfc_alt_name)).split(" ")
        if len(query_words) != len(alt_name_words):
            query_words, alt_name_words = [" ".join(query_words)], [" ".join(alt_name_words)]
        similarity = sum(
            _abbreviated_word_similarity(query_word, alt_name_word)
            for query_word, alt_name_word in zip(query_words, alt_name_words)
        ) / len(query_words)
        best_similarity = max(best_similarity, similarity)
    return best_similarity

def search_normalized_similarity_and_distance(nfc_query : str, nfc_alt_name : str) -> tuple[int, float]:
    """
    Edit distance and similarity between the names normalized as for search
    (see normalized_search_strings), keeping whichever of the basic and
    german ascii normalizations yields the higher similarity.
    """
    best = None
    for normalize in (ascii_normalize, german_normalize):
        edit_distance, similarity = similarity_and_distance(
            normalize(_remove_stop_words(nfc_query)), normalize(_remove_stop_words(nfc_alt_name)), 10)
        if best is None or similarity > best[1]:
            best = (edit_distance, similarity)
    return best

def score_name_similarity(
        nfc_query : str, nfc_alt_name : str, logger : logging.Logger = _NAME_SCORING_LOGGER,
        is_abbreviation_match : bool = False
    ) -> NameSimilarity:
    """
    Similarity between a query and a retrieved name, as used to score search
    matches (see GeoDBSearch._parse_data) and to rescore them against part of
    the query (see Disambiguator._split_entity): the edit distance and fuzzy
    score between the names, and the phonetic score between their phonetic
    keys. Both names have the abbreviations of the expansion list expanded
    first (see abbrev_list_expander). The fuzzy score is that of
    abbreviation_similarity for an abbreviation match, and otherwise the
    similarity by edit distance: between the names normalized for scoring
    (see normalize_for_scoring) for a query of fewer than
    STRICT_SCORING_MAX_QUERY_LETTERS letters, or else between the names
    normalized as for search (see search_normalized_similarity_and_distance).
    """
    nfc_query, _ = abbrev_list_expander.expand_abbreviations(nfc_query)
    nfc_alt_name, _ = abbrev_list_expander.expand_abbreviations(nfc_alt_name)
    query_letters = sum(c.isalpha() for c in nfc_query)
    if is_abbreviation_match:
        edit_distance, _ = search_normalized_similarity_and_distance(nfc_query, nfc_alt_name)
        fuzzy_score = abbreviation_similarity(nfc_query, nfc_alt_name)
        method = "abbreviation prefix"
    elif query_letters < STRICT_SCORING_MAX_QUERY_LETTERS:
        edit_distance, fuzzy_score = similarity_and_distance(
            normalize_for_scoring(nfc_query), normalize_for_scoring(nfc_alt_name), 10,
            distance_function=levenshtein_for_scoring)
        method = "strict edit distance"
    else:
        edit_distance, fuzzy_score = search_normalized_similarity_and_distance(nfc_query, nfc_alt_name)
        method = "search normalized edit distance"
    logger.debug(
        "Computed edit distance %d and fuzzy score %.3f (%s) for query %r vs alt name %r",
        edit_distance, fuzzy_score, method, nfc_query, nfc_alt_name
    )
    query_phonetic_key = phonetics_for_scoring(normalize_for_phonetics(nfc_query))
    alt_name_phonetic_key = phonetics_for_scoring(normalize_for_phonetics(nfc_alt_name))
    phonetic_dist, phonetic_score = similarity_and_distance_with_onset_penalty(
        query_phonetic_key, alt_name_phonetic_key, 10)
    logger.debug(
        "Computed phonetic distance %d and score %.3f for query phonetic key %r vs alt name phonetic key %r",
        phonetic_dist, phonetic_score, query_phonetic_key, alt_name_phonetic_key
    )
    return NameSimilarity(edit_distance, fuzzy_score, phonetic_score)

# Max per-word edit distance tolerated when pairing a query word against a
# candidate word in the partial word match (character-level typo tolerance,
# e.g. "Frankfrut" vs "Frankfurt").
PARTIAL_MATCH_WORD_DISTANCE = 1

def pair_words(
        query_words : list[str], candidate_words : list[str], max_distance : int = PARTIAL_MATCH_WORD_DISTANCE,
        abbreviated_query_words : Collection[str] = frozenset()
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Greedily pairs up words between the two lists, closest edit distance
    first (within max_distance), each word used in at most one pair. A query
    word among abbreviated_query_words (see abbreviated_words) also pairs
    with the candidate words it is a prefix of (e.g. "bergstr" with
    "bergstrasse"), ranked as if at max_distance. Returns the (query word
    index, candidate word index) pairs and the indices of the words on either
    side left unpaired.
    """
    candidate_pairs = []
    for qi, qword in enumerate(query_words):
        for ci, cword in enumerate(candidate_words):
            distance = levenshtein(qword, cword, max_distance)
            if distance <= max_distance:
                candidate_pairs.append((distance, qi, ci))
            elif qword in abbreviated_query_words and cword.startswith(qword):
                candidate_pairs.append((max_distance, qi, ci))
    candidate_pairs.sort(key=lambda p: p[0])
    used_query, used_candidate = set(), set()
    pairs = []
    for _, qi, ci in candidate_pairs:
        if qi in used_query or ci in used_candidate:
            continue
        used_query.add(qi)
        used_candidate.add(ci)
        pairs.append((qi, ci))
    unmatched_query = [i for i in range(len(query_words)) if i not in used_query]
    unmatched_candidate = [i for i in range(len(candidate_words)) if i not in used_candidate]
    return pairs, unmatched_query, unmatched_candidate

_WORD_REGEX = re.compile(r"\w+")

def partial_match_query_span(nfc_query : str, matched_key : str) -> Optional[tuple[int, int]]:
    """
    The part of nfc_query, as a (start, end) character span, that a partial
    word match (see TantivySearchIndex._is_partial_word_match) shares with
    the name it retrieved, matched_key being that name's normalized search
    string: from the first to the last query word paired with one of the
    name's words (stop words aside). The query words are paired under both
    the basic and the german ascii normalization, as when searching (see
    normalized_search_strings), keeping whichever pairs more words, and
    abbreviated ones with the words they are a prefix of (see
    abbreviated_words). None if no query word pairs with the name.
    """
    tokens = list(_WORD_REGEX.finditer(nfc_query))
    abbreviated = abbreviated_words(nfc_query)
    candidate_words = [w for w in matched_key.split(" ") if w and w not in _STOP_WORDS]
    best_paired_tokens : list[int] = []
    for normalize in (ascii_normalize, german_normalize):
        token_indices, query_words = [], []
        for i, token in enumerate(tokens):
            word = normalize(token.group())
            if word and word not in _STOP_WORDS:
                token_indices.append(i)
                query_words.append(word)
        pairs, _, _ = pair_words(query_words, candidate_words, abbreviated_query_words=abbreviated)
        if len(pairs) > len(best_paired_tokens):
            best_paired_tokens = [token_indices[qi] for qi, _ in pairs]
    if len(best_paired_tokens) == 0:
        return None
    return tokens[min(best_paired_tokens)].start(), tokens[max(best_paired_tokens)].end()

# Countries whose place names regional terms are collected from (see
# build_regional_terms). The geometry of a term is estimated from the places
# it qualifies in these and their neighboring countries, since regions named
# in the addresses often extend across today's borders (e.g. "Ostpreussen",
# now in Poland, Russia and Lithuania).
REGIONAL_TERM_COUNTRIES = ("DE", "PL", "CZ", "IL")
# Prepositions introducing the qualifier of a place name (German, then
# Polish/Czech), e.g. "Neustadt an der Donau", "Kazimierz nad Wisłą",
# "Rožnov pod Radhoštěm", "Hradec u Kadaně", "Nové Město na Moravě"
_REGIONAL_TERM_PREPOSITIONS = (
    r"am|an\s+der|an\s+dem|an|in\s+der|in\s+dem|im|in|bei|b\.|ob\s+der|auf\s+der|auf\s+dem|"
    r"vor\s+der|vor\s+dem|unter\s+der|über|a\.\s*d\.|a\.|i\.\s*d\.|i\.|nad|pod|u|na"
)
# The qualifier of a place name: after a preposition, in parentheses or after
# a slash. Dashes are left out, since what follows them is mostly a generic
# suffix (e.g. "-Siedlung", "-Dorf") or a district's name.
_REGIONAL_TERM_PATTERNS = (
    re.compile(rf"^.+\s+(?:{_REGIONAL_TERM_PREPOSITIONS})\s+(?P<qualifier>[^()/]+)$"),
    re.compile(rf"^[^()]+?\s*\(\s*(?:(?:{_REGIONAL_TERM_PREPOSITIONS})\s+)?(?P<qualifier>[^()]+?)\s*\)\s*$"),
    re.compile(rf"^[^/()]+?\s*/\s*(?:(?:{_REGIONAL_TERM_PREPOSITIONS})\s+)?(?P<qualifier>[^/()]+?)\s*$"),
)
# Min number of distinct entities a term must qualify to be a regional term
REGIONAL_TERM_MIN_ENTITIES = 2
# Max distance between the places qualified by a term for them to be merged
# into the same cluster of its geometry
REGIONAL_TERM_CLUSTER_DISTANCE_KM = 50
# Grid (in degrees, ~5 km) the points of a geometry are snapped to and
# deduplicated on, to keep the geometry of widespread terms small
REGIONAL_TERM_GRID_DEGREES = 0.05
# Terms naming a populated place at least this large are left out: a city
# qualifying its surroundings (e.g. "Garching bei München") is a place to look
# up in its own right (e.g. "München-Pasing"), not only a region
REGIONAL_TERM_MAX_PLACE_POPULATION = 50_000
# Divisions whose center coordinates make up the geometry of a German state.
# Geonames' ADM2 (Regierungsbezirk) only exists in 4 of the 16 states, so
# ADM3 (Kreis) is used instead.
GERMAN_STATE_GEOMETRY_DIVISION = "A.ADM3"
# City states, which are cities first: their names are left out of the
# regional terms
_GERMAN_CITY_STATE_ADMIN1_CODES = ("03", "04", "16")  # Bremen, Hamburg, Berlin
# Entity types whose whole text may be matched as a regional term (see
# GeoDBSearch._regional_term_match)
REGIONAL_TERM_ENTITY_TYPES = (
    GeographicalEntityType.AboveCity, GeographicalEntityType.Region,
    GeographicalEntityType.State, GeographicalEntityType.District,
)
# MatchedName.matching_method of a match on a regional term
REGIONAL_TERM_MATCHING_METHOD = "regional_term"
# IRI prefix of the synthetic entities standing for a regional term without
# an entity of its own, which can thus never be linked
REGIONAL_TERM_IRI_PREFIX = "regional-term:"
# Bump to invalidate the cached regional terms after changing how they are built
_REGIONAL_TERMS_VERSION = 2


@dataclass(frozen=True)
class RegionalTerm:
    # The name matched on the term: a synthetic entity (see
    # REGIONAL_TERM_IRI_PREFIX) or, for a German state, the state itself,
    # either way carrying the term's geometry
    geographical_name : GeographicalName
    # Number of distinct entities the geometry was estimated from
    entity_count : int


class RegionalTerms:
    """
    Normalized regional terms (e.g. "donau", "bergstrasse", "hessen") mapped
    to the regions they name, located by the places they qualify (see
    build_regional_terms). Matched on instead of being looked up in the
    index: see GeoDBSearch._regional_term_match and
    TantivySearchIndex._partial_word_match.
    """
    def __init__(self, terms : dict[str, RegionalTerm]):
        self.terms = terms
        self._max_words = max((len(term.split(" ")) for term in terms), default=0)
        self._single_word_terms = sorted(term for term in terms if " " not in term)

    def __getitem__(self, term : str) -> RegionalTerm:
        return self.terms[term]

    def __contains__(self, term : str) -> bool:
        return term in self.terms

    def __len__(self) -> int:
        return len(self.terms)

    def _abbreviated_term(self, word : str) -> Optional[str]:
        """The regional term the abbreviated word stands for (e.g. "bergstr" for "bergstrasse"), the one qualifying the most entities if several."""
        candidates = [term for term in self._single_word_terms if term.startswith(word)]
        return max(candidates, key=lambda term: self.terms[term].entity_count, default=None)

    def find(self, query_string : str, abbreviated_query_words : Collection[str] = frozenset()) -> Optional[str]:
        """The regional term the whole query string is, if any."""
        if query_string in self.terms:
            return query_string
        if " " not in query_string and query_string in abbreviated_query_words:
            return self._abbreviated_term(query_string)
        return None

    def find_in_words(
            self, words : list[str], abbreviated_query_words : Collection[str] = frozenset()
        ) -> tuple[list[str], list[str]]:
        """
        The regional terms found among the words (longest first, left to
        right, each word in at most one term), and the words left over. Only
        looked for after the first word, which names the place itself (e.g.
        "steinbach" in "steinbach glan", though "steinbach" also qualifies
        other places), as terms are collected from qualifiers.
        """
        terms, remaining = [], words[:1]
        i = 1
        while i < len(words):
            for n in range(min(self._max_words, len(words) - i), 0, -1):
                term = self.find(" ".join(words[i:i + n]), abbreviated_query_words if n == 1 else frozenset())
                if term is not None:
                    terms.append(term)
                    i += n
                    break
            else:
                remaining.append(words[i])
                i += 1
        return terms, remaining


def _snap_to_grid(points : Iterable[Coordinates]) -> list[Coordinates]:
    """The points snapped to REGIONAL_TERM_GRID_DEGREES and deduplicated."""
    snapped = {
        (round(p.latitude / REGIONAL_TERM_GRID_DEGREES) * REGIONAL_TERM_GRID_DEGREES,
         round(p.longitude / REGIONAL_TERM_GRID_DEGREES) * REGIONAL_TERM_GRID_DEGREES): None
        for p in points
    }
    return [Coordinates(round(lat, 4), round(lon, 4)) for lat, lon in snapped]


def _regional_term_entry(
        display_name : str, points_by_iri : dict[str, tuple[Coordinates, str]], state_iri : Optional[str] = None
    ) -> dict:
    """The cacheable description of a regional term (see _regional_terms_from_entries)."""
    clusters = cluster_coordinates(_snap_to_grid(p for p, _ in points_by_iri.values()), REGIONAL_TERM_CLUSTER_DISTANCE_KM)
    countries = Counter(country for _, country in points_by_iri.values())
    return {
        "name": display_name,
        "clusters": [[[p.latitude, p.longitude] for p in cluster] for cluster in clusters],
        "countries": [country for country, _ in countries.most_common()],
        "entities": len(points_by_iri),
        "state_iri": state_iri,
    }


def build_regional_terms(connection : duckdb.DuckDBPyConnection, search_index : "TantivySearchIndex") -> dict[str, dict]:
    """
    Collects the regional terms of the indexed place names in
    REGIONAL_TERM_COUNTRIES: the informative words (see
    TantivySearchIndex.partial_match_idf_threshold) qualifying a place's name
    (see _REGIONAL_TERM_PATTERNS, e.g. "donau" in "Neustadt an der Donau",
    "bergstrasse" in "Heppenheim (Bergstraße)"), or the whole qualifier if
    made up of several words (e.g. "thuringer wald"), qualifying at least
    REGIONAL_TERM_MIN_ENTITIES distinct entities there. Each is mapped to a
    geometry estimated from the locations of the entities it qualifies, in
    REGIONAL_TERM_COUNTRIES and their neighboring countries, clustered (see
    cluster_coordinates). Also maps the names of every German
    state but the city states to its geometry, estimated from the centers of
    its divisions (see GERMAN_STATE_GEOMETRY_DIVISION), overriding a term of
    the same name. Returns the cacheable entries of the terms, keyed by their
    basic normalization (see normalized_search_strings), which every search
    tries.
    """
    logger = ENTITY_LINKING_LOGGER.getChild("RegionalTerms")
    countries_sql = ", ".join(f"'{country}'" for country in REGIONAL_TERM_COUNTRIES)
    neighboring_countries = sorted({
        neighbor
        for (neighbors,) in connection.execute(f"""
            SELECT DISTINCT entity.country.neighboring_countries_iso_codes FROM geo_db.geographical_names_with_entities
            WHERE entity.country.iso_code IN ({countries_sql}) AND entity.classification = 'A.PCLI'
        """).fetchall()
        for neighbor in neighbors
    } - set(REGIONAL_TERM_COUNTRIES))
    point_countries_sql = ", ".join(f"'{country}'" for country in (*REGIONAL_TERM_COUNTRIES, *neighboring_countries))
    rows = connection.execute(f"""
        SELECT DISTINCT name, entity.iri, entity.coordinates.latitude, entity.coordinates.longitude,
            entity.country.iso_code
        FROM ({POP_LANGUAGE_FILTERED_NAMES_SELECT})
        WHERE entity.country.iso_code IN ({point_countries_sql}) AND entity.coordinates.latitude IS NOT NULL
            AND regexp_matches(name, '[(/]|\\s\\S+\\s')
    """).fetchall()
    logger.info(
        "Collecting regional terms from names in %s, locating them in those and %s (%d names)",
        REGIONAL_TERM_COUNTRIES, neighboring_countries, len(rows))
    # term -> iri -> (coordinates, country), in every country; term -> iris
    # in REGIONAL_TERM_COUNTRIES; term -> raw spellings
    qualified : dict[str, dict[str, tuple[Coordinates, str]]] = defaultdict(dict)
    qualified_in_countries : dict[str, set[str]] = defaultdict(set)
    spellings : dict[str, Counter] = defaultdict(Counter)
    for name, iri, latitude, longitude, country in rows:
        name = unicodedata.normalize("NFC", name).strip()
        for pattern in _REGIONAL_TERM_PATTERNS:
            match = pattern.match(name)
            if match is None:
                continue
            qualifier = match.group("qualifier").strip()
            raw_words = [w for w in _WORD_REGEX.findall(qualifier) if not any(c.isdigit() for c in w)]
            terms = []
            for raw_word in raw_words:
                word = ascii_normalize(raw_word)
                if len(word) >= 3 and word not in _STOP_WORDS:
                    terms.append((word, raw_word))
            if len(terms) > 1:
                terms.append((" ".join(word for word, _ in terms), " ".join(raw for _, raw in terms)))
            for term, spelling in terms:
                qualified[term][iri] = (Coordinates(latitude, longitude), country)
                if country in REGIONAL_TERM_COUNTRIES:
                    qualified_in_countries[term].add(iri)
                    spellings[term][spelling] += 1
            break
    large_places = {
        ascii_normalize(name) for (name,) in connection.execute(f"""
            SELECT DISTINCT name FROM ({POP_LANGUAGE_FILTERED_NAMES_SELECT})
            WHERE entity.country.iso_code IN ({countries_sql}) AND entity.classification LIKE 'P.%'
                AND entity.population >= {REGIONAL_TERM_MAX_PLACE_POPULATION}
        """).fetchall()
    }
    searcher = search_index.index.searcher()
    threshold = search_index.partial_match_idf_threshold
    entries : dict[str, dict] = {}
    for term, points_by_iri in qualified.items():
        if len(qualified_in_countries[term]) < REGIONAL_TERM_MIN_ENTITIES:
            continue
        if term in large_places:
            logger.debug("Regional term %r left out: also the name of a large place", term)
            continue
        if not any(search_index._word_idf(searcher, word) > threshold for word in term.split(" ")):
            logger.debug("Regional term %r left out: no informative word", term)
            continue
        entries[term] = _regional_term_entry(spellings[term].most_common(1)[0][0], points_by_iri)
    logger.info("Collected %d regional terms qualifying place names", len(entries))
    # German states, located by the centers of their divisions
    division_points : dict[str, dict[str, tuple[Coordinates, str]]] = defaultdict(dict)
    for iri, admin1_code, latitude, longitude in connection.execute(f"""
            SELECT iri, admin_codes.admin1_code, coordinates.latitude, coordinates.longitude
            FROM geo_db.geographical_entities
            WHERE iso_country_code = 'DE' AND classification = '{GERMAN_STATE_GEOMETRY_DIVISION}'
                AND coordinates.latitude IS NOT NULL
        """).fetchall():
        division_points[admin1_code][iri] = (Coordinates(latitude, longitude), "DE")
    state_rows = connection.execute(f"""
        SELECT DISTINCT name, entity.iri, entity.name, entity.admin_codes.admin1_code
        FROM ({POP_LANGUAGE_FILTERED_NAMES_SELECT})
        WHERE entity.country.iso_code = 'DE' AND entity.classification = 'A.ADM1'
    """).fetchall()
    states = 0
    for name, iri, state_name, admin1_code in state_rows:
        if admin1_code in _GERMAN_CITY_STATE_ADMIN1_CODES or admin1_code not in division_points:
            continue
        entry = _regional_term_entry(state_name, division_points[admin1_code], state_iri=iri)
        for key in normalized_search_strings(_remove_stop_words(unicodedata.normalize("NFC", name))):
            if len(key) >= 3:
                entries[key] = entry
                states += 1
    logger.info("Added %d names of German states", states)
    return entries


def _regional_terms_from_entries(
        connection : duckdb.DuckDBPyConnection, entries : dict[str, dict]
    ) -> RegionalTerms:
    """The regional terms described by entries (see build_regional_terms)."""
    country_data = {}
    countries_sql = ", ".join(
        f"'{country}'" for country in sorted({c for entry in entries.values() for c in entry["countries"]}))
    for (country,) in connection.execute(f"""
            SELECT DISTINCT entity.country FROM geo_db.geographical_names_with_entities
            WHERE entity.country.iso_code IN ({countries_sql}) AND entity.classification = 'A.PCLI'
        """).fetchall():
        country_data[country["iso_code"]] = CountryData(**country)
    state_names : dict[str, GeographicalName] = {}
    terms = {}
    for term, entry in entries.items():
        geometry = RegionGeometry(
            clusters=tuple(tuple(Coordinates(lat, lon) for lat, lon in cluster) for cluster in entry["clusters"]),
            country_iso_codes=tuple(entry["countries"]),
        )
        if entry["state_iri"] is not None:
            state_iri = entry["state_iri"]
            if state_iri not in state_names:
                row = connection.execute(preferred_name_by_entity_iri_select("geo_db."), [state_iri]).fetchone()
                columns = [description[0] for description in connection.description]
                state_names[state_iri] = decode_from_dict(dict(zip(columns, row)), GeographicalName)
            state_name = state_names[state_iri]
            geographical_name = dataclasses.replace(
                state_name, entity=dataclasses.replace(state_name.entity, geometry=geometry))
        else:
            largest_cluster = geometry.clusters[0]
            # the term was found in REGIONAL_TERM_COUNTRIES, which come first
            countries = sorted(entry["countries"], key=lambda country: country not in REGIONAL_TERM_COUNTRIES)
            entity = GeographicalEntity(
                iri=f"{REGIONAL_TERM_IRI_PREFIX}{term}",
                provider=None,
                name=entry["name"],
                asciiname=term,
                classification="REGIONAL_TERM",
                possible_entity_types=(GeographicalEntityType.Region,),
                coordinates=Coordinates(
                    sum(p.latitude for p in largest_cluster) / len(largest_cluster),
                    sum(p.longitude for p in largest_cluster) / len(largest_cluster)),
                population=None,
                geonames_id=None,
                closest_geonames_id=None,
                country=country_data[countries[0]],
                alternate_iso_country_codes=tuple(countries[1:]),
                admin_codes=GeonamesAdminCodes(None, None, None, None, None),
                other_parent_iris=(),
                geometry=geometry,
            )
            geographical_name = GeographicalName(
                name_id=-1, name=entry["name"], entity=entity, is_preferred_name=True,
                is_short_name=None, is_colloquial=None, name_provider=None, isolanguage=None)
        terms[term] = RegionalTerm(geographical_name=geographical_name, entity_count=entry["entities"])
    return RegionalTerms(terms)


def is_regional_term_entity(entity : GeographicalEntity) -> bool:
    """Whether the entity is the synthetic one of a regional term (see RegionalTerm), which cannot be linked."""
    return entity.iri.startswith(REGIONAL_TERM_IRI_PREFIX)


class IndexSearchMatch(NamedTuple):
    score : float # score as returned by the the specific index search, meaning differs
    matched_key: str
    nfc_name: str
    levenshtein_distance: Optional[int] = None,
    retrieved_data : Optional[dict] = None
    metadata : Optional[dict] = None
    # True if this match was only found through the (lower priority) phonetic search key
    is_phonetic_match : bool = False
    # True if this match was found by allowing one word to be missing/added/
    # substituted (see TantivySearchIndex._partial_word_match)
    is_partial_word_match : bool = False
    # The name matched, when not retrieved from the index (retrieved_data is
    # then None), i.e. a regional term's (see RegionalTerms)
    geographical_name : Optional[GeographicalName] = None


# Names of the phases of TantivySearchIndex.search (see MatchedName.search_phase)
# which retrieve their matches in the same phase, told apart per match by
# _match_search_phase
PHONETIC_SEARCH_PHASE = "phonetic"
FUZZY_DISTANCE_1_SEARCH_PHASE = "fuzzy(distance=1)"


def _match_search_phase(phase : Optional[str], index_match : "IndexSearchMatch") -> Optional[str]:
    """
    The search phase of a single match of a phase of TantivySearchIndex.search:
    the phonetic phase also runs the edit distance 1 fuzzy query, so its
    matches are told apart by whether they came from the phonetic key (which
    takes precedence for a name retrieved by both, consistently with
    MatchedName.is_phonetic_match) or from the fuzzy query.
    """
    if phase is not None and phase.startswith(PHONETIC_SEARCH_PHASE):
        return PHONETIC_SEARCH_PHASE if index_match.is_phonetic_match else FUZZY_DISTANCE_1_SEARCH_PHASE
    return phase


class IndexSearchResult(NamedTuple):
    nfc_query : str
    query_strings : list[str]
    abbreviation_pattern : Optional[str]
    matches : list[IndexSearchMatch]
    # The phase of TantivySearchIndex.search that retrieved the matches
    phase : Optional[str] = None


def _describe_index_matches(matches : Iterable[IndexSearchMatch]) -> list[str]:
    """
    Compact, human-readable summary of index matches for debug logging. Only
    call this behind a logger.isEnabledFor(logging.DEBUG) check, since it
    walks every match.
    """
    descriptions = []
    for match in matches:
        entity = (match.retrieved_data or {}).get("entity") or {}
        country = (entity.get("country") or {}).get("iso_code")
        descriptions.append(
            f"{match.nfc_name!r} (key={match.matched_key!r}, iri={entity.get('iri')}, "
            f"country={country}, types={entity.get('possible_entity_types')}, score={match.score:.3f})"
        )
    return descriptions


class GeoSearchIndex(ABC):
    index_descriptor : str
    @abstractmethod
    def populate_index(self, row_retriever : Iterable[dict], skip_if_exists=True):
        pass
    @abstractmethod
    def search(
            self, 
            query_string, 
            distance_threshold : int, 
            similarity_threshold : float,
            limit : int = 10,
            expand_abbreviations : bool = True,
            entity_types : Optional[Collection[GeographicalEntityType]] = None,
            country_codes : Optional[Collection[str]] = None,
            strict_country_filtering : bool = False,
            admin_codes : Optional[Collection[GeonamesAdminCodes]] = None,
            accept_matches : Optional[Callable[[list["IndexSearchMatch"]], bool]] = None
        ) -> IndexSearchResult:
        pass

def _serialized_searcher_access(method):
    """
    Serializes calls to a TantivySearchIndex method using a tantivy searcher,
    which cannot be used from several threads at once, under the index's
    _searcher_lock. Everything else about a search (query building, scoring
    and the GeoDBSearch logic around it) runs concurrently.
    """
    @functools.wraps(method)
    def wrapper(self : "TantivySearchIndex", *args, **kwargs):
        with self._searcher_lock:
            return method(self, *args, **kwargs)
    return wrapper


class TantivySearchIndex(GeoSearchIndex):
    index_descriptor = "tantivy"
    logger = ENTITY_LINKING_LOGGER.getChild("TantivySearchIndex")

    # Reference words used to calibrate, from the index's own word statistics,
    # how rare a word must be to count as informative enough to support a
    # partial word match (see _compute_partial_match_idf_threshold): common
    # German place-name qualifiers, none of which should on their own be
    # treated as the "real" identifying word of a name, along with generic
    # words found trailing names that would otherwise make for region words
    # (e.g. "garden" in "Clover Garden"). River names (e.g. "Main") are left
    # out on purpose: they are regional terms (see build_regional_terms).
    _PARTIAL_MATCH_IDF_REFERENCE_WORDS = ("Alt", "Neu", "Bad", "Garden", "See")
    # see PARTIAL_MATCH_WORD_DISTANCE
    _PARTIAL_MATCH_WORD_DISTANCE = PARTIAL_MATCH_WORD_DISTANCE

    def __init__(
            self, index_path : str | Path, read_threads : int | Literal['auto'] = 'auto', write_threads : int = 8,
        ):
        if read_threads == 'auto':
            read_threads = (max(1, getattr(os, "process_cpu_count", lambda : None)() or os.cpu_count() or 8) * 3) // 2
        self.logger.info("Initialized TantivySearchIndex object with %d read threads", read_threads)
        self.index_path = Path(index_path)
        self.already_exists = self.index_path.exists()
        if not self.already_exists:
            self.index_path.mkdir(parents=True)
        self.read_threads = read_threads
        self.write_threads = write_threads
        # set by GeoDBSearch.initialize (see build_regional_terms)
        self.regional_terms : RegionalTerms = RegionalTerms({})
        self._open_index()

    def _open_index(self):
        """Opens the index on disk at index_path, along with what goes with it."""
        self.logger.info(
            "Opening tantivy index at %s (already exists: %s, read threads: %d, write threads: %d)",
            self.index_path, self.already_exists, self.read_threads, self.write_threads)
        self.schema = self.create_schema()
        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        self.index.config_reader(num_warmers=self.read_threads)
        # see _serialized_searcher_access
        self._searcher_lock = threading.Lock()
        self.index.register_tokenizer(
            "whitespace",
            tantivy.TextAnalyzerBuilder(
                tantivy.Tokenizer.whitespace()
            ).build()
        )

    def __getstate__(self):
        # The index object only refers to the index on disk, and neither it
        # nor the lock can be pickled (e.g. when sent to a worker process);
        # each copy reopens the index instead. A copy is only ever searched
        # from a single thread, hence a single warmer.
        state = self.__dict__.copy()
        for attribute in ("schema", "index", "_searcher_lock"):
            del state[attribute]
        state["read_threads"] = 1
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_index()

    def populate_index(self, row_retriever : Iterable[dict], skip_if_exists=True):
        if skip_if_exists and self.already_exists:
            self.logger.info("Index already exists at %s; skipping population", self.index_path)
            self.index.reload()
        else:
            self.logger.info("Populating index at %s", self.index_path)
            # Group all entities sharing the same (NFC normalized) name together, so that
            # a single document maps a name to every entity known under that name.
            entities_by_name : dict[str, list[dict]] = defaultdict(list)
            for row in row_retriever:
                nfc_name = unicodedata.normalize("NFC", row["name"])
                entities_by_name[nfc_name].append(row)
            with self.index.writer(num_threads=self.write_threads) as writer:
                for nfc_name, rows in tqdm(entities_by_name.items(), desc="Writing search index"):
                    doc = tantivy.Document()
                    doc.add_text("nfc_name", nfc_name)
                    search_keys = set(normalized_search_strings(nfc_name))
                    search_keys.update(normalized_search_strings(_remove_stop_words(nfc_name)))
                    for search_key in search_keys:
                        if search_key is None or search_key.strip() == "":
                            continue
                        doc.add_text("search_key", search_key)
                        doc.add_text("search_words", search_key)
                    phonetic_key = cologne_phonetic_normalize(normalize_for_phonetics(nfc_name))
                    if phonetic_key.strip() != "":
                        doc.add_text("phonetic_key", phonetic_key)
                    # blob with the complete data of every entity known under this name; not indexed
                    doc.add_bytes("name_data", json.dumps(rows).encode("utf-8"))

                    # extra fields for search restriction, aggregated over all entities sharing this name
                    for row in rows:
                        entity_types = row["entity"].get("possible_entity_types", [])
                        for entity_type in GeographicalEntityType:
                            if entity_type.entity_type in entity_types:
                                doc.add_boolean(entity_type.entity_type, True)
                        country_code = row["entity"]["country"]["iso_code"]
                        if country_code:
                            doc.add_text("country_code", country_code)
                        admin_codes = row["entity"].get("admin_codes", {})
                        for admin_level in range(1, 6):
                            admin_code = admin_codes.get(f"admin{admin_level}_code")
                            if admin_code:
                                doc.add_text(f"admin{admin_level}_code", admin_code)
                    writer.add_document(doc)
            self.index.reload()
            self.logger.info("Finished writing %d name documents to the index", len(entities_by_name))
        self.partial_match_idf_threshold = self._compute_partial_match_idf_threshold()
        self.logger.info(
            "Partial word match IDF threshold set to %.4f (from reference words %s)",
            self.partial_match_idf_threshold, self._PARTIAL_MATCH_IDF_REFERENCE_WORDS)

    def create_schema(self):
        schema_builder = tantivy.SchemaBuilder()
        text_field_options = dict( 
            # Use raw tokenizer; tantivy tokenizer is a full text search feature, 
            # we only need single term matching
            tokenizer_name = 'raw',
            index_option = 'basic'
        )
        # search keys
        schema_builder.add_text_field("search_key", stored=True, fast=True, **text_field_options)
        # cologne phonetic search key, lower priority fallback used when the regular search keys find nothing
        schema_builder.add_text_field("phonetic_key", stored=True, fast=True, **text_field_options)
        # search key words, split on whitespace into individual terms (unlike
        # search_key's raw/single-token indexing); used for the partial word
        # match fallback, both to retrieve candidates sharing a word with the
        # query and to look up per-word document frequency (see
        # _compute_partial_match_idf_threshold)
        schema_builder.add_text_field("search_words", tokenizer_name="whitespace", index_option="basic")
        # json blob with a list of the complete data of every entity sharing this name; not indexed
        schema_builder.add_bytes_field("name_data", stored=True, indexed=False)
        # nfc name, no char stripping
        schema_builder.add_text_field("nfc_name", stored=True, **text_field_options)

        # extra fields for search restriction
        for entity_type in GeographicalEntityType:
            if entity_type in (GeographicalEntityType.AboveCity, GeographicalEntityType.Unknown):
                continue
            schema_builder.add_boolean_field(entity_type.entity_type, fast=True, indexed=True)
        schema_builder.add_text_field("country_code", fast=True, **text_field_options)
        for code in ["admin1_code", "admin2_code", "admin3_code", "admin4_code", "admin5_code"]:
            schema_builder.add_text_field(code, fast=True, **text_field_options)
        return schema_builder.build()

    def _word_idf(self, searcher : tantivy.Searcher, word : str) -> float:
        """
        Inverse document frequency of `word` among search_words, i.e. how
        rare it is across indexed names: high for a word that identifies a
        specific place, low for a common qualifier shared by many names.
        """
        doc_freq = searcher.doc_freq("search_words", word)
        # A word absent from the index is at least as rare as the rarest word
        # actually present; treat it as maximally informative rather than
        # dividing by zero.
        doc_freq = max(doc_freq, 1)
        return math.log(searcher.num_docs / doc_freq)

    def _compute_partial_match_idf_threshold(self) -> float:
        """
        The minimum IDF a word must exceed to be treated as meaningful enough
        to support a partial word match. Set to the worst (highest) IDF among
        _PARTIAL_MATCH_IDF_REFERENCE_WORDS, computed from this index's own
        word frequencies rather than hardcoded, so that anything at least as
        common as the most rarely-shared of those reference qualifiers is
        still treated as a droppable filler.
        """
        searcher = self.index.searcher()
        return max(
            self._word_idf(searcher, ascii_normalize(word))
            for word in self._PARTIAL_MATCH_IDF_REFERENCE_WORDS
        )

    def _is_partial_word_match(
            self, searcher : tantivy.Searcher, query_words : list[str], candidate_words : list[str],
            abbreviated_query_words : Collection[str] = frozenset()
        ) -> bool:
        """
        Whether the candidate is a partial word match of the query.
        """
        if len(query_words) == 0 or len(candidate_words) == 0:
            self.logger.debug(
                "Partial word match %s vs %s rejected: empty word list", query_words, candidate_words)
            return False
        # a candidate whose words appear verbatim and in order in the query
        # is accepted regardless of how informative its words are (e.g.
        # "Neustadt" in "Neustadt Weinstrasse"), as long as it is not made
        # up only of stop words
        if any(w not in _STOP_WORDS for w in candidate_words) and \
                f" {' '.join(candidate_words)} " in f" {' '.join(query_words)} ":
            self.logger.debug(
                "Partial word match %s vs %s accepted: candidate fully contained in query",
                query_words, candidate_words)
            return True
        index_pairs, unmatched_query_idx, unmatched_candidate_idx = pair_words(
            query_words, candidate_words, self._PARTIAL_MATCH_WORD_DISTANCE,
            abbreviated_query_words=abbreviated_query_words)
        pairs = [(query_words[qi], candidate_words[ci]) for qi, ci in index_pairs]
        if len(pairs) == 0:
            # nothing matched at all: not even a partial match
            self.logger.debug(
                "Partial word match %s vs %s rejected: no words paired", query_words, candidate_words)
            return False
        # unpaired stop words carry no meaning and are ignored altogether
        unmatched_query_idx = [i for i in unmatched_query_idx if query_words[i] not in _STOP_WORDS]
        unmatched_candidate_idx = [
            i for i in unmatched_candidate_idx if candidate_words[i] not in _STOP_WORDS]
        unmatched_query = [query_words[i] for i in unmatched_query_idx]
        unmatched_candidate = [candidate_words[i] for i in unmatched_candidate_idx]
        # a candidate made up only of informative words, all of them found in
        # the query (in any order), is accepted regardless of how many query
        # words are left over or where they are (e.g. "Homburg" in
        # "Homburg Saarpfalz Kreis")
        if len(unmatched_candidate) == 0:
            candidate_content_words = [w for w in candidate_words if w not in _STOP_WORDS]
            if len(candidate_content_words) > 0 and all(
                    self._word_idf(searcher, w) > self.partial_match_idf_threshold
                    for w in candidate_content_words):
                self.logger.debug(
                    "Partial word match %s vs %s accepted: candidate of informative words fully "
                    "contained in query (unpaired query words: %s)",
                    query_words, candidate_words, unmatched_query)
                return True
        # IDF is taken on the candidate side, since the candidate word is
        # known to be in the index while the query word may be a typo.
        paired_idfs = [(cword, self._word_idf(searcher, cword)) for _, cword in pairs]
        # unpaired words all less informative than every informative paired
        # word are mere qualifiers of the shared name, however many or
        # wherever they are, so the rules below no longer apply
        informative_paired_idfs = [
            idf for _, idf in paired_idfs if idf > self.partial_match_idf_threshold]
        if len(informative_paired_idfs) > 0:
            unmatched_idfs = [
                (w, self._word_idf(searcher, w)) for w in unmatched_query + unmatched_candidate]
            min_paired_idf = min(informative_paired_idfs)
            if all(idf < min_paired_idf for _, idf in unmatched_idfs):
                self.logger.debug(
                    "Partial word match %s vs %s accepted: unpaired words (%s) all less informative "
                    "than the paired informative words (min idf %.4f)",
                    query_words, candidate_words,
                    ", ".join(f"{w!r} idf {idf:.4f}" for w, idf in unmatched_idfs), min_paired_idf)
                return True
        # at most one word may be missing/added/substituted on either side
        if len(unmatched_query) > 1 or len(unmatched_candidate) > 1:
            self.logger.debug(
                "Partial word match %s vs %s rejected: too many unpaired words (query: %s, candidate: %s)",
                query_words, candidate_words, unmatched_query, unmatched_candidate)
            return False
        # the shared words must include at least one informative word: names
        # sharing only a common qualifier (e.g. "Bad Homburg" vs "Bad Tölz")
        # are unrelated places.
        if all(idf <= self.partial_match_idf_threshold for _, idf in paired_idfs):
            self.logger.debug(
                "Partial word match %s vs %s rejected: only non-informative words paired "
                "(%s, threshold %.4f)",
                query_words, candidate_words,
                ", ".join(f"{w!r} idf {idf:.4f}" for w, idf in paired_idfs),
                self.partial_match_idf_threshold)
            return False
        # an informative word may only be dropped if it is the last word of
        # its name (e.g. "Frankfurt" vs "Frankfurt Oder"), not counting
        # trailing stop words. When a leading or middle word is dropped, the
        # names only share their trailing part (e.g. "Weilheim in Oberbayern"
        # vs "Tiefenbach Oberbayern"), which is no match for the place itself.
        for words, unmatched_idx in ((query_words, unmatched_query_idx), (candidate_words, unmatched_candidate_idx)):
            last_content_idx = max(
                (i for i, w in enumerate(words) if w not in _STOP_WORDS), default=len(words) - 1)
            for i in unmatched_idx:
                if i == last_content_idx:
                    continue
                idf = self._word_idf(searcher, words[i])
                if idf > self.partial_match_idf_threshold:
                    self.logger.debug(
                        "Partial word match %s vs %s rejected: unpaired non-final word %r is too "
                        "informative (idf %.4f > threshold %.4f)",
                        query_words, candidate_words, words[i], idf, self.partial_match_idf_threshold)
                    return False
        self.logger.debug(
            "Partial word match %s vs %s accepted (unpaired query words: %s, unpaired candidate words: %s)",
            query_words, candidate_words, unmatched_query, unmatched_candidate)
        return True

    @_serialized_searcher_access
    def _partial_word_match(
            self,
            query_strings : list[str],
            hints : list[tuple[tantivy.Occur, tantivy.Query]],
            limit : int,
            abbreviated_query_words : Collection[str] = frozenset(),
        ) -> list[IndexSearchMatch]:
        """
        Lowest-priority fallback: matches names that share most, but not
        necessarily all, of their words with the query (see _is_partial_word_match).
        The query words written abbreviated (abbreviated_query_words, see
        abbreviated_words) also match the words they are a prefix of (e.g.
        "bergstr" the word "bergstrasse").
        The query words making up a regional term (see RegionalTerms, e.g.
        "donau" in "Ulm/Donau") are not looked up: the regional term itself is
        matched on them instead, and the other words are matched without them.
        """
        searcher = self.index.searcher()
        matches = []
        remaining_query_strings = []
        matched_terms = {}
        for query_string in query_strings:
            query_words = [w for w in query_string.split(" ") if w]
            terms, remaining_words = self.regional_terms.find_in_words(query_words, abbreviated_query_words)
            for term in terms:
                matched_terms.setdefault(term, None)
            if len(remaining_words) > 0:
                remaining_query_strings.append(" ".join(remaining_words))
        for term in matched_terms:
            regional_term = self.regional_terms[term]
            self.logger.debug(
                "Partial word match on regional term %r (%s, %d entities); not looking it up",
                term, regional_term.geographical_name.entity.iri, regional_term.entity_count)
            matches.append(IndexSearchMatch(
                score=0.0,
                matched_key=term,
                nfc_name=regional_term.geographical_name.name,
                is_partial_word_match=True,
                geographical_name=regional_term.geographical_name
            ))
        query_strings = remaining_query_strings
        words = [
            word
            for query_string in query_strings
            for word in query_string.split(" ") if word != ""
        ]
        word_queries = []
        for word in words:
            # tantivy's fuzzy term query scores every hit a constant 1.0
            # regardless of edit distance, so on its own exact hits are not
            # ranked above fuzzy ones in the top k (and can even rank below
            # them, since an exact BM25 score on a common word may be < 1).
            # Exact hits get the same constant plus their BM25 score, putting
            # them above every fuzzy-only hit while still favouring rarer words.
            exact_query = tantivy.Query.term_query(
                self.schema, "search_words", word, index_option='basic')
            word_queries.append(tantivy.Query.boolean_query([
                (tantivy.Occur.Should, tantivy.Query.const_score_query(exact_query, 1.0)),
                (tantivy.Occur.Should, exact_query),
            ]))
            word_queries.append(tantivy.Query.fuzzy_term_query(
                self.schema, "search_words", word, distance=self._PARTIAL_MATCH_WORD_DISTANCE))
            if word in abbreviated_query_words:
                word_queries.append(tantivy.Query.regex_query(
                    self.schema, "search_words", f"{re.escape(word)}[a-z0-9]*"))
        if len(word_queries) == 0:
            self.logger.debug("Partial word match lookup skipped: no words in query strings %s", query_strings)
            return matches
        final_name_query = tantivy.Query.disjunction_max_query(word_queries)
        if len(hints) > 0:
            final_query = tantivy.Query.boolean_query(
                [(tantivy.Occur.Must, final_name_query)] + hints)
        else:
            final_query = final_name_query
        search_results = searcher.search(final_query, limit=limit)
        for score, doc_address in search_results.hits:
            doc = searcher.doc(doc_address)
            nfc_name = doc.get_first("nfc_name")
            matched_key = None
            for query_string in query_strings:
                query_words = [w for w in query_string.split(" ") if w]
                for candidate_string in normalized_search_strings(nfc_name):
                    candidate_words = [w for w in candidate_string.split(" ") if w]
                    if self._is_partial_word_match(searcher, query_words, candidate_words, abbreviated_query_words):
                        matched_key = candidate_string
                        break
                if matched_key is not None:
                    break
            if matched_key is None:
                self.logger.debug("Partial word match candidate %r discarded", nfc_name)
                continue
            entities_data = json.loads(doc.get_first("name_data").decode("utf-8"))
            for entity_data in entities_data:
                matches.append(IndexSearchMatch(
                    score=score,
                    matched_key=matched_key,
                    nfc_name=nfc_name,
                    retrieved_data=entity_data,
                    is_partial_word_match=True
                ))
        return matches

    def _build_entity_type_restriction(
            self, entity_types : Collection[GeographicalEntityType]) -> tantivy.Query:
        if len(entity_types) == 1:
            entity_type = next(iter(entity_types))
            return tantivy.Query.term_query(
                self.schema, entity_type.entity_type, True, index_option='basic')
        else:
            disjuction = []
            for entity_type in entity_types:
                disjuction.append(tantivy.Query.term_query(self.schema, entity_type.entity_type, True))
            return tantivy.Query.disjunction_max_query(disjuction)

    def _build_country_code_restriction(self, country_codes : Collection[str]) -> tantivy.Query:
        if len(country_codes) == 1:
            country_code = next(iter(country_codes))
            return tantivy.Query.term_query(self.schema, "country_code", country_code, index_option='basic')
        else:
            disjuction = []
            for country_code in country_codes:
                disjuction.append(tantivy.Query.term_query(self.schema, "country_code", country_code))
            return tantivy.Query.disjunction_max_query(disjuction)
    
    def _build_admin_code_restriction(self, admin_codes : Collection[GeonamesAdminCodes]) -> tantivy.Query:
        def _admin_code_to_query(admin_code : GeonamesAdminCodes) -> tantivy.Query:
            admin_code_queries = []
            for k, v in admin_code._asdict().items():
                if v is not None and v != '':
                    admin_code_queries.append((
                        tantivy.Occur.Must, tantivy.Query.term_query(self.schema, k, v, index_option='basic')
                    ))
            if len(admin_code_queries) == 0:
                return None
            if len(admin_code_queries) == 1:
                return admin_code_queries[0][1]
            else:
                return tantivy.Query.boolean_query(admin_code_queries)
        if len(admin_codes) == 1:
            return _admin_code_to_query(next(iter(admin_codes)))
        else:
            disjuction = []
            for admin_code in admin_codes:
                admin_code_query = _admin_code_to_query(admin_code)
                if admin_code_query is not None:
                    disjuction.append(admin_code_query)
            return tantivy.Query.disjunction_max_query(disjuction)


    @_serialized_searcher_access
    def _search_inner(
            self,
            limit : int,
            query_strings : list[str] = [],
            abbrev_pattern : Optional[str] = None,
            hints : list[tuple[tantivy.Occur, tantivy.Query]] = [],
            distance_threshold : int = 0,
            field : str = "search_key",
            is_phonetic : bool = False,
        ) -> list[IndexSearchMatch]:
        if distance_threshold == 0:
            name_queries = [
                tantivy.Query.term_query(self.schema, field, query, index_option='basic')
                for query in query_strings
            ]
        else:

            name_queries = [
                tantivy.Query.fuzzy_term_query(self.schema,
                    field, query, distance=distance_threshold)
                for query in query_strings
            ]
        if abbrev_pattern:
            name_queries.append(tantivy.Query.regex_query(self.schema, field, abbrev_pattern))
        final_name_query = tantivy.Query.disjunction_max_query(name_queries)
        if len(hints) > 0:
            final_query = tantivy.Query.boolean_query(
                [(tantivy.Occur.Must, final_name_query)] + hints)
        else:
            final_query = final_name_query
        searcher = self.index.searcher()
        # limit applies to the retrieved name documents; since each one expands into every
        # entity known under that name, the number of resulting matches may exceed it
        search_results = searcher.search(final_query, limit=limit)
        matches = []
        for score, doc_address in search_results.hits:
            doc = searcher.doc(doc_address)
            matched_key = doc.get_first(field)
            if abbrev_pattern:
                # a name has several search keys (e.g. with and without stop
                # words); keep the one the abbreviation pattern matched, which
                # GeoDBSearch._parse_data checks to flag abbreviation matches
                matched_key = next(
                    (key for key in doc.get_all(field) if re.fullmatch(abbrev_pattern, key)), matched_key)
            nfc_name = doc.get_first("nfc_name")
            entities_data = json.loads(doc.get_first("name_data").decode("utf-8"))
            for entity_data in entities_data:
                matches.append(IndexSearchMatch(
                    score=score,
                    matched_key=matched_key,
                    nfc_name=nfc_name,
                    retrieved_data=entity_data,
                    is_phonetic_match=is_phonetic
                ))
        return matches

    def search(
            self, 
            query_string, 
            distance_threshold : int, 
            similarity_threshold : float,
            callback : Optional[Callable[[IndexSearchResult], bool]],
            limit : int = 10,
            expand_abbreviations : bool = True,
            entity_types : Optional[Collection[GeographicalEntityType]] = None,
            strict_entity_type_filtering : bool = False,
            country_codes : Optional[Collection[str]] = None,
            strict_country_filtering : bool = False,
            admin_codes : Optional[Collection[GeonamesAdminCodes]] = None
        ):
        if similarity_threshold == 1.0:
            self.logger.debug("Similarity threshold is 1.0; disabling fuzzy search (distance threshold 0)")
            distance_threshold = 0
        expanded_query = unicodedata.normalize("NFC", query_string)
        expanded_query, direct_link_iri = abbrev_list_expander.expand_abbreviations(expanded_query)
        # TODO the code bellow has a new field, data schema needs update. Upstream code needs to fetch the entity data
        # if direct_link_iri is not None:
        #     self.logger.debug(
        #         "Query %r expanded to direct link IRI %r; returning only that entity",
        #         query_string, direct_link_iri)
        #     return IndexSearchResult(
        #         nfc_query=expanded_query,
        #         query_strings=[],
        #         abbreviation_pattern=None,
        #         matches=[IndexSearchMatch(
        #             score=1.0,
        #             matched_key=expanded_query,
        #             nfc_name=expanded_query,
        #             retrieved_data={"entity": {"iri": direct_link_iri}},
        #             is_direct_link=True
        #         )]
        #     )
        query_strings = normalized_search_strings(_remove_stop_words(expanded_query))
        phonetic_query_string = cologne_phonetic_normalize(normalize_for_phonetics(expanded_query))
        if expand_abbreviations:
            abbrev_pattern = abbreviation_pattern_to_regexes(expanded_query)
            abbreviated_query_words = abbreviated_words(expanded_query)
        else:
            abbrev_pattern = None
            abbreviated_query_words = frozenset()
        self.logger.debug(
            "Searching %r: query strings %s, phonetic key %r, abbreviation pattern %r, "
            "distance threshold %d, limit %d, entity types %s (strict: %s), "
            "country codes %s (strict: %s), admin codes %s",
            expanded_query, query_strings, phonetic_query_string, abbrev_pattern,
            distance_threshold, limit, entity_types, strict_entity_type_filtering,
            country_codes, strict_country_filtering, admin_codes)
        
        hints = []
        if entity_types is not None:
            strictness = tantivy.Occur.Should
            if strict_entity_type_filtering:
                strictness = tantivy.Occur.Must
            hints.append((strictness, self._build_entity_type_restriction(entity_types)))
        if country_codes is not None:
            strictness = tantivy.Occur.Should
            if strict_country_filtering:
                strictness = tantivy.Occur.Must
            hints.append((strictness, self._build_country_code_restriction(country_codes)))
        if admin_codes is not None:
            admin_code_restriction = self._build_admin_code_restriction(admin_codes)
            if admin_code_restriction is not None:
                hints.append((tantivy.Occur.Should, admin_code_restriction))
        def _falling_queries():
            other_params = dict(hints=hints, limit=limit)
            # exact matches first
            self.logger.debug(
                "Phase 'exact' for %r: query strings %s, abbreviation pattern %r",
                expanded_query, query_strings, abbrev_pattern)
            yield "exact", self._search_inner(query_strings=query_strings, distance_threshold=0, **other_params)

            #abbreviation matches next: catches names that are abbreviated in the query but not in the index, e.g. "Rum." vs "Rumanien"
            if abbrev_pattern is not None:
                self.logger.debug(
                    "Phase 'abbreviation' for %r: abbreviation pattern %r", expanded_query, abbrev_pattern)
                yield "abbreviation", self._search_inner(
                    abbrev_pattern=abbrev_pattern, query_strings=[], distance_threshold=0, **other_params)
            
            # phonetic and edit distance 1 fuzzy matches next, together in the
            # same phase with their results concatenated: phonetic matching
            # catches names that were misheard/misspelled in a way plain edit
            # distance on the raw string would not, and vice versa, so neither
            # should preempt the other
            phonetic_phase_name = PHONETIC_SEARCH_PHASE
            phonetic_phase_matches = []
            if phonetic_query_string.strip() != "":
                self.logger.debug(
                    "Phase 'phonetic' for %r: phonetic key %r", expanded_query, phonetic_query_string)
                phonetic_phase_matches.extend(self._search_inner(
                    query_strings=[phonetic_query_string], distance_threshold=0,
                    field="phonetic_key", is_phonetic=True, **other_params))
            else:
                self.logger.debug("Phase 'phonetic' for %r skipped: empty phonetic key", expanded_query)
            if distance_threshold >= 1:
                phonetic_phase_name = f"{PHONETIC_SEARCH_PHASE}+{FUZZY_DISTANCE_1_SEARCH_PHASE}"
                self.logger.debug(
                    "Phase 'phonetic' for %r: query strings %s, edit distance 1", expanded_query, query_strings)
                phonetic_phase_matches.extend(self._search_inner(
                    query_strings=query_strings, distance_threshold=1, **other_params))
            yield phonetic_phase_name, phonetic_phase_matches
            # larger edit distance fuzzy matches next, only tried once exact,
            # abbreviation, phonetic and distance 1 fuzzy matching have failed
            # to find anything
            for i in range(2, distance_threshold + 1):
                self.logger.debug(
                    "Phase 'fuzzy' for %r: query strings %s, edit distance %d",
                    expanded_query, query_strings, i)
                yield f"fuzzy(distance={i})", self._search_inner(
                    query_strings=query_strings, distance_threshold=i, **other_params)
            # partial word match last: lowest priority and riskiest for false
            # positives, only tried once nothing else has found anything
            self.logger.debug(
                "Phase 'partial_word' for %r: query strings %s, max word distance %d, abbreviated words %s",
                expanded_query, query_strings, self._PARTIAL_MATCH_WORD_DISTANCE, abbreviated_query_words)
            yield "partial_word", self._partial_word_match(
                query_strings, hints=hints, limit=limit, abbreviated_query_words=abbreviated_query_words)

        for phase, matches in _falling_queries():
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Phase '%s' for %r retrieved %d matches: %s",
                    phase, expanded_query, len(matches), _describe_index_matches(matches))
            if len(matches) > 0:
                if callback(IndexSearchResult(
                    nfc_query=expanded_query,
                    query_strings=query_strings,
                    abbreviation_pattern=abbrev_pattern,
                    matches=matches,
                    phase=phase,
                )):
                    self.logger.debug("Phase '%s' for %r settled the search; stopping", phase, expanded_query)
                    break
                self.logger.debug("Phase '%s' for %r did not settle the search; continuing", phase, expanded_query)
        else:
            self.logger.debug("All search phases for %r exhausted without settling", expanded_query)
        return IndexSearchResult(
            nfc_query=expanded_query,
            query_strings=query_strings,
            abbreviation_pattern=abbrev_pattern,
            matches=[]
        )

# Based on observation of the distribution of countries
PRIORITY_COUNTRIES = [
    "DE", # Germany
    "US", # United States
    "IL", # Israel
    "PL", # Poland
    "FR", # France
    "RO", # Romania
    "CZ", # Czech Republic
    "HU", # Hungary
    "RU", # Russia
]

# Matches geonames iris regardless of scheme (http/https), as attached by
# address parsing (see entity_linking._geonames_iri) or held by the geo
# duckdb (built as "https://sws.geonames.org/{geonameId}", see
# build_geonames_db.py) -- these two do not always agree on scheme, so
# geonames iris are resolved by id instead of by direct iri comparison.
_GEONAMES_IRI_REGEX = re.compile(r"^https?://sws\.geonames\.org/(\d+)/?$")


def _normalize_iri(iri: str) -> str:
    """
    Normalizes a non-geonames iri to the form used by the geo duckdb: https,
    no trailing slash.
    """
    if iri.startswith("http://"):
        iri = "https://" + iri[len("http://"):]
    return iri.rstrip("/")


SEARCHABLE_ENTITY_TYPES = [
    GeographicalEntityType.Country,
    GeographicalEntityType.State,
    GeographicalEntityType.Region ,
    GeographicalEntityType.District,
    GeographicalEntityType.City,
    GeographicalEntityType.Neighborhood,
    GeographicalEntityType.AboveCity,
    GeographicalEntityType.Unknown,
]


# Every real (searchable) feature type, i.e. what Unknown (a wildcard with
# no dedicated feature type of its own) expands to when searched.
ANY_ENTITY_TYPES = tuple(
    entity_type for entity_type in SEARCHABLE_ENTITY_TYPES
    if entity_type not in (GeographicalEntityType.AboveCity, GeographicalEntityType.Unknown)
)

# Every real feature type coarser than City, i.e. what AboveCity (a
# parsing-time hint with no dedicated feature type of its own) expands to
# when searched: District, Region, State, Country.
ABOVE_CITY_ENTITY_TYPES = tuple(
    entity_type for entity_type in ANY_ENTITY_TYPES
    if entity_type < GeographicalEntityType.City
)

class GeoDBSearch(LinkingStep):
    logger = ENTITY_LINKING_LOGGER.getChild("GeoDBSearch")

    def __init__(
            self, search_cache_db,
            search_index : GeoSearchIndex, materialize_table : bool = True,
            cleaned_distance_threshold : int = 2,
            prune_score_threshold : float = 0.5,
            settle_score_threshold : float = 0.9,
            topk : int = 20,
            priority_countries : Optional[list[str]] = PRIORITY_COUNTRIES,
            geo_db_path : str = "geo.duckdb",
            cache_dir : str | Path = "cache"
        ):
        self.search_cache_db_path = search_cache_db
        self.search_index = search_index
        self.materialize_table = materialize_table
        self.cleaned_distance_threshold = cleaned_distance_threshold
        self.prune_score_threshold = prune_score_threshold
        self.settle_score_threshold = settle_score_threshold
        self.topk = topk
        self.priority_countries = priority_countries
        self.geo_db_path = geo_db_path
        self.cache_dir = Path(cache_dir)
        self._pre_linked_cache_path = self.cache_dir / "pre_linked_entities.json"
        self._pre_linked_cache : dict[str, Optional[GeographicalName]] = {}
        # apply runs concurrently from several threads, while the duckdb
        # connection is not safe to use from several threads at once; cache
        # misses of _retrieve_pre_linked, the only use of the connection
        # during apply, go through this lock (see _retrieve_pre_linked)
        self._pre_linked_lock = threading.Lock()
        # set by initialize (see build_regional_terms)
        self.regional_terms = RegionalTerms({})

    def initialize(self):
        self.logger.info(
            "Initializing with search cache db %s, geo db %s, index %s "
            "(distance threshold %d, prune score threshold %.2f, settle score threshold %.2f, "
            "topk %d, priority countries %s)",
            self.search_cache_db_path, self.geo_db_path, self.search_index.index_descriptor,
            self.cleaned_distance_threshold, self.prune_score_threshold, self.settle_score_threshold,
            self.topk, self.priority_countries)
        self._connect()
        def row_retriever(sql_query : str) -> Iterable[dict]:
            total_rows = self.connection.execute("SELECT COUNT(*) FROM (" + sql_query + ")").fetchone()[0]
            self.logger.info(
                "Populating %s index with %d names from the geo database...",
                self.search_index.index_descriptor, total_rows)
            batch_iterator = self.connection.execute(sql_query).to_arrow_reader(5_000)
            pbar = tqdm(total=total_rows, desc="Populating search index")
            for batch in batch_iterator:
                columns = [col.to_pylist() for col in batch.columns]
                for row_tuple in zip(*columns):
                    row = {name : value for name, value in zip(batch.column_names, row_tuple)}
                    yield row
                pbar.update(len(batch))
            pbar.close()
        self.search_index.populate_index(row_retriever(POP_LANGUAGE_FILTERED_NAMES_SELECT), skip_if_exists=True)
        self._load_pre_linked_cache()
        self.regional_terms = self._load_or_build_regional_terms()
        if isinstance(self.search_index, TantivySearchIndex):
            self.search_index.regional_terms = self.regional_terms
        return super().initialize()

    def _regional_terms_cache_parameters(self) -> dict:
        """Everything the regional terms depend on, invalidating their cache when changed."""
        index_filters = POP_LANGUAGE_FILTERED_NAMES_SELECT.encode("utf-8")
        return {
            "version": _REGIONAL_TERMS_VERSION,
            "countries": list(REGIONAL_TERM_COUNTRIES),
            "patterns": [pattern.pattern for pattern in _REGIONAL_TERM_PATTERNS],
            "min_entities": REGIONAL_TERM_MIN_ENTITIES,
            "cluster_distance_km": REGIONAL_TERM_CLUSTER_DISTANCE_KM,
            "grid_degrees": REGIONAL_TERM_GRID_DEGREES,
            "max_place_population": REGIONAL_TERM_MAX_PLACE_POPULATION,
            "state_division": GERMAN_STATE_GEOMETRY_DIVISION,
            "idf_threshold": round(self.search_index.partial_match_idf_threshold, 4),
            "index_filters_sha1": hashlib.sha1(index_filters).hexdigest(),
        }

    def _load_or_build_regional_terms(self) -> RegionalTerms:
        """
        The regional terms (see build_regional_terms), cached to disk since
        collecting them from the geo database takes a while.
        """
        if not isinstance(self.search_index, TantivySearchIndex):
            return RegionalTerms({})
        path = self.cache_dir / "regional_terms.json"
        parameters = self._regional_terms_cache_parameters()
        entries = None
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("parameters") == parameters:
                entries = cached["terms"]
                self.logger.info("Loaded %d regional terms from %s", len(entries), path)
            else:
                self.logger.info("Ignoring cached regional terms at %s: built with other parameters", path)
        if entries is None:
            entries = build_regional_terms(self.connection, self.search_index)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # written to a temporary file first, so that an interrupted write
            # does not leave a truncated cache behind
            temporary_path = path.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as f:
                json.dump({"parameters": parameters, "terms": entries}, f, ensure_ascii=False)
            temporary_path.replace(path)
            self.logger.info("Cached %d regional terms to %s", len(entries), path)
        return _regional_terms_from_entries(self.connection, entries)

    def _regional_term_match(self, entity : RawEntity) -> Optional[MatchedName]:
        """
        A match on the regional term the whole entity is (see RegionalTerms),
        if it is one and coarser than a city (e.g. AboveCity "Pfalz" or
        "Bergstr."). A City or Neighborhood is left to the search, its name
        being a place in its own right.
        """
        if entity.entity_type not in REGIONAL_TERM_ENTITY_TYPES or len(self.regional_terms) == 0:
            return None
        expanded_query, _ = abbrev_list_expander.expand_abbreviations(unicodedata.normalize("NFC", entity.raw_text))
        abbreviated = abbreviated_words(expanded_query)
        for query_string in normalized_search_strings(_remove_stop_words(expanded_query)):
            term = self.regional_terms.find(query_string, abbreviated)
            if term is None:
                continue
            geographical_name = self.regional_terms[term].geographical_name
            edit_distance, fuzzy_score, phonetic_score = score_name_similarity(
                expanded_query, geographical_name.name, self.logger,
                is_abbreviation_match=term != query_string)
            self.logger.debug(
                "Entity %r (%s) is the regional term %r (%s); matching it instead of searching",
                entity.raw_text, entity.entity_type, term, geographical_name.entity.iri)
            return MatchedName(
                geographical_name=geographical_name,
                nfc_query=expanded_query,
                nfc_alt_name=geographical_name.name,
                cleaned_query=None,
                cleaned_alt_name=term,
                cleaned_edit_distance=None,
                matching_method=REGIONAL_TERM_MATCHING_METHOD,
                search_phase="regional_term",
                matching_score=1.0,
                fuzzy_score=fuzzy_score,
                phonetic_score=phonetic_score,
                abbreviation_pattern=None,
                edit_distance=edit_distance,
                is_abbreviation_match=term != query_string,
                is_phonetic_match=False,
                is_partial_word_match=False,
            )
        return None

    def finalize(self):
        self.connection.close()
        return super().finalize()

    def _connect(self, **settings):
        self.connection = duckdb.connect(self.search_cache_db_path)
        for name, value in settings.items():
            self.connection.execute(f"SET {name} = {value}")
        self.connection.execute(f"ATTACH DATABASE '{self.geo_db_path}' AS geo_db (READ_ONLY)")

    def __getstate__(self):
        # Neither the connection nor the lock can be pickled (e.g. when sent
        # to a worker process); each copy opens its own connection instead,
        # single threaded and without a progress bar since it only serves the
        # lookups of _retrieve_pre_linked, alongside many other copies. Everything initialize built (e.g. the regional
        # terms) is copied as is.
        state = self.__dict__.copy()
        for attribute in ("connection", "_pre_linked_lock"):
            state.pop(attribute, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pre_linked_lock = threading.Lock()
        if self.search_cache_db_path != ":memory:":
            # the original's connection holds the file's lock
            raise ValueError(
                f"Cannot copy a GeoDBSearch whose search cache db is a file ({self.search_cache_db_path}); "
                "use ':memory:'")
        self._connect(threads=1, enable_progress_bar=False)

    def _load_pre_linked_cache(self):
        if not self._pre_linked_cache_path.exists():
            self.logger.info("No pre-linked cache found at %s", self._pre_linked_cache_path)
            return
        with self._pre_linked_cache_path.open("r", encoding="utf-8") as f:
            raw_cache = json.load(f)
        self._pre_linked_cache = {
            iri: (decode_from_dict(data, GeographicalName) if data is not None else None)
            for iri, data in raw_cache.items()
        }
        self.logger.info(
            "Loaded %d pre-linked entities from %s", len(self._pre_linked_cache), self._pre_linked_cache_path)

    def _save_pre_linked_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw_cache = {
            iri: (encode_as_dict(geographical_name) if geographical_name is not None else None)
            for iri, geographical_name in self._pre_linked_cache.items()
        }
        # copies in other worker processes may save the cache at the same
        # time: each writes its own temporary file first, then atomically
        # replaces the cache with it, so that the cache is never left
        # truncated (an entry only another copy looked up may get lost, to be
        # looked up again)
        temporary_path = self._pre_linked_cache_path.with_suffix(f".json.{os.getpid()}.tmp")
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(raw_cache, f, ensure_ascii=False)
        temporary_path.replace(self._pre_linked_cache_path)

    def _search_entity(
        self, 
        entity : MatchedEntity, 
        country_codes : Optional[Collection[str]], 
        admin_codes : Optional[Collection[GeonamesAdminCodes]],
        search_callback : Callable[[IndexSearchResult], bool],
        hint_country_codes : Optional[Collection[str]] = None,
    ) -> IndexSearchResult:
        entity_type = entity.entity_type
        # with no already resolved countries, the search is only hinted
        # towards the countries hinted by the address (see apply), or else
        # towards the priority countries, not restricted to them
        country_strict = False
        if entity_type != GeographicalEntityType.Country and country_codes is None and hint_country_codes:
            self.logger.debug(
                "No resolved country for %r (%s); hinting search towards countries %s named in the address",
                entity.raw_text, entity_type, sorted(hint_country_codes))
            country_codes = hint_country_codes
        elif entity_type != GeographicalEntityType.Country and country_codes is None:
            self.logger.debug(
                "No known country for %r (%s); hinting search towards priority countries %s",
                entity.raw_text, entity_type, self.priority_countries)
            country_codes = self.priority_countries
        elif country_codes is not None:
            self.logger.debug(
                "Restricting search for %r (%s) strictly to already resolved countries %s",
                entity.raw_text, entity_type, country_codes)
            country_strict = True
        # AboveCity and Unknown have no dedicated feature type of their own;
        # search for them as a disjunction over every real type coarser than
        # City, or over every real type at all, respectively (the search index
        # already treats multiple entity_types as an OR).
        if entity_type == GeographicalEntityType.AboveCity:
            search_entity_types = ABOVE_CITY_ENTITY_TYPES
        elif entity_type == GeographicalEntityType.Unknown:
            search_entity_types = ANY_ENTITY_TYPES
        else:
            search_entity_types = [entity_type]
        if entity_type in (GeographicalEntityType.AboveCity, GeographicalEntityType.Unknown):
            self.logger.debug(
                "Expanding %s entity %r to entity types %s", entity_type.name, entity.raw_text, search_entity_types)
        index_matches = self.search_index.search(
            query_string=entity.raw_text,
            distance_threshold=self.cleaned_distance_threshold,
            similarity_threshold=self.prune_score_threshold,
            callback=search_callback,
            limit=self.topk,
            expand_abbreviations=True,
            entity_types=search_entity_types,
            country_codes=country_codes,
            strict_country_filtering=country_strict,
            admin_codes=admin_codes
        )
        return index_matches
        
    def _parse_data(self, index_result : IndexSearchResult) -> Iterable[MatchedName]:
        for index_match in index_result.matches:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Parsing index match with IRI %r for query %r (abbreviation pattern %r)",
                    (index_match.retrieved_data or {}).get("entity", {}).get("iri", None)
                    if index_match.geographical_name is None else index_match.geographical_name.entity.iri,
                    index_result.nfc_query, index_result.abbreviation_pattern)
            is_abreviation_match = False
            if index_result.abbreviation_pattern is not None:
               is_abreviation_match = re.fullmatch(index_result.abbreviation_pattern, index_match.matched_key) is not None

            if index_match.geographical_name is not None:
                geographical_name = index_match.geographical_name
            else:
                geographical_name = decode_from_dict(index_match.retrieved_data, GeographicalName)
            nfc_alt_name = unicodedata.normalize("NFC", geographical_name.name)
            edit_distance, fuzzy_score, phonetic_score = score_name_similarity(
                index_result.nfc_query, nfc_alt_name, self.logger,
                is_abbreviation_match=is_abreviation_match)
            matched_name = MatchedName(
                geographical_name=geographical_name,
                nfc_query=index_result.nfc_query,
                nfc_alt_name=nfc_alt_name,
                cleaned_query=None,  # Deprecated
                cleaned_alt_name=index_match.matched_key,
                cleaned_edit_distance=None, # Deprecated
                edit_distance = edit_distance,
                fuzzy_score = fuzzy_score,
                phonetic_score = phonetic_score,
                abbreviation_pattern=index_result.abbreviation_pattern,
                is_abbreviation_match=is_abreviation_match,
                is_phonetic_match=index_match.is_phonetic_match,
                is_partial_word_match=index_match.is_partial_word_match,
                matching_method=(
                    REGIONAL_TERM_MATCHING_METHOD if index_match.geographical_name is not None
                    else self.search_index.index_descriptor),
                matching_score=index_match.score,
                search_phase=_match_search_phase(index_result.phase, index_match),
            )

            yield matched_name

    def _retrieve_pre_linked(self, iri : str) -> Optional[GeographicalName]:
        """
        Resolves an entity already known by iri (attached during address
        parsing) via a direct id lookup against the geo duckdb, instead of a
        text search against the tantivy index (which maps text to possibly
        many entities and is not suited for an exact id lookup). The small,
        bounded set of ids that ever shows up this way is cached to disk
        (see _pre_linked_cache_path), since the duckdb lookup is comparatively
        slow and is otherwise repeated for the same handful of entities.
        """
        # Checked without the lock first: reading a dict is thread safe, and
        # cache hits are the common case
        if iri in self._pre_linked_cache:
            self.logger.debug("Pre-linked iri %s found in cache", iri)
            return self._pre_linked_cache[iri]
        with self._pre_linked_lock:
            # another thread may have looked it up while waiting for the lock
            if iri in self._pre_linked_cache:
                self.logger.debug("Pre-linked iri %s found in cache", iri)
                return self._pre_linked_cache[iri]
            return self._look_up_pre_linked(iri)

    def _look_up_pre_linked(self, iri : str) -> Optional[GeographicalName]:
        """
        Cache miss of _retrieve_pre_linked: looks the iri up in the geo duckdb
        and caches the result. Only called holding _pre_linked_lock.
        """
        self.logger.debug("Pre-linked iri %s is not in the cache; looking it up in the geo database", iri)
        geonames_match = _GEONAMES_IRI_REGEX.match(iri)
        if geonames_match is not None:
            # The duckdb's own geonames iris and the ones attached during
            # parsing do not always agree on http vs https; reconstruct the
            # canonical https form the geo duckdb itself stores (rather than
            # matching on geonames_id, which -- unlike iri -- is not a
            # primary/unique key and would force a full table scan).
            lookup_iri = f"https://sws.geonames.org/{geonames_match.group(1)}"
        else:
            lookup_iri = _normalize_iri(iri)
        row = self.connection.execute(preferred_name_by_entity_iri_select("geo_db."), [lookup_iri]).fetchone()
        if row is None:
            self.logger.debug("Pre-linked iri %s (looked up as %s) not found in the geo database", iri, lookup_iri)
            geographical_name = None
        else:
            columns = [description[0] for description in self.connection.description]
            geographical_name = decode_from_dict(dict(zip(columns, row)), GeographicalName)
        self._pre_linked_cache[iri] = geographical_name
        self._save_pre_linked_cache()
        return geographical_name

    def _pre_linked_match(self, entity : RawEntity) -> Optional[MatchedName]:
        geographical_name = self._retrieve_pre_linked(entity.pre_linked_iri)
        if geographical_name is None:
            return None
        return MatchedName(
            geographical_name=geographical_name,
            nfc_query=entity.raw_text,
            nfc_alt_name=geographical_name.name,
            cleaned_query=None,
            cleaned_alt_name=None,
            cleaned_edit_distance=None,
            matching_method="pre_linked",
            search_phase="pre_linked",
            matching_score=1.0,
            fuzzy_score=1.0,
            phonetic_score=1.0,
            abbreviation_pattern=None,
            edit_distance=0,
            is_abbreviation_match=False,
            is_phonetic_match=False,
            is_partial_word_match=False,
        )

    def _is_trusted_missed_word_match(self, index_result : IndexSearchResult, match : MatchedName) -> bool:
        """
        Whether a match is trusted for a missed word entity (see
        RawEntity.is_missed_word): only an exact match, an abbreviation match,
        or a name containing the query as is (e.g. "Pegnitz" in "Lauf an der
        Pegnitz"). A missed word being only a guess at a place, the looser
        phonetic and edit distance matches are mostly unrelated places (e.g.
        "Ostrach" for "Postfach").
        """
        if match.is_abbreviation_match:
            return True
        if match.is_phonetic_match or match.cleaned_alt_name is None:
            return False
        return any(
            f" {query_string} " in f" {match.cleaned_alt_name} " for query_string in index_result.query_strings)

    def _settle_for_search_match(
            self, entity : RawEntity, match : MatchedName, country_codes : Collection[str]
        ) -> bool:
        if is_entity_type_ruled_out(entity.entity_type, match.geographical_name.entity.possible_entity_types):
            self.logger.debug(
                "Not settling search for %r on %r (%s): expected entity type %r but the possible entity types %r "
                "do not include City or Neighborhood",
                entity.raw_text, match.nfc_alt_name, match.geographical_name.entity.iri,
                entity.entity_type, match.geographical_name.entity.possible_entity_types)
            return False
        # only a match in a country established by an authoritative entity of
        # the address (see apply) may settle the search, so that the matches
        # of later phases remain available to the Disambiguator otherwise
        if match.geographical_name.entity.country.iso_code not in country_codes:
            self.logger.debug(
                "Not settling search for %r on %r (%s): country %s is not among the authoritative country codes %s",
                entity.raw_text, match.nfc_alt_name, match.geographical_name.entity.iri,
                match.geographical_name.entity.country.iso_code, sorted(country_codes))
            return False
        if match.fuzzy_score >= self.settle_score_threshold:
            self.logger.debug(
                "Settling search for %r on %r (%s): fuzzy score %.3f >= settle threshold %.3f",
                entity.raw_text, match.nfc_alt_name, match.geographical_name.entity.iri,
                match.fuzzy_score, self.settle_score_threshold)
            return True
        # very short abbreviations (e.g. "Rum.") inherently score low against
        # their expansion, so the fuzzy score is not a useful signal for them
        query_letters = sum(c.isalpha() for c in entity.raw_text)
        if match.is_abbreviation_match and query_letters < 4:
            self.logger.debug(
                "Settling search for %r on %r (%s): abbreviation match on a %d-letter query "
                "(fuzzy score %.3f ignored)",
                entity.raw_text, match.nfc_alt_name, match.geographical_name.entity.iri,
                query_letters, match.fuzzy_score)
            return True
        return False

    def _prune_search_match(self, entity : RawEntity, match : MatchedName) -> bool:
        """
        Prune a match if it is unlikely to be the correct match for the given address and entity.
        """
        # if match.fuzzy_score < self.prune_score_threshold:
        #     return True
        if entity.entity_type == GeographicalEntityType.Country and GeographicalEntityType.Country not in match.geographical_name.entity.possible_entity_types:
            self.logger.debug(
                "Pruned %r (%s) for %r: searched for a country but match is not a country (types %s)",
                match.nfc_alt_name, match.geographical_name.entity.iri, entity.raw_text,
                match.geographical_name.entity.possible_entity_types)
            return True
        # matches of an entity type ruled out for the entity are not pruned
        # here but by the Disambiguator (see is_entity_type_ruled_out), and
        # neither are weak matches in an unlikely country, which other
        # entities of the address may corroborate (see
        # Disambiguator._uncorroborated_unlikely_country_reason)
        return False

    @staticmethod
    def _is_verbatim_match(entity : RawEntity, match : MatchedName) -> bool:
        """
        Whether a match's name is the entity's text as is, punctuation
        included (e.g. "Polen" for "Polen", but not "Calif" for "Calif."),
        ignoring only case and surrounding/repeated whitespace.
        """
        def _key(text : str) -> str:
            return " ".join(unicodedata.normalize("NFC", text).split()).casefold()
        return _key(entity.raw_text) == _key(match.nfc_alt_name)

    def _country_restriction_and_hints(
            self, entity : RawEntity, matched_names : Iterable[MatchedName]
        ) -> tuple[list[MatchedName], set[str]]:
        """
        (matches restricting later searches to their country, countries later
        searches are only hinted towards) for the matches of a Country or
        AboveCity entity. Only independent countries (A.PCLI) count. A
        verbatim match (see _is_verbatim_match) restricts later searches when
        every verbatim match is in the same country; otherwise every country
        matched (verbatim or not) is only hinted at, along with its
        neighboring countries. A historical country (e.g. Czechoslovakia)
        never restricts, but hints at the current countries on its former
        territory (see GeographicalEntity.successor_country_iso_codes) along
        with its neighboring countries.
        """
        country_matches = [
            matched_name for matched_name in matched_names
            if matched_name.geographical_name.entity.classification == "A.PCLI"
            and matched_name.geographical_name.entity.country is not None
        ]
        verbatim_matches = [m for m in country_matches if self._is_verbatim_match(entity, m)]
        if len({m.geographical_name.entity.country.iso_code for m in verbatim_matches}) == 1:
            return verbatim_matches, set()
        hint_country_codes = set()
        for matched_name in country_matches:
            country = matched_name.geographical_name.entity.country
            hint_country_codes.add(country.iso_code)
            hint_country_codes.update(country.neighboring_countries_iso_codes or ())
        for matched_name in matched_names:
            hint_country_codes.update(self._historical_country_hints(matched_name))
        return [], hint_country_codes

    @staticmethod
    def _historical_country_hints(matched_name : MatchedName) -> set[str]:
        """
        For a match of a historical country, the current countries on its
        former territory and its neighboring countries; empty for any other match.
        """
        entity = matched_name.geographical_name.entity
        if not entity.successor_country_iso_codes:
            return set()
        hint_country_codes = set(entity.successor_country_iso_codes)
        if entity.country is not None:
            hint_country_codes.update(entity.country.neighboring_countries_iso_codes or ())
        return hint_country_codes

    def apply(self, address):
        new_entities = []
        # countries later searches are restricted to (see _search_entity),
        # and countries they are only hinted towards
        country_codes = set()
        hint_country_codes = set()
        admin_codes = set()
        self.logger.debug("Searching entities of address %s (%r)", address.id, address.full_address)
        for entity in sorted(address.entities, key=lambda e: e.entity_type):
            if entity.entity_type not in SEARCHABLE_ENTITY_TYPES:
                self.logger.debug("Skipping entity %r: type %s is not searchable", entity.raw_text, entity.entity_type)
                new_entities.append(entity)
                continue
            if entity.pre_linked_iri is not None:
                # Already known by id from address parsing: resolve it
                # directly instead of running a text search.
                self.logger.debug(
                    "Entity %r (%s) is pre-linked to %s; resolving directly",
                    entity.raw_text, entity.entity_type, entity.pre_linked_iri)
                pre_linked_match = self._pre_linked_match(entity)
                matched_names = [pre_linked_match] if pre_linked_match is not None else []
                # A pre-linked entity is as authoritative as a resolved
                # Country search match, regardless of its own entity type.
                authoritative_matches = matched_names
            elif (regional_term_match := self._regional_term_match(entity)) is not None:
                matched_names = [regional_term_match]
                authoritative_matches = []
            else:
                matched_names : list[MatchedName] = []
                def callback(index_result : IndexSearchResult) -> bool:
                    nonlocal matched_names
                    settle_here = False
                    for matched_name in self._parse_data(index_result):
                        if entity.is_missed_word and not self._is_trusted_missed_word_match(index_result, matched_name):
                            self.logger.debug(
                                "Discarded %r (%s) for missed word %r: neither an exact, abbreviation nor "
                                "containing match (key %r)",
                                matched_name.nfc_alt_name, matched_name.geographical_name.entity.iri,
                                entity.raw_text, matched_name.cleaned_alt_name)
                            continue
                        if not self._prune_search_match(entity, matched_name):
                            self.logger.debug(
                                "Kept %r (%s, country %s) for %r: fuzzy %.3f, phonetic %.3f, "
                                "abbreviation %s, phonetic match %s, partial word match %s",
                                matched_name.nfc_alt_name, matched_name.geographical_name.entity.iri,
                                matched_name.geographical_name.entity.country.iso_code, entity.raw_text,
                                matched_name.fuzzy_score, matched_name.phonetic_score,
                                matched_name.is_abbreviation_match, matched_name.is_phonetic_match,
                                matched_name.is_partial_word_match)
                            matched_names.append(matched_name)
                            if self._settle_for_search_match(entity, matched_name, country_codes):
                                settle_here = True
                    return settle_here

                self.logger.debug("Searching for entity %r (%s)", entity.raw_text, entity.entity_type)
                self._search_entity(
                    entity,
                    country_codes=country_codes if len(country_codes) > 0 else None,
                    admin_codes=admin_codes if len(admin_codes) > 0 else None,
                    search_callback=callback,
                    hint_country_codes=hint_country_codes,
                )
                if entity.entity_type in (GeographicalEntityType.Country, GeographicalEntityType.AboveCity):
                    # a Country or AboveCity entity naming a country (e.g.
                    # "Polen" in "Stol/Polen") restricts later searches to it
                    # or hints them towards it (see _country_restriction_and_hints)
                    authoritative_matches, entity_hint_country_codes = self._country_restriction_and_hints(
                        entity, matched_names)
                    if entity_hint_country_codes:
                        self.logger.debug(
                            "Entity %r hints later searches towards countries %s",
                            entity.raw_text, sorted(entity_hint_country_codes))
                        hint_country_codes.update(entity_hint_country_codes)
                else:
                    authoritative_matches = []
            self.logger.debug(
                "Entity %r (%s) ended with %d matches", entity.raw_text, entity.entity_type, len(matched_names))
            if len(authoritative_matches) > 0:
                self.logger.debug(
                    "Entity %r is authoritative; %d of its matches restrict countries/admin codes of later searches",
                    entity.raw_text, len(authoritative_matches))
                for matched_name in authoritative_matches:
                    if matched_name.geographical_name.entity.classification == "A.PCLH":
                        historical_hint_country_codes = self._historical_country_hints(matched_name)
                        self.logger.debug(
                            "Entity %r is an historical country; its country code %s will not be used to restrict "
                            "later searches, which are hinted towards countries %s instead",
                            entity.raw_text, matched_name.geographical_name.entity.all_country_iso_codes,
                            sorted(historical_hint_country_codes))
                        hint_country_codes.update(historical_hint_country_codes)
                        continue
                    self.logger.debug("Entity %r adds country codes %s to later searches", 
                                      entity.raw_text, matched_name.geographical_name.entity.all_country_iso_codes)
                    country_codes.update(matched_name.geographical_name.entity.all_country_iso_codes)
                    if matched_name.geographical_name.entity.admin_codes:
                        self.logger.debug("Entity %r adds admin codes %s to later searches", 
                                          entity.raw_text, matched_name.geographical_name.entity.admin_codes)
                        admin_codes.add(matched_name.geographical_name.entity.admin_codes)
            new_entities.append(entity.with_matches(tuple(matched_names)))
        return dataclasses.replace(address, entities=tuple(new_entities))
