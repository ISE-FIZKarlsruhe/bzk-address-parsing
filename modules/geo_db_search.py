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
import enum
import dataclasses
import textwrap
import warnings
import textwrap
import itertools
import time
from collections import defaultdict
from modules.pipeline.geographical_entity import CountryData, GeographicalBranch, GeographicalEntity, GeographicalEntityType, GeographicalName, GeonamesAdminCodes
from modules.pipeline.linked_data import MatchedEntity, MatchedName, RawEntity, RegionHintEntity
from modules.pipeline.linking_steps import LinkingStep
import tantivy
from tqdm.auto import tqdm
import unicodedata
import re
import os
import editdistpy
import sys
import dataclasses
import json
import logging
import pyarrow
import unidecode
import math
import cologne_phonetics
import uuid
from modules import phonetics_fuzzy_scoring
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


POP_LANGUAGE_FILTERED_NAMES_SELECT = """
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE (
    isolanguage == '' OR
    isolanguage LIKE 'en%' OR
    isolanguage LIKE 'de%' OR
    isolanguage == 'abbr' OR
    list_bool_or([(isolanguage IN lang) FOR lang IN entity.country.iso_languages])
) AND len(entity.possible_entity_types) > 0
"""

OTHER_LANGUAGE_FILTERED_NAMES_SELECT = """
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE (
    isolanguage == '' OR
    isolanguage LIKE 'en%' OR
    isolanguage LIKE 'de%' OR
    isolanguage == 'abbr' OR
    list_bool_or([(isolanguage IN lang) FOR lang IN entity.country.iso_languages])
) AND len(entity.possible_entity_types) = 0
"""

# match periods follwoing an isolated letter
_STRIP_PERIODS_ABBREV_REGEX = re.compile(r'((?<=\W\w)|(?<=^\w))\.')
_STRIP_PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
_DEDUPE_WHITESPACE_REGEX = re.compile(r'\s+')

# Most addresses are in german, so german stop words take priority, but a few
# common spanish and english ones are included too since some addresses use
# those languages instead.
_STOP_WORDS = frozenset((
    "der", "die", "das", "des", "dem", "den",
    "und", "oder", "in", "im", "am", "an", "auf", "bei",
    "zu", "zum", "zur", "von", "vom", "nach", "fuer", "fur",
    "the", "and", "of", "at",
    "el", "la", "los", "las", "de", "del", "y", "en",
))

def _remove_stop_words(normalized_string : str) -> str:
    """
    Removes stop words from an already normalized (lowercase, accent/punctuation
    stripped) string. If every word is a stop word, the string is returned
    unchanged rather than reduced to an empty search key.
    """
    words = [w for w in normalized_string.split(" ") if w.lower() not in _STOP_WORDS]
    if not words:
        return normalized_string
    return " ".join(words)

def ascii_normalize(nfc_string : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_string.lower()
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


class IndexSearchResult(NamedTuple):
    nfc_query : str
    query_strings : list[str]
    abbreviation_pattern : Optional[str]
    matches : list[IndexSearchMatch]
    # (word, branches) for each informative word that names retrieved
    # by a partial word match only shared with the trailing part of the
    # query, pointing at a region the queried place is in rather than at the
    # place itself (see TantivySearchIndex._is_partial_word_match)
    region_hints : tuple[tuple[str, tuple[GeographicalBranch, ...]], ...] = ()


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

class TantivySearchIndex(GeoSearchIndex):
    index_descriptor = "tantivy"
    logger = ENTITY_LINKING_LOGGER.getChild("TantivySearchIndex")

    # Reference words used to calibrate, from the index's own word statistics,
    # how rare a word must be to count as informative enough to support a
    # partial word match (see _compute_partial_match_idf_threshold): common
    # German place-name qualifiers, none of which should on their own be
    # treated as the "real" identifying word of a name.
    _PARTIAL_MATCH_IDF_REFERENCE_WORDS = ("Alt", "Neu", "Bad", "Main")
    # Max per-word edit distance tolerated when pairing a query word against
    # a candidate word in the partial word match (character-level typo
    # tolerance, e.g. "Frankfrut" vs "Frankfurt").
    _PARTIAL_MATCH_WORD_DISTANCE = 1
    # Max number of indexed names containing a word for their branches to
    # still be extracted as a region hint (see _region_branches_for_word); a
    # word shared by more names than this is too widespread to point at
    # specific regions.
    _REGION_HINT_MAX_DOCS = 1000

    def __init__(self, index_path : str | Path, read_threads : int | Literal['auto'] = 'auto', write_threads : int = 8):
        if read_threads == 'auto':
            read_threads = (max(1, getattr(os, "process_cpu_count", lambda : None)() or os.cpu_count() or 8) * 3) // 2
        self.schema = self.create_schema()
        self.index_path = Path(index_path)
        self.already_exists = self.index_path.exists()
        if not self.already_exists:
            self.index_path.mkdir(parents=True)
        self.read_threads = read_threads
        self.write_threads = write_threads
        self.logger.info(
            "Opening tantivy index at %s (already exists: %s, read threads: %d, write threads: %d)",
            self.index_path, self.already_exists, self.read_threads, self.write_threads)
        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        self.index.config_reader(num_warmers=self.read_threads)
        self._region_branch_cache : dict[str, tuple[GeographicalBranch, ...]] = {}
        self.index.register_tokenizer(
            "whitespace",
            tantivy.TextAnalyzerBuilder(
                tantivy.Tokenizer.whitespace()
            ).build()
        )

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
                    phonetic_key = cologne_phonetic_normalize(nfc_name)
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
            if entity_type == GeographicalEntityType.AboveCity:
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

    def _pair_words(
            self, query_words : list[str], candidate_words : list[str]
        ) -> tuple[list[tuple[str, str]], list[int], list[int]]:
        """
        Greedily pairs up words between the two lists, closest edit distance
        first (within _PARTIAL_MATCH_WORD_DISTANCE), each word used in at
        most one pair. Returns the (query word, candidate word) pairs and the
        indices of the words on either side left unpaired.
        """
        candidate_pairs = []
        for qi, qword in enumerate(query_words):
            for ci, cword in enumerate(candidate_words):
                distance = levenshtein(qword, cword, self._PARTIAL_MATCH_WORD_DISTANCE)
                if distance <= self._PARTIAL_MATCH_WORD_DISTANCE:
                    candidate_pairs.append((distance, qi, ci))
        candidate_pairs.sort(key=lambda p: p[0])
        used_query, used_candidate = set(), set()
        pairs = []
        for _, qi, ci in candidate_pairs:
            if qi in used_query or ci in used_candidate:
                continue
            used_query.add(qi)
            used_candidate.add(ci)
            pairs.append((query_words[qi], candidate_words[ci]))
        unmatched_query = [i for i in range(len(query_words)) if i not in used_query]
        unmatched_candidate = [i for i in range(len(candidate_words)) if i not in used_candidate]
        return pairs, unmatched_query, unmatched_candidate

    def _is_partial_word_match(
            self, searcher : tantivy.Searcher, query_words : list[str], candidate_words : list[str]
        ) -> tuple[bool, list[str]]:
        """
        Returns whether the candidate is a partial word match of the query
        and, when it is not a match only because the words it shares with the
        query are just the trailing part of either name (e.g. "Weilheim in
        Oberbayern" vs "Tiefenbach Oberbayern"), those shared informative
        words: region words, pointing at a region the queried place is in
        (see _region_branches_for_word) rather than at the place itself.
        """
        if len(query_words) == 0 or len(candidate_words) == 0:
            self.logger.debug(
                "Partial word match %s vs %s rejected: empty word list", query_words, candidate_words)
            return False, []
        pairs, unmatched_query_idx, unmatched_candidate_idx = self._pair_words(query_words, candidate_words)
        if len(pairs) == 0:
            # nothing matched at all: not even a partial match
            self.logger.debug(
                "Partial word match %s vs %s rejected: no words paired", query_words, candidate_words)
            return False, []
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
                return True, []
        # at most one word may be missing/added/substituted on either side
        if len(unmatched_query) > 1 or len(unmatched_candidate) > 1:
            self.logger.debug(
                "Partial word match %s vs %s rejected: too many unpaired words (query: %s, candidate: %s)",
                query_words, candidate_words, unmatched_query, unmatched_candidate)
            return False, []
        # the shared words must include at least one informative word: names
        # sharing only a common qualifier (e.g. "Bad Homburg" vs "Bad Tölz")
        # are unrelated places. IDF is taken on the candidate side, since the
        # candidate word is known to be in the index while the query word may
        # be a typo.
        paired_idfs = [(cword, self._word_idf(searcher, cword)) for _, cword in pairs]
        if all(idf <= self.partial_match_idf_threshold for _, idf in paired_idfs):
            self.logger.debug(
                "Partial word match %s vs %s rejected: only non-informative words paired "
                "(%s, threshold %.4f)",
                query_words, candidate_words,
                ", ".join(f"{w!r} idf {idf:.4f}" for w, idf in paired_idfs),
                self.partial_match_idf_threshold)
            return False, []
        # an informative word may only be dropped if it is the last word of
        # its name (e.g. "Frankfurt" vs "Frankfurt Oder"), not counting
        # trailing stop words. When a leading or middle query word is
        # dropped, the candidate only shares the trailing part of the query,
        # which is then no match for the place itself, but its shared
        # informative words are kept as region words (e.g. "oberbayern" from
        # "Weilheim in Oberbayern" vs "Tiefenbach Oberbayern"). Not so for a
        # dropped candidate word, where the shared words may well be the
        # place's own name (e.g. "Tiefenbach" in "Kleinwalsertal Tiefenbach")
        for words, unmatched_idx, keeps_region_words in (
                (query_words, unmatched_query_idx, True), (candidate_words, unmatched_candidate_idx, False)):
            last_content_idx = max(
                (i for i, w in enumerate(words) if w not in _STOP_WORDS), default=len(words) - 1)
            for i in unmatched_idx:
                if i == last_content_idx:
                    continue
                idf = self._word_idf(searcher, words[i])
                if idf > self.partial_match_idf_threshold:
                    region_words = [
                        w for w, word_idf in paired_idfs if word_idf > self.partial_match_idf_threshold
                    ] if keeps_region_words else []
                    self.logger.debug(
                        "Partial word match %s vs %s rejected: unpaired non-final word %r is too "
                        "informative (idf %.4f > threshold %.4f); keeping region words %s",
                        query_words, candidate_words, words[i], idf, self.partial_match_idf_threshold,
                        region_words)
                    return False, region_words
        self.logger.debug(
            "Partial word match %s vs %s accepted (unpaired query words: %s, unpaired candidate words: %s)",
            query_words, candidate_words, unmatched_query, unmatched_candidate)
        return True, []

    def _region_branches_for_word(self, searcher : tantivy.Searcher, word : str) -> tuple[GeographicalBranch, ...]:
        """
        Aggregates every indexed name containing `word`, groups the entities
        known under those names by (country, admin1 code), and returns one
        branch per group, down to the longest sequence of admin codes common
        to every entity in the group. Returns no branches if the word is in
        too many names (_REGION_HINT_MAX_DOCS) to point at specific regions.
        """
        self.logger.debug("Looking up region branches for word %r", word)
        if word in self._region_branch_cache:
            self.logger.debug("Cache hit: region branches for word %r", word)
            return self._region_branch_cache[word]
        self.logger.debug("Cache miss: fetching all matches containing word %r", word)
        query = tantivy.Query.term_query(self.schema, "search_words", word, index_option='basic')
        search_results = searcher.search(query, limit=self._REGION_HINT_MAX_DOCS, count=True)
        branches : tuple[GeographicalBranch, ...] = ()
        if search_results.count == 0:
            self.logger.debug("Region word %r: not found in any indexed name", word)
        elif search_results.count > self._REGION_HINT_MAX_DOCS:
            self.logger.debug(
                "Region word %r: found in %d indexed names (> %d); too widespread for a region hint",
                word, search_results.count, self._REGION_HINT_MAX_DOCS)
        else:
            # (country, admin1 code) -> admin codes common to the group so far
            common_codes_by_group : dict[tuple[str, Optional[str]], list[str]] = {}
            for _, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)
                for row in json.loads(doc.get_first("name_data").decode("utf-8")):
                    entity = row["entity"]
                    iri = entity.get("iri")
                    country_code = (entity.get("country") or {}).get("iso_code")
                    if not country_code:
                        self.logger.debug("Skipping entity %r without a country", iri)
                        continue
                    admin_codes_data = entity.get("admin_codes") or {}
                    codes = []
                    for admin_level in range(1, 6):
                        code = admin_codes_data.get(f"admin{admin_level}_code")
                        if code is None or code == "" or all(c == "0" for c in code):
                            break
                        codes.append(code)
                    group = (country_code, codes[0] if codes else None)
                    common_codes = common_codes_by_group.get(group)
                    if common_codes is None:
                        self.logger.debug("New group %s with codes %s from entity %r", group, codes, iri)
                        common_codes_by_group[group] = codes
                        continue
                    common_length = 0
                    for a, b in zip(common_codes, codes):
                        if a != b:
                            break
                        common_length += 1
                    if common_length < len(common_codes):
                        self.logger.debug(
                            "Group %s codes reduced from %s to %s by entity %r (codes: %s)",
                            group, common_codes, common_codes[:common_length], iri, codes)
                        common_codes_by_group[group] = common_codes[:common_length]
            distinct_branches = {
                GeographicalBranch(country_code, tuple(codes)): None
                for (country_code, _), codes in common_codes_by_group.items()
            }
            branches = tuple(distinct_branches)
            self.logger.debug(
                "Region word %r: %d indexed names span branches %s",
                word, search_results.count, branches)
        self._region_branch_cache[word] = branches
        return branches

    def _partial_word_match(
            self,
            query_strings : list[str],
            hints : list[tuple[tantivy.Occur, tantivy.Query]],
            limit : int,
        ) -> tuple[list[IndexSearchMatch], tuple[tuple[str, tuple[GeographicalBranch, ...]], ...]]:
        """
        Lowest-priority fallback: matches names that share most, but not
        necessarily all, of their words with the query (see _is_partial_word_match).
        Also returns the region hints (region word, branches) gathered
        from the retrieved names rejected for sharing only a trailing part of
        the query.
        """
        searcher = self.index.searcher()
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
        if len(word_queries) == 0:
            self.logger.debug("Partial word match skipped: no words in query strings %s", query_strings)
            return [], ()
        final_name_query = tantivy.Query.disjunction_max_query(word_queries)
        if len(hints) > 0:
            final_query = tantivy.Query.boolean_query(
                [(tantivy.Occur.Must, final_name_query)] + hints)
        else:
            final_query = final_name_query
        search_results = searcher.search(final_query, limit=limit)
        matches = []
        region_words = []
        for score, doc_address in search_results.hits:
            doc = searcher.doc(doc_address)
            nfc_name = doc.get_first("nfc_name")
            matched_key = None
            for query_string in query_strings:
                query_words = [w for w in query_string.split(" ") if w]
                for candidate_string in normalized_search_strings(nfc_name):
                    candidate_words = [w for w in candidate_string.split(" ") if w]
                    is_match, candidate_region_words = self._is_partial_word_match(
                        searcher, query_words, candidate_words)
                    region_words.extend(w for w in candidate_region_words if w not in region_words)
                    if is_match:
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
        region_hints = []
        for word in region_words:
            branches = self._region_branches_for_word(searcher, word)
            if len(branches) > 0:
                region_hints.append((word, branches))
        return matches, tuple(region_hints)

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
        nfc_query = unicodedata.normalize("NFC", query_string)
        query_strings = normalized_search_strings(_remove_stop_words(nfc_query))
        phonetic_query_string = cologne_phonetic_normalize(nfc_query)
        if expand_abbreviations:
            abbrev_pattern = abbreviation_pattern_to_regexes(nfc_query)
        else: abbrev_pattern = None
        self.logger.debug(
            "Searching %r: query strings %s, phonetic key %r, abbreviation pattern %r, "
            "distance threshold %d, limit %d, entity types %s (strict: %s), "
            "country codes %s (strict: %s), admin codes %s",
            nfc_query, query_strings, phonetic_query_string, abbrev_pattern,
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
                nfc_query, query_strings, abbrev_pattern)
            yield "exact", self._search_inner(query_strings=query_strings, distance_threshold=0, **other_params), ()

            #abbreviation matches next: catches names that are abbreviated in the query but not in the index, e.g. "Rum." vs "Rumanien"
            if abbrev_pattern is not None:
                self.logger.debug(
                    "Phase 'abbreviation' for %r: abbreviation pattern %r", nfc_query, abbrev_pattern)
                yield "abbreviation", self._search_inner(
                    abbrev_pattern=abbrev_pattern, query_strings=[], distance_threshold=0, **other_params), ()
            
            # phonetic matches next: catches names that were misheard/misspelled
            # in a way plain edit distance on the raw string would not
            if phonetic_query_string.strip() != "":
                self.logger.debug(
                    "Phase 'phonetic' for %r: phonetic key %r", nfc_query, phonetic_query_string)
                yield "phonetic", self._search_inner(
                    query_strings=[phonetic_query_string], distance_threshold=0,
                    field="phonetic_key", is_phonetic=True, **other_params), ()
            else:
                self.logger.debug("Phase 'phonetic' for %r skipped: empty phonetic key", nfc_query)
            # fuzzy matches next, only tried once exact, abbreviation and
            # phonetic matching have failed to find anything
            for i in range(1, distance_threshold + 1):
                self.logger.debug(
                    "Phase 'fuzzy' for %r: query strings %s, edit distance %d",
                    nfc_query, query_strings, i)
                yield f"fuzzy(distance={i})", self._search_inner(
                    query_strings=query_strings, distance_threshold=i, **other_params), ()
            # partial word match last: lowest priority and riskiest for false
            # positives, only tried once nothing else has found anything
            self.logger.debug(
                "Phase 'partial_word' for %r: query strings %s, max word distance %d",
                nfc_query, query_strings, self._PARTIAL_MATCH_WORD_DISTANCE)
            yield ("partial_word", *self._partial_word_match(query_strings, hints=hints, limit=limit))

        for phase, matches, region_hints in _falling_queries():
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Phase '%s' for %r retrieved %d matches: %s (region hints: %s)",
                    phase, nfc_query, len(matches), _describe_index_matches(matches), region_hints)
            if len(matches) > 0 or len(region_hints) > 0:
                if callback(IndexSearchResult(
                    nfc_query=nfc_query,
                    query_strings=query_strings,
                    abbreviation_pattern=abbrev_pattern,
                    matches=matches,
                    region_hints=region_hints
                )):
                    self.logger.debug("Phase '%s' for %r settled the search; stopping", phase, nfc_query)
                    break
                self.logger.debug("Phase '%s' for %r did not settle the search; continuing", phase, nfc_query)
        else:
            self.logger.debug("All search phases for %r exhausted without settling", nfc_query)
        return IndexSearchResult(
            nfc_query=nfc_query,
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
]



# Every real feature type coarser than City, i.e. what AboveCity (a
# parsing-time hint with no dedicated feature type of its own) expands to
# when searched: District, Region, State, Country.
ABOVE_CITY_ENTITY_TYPES = tuple(
    entity_type for entity_type in GeographicalEntityType
    if entity_type != GeographicalEntityType.AboveCity and entity_type < GeographicalEntityType.City
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

    def initialize(self):
        self.logger.info(
            "Initializing with search cache db %s, geo db %s, index %s "
            "(distance threshold %d, prune score threshold %.2f, settle score threshold %.2f, "
            "topk %d, priority countries %s)",
            self.search_cache_db_path, self.geo_db_path, self.search_index.index_descriptor,
            self.cleaned_distance_threshold, self.prune_score_threshold, self.settle_score_threshold,
            self.topk, self.priority_countries)
        self.connection = duckdb.connect(self.search_cache_db_path)
        self.connection.execute(f"ATTACH DATABASE '{self.geo_db_path}' AS geo_db (READ_ONLY)")
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
        return super().initialize()

    def finalize(self):
        self.connection.close()
        return super().finalize()

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
        with self._pre_linked_cache_path.open("w", encoding="utf-8") as f:
            json.dump(raw_cache, f, ensure_ascii=False)

    def _search_entity(
        self, 
        entity : MatchedEntity, 
        country_codes : Optional[Collection[str]], 
        admin_codes : Optional[Collection[GeonamesAdminCodes]],
        search_callback : Callable[[IndexSearchResult], bool]
    ) -> IndexSearchResult:
        entity_type = entity.entity_type
        # country_codes as passed in is what _prune_search_match should judge
        # plausibility against; it is only widened to the priority countries
        # below for restricting the search itself.
        country_strict = False
        if entity_type != GeographicalEntityType.Country and country_codes is None:
            self.logger.debug(
                "No known country for %r (%s); hinting search towards priority countries %s",
                entity.raw_text, entity_type, self.priority_countries)
            country_codes = self.priority_countries
        elif country_codes is not None:
            self.logger.debug(
                "Restricting search for %r (%s) strictly to already resolved countries %s",
                entity.raw_text, entity_type, country_codes)
            country_strict = True
        # AboveCity has no dedicated feature type of its own; search for it
        # as a disjunction over every real type coarser than City instead
        # (the search index already treats multiple entity_types as an OR).
        search_entity_types = (
            ABOVE_CITY_ENTITY_TYPES if entity_type == GeographicalEntityType.AboveCity else [entity_type]
        )
        if entity_type == GeographicalEntityType.AboveCity:
            self.logger.debug(
                "Expanding AboveCity entity %r to entity types %s", entity.raw_text, search_entity_types)
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
            is_abreviation_match = False
            if index_result.abbreviation_pattern is not None:
               is_abreviation_match = re.fullmatch(index_result.abbreviation_pattern, index_match.matched_key) is not None

            geographical_name=decode_from_dict(index_match.retrieved_data, GeographicalName)
            nfc_alt_name = unicodedata.normalize("NFC", geographical_name.name)
            query_for_scoring = normalize_for_scoring(index_result.nfc_query)
            alt_name_for_scoring = normalize_for_scoring(nfc_alt_name)
            edit_distance, fuzzy_score = similarity_and_distance(query_for_scoring, alt_name_for_scoring, 10, distance_function=levenshtein_for_scoring)
            query_phonetic_key = phonetics_for_scoring(index_result.nfc_query)
            alt_name_phonetic_key = phonetics_for_scoring(nfc_alt_name)
            phonetic_dist, phonetic_score = similarity_and_distance(query_phonetic_key, alt_name_phonetic_key, 10)
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
                matching_method=self.search_index.index_descriptor,
                matching_score=index_match.score
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
        if iri in self._pre_linked_cache:
            self.logger.debug("Pre-linked iri %s found in cache", iri)
            return self._pre_linked_cache[iri]
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
        row = self.connection.execute(
            "SELECT * FROM geo_db.geographical_names_with_entities WHERE entity.iri = ? "
            "ORDER BY is_preferred_name DESC NULLS LAST, name_id LIMIT 1",
            [lookup_iri]
        ).fetchone()
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
            matching_score=1.0,
            fuzzy_score=1.0,
            phonetic_score=1.0,
            abbreviation_pattern=None,
            edit_distance=0,
            is_abbreviation_match=False,
            is_phonetic_match=False,
            is_partial_word_match=False,
        )

    def _settle_for_search_match(self, entity : RawEntity, match : MatchedName) -> bool:
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

    def _prune_search_match(self, entity : RawEntity, match : MatchedName, country_codes : set) -> bool:
        """
        Prune a match if it is unlikely to be the correct match for the given address and entity.
        """
        country_is_unlikely = (
            match.geographical_name.entity.country.iso_code not in ("IL", "US") and 
            match.geographical_name.entity.country.continent != "EU"
        )
        # if match.fuzzy_score < self.prune_score_threshold:
        #     return True
        if entity.entity_type == GeographicalEntityType.Country and GeographicalEntityType.Country not in match.geographical_name.entity.possible_entity_types:
            self.logger.debug(
                "Pruned %r (%s) for %r: searched for a country but match is not a country (types %s)",
                match.nfc_alt_name, match.geographical_name.entity.iri, entity.raw_text,
                match.geographical_name.entity.possible_entity_types)
            return True
        elif (
            entity.entity_type != GeographicalEntityType.Country and
            match.geographical_name.entity.country.iso_code not in country_codes and
            country_is_unlikely and
            (
                ((match.geographical_name.entity.population or 1) < 100_000 and match.fuzzy_score < 0.95)
                or
                match.fuzzy_score < 0.9
            )
            
        ):
            self.logger.debug(
                "Pruned %r (%s) for %r: unlikely country %s (continent %s) outside known countries %s "
                "with population %s and fuzzy score %.3f",
                match.nfc_alt_name, match.geographical_name.entity.iri, entity.raw_text,
                match.geographical_name.entity.country.iso_code, match.geographical_name.entity.country.continent,
                country_codes, match.geographical_name.entity.population, match.fuzzy_score)
            return True
        return False

    def apply(self, address):
        new_entities = []
        region_hint_entities = []
        country_codes = set()
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
                is_authoritative = pre_linked_match is not None
            else:
                matched_names : list[MatchedName] = []
                region_hints : dict[str, tuple[GeographicalBranch, ...]] = {}
                def callback(index_result : IndexSearchResult) -> bool:
                    nonlocal matched_names
                    settle_here = False
                    region_hints.update(index_result.region_hints)
                    for matched_name in self._parse_data(index_result):
                        if not self._prune_search_match(entity, matched_name, country_codes):
                            self.logger.debug(
                                "Kept %r (%s, country %s) for %r: fuzzy %.3f, phonetic %.3f, "
                                "abbreviation %s, phonetic match %s, partial word match %s",
                                matched_name.nfc_alt_name, matched_name.geographical_name.entity.iri,
                                matched_name.geographical_name.entity.country.iso_code, entity.raw_text,
                                matched_name.fuzzy_score, matched_name.phonetic_score,
                                matched_name.is_abbreviation_match, matched_name.is_phonetic_match,
                                matched_name.is_partial_word_match)
                            matched_names.append(matched_name)
                            if self._settle_for_search_match(entity, matched_name):
                                settle_here = True
                    return settle_here

                self.logger.debug("Searching for entity %r (%s)", entity.raw_text, entity.entity_type)
                self._search_entity(
                    entity,
                    country_codes=country_codes if len(country_codes) > 0 else None,
                    admin_codes=admin_codes if len(admin_codes) > 0 else None,
                    search_callback=callback
                )
                is_authoritative = entity.entity_type == GeographicalEntityType.Country
                for word, branches in region_hints.items():
                    self.logger.debug(
                        "Attaching region hint %r (branches %s) to entity %r", word, branches, entity.raw_text)
                    region_hint_entities.append(RegionHintEntity(
                        address_entity_id=str(uuid.uuid4()),
                        raw_text=word,
                        entity_type=GeographicalEntityType.Region,
                        span=None,
                        nearby=None,
                        source_entity_id=entity.address_entity_id,
                        branches=branches
                    ))
            self.logger.debug(
                "Entity %r (%s) ended with %d matches", entity.raw_text, entity.entity_type, len(matched_names))
            if is_authoritative and len(matched_names) > 0:
                self.logger.debug(
                    "Entity %r is authoritative; its matches restrict countries/admin codes of later searches",
                    entity.raw_text)
                for matched_name in matched_names:
                    country_codes.update(matched_name.geographical_name.entity.all_country_iso_codes)
                    if matched_name.geographical_name.entity.admin_codes:
                        admin_codes.add(matched_name.geographical_name.entity.admin_codes)
            new_entities.append(entity.with_matches(tuple(matched_names)))
        return dataclasses.replace(
            address, entities=tuple(new_entities), region_hints=tuple(address.region_hints) + tuple(region_hint_entities))
