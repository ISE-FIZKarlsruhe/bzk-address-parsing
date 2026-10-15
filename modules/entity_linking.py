"""
A full entity linking pipeline from parsed addresses such as this example:

{"filename": "1863_06_29_2_0.jpg", "vbp.raw": "Putzig (Danzig)", "vbp.status": "llm_parsed",
 "vbp.AboveCity.text": null, "vbp.City.text": "Putzig", "vbp.Country.text": null,
 "vbp.Neighborhood.text": null, "vbp.AboveCity.end": null, "vbp.AboveCity.start": null,
 "vbp.City.end": null, "vbp.City.geonames_id": null, "vbp.City.special_regex": null,
 "vbp.City.start": null, "vbp.global_regex": null, "vbp.llm_parser": "Qwen/Qwen3.5-9B"}

to linked and disambiguated addresses.

The parsed addresses may already bring IRIs (in the form of "vbp.City.geonames_id": id),
which are respected instead of being re-resolved.

"vbp" is the column prefix for VictimBirthPlace; the other prefixes (see FIELD_PREFIXES)
are for VictimCurrentAddress, VictimDeathPlace, ApplicantBirthPlace and
ApplicantCurrentAddress.

For each address field, the pipeline:
1. Tries matching the full address (or, failing that, the parsed city name)
   against the reference list of concentration camps and ghettos from
   wikidata, via camp_search.CampReferenceMatcher.
2. If no match, tags the value with the thematic categories it mentions (e.g.
   deportation, a generic/unnamed camp or ghetto, ...) via
   address_tagging.tag_address, and stops there if that leaves no place name.
3. If a place name remains, tries matching against GeoDBSearch.
4. Applies the disambiguation described in geo_disambiguation.

Output records keep the input schema (same "{prefix}.*" columns) and add new
"{prefix}.*" columns modelled after open_data/bzkopen_linking_groundtruth.jsonl:
an overall "{prefix}.iri" for the address plus a "{prefix}.{EntityType}_iri" for
every entity that was successfully linked, alongside metadata about how that
link was produced (see LinkedEntityMetadata and apply_outcome_to_row).

The steps are initialized once, in the main process, and addresses are then
linked in parallel by worker processes, each with its own copy of the steps
(see LinkingPool).
"""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
import json
import logging
import re
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
import time
import logging.handlers
import multiprocessing
import os
import pickle
from typing import Callable, Iterable, Iterator, NamedTuple, Optional

import duckdb
import pandas as pd
from tqdm.contrib.logging import tqdm_logging_redirect

from modules import abbrev_list_expander
from modules.address_tagging import tag_address
from modules.camp_search import DEFAULT_CAMPS_REFERENCE_PATH, CampReferenceMatcher
from modules.geo_db_search import FUZZY_DISTANCE_1_SEARCH_PHASE, PHONETIC_SEARCH_PHASE, REGIONAL_TERM_MATCHING_METHOD, _STOP_WORDS, _compiled_stop_words, GeoDBSearch, TantivySearchIndex, ascii_normalize, german_normalize
from modules.geo_disambiguation import MISSED_WORD_ENTITY_WEIGHT, Disambiguator, deciding_factor, factor_of_label
from modules.entity_linking_eval_metrics import normalize_iri
from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import AddressProcessingData, AddressSpan, BZKFieldName, LinkedAddress, MatchedEntity, MatchedName, RawEntity
from modules.regex_patterns import CITY_SOMETHING_REGEX

logger = logging.getLogger("entity_linking")

FIELD_PREFIXES: dict[str, BZKFieldName] = {
    "abp": BZKFieldName.APPLICANT_BIRTH_PLACE,
    "aca": BZKFieldName.APPLICANT_CURRENT_ADDRESS,
    "vbp": BZKFieldName.VICTIM_BIRTH_PLACE,
    "vdp": BZKFieldName.VICTIM_DEATH_PLACE,
    "vca": BZKFieldName.VICTIM_CURRENT_ADDRESS,
}

ENTITY_TYPE_COLUMNS = [entity_type.name for entity_type in GeographicalEntityType]

# Finest to coarsest; used to pick which pre-existing geonames id to respect
# when several are given for the same address field.
PRE_LINKED_ENTITY_ORDER = ["Neighborhood", "City", "District", "Region", "State", "Country"]

# Parameters of the pipeline's components, shared by main() and the
# entity_linking notebook (through the build_* functions below) so that
# production runs with the same configuration that was evaluated.
SEARCH_PRUNE_SCORE_THRESHOLD = 0.2
# Minimum difference between two candidates' scores on a disambiguation factor
# for that factor to decide between them (see geo_disambiguation._compare_scores).
DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05
MIN_POPULATION_ORDER_OF_MAGNITUDE = 100_000
PARENT_CHILD_MAX_DISTANCE_KM_BY_ADMIN_LEVEL = {
    0: 100,
    1: 100,
    2: 34,
    3: 14,
    4: 6,
    5: 5,
}
WEIGHTED_DISAMBIGUATION_FACTORS = {
    'phonetic_score': 2,
    'fuzzy_similarity_score': 3,
    'entity_types_matching_preferred': 1,
}
DISAMBIGUATION_FACTOR_PRIMARY_PRIORITY = [
    'weighted_score',
    'child_parent_likelihood',
    'population_order_of_magnitude',
    'fuzzy_similarity_score',
    'entity_types_matching_preferred',
    'country_likelihood_rank',
    'phonetic_score',
    'entity_types_matching',
    'is_preferred_name',
    'population_count',
]
DISAMBIGUATION_FACTOR_SECONDARY_PRIORITY = [
]
SIGNIFICANCE_THRESHOLDS = {
    "fuzzy_similarity_score" : 0,
    "population_order_of_magnitude" : 1
}
DISAMBIGUATION_POPULATION_ROUNDING_FACTOR = 1
DISAMBIGUATION_SCORE_PRUNE_THRESHOLDS = {
    "fuzzy_similarity_score": 0.4,
    "child_parent_likelihood": 0.4,
}


class SearchStatus:
    """
    Describes which method was used (or why none was) to find candidates for
    an address field.
    """
    PRE_LINKED_DURING_PARSING="PRE_LINKED_DURING_PARSING"
    CAMP_REFERENCE = "CAMP_AND_GHETTO_REFERENCE"
    NO_ENTITIES = "EMPTY_PARSING_RESULT"
    NO_LOCATION = "TAGGED_NOT_A_LOCATION"
    # GeoDBSearch found no candidates at all for the address.
    GEO_DB_NO_CANDIDATES = "GEO_DB_NO_CANDIDATES"
    # GeoDBSearch found candidates and one was linked; the suffix says which
    # of the progressively looser levels tried by TantivySearchIndex.search
    # produced the winning match for the finest-grain linked entity.
    GEO_DB_EXACT_MATCH = "GEO_DB_EXACT_MATCH"
    GEO_DB_ABBREVIATION_MATCH = "GEO_DB_ABBREVIATION_MATCH"
    GEO_DB_PHONETIC_MATCH = "GEO_DB_PHONETIC_MATCH"
    # retrieved with an edit distance of 1 (in the same phase as phonetic
    # matches, see geo_db_search._match_search_phase), or of 2 and more
    GEO_DB_FUZZY_DISTANCE_1_MATCH = "GEO_DB_FUZZY_DISTANCE_1_MATCH"
    GEO_DB_FUZZY_MATCH = "GEO_DB_FUZZY_MATCH"
    GEO_DB_PARTIAL_WORD_MATCH = "GEO_DB_PARTIAL_WORD_MATCH"
    # matched as a regional term (see geo_db_search.RegionalTerms), e.g. a
    # German state named as a whole
    GEO_DB_REGIONAL_TERM_MATCH = "GEO_DB_REGIONAL_TERM_MATCH"
    # Linking the address raised an exception (see process_row); such
    # addresses are linked again when resuming a run (see main).
    EXCEPTION = "EXCEPTION"


class DisambiguationStatus:
    """
    Describes the outcome of disambiguating the candidates found by search.
    """
    PRE_LINKED_DURING_PARSING = "PRE_LINKED_DURING_PARSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    # No disambiguation was needed in the first place since there was only one candidate.
    UNAMBIGUOUS = "UNAMBIGUOUS"
    # There were several candidates, but disambiguation could settle on one
    # using only the entity type and matching with other entities in the same address.
    DISAMBIGUATED_BY_CONTEXT = "DISAMBIGUATED_BY_CONTEXT"
    # There were several candidates, and disambiguation could settle on one
    # only by using the remaining criteria, e.g. population, country, ...;
    # this is the most error-prone case.
    DISAMBIGUATED_HEURISTICALLY = "DISAMBIGUATED_HEURISTICALLY"
    # Disambiguation could not settle on one of several candidates, but they
    # all lie within the same branch of the geographical hierarchy (e.g. the
    # same state), so the entity at that branch was linked instead.
    DISAMBIGUATED_BY_COMMON_PARENT = "DISAMBIGUATED_BY_COMMON_PARENT"

    AMBIGUOUS = "AMBIGUOUS"
    NO_CANDIDATES = "NO_CANDIDATES"
    # Linking the address raised an exception (see SearchStatus.EXCEPTION).
    EXCEPTION = "EXCEPTION"


# Disambiguation factors (see geo_disambiguation.DISAMBIGUATION_FACTOR_PRIORITY) that reflect the entity's own type
# match or agreement with sibling entities in the same address, as opposed to
# generic ranking criteria (population, country, preferred name, ...).
_CONTEXT_DISAMBIGUATION_FACTORS = {
    "child_parent_likelihood",
    "entity_types_matching_preferred",
    "entity_types_matching",
}


# LinkingOutcome.deciding_factor of an address linked without disambiguating
# between candidates: the only candidate entity, or the common parent of tied ones
UNAMBIGUOUS_DECIDING_FACTOR = "(single candidate)"
COMMON_PARENT_DECIDING_FACTOR = "(common parent of tied candidates)"


def _candidate_iri(candidate: LinkedAddress) -> str:
    return candidate.finest_grain_entity.linked_to.geographical_name.entity.iri


def _deciding_factor(address: AddressProcessingData, disambiguator: Disambiguator) -> str:
    """
    The disambiguation factor the choice of address.linked_to hinged on, by
    the same comparison steps Disambiguator.disambiguate ranks candidates
    with (see geo_disambiguation.deciding_factor), labelled as the step that
    decided (e.g. "fuzzy_similarity_score (no threshold)" when decided only
    once the primary factors are compared with no threshold). Candidates for
    the same entity as the linked one do not compete with it (disambiguate
    only treats distinct entities as ambiguous). Against each competitor, the
    deciding factor is that of the first step whose difference is
    significant; the address' is the latest step of those, i.e. the one that separated the
    linked candidate from its closest competitor. Only meaningful when
    address.linked_to is not None.
    """
    if address.linked_to_common_parent:
        return COMMON_PARENT_DECIDING_FACTOR
    best = address.linked_to
    best_iri = _candidate_iri(best)
    step_order = {step.label: i for i, step in enumerate(disambiguator.comparison_steps)}
    closest = None
    for competitor in address.possible_links or ():
        if _candidate_iri(competitor) == best_iri:
            continue
        decided = deciding_factor(best.scores, competitor.scores, disambiguator.comparison_steps)
        # a tie would have left the address ambiguous instead
        if decided is not None and (closest is None or step_order[decided[0]] > step_order[closest[0][0]]):
            closest = (decided, competitor)
    if closest is None:
        return UNAMBIGUOUS_DECIDING_FACTOR
    (factor, best_score, competitor_score), competitor = closest
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Address %s: best candidate beats its closest competitor %s on '%s' (%.3f vs %.3f)",
            address.id, _describe_candidate(competitor), factor, best_score, competitor_score)
    return factor


def _resolved_disambiguation_status(factor: str) -> str:
    """
    Distinguish why disambiguation was able to settle on a single candidate
    (see DisambiguationStatus.UNAMBIGUOUS/DISAMBIGUATED_BY_CONTEXT/
    DISAMBIGUATED_HEURISTICALLY/DISAMBIGUATED_BY_COMMON_PARENT) from the
    factor it hinged on (see _deciding_factor).
    """
    if factor == COMMON_PARENT_DECIDING_FACTOR:
        return DisambiguationStatus.DISAMBIGUATED_BY_COMMON_PARENT
    if factor == UNAMBIGUOUS_DECIDING_FACTOR:
        return DisambiguationStatus.UNAMBIGUOUS
    if factor_of_label(factor) in _CONTEXT_DISAMBIGUATION_FACTORS:
        return DisambiguationStatus.DISAMBIGUATED_BY_CONTEXT
    return DisambiguationStatus.DISAMBIGUATED_HEURISTICALLY


def _weighted_score_contributions(linked_address: LinkedAddress, disambiguator: Disambiguator) -> dict[str, float]:
    """
    The contribution of each weighted factor (see
    Disambiguator.weighted_score_contributions) to the address-level
    weighted score of a candidate, which is the sum of these contributions:
    the per-entity contributions averaged the way the per-entity scores are
    (see geo_disambiguation._average_scores, with missed words weighing
    MISSED_WORD_ENTITY_WEIGHT). Entities left without a match only lower the
    weighted score, so the contributions are scaled from their shares of it.
    """
    sums = defaultdict(float)
    for entity in linked_address.entities:
        weight = MISSED_WORD_ENTITY_WEIGHT if entity.is_missed_word else 1.0
        for factor, contribution in disambiguator.weighted_score_contributions(entity.scores).items():
            sums[factor] += weight * contribution
    total = sum(sums.values())
    if total == 0:
        return {}
    weighted_score = linked_address.scores.get("weighted_score", 0.0)
    return {factor: weighted_score * contribution / total for factor, contribution in sums.items()}


def _describe_raw_entity(entity: RawEntity) -> str:
    pre_linked = f" (pre-linked {entity.pre_linked_iri})" if entity.pre_linked_iri else ""
    return f"{entity.entity_type.name} {entity.raw_text!r}{pre_linked}"


def _describe_candidate(candidate: LinkedAddress) -> str:
    """
    Compact, human-readable summary of every entity of a candidate address
    and its scores, for debug logging. Only call this behind a
    logger.isEnabledFor(logging.DEBUG) check.
    """
    entities = ", ".join(
        f"{entity.entity_type.name} {entity.linked_to.geographical_name.name!r} "
        f"({normalize_iri(entity.linked_to.geographical_name.entity.iri)})"
        for entity in candidate.entities
    )
    scores = {factor: round(score, 3) for factor, score in candidate.scores.items()}
    return f"[{entities}] scores {scores}"


def _geo_db_search_status(matched_name: MatchedName) -> str:
    if matched_name.matching_method == "pre_linked":
        return SearchStatus.PRE_LINKED_DURING_PARSING
    if matched_name.matching_method == REGIONAL_TERM_MATCHING_METHOD:
        return SearchStatus.GEO_DB_REGIONAL_TERM_MATCH
    if matched_name.is_abbreviation_match:
        return SearchStatus.GEO_DB_ABBREVIATION_MATCH
    if matched_name.is_phonetic_match:
        return SearchStatus.GEO_DB_PHONETIC_MATCH
    if matched_name.is_partial_word_match:
        return SearchStatus.GEO_DB_PARTIAL_WORD_MATCH
    if matched_name.search_phase == "exact":
        return SearchStatus.GEO_DB_EXACT_MATCH
    if matched_name.search_phase == FUZZY_DISTANCE_1_SEARCH_PHASE:
        return SearchStatus.GEO_DB_FUZZY_DISTANCE_1_MATCH
    return SearchStatus.GEO_DB_FUZZY_MATCH


def _clean_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and value == "":
        return None
    return value


def _geonames_iri(geonames_id) -> Optional[str]:
    geonames_id = _clean_optional_str(geonames_id)
    if geonames_id is None:
        return None
    if isinstance(geonames_id, float):
        geonames_id = int(geonames_id)
    geonames_id = str(geonames_id).strip()
    if geonames_id == "":
        return None
    return f"http://sws.geonames.org/{geonames_id}"


def _extract_entity_texts(row, prefix: str) -> dict[str, Optional[str]]:
    # The regex parser's own "Unknown" group marks a location left
    # unspecified (e.g. "unbekannt", see regex_patterns.UNKNOWN_PATTERN), not
    # a place name of unknown type, so is never read as an Unknown entity;
    # those only come from _single_word_regex_city_as_unknown.
    return {
        entity_type: _clean_optional_str(row.get(f"{prefix}.{entity_type}.text"))
        for entity_type in ENTITY_TYPE_COLUMNS
        if entity_type != GeographicalEntityType.Unknown.name
    }


def _hotfix_truncated_regex_city(row, prefix: str):
    """
    HOTFIX: the corpus was regex-parsed while
    modules.regex_patterns.get_words_left split words after their first
    letter and dropped the last one, so a 2-3 letter address left as a single
    unparsed word got only its first letter as City (e.g. "Lom" -> "L", "KZ"
    -> "K"), marked "fully_parsed". Rather than re-parsing the corpus, the
    truncated City is widened here back to the whole word it starts, stopping
    at text assigned to any other entity.

    The bug is recognized as a regex (not LLM) parse whose City came from
    neither a common-name pattern nor a global regex, and does not end at a
    word boundary. Returns `row` unchanged when it does not apply, otherwise
    a copy with the corrected "{prefix}.City.*" columns.
    """
    raw_address = _clean_optional_str(row.get(f"{prefix}.raw"))
    city_text = _clean_optional_str(row.get(f"{prefix}.City.text"))
    city_start = _clean_optional_str(row.get(f"{prefix}.City.start"))
    city_end = _clean_optional_str(row.get(f"{prefix}.City.end"))
    if (
        raw_address is None or city_text is None or city_start is None or city_end is None
        or row.get(f"{prefix}.status") != "fully_parsed"
        or _clean_optional_str(row.get(f"{prefix}.global_regex")) is not None
        or _clean_optional_str(row.get(f"{prefix}.City.special_regex")) is not None
    ):
        return row
    city_start, city_end = int(city_start), int(city_end)
    if raw_address[city_start:city_end] != city_text:
        return row
    covered = [False] * len(raw_address)
    for column, value in row.items():
        if not (column.startswith(f"{prefix}.") and column.endswith(".start")) or column == f"{prefix}.City.start":
            continue
        start = _clean_optional_str(value)
        end = _clean_optional_str(row.get(column[:-len(".start")] + ".end"))
        if start is not None and end is not None:
            covered[int(start):int(end)] = [True] * (int(end) - int(start))
    fixed_end = city_end
    while fixed_end < len(raw_address) and raw_address[fixed_end].isalpha() and not covered[fixed_end]:
        fixed_end += 1
    if fixed_end == city_end:
        return row
    fixed_text = raw_address[city_start:fixed_end]
    logger.debug("Hotfix: widened truncated regex City %r to %r in %r", city_text, fixed_text, raw_address)
    return {**row, f"{prefix}.City.text": fixed_text, f"{prefix}.City.end": float(fixed_end)}


# As stored in the "{prefix}.global_regex" column by regex_patterns.regex_parse.
_CITY_SOMETHING_GLOBAL_REGEX = CITY_SOMETHING_REGEX.pattern.replace('\n', ' ')


def _single_word_regex_city_as_unknown(row, prefix: str, entity_texts: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
    """
    A single-word address the regex parser could not match against any
    pattern ends up entirely as City (see regex_patterns.regex_parse), though
    it is sometimes a Country instead (e.g. "Polen"). Such a City is
    relabelled Unknown, which search and disambiguation treat as a wildcard
    matching any entity type.

    Only applies to regex (not LLM) parses whose City came from neither a
    common-name pattern nor a global regex (as in _hotfix_truncated_regex_city)
    and carries no direct geonames link. The exception is the generic
    "City[/AboveCity]" global regex (CITY_SOMETHING_REGEX): on a single word
    it just captures the whole word as City, no more telling of its type.
    Most single-word addresses end up there instead of in regex_parse's own
    single-word branch, since get_words_left splits every word after its
    first letter. Returns `entity_texts` unchanged
    when it does not apply, otherwise a copy with the City text moved to
    Unknown.
    """
    raw_address = _clean_optional_str(row.get(f"{prefix}.raw"))
    city_text = entity_texts.get("City")
    if (
        raw_address is None or city_text is None
        or row.get(f"{prefix}.status") != "fully_parsed"
        or _clean_optional_str(row.get(f"{prefix}.global_regex")) not in (None, _CITY_SOMETHING_GLOBAL_REGEX)
        or _clean_optional_str(row.get(f"{prefix}.City.special_regex")) is not None
        or _clean_optional_str(row.get(f"{prefix}.City.geonames_id")) is not None
        or len(_WORD_PATTERN.findall(raw_address)) != 1
    ):
        return entity_texts
    logger.debug("Relabelled single-word regex City %r as Unknown in %r", city_text, raw_address)
    return {**entity_texts, "City": None, GeographicalEntityType.Unknown.name: city_text}


def _pre_linked_iris(row, prefix: str) -> dict[str, str]:
    """
    All geonames ids already attached to this address field's entities by
    parsing, keyed by entity type.
    """
    return {
        entity_type: iri
        for entity_type in PRE_LINKED_ENTITY_ORDER
        for iri in (_geonames_iri(row.get(f"{prefix}.{entity_type}.geonames_id")),)
        if iri is not None
    }


def _finest_entity_type_with_text(entity_texts: dict[str, Optional[str]]) -> Optional[str]:
    for entity_type in PRE_LINKED_ENTITY_ORDER:
        if entity_texts.get(entity_type):
            return entity_type
    return None


def _pre_linked_finest_entity(
    pre_linked_iris: dict[str, str], entity_texts: dict[str, Optional[str]]
) -> Optional[tuple[str, str]]:
    """
    Only when the pre-linked iri is for the finest grain entity actually
    present in the address (e.g. Neighborhood if parsed, else City, ...) can
    linking be short-circuited entirely: a pre-link for a coarser entity
    (e.g. Country) while a finer one (e.g. City) is still unresolved must
    instead go through the full pipeline, which resolves the pre-linked
    entity by id and propagates it downstream (see GeoDBSearch.apply).
    """
    finest_type = _finest_entity_type_with_text(entity_texts)
    if finest_type is not None and finest_type in pre_linked_iris:
        return finest_type, pre_linked_iris[finest_type]
    return None


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.split()).casefold()


# Candidate words of the raw address, delimited by whitespace and the
# separators parsed addresses commonly use between their parts.
_WORD_PATTERN = re.compile(r"[^\s/,;()\[\]<>\"]+")
# Only words with more than this many letters (digits and punctuation not
# counting) are recovered as missed AboveCity words.
_MISSED_WORD_MIN_LETTERS = 4
# Words that commonly go unassigned by parsing but are not a broader place
# around the City (administrative qualifiers, institutions, ...), so are
# never recovered as missed AboveCity words; nor are geo_db_search's stop
# words. Compared after ascii/german normalization (see _is_excluded_missed_word).
MISSED_WORD_EXCLUSION_LIST = (
    "Kreis",
    "Reg.Bez.",
    "Prov.",
    "Todeserklärung",
    "Altersheim",
    "DP-Lager",
    "Parz.",
    "Maria",
    "Haus",
    "Rh.",
    "Strasse",
    "Street",
    "St."
)
_NORMALIZED_MISSED_WORD_EXCLUSIONS = frozenset(
    normalized
    for word in MISSED_WORD_EXCLUSION_LIST
    for normalized in (ascii_normalize(word), german_normalize(word))
) | _STOP_WORDS


def _is_excluded_missed_word(word: str) -> bool:
    ascii_normalized = ascii_normalize(word)
    german_normalized = german_normalize(word)
    return (
        ascii_normalized in _NORMALIZED_MISSED_WORD_EXCLUSIONS
        or german_normalized in _NORMALIZED_MISSED_WORD_EXCLUSIONS
    )


def _count_non_stop_words(raw_address: Optional[str]) -> int:
    """Number of words in `raw_address`, ignoring geo_db_search's stop words."""
    if not raw_address:
        return 0
    count = 0
    for word_match in _WORD_PATTERN.finditer(raw_address):
        inner = re.search(r"[^\W_](?:.*[^\W_])?\.?", word_match.group())
        if inner is None:
            continue
        # Also match the raw word: normalization drops the period some stop
        # words (e.g. "i.") are written with.
        forms = (inner.group(), ascii_normalize(inner.group()))
        if any(stop_word.fullmatch(form) for stop_word in _compiled_stop_words for form in forms):
            continue
        count += 1
    return count


def _assigned_texts(row, prefix: str) -> list[str]:
    """Every text parsing assigned to some entity type (any "{prefix}.*.text" column)."""
    return [
        text
        for column, value in row.items()
        if column.startswith(f"{prefix}.") and column.endswith(".text")
        for text in (_clean_optional_str(value),)
        if isinstance(text, str)
    ]


def _missed_words(raw_address: Optional[str], assigned_texts: list[str]) -> list[tuple[str, AddressSpan]]:
    """
    Words of `raw_address` with more than _MISSED_WORD_MIN_LETTERS letters
    that parsing did not assign to any entity type, with their span in
    `raw_address`. Leading/trailing punctuation is left out of the word,
    except for a trailing period (e.g. the abbreviation "Bergstr."), which
    is kept.

    A word counts as assigned when it overlaps an occurrence of an assigned
    text in `raw_address`, or when it equals one of the words of an
    assigned text (which may not appear verbatim in the raw address, e.g.
    when the LLM parser normalized its surroundings).
    """
    if not raw_address:
        return []
    covered = [False] * len(raw_address)
    assigned_words = set()
    for text in assigned_texts:
        for occurrence in re.finditer(re.escape(text), raw_address, re.IGNORECASE):
            covered[occurrence.start():occurrence.end()] = [True] * (occurrence.end() - occurrence.start())
        assigned_words.update(word.casefold() for word in re.findall(r"[^\W\d_]+", text))

    missed = []
    for word_match in _WORD_PATTERN.finditer(raw_address):
        # From the word's first to its last alphanumeric character.
        inner = re.search(r"[^\W_](?:.*[^\W_])?", word_match.group())
        if inner is None:
            continue
        start, end = word_match.start() + inner.start(), word_match.start() + inner.end()
        if end < word_match.end() and raw_address[end] == ".":
            end += 1
        letters = [c for c in raw_address[start:end] if c.isalpha()]
        if len(letters) <= _MISSED_WORD_MIN_LETTERS:
            continue
        if any(covered[start:end]):
            continue
        if "".join(letters).casefold() in assigned_words:
            continue
        if _is_excluded_missed_word(raw_address[start:end]):
            continue
        missed.append((raw_address[start:end], AddressSpan(start, end)))
    return missed


def _abbreviation_iri(text: str) -> Optional[str]:
    """
    The iri of the place an entity's whole text abbreviates (e.g. Bavaria for
    "Bay."), from abbrev_list_expander, if any: such an entity is pre-linked
    to it instead of searched for.
    """
    # entity texts of types not searched for may be left NaN by parsing
    if not isinstance(text, str):
        return None
    return abbrev_list_expander.expand_abbreviations(text)[1]


def _build_entities(
    entity_texts: dict[str, Optional[str]], above_city_text: Optional[str], pre_linked_iris: dict[str, str],
    missed_words: list[tuple[str, AddressSpan]] = (), raw_address: Optional[str] = None,
) -> list[RawEntity]:
    entities = [
        RawEntity.with_parsed(
            entity_type=GeographicalEntityType[entity_type], raw_text=text,
            pre_linked_iri=pre_linked_iris.get(entity_type) or _abbreviation_iri(text)
        )
        for entity_type, text in entity_texts.items()
        if text
    ]
    if above_city_text:
        # AboveCity (e.g. "Danzig" in "Putzig (Danzig)") names a broader place
        # around the target, without parsing knowing which entity type above
        # City it actually is; GeoDBSearch expands it into a disjunction over
        # every such type (District, Region, State, Country) when searching.
        # However parsing sometimes captures the same text into AboveCity and
        # into one of the typed columns (e.g. "Sofia, Bulg." -> City="Sofia",
        # AboveCity="Bulg." *and* Country="Bulg."); adding it again then would
        # only duplicate an entity already covered (and possibly pre-linked)
        # above, at the cost of a redundant search.
        normalized_above_city = _normalize_for_dedup(above_city_text)
        is_duplicate = any(
            _normalize_for_dedup(entity.raw_text) == normalized_above_city for entity in entities
        )
        if not is_duplicate:
            entities.append(RawEntity.with_parsed(
                entity_type=GeographicalEntityType.AboveCity, raw_text=above_city_text,
                pre_linked_iri=_abbreviation_iri(above_city_text)))
    if entity_texts.get("City"):
        # Words parsing left unassigned next to an identified City (e.g.
        # "Weinheim/Bergstr." parsed as just City="Weinheim") are most often
        # a broader place around it; see _missed_words. Only trusted for
        # two-word addresses (ignoring stop words): City plus the missed word.
        for text, span in missed_words:
            entities.append(RawEntity.with_parsed(
                entity_type=GeographicalEntityType.AboveCity, raw_text=text, span=span, is_missed_word=True,
                pre_linked_iri=_abbreviation_iri(text)))
    return entities


@dataclass(frozen=True)
class LinkedEntityMetadata:
    """
    Metadata about a single entity that ended up linked, either as the result
    of GeoDBSearch + disambiguation, or as a direct camp/pre-linked match.
    """
    entity_type: str
    iri: str
    # The original (query) text for this entity, when known.
    raw_text: Optional[str] = None
    # Country name of the matched entity, when known.
    country: Optional[str] = None
    # String-matching metadata produced by GeoDBSearch for this entity (empty
    # for camp/pre-linked matches, which do not go through it).
    matching: dict = field(default_factory=dict)
    # Per-criterion disambiguation scores for this entity (see
    # geo_disambiguation.Disambiguator._score_individual_match).
    disambiguation_scores: dict = field(default_factory=dict)
    # Number of candidates GeoDBSearch found for this entity, before dedup
    # and disambiguation.
    search_candidate_count: Optional[int] = None
    
    total_time : Optional[float] = None

@dataclass(frozen=True)
class LinkingOutcome:
    iri: Optional[str]
    entity_type: Optional[str]
    tags: tuple[str, ...]
    # Which method was used to search for candidates (see SearchStatus).
    search_status: str
    # Outcome of disambiguating those candidates (see DisambiguationStatus).
    disambiguation_status: str
    linked_entities: tuple[LinkedEntityMetadata, ...] = ()
    # All candidate IRIs, when disambiguation could not settle on a single one.
    ambiguous_iris: Optional[list[str]] = None
    # Candidate counts for the two steps between search and disambiguation:
    # every scored candidate address, and the ones within the ambiguity
    # threshold of the best one.
    possible_links_count: Optional[int] = None
    likely_links_count: Optional[int] = None
    # Address-level (average of per-entity) disambiguation scores for the
    # winning candidate.
    disambiguation_scores: dict = field(default_factory=dict)
    # The disambiguation factor the choice of the linked candidate hinged on
    # (see _deciding_factor), when one was linked through GeoDBSearch.
    deciding_factor: Optional[str] = None
    # The phase of GeoDBSearch (see MatchedName.search_phase) that retrieved
    # the match of the reference entity of the linked candidate, or of the
    # first likely candidate when disambiguation could not settle on one.
    reference_search_phase: Optional[str] = None
    # The contribution of each weighted factor to the linked candidate's
    # weighted score (see _weighted_score_contributions), when one was
    # linked through GeoDBSearch.
    weighted_score_contributions: dict = field(default_factory=dict)
    total_time : Optional[float] = None
    # The address as it was when linking stopped (after disambiguation, when
    # it got that far), with every scored candidate; only kept when
    # link_field is called with keep_address=True, for debugging.
    address: Optional[AddressProcessingData] = field(default=None, repr=False, compare=False)


def _matching_metadata(matched_name: MatchedName) -> dict:
    return {
        "method": matched_name.matching_method,
        "score": matched_name.matching_score,
        "matched_name": matched_name.geographical_name.name,
        "edit_distance": matched_name.edit_distance,
        "cleaned_edit_distance": matched_name.cleaned_edit_distance,
        "is_abbreviation_match": matched_name.is_abbreviation_match,
        "is_phonetic_match": matched_name.is_phonetic_match,
        "is_partial_word_match": matched_name.is_partial_word_match,
        "fuzzy_score": matched_name.fuzzy_score,
        "cleaned_similarity": matched_name.cleaned_similarity,
        "search_phase": matched_name.search_phase,
    }


def _linked_entities_metadata(
    linked_address: LinkedAddress, search_candidate_counts: dict[str, int]
) -> tuple[LinkedEntityMetadata, ...]:
    return tuple(
        LinkedEntityMetadata(
            entity_type=entity.entity_type.name,
            iri=normalize_iri(entity.linked_to.geographical_name.entity.iri),
            raw_text=entity.raw_text,
            country=(
                entity.linked_to.geographical_name.entity.country.country_name
                if entity.linked_to.geographical_name.entity.country is not None
                else None
            ),
            matching=_matching_metadata(entity.linked_to),
            disambiguation_scores={criterion: score.score for criterion, score in entity.scores.items()},
            search_candidate_count=search_candidate_counts.get(entity.entity_type.name),
        )
        for entity in linked_address.entities
    )



def link_field(
    row,
    prefix: str,
    camp_matcher: CampReferenceMatcher,
    geo_db_searcher: GeoDBSearch,
    disambiguator: Disambiguator,
    keep_address: bool = True,
) -> LinkingOutcome:
    """
    Run the full linking pipeline for a single address field of a single row.
    With keep_address=True, the returned outcome also carries the address
    as it was when linking stopped (see LinkingOutcome.address).
    """
    outcome, address = _link_field(row, prefix, camp_matcher, geo_db_searcher, disambiguator)
    if keep_address:
        outcome = replace(outcome, address=address)
    return outcome


class ParsedFieldEntities(NamedTuple):
    # The row after the parsing hotfixes (see _hotfix_truncated_regex_city).
    row: dict
    raw_address: Optional[str]
    entity_texts: dict[str, Optional[str]]
    pre_linked_iris: dict[str, str]
    # The entities linking searches for: the parsed ones plus AboveCity and
    # the words recovered from those parsing left unassigned (see _build_entities).
    entities: list[RawEntity]


def parse_field_entities(row, prefix: str) -> ParsedFieldEntities:
    """The entities to link for one address field of a parsed row, as link_field builds them."""
    row = _hotfix_truncated_regex_city(row, prefix)
    raw_address = _clean_optional_str(row.get(f"{prefix}.raw"))
    entity_texts = _single_word_regex_city_as_unknown(row, prefix, _extract_entity_texts(row, prefix))
    above_city_text = _clean_optional_str(row.get(f"{prefix}.AboveCity.text"))
    pre_linked_iris = _pre_linked_iris(row, prefix)
    missed_words = _missed_words(raw_address, _assigned_texts(row, prefix))
    entities = _build_entities(entity_texts, above_city_text, pre_linked_iris, missed_words, raw_address)
    return ParsedFieldEntities(row, raw_address, entity_texts, pre_linked_iris, entities)


def _link_field(
    row,
    prefix: str,
    camp_matcher: CampReferenceMatcher,
    geo_db_searcher: GeoDBSearch,
    disambiguator: Disambiguator,
) -> tuple[LinkingOutcome, AddressProcessingData]:
    start = time.monotonic()
    def _elapsed():
        nonlocal start
        return time.monotonic() - start
    row, raw_address, entity_texts, pre_linked_iris, entities = parse_field_entities(row, prefix)
    card_id = row.get("card_id", row.get("filename"))
    bzk_field_name = FIELD_PREFIXES[prefix]
    address_id = str(row.get("address_id", f"{card_id}/{bzk_field_name.name}"))
    address = AddressProcessingData(
        card_id=card_id,
        id=address_id,
        full_address=raw_address,
        bzk_field_name=bzk_field_name,
        entities=entities
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Linking address %s (%r), field %s, parse status %s: entities %s",
            address.id, raw_address, address.bzk_field_name, _clean_optional_str(row.get(f"{prefix}.status")),
            [_describe_raw_entity(entity) for entity in entities])

    pre_linked = _pre_linked_finest_entity(pre_linked_iris, entity_texts)
    if pre_linked is not None:
        entity_type, iri = pre_linked
        logger.debug(
            "Address %s: parsing already linked the finest entity (%s) to %s; skipping search",
            address.id, entity_type, iri)
        return LinkingOutcome(
            iri=iri, entity_type=entity_type, tags=(),
            search_status=SearchStatus.PRE_LINKED_DURING_PARSING,
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            linked_entities=(LinkedEntityMetadata(entity_type=entity_type, iri=iri),),
            total_time=_elapsed()
        ), address

    tagged_text = raw_address or entity_texts.get("City")
    address_tags = tag_address(tagged_text)
    logger.debug("Address %s: tagged %r as %s", address.id, tagged_text, sorted(address_tags))
    if "location_unspecified" in address_tags:
        logger.debug(
            "Address %s: nothing resembling a place name is left after stripping the tagged terms, "
            "filler words and punctuation; treating it as not a location", address.id)
        reasons = sorted(address_tags - {"location_unspecified"})
        return LinkingOutcome(
            iri=None, entity_type=None, tags=("unresolved", *sorted(address_tags)),
            search_status=f"{SearchStatus.NO_LOCATION} ({', '.join(reasons)})" if reasons else SearchStatus.NO_LOCATION,
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            total_time=_elapsed()
        ), address
    # Tags that describe the value without ruling out a location (e.g.
    # "displaced_persons_camp" for "DP-Lager Foehrenwald"); carried over
    # regardless of how the rest of the pipeline resolves the location.
    extra_tags = tuple(sorted(address_tags))

    # The full raw text is tried first since camp/ghetto labels often include
    # a "KZ "/"Ghetto " prefix (e.g. "KZ Auschwitz" is itself a known label);
    # the parsed city name is tried too since it may already have been
    # cleaned of surrounding words (e.g. "deportiert nach Auschwitz").
    camp_match = camp_matcher.match(address, extra_tags)
    if camp_match is not None:
        logger.debug(
            "Address %s: matched camp/ghetto %r -> %s (wikidata %s, tags %s); skipping search",
            address.id, camp_match.label, camp_match.iri, camp_match.wikidata_iri, sorted(camp_match.tags))
        camp_tags = set(camp_match.tags)
        if not camp_tags & {"concentration_camp", "ghetto"}:
            # Sanity check: every indexed camp/ghetto carries one of these
            # (see camp_search._HIGHER_CLASS_TAGS).
            logger.warning(
                "Address %s: camp/ghetto match %r -> %s (wikidata %s) has neither a "
                "concentration_camp nor a ghetto tag (tags %s)",
                address.id, camp_match.label, camp_match.iri, camp_match.wikidata_iri, sorted(camp_match.tags))
        camp_tags.update(extra_tags)
        camp_tags = tuple(sorted(camp_tags))
        return LinkingOutcome(
            iri=camp_match.iri, entity_type="Camp", tags=camp_tags,
            search_status=f"{SearchStatus.CAMP_REFERENCE}",
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            linked_entities=(LinkedEntityMetadata(entity_type="Camp", iri=camp_match.iri),),
            total_time=_elapsed()
        ), address
    
    if len(entities) == 0:
        logger.debug("Address %s: parsing yielded no entities to search for", address.id)
        return LinkingOutcome(
            iri=None, entity_type=None, tags=("unresolved", *extra_tags),
            search_status=SearchStatus.NO_ENTITIES,
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            total_time=_elapsed()
        ), address

    address = AddressProcessingData(
        card_id=row.get("card_id"),
        id=str(row.get("address_id", row.get("filename"))),
        full_address=raw_address,
        bzk_field_name=FIELD_PREFIXES[prefix],
        entities=entities,
    )
    address = geo_db_searcher.apply(address)
    search_candidate_counts = {
        entity.entity_type.name: len(entity.matches)
        for entity in address.entities
        if isinstance(entity, MatchedEntity) and entity.matches is not None
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Address %s: GeoDBSearch candidates per entity: %s", address.id,
            [f"{entity.entity_type.name} {entity.raw_text!r}: "
             f"{len(entity.matches) if isinstance(entity, MatchedEntity) and entity.matches is not None else 0}"
             for entity in address.entities])
    address = disambiguator.disambiguate(address)
    possible_links_count = len(address.possible_links or ())
    likely_links_count = len(address.likely_links or ())

    if address.linked_to is not None:
        linked_address = address.linked_to
        linked_entities = _linked_entities_metadata(linked_address, search_candidate_counts)
        finest_type = linked_address.finest_grain_entity.entity_type.name
        finest_iri = next(m.iri for m in linked_entities if m.entity_type == finest_type)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Address %s: linked to %s out of %d possible link(s)",
                address.id, _describe_candidate(linked_address), possible_links_count)
        decided_by = _deciding_factor(address, disambiguator)
        disambiguation_status = _resolved_disambiguation_status(decided_by)
        ambiguous_iris = None
        search_status_match = linked_address.finest_grain_entity.linked_to
        reference_match = linked_address.reference_entity.linked_to
        if address.linked_to_common_parent:
            # The linked entity was not itself found by search, but inferred
            # from the ambiguous candidates, which are kept for reference.
            ambiguous_iris = [
                normalize_iri(candidate.finest_grain_entity.linked_to.geographical_name.entity.iri)
                for candidate in address.likely_links
            ]
            search_status_match = address.likely_links[0].finest_grain_entity.linked_to
            reference_match = address.likely_links[0].reference_entity.linked_to
        return LinkingOutcome(
            iri=finest_iri,
            entity_type=finest_type,
            tags=extra_tags,
            search_status=_geo_db_search_status(search_status_match),
            disambiguation_status=disambiguation_status,
            linked_entities=linked_entities,
            ambiguous_iris=ambiguous_iris,
            possible_links_count=possible_links_count,
            likely_links_count=likely_links_count,
            disambiguation_scores=dict(linked_address.scores),
            deciding_factor=decided_by,
            reference_search_phase=reference_match.search_phase,
            weighted_score_contributions=_weighted_score_contributions(linked_address, disambiguator),
            total_time=_elapsed()
        ), address
    if likely_links_count > 1:
        first_candidate = address.likely_links[0]
        ambiguous_iris = [
            normalize_iri(candidate.finest_grain_entity.linked_to.geographical_name.entity.iri)
            for candidate in address.likely_links
        ]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Address %s: disambiguation could not choose between %d likely candidates "
                "(within the %s ambiguity threshold of each other) out of %d possible link(s): %s",
                address.id, likely_links_count, disambiguator.score_threshold, possible_links_count,
                [_describe_candidate(candidate) for candidate in address.likely_links])
        return LinkingOutcome(
            iri=None,
            entity_type=first_candidate.finest_grain_entity.entity_type.name,
            tags=("unresolved", *extra_tags),
            search_status=_geo_db_search_status(first_candidate.finest_grain_entity.linked_to),
            disambiguation_status=DisambiguationStatus.AMBIGUOUS,
            linked_entities=_linked_entities_metadata(first_candidate, search_candidate_counts),
            ambiguous_iris=ambiguous_iris,
            possible_links_count=possible_links_count,
            likely_links_count=likely_links_count,
            disambiguation_scores=dict(first_candidate.scores),
            reference_search_phase=first_candidate.reference_entity.linked_to.search_phase,
            total_time=_elapsed()
        ), address
    logger.debug(
        "Address %s: no candidates left after search and disambiguation (%d possible link(s))",
        address.id, possible_links_count)
    return LinkingOutcome(
        iri=None, entity_type=None, tags=("unresolved", *extra_tags),
        search_status=SearchStatus.GEO_DB_NO_CANDIDATES,
        disambiguation_status=DisambiguationStatus.NO_CANDIDATES,
        possible_links_count=possible_links_count, likely_links_count=likely_links_count,
        total_time=_elapsed()
    ), address


def apply_outcome_to_row(row: dict, prefix: str, outcome: LinkingOutcome) -> dict:
    """
    Build the new "{prefix}.*" columns to add to an input row for one address
    field, keeping the same dotted naming convention as the input schema.
    """
    updates = {
        f"{prefix}.iri": outcome.iri,
        f"{prefix}.entity_type": outcome.entity_type,
        f"{prefix}.tags": list(outcome.tags),
        # How the address field was parsed out of the raw text, propagated
        # as-is from the input (e.g. "fully_parsed", "llm_parsed", ...).
        f"{prefix}.parsing_status": _clean_optional_str(row.get(f"{prefix}.status")),
        f"{prefix}.search_status": outcome.search_status,
        f"{prefix}.disambiguation_status": outcome.disambiguation_status,
    }
    if outcome.ambiguous_iris is not None:
        updates[f"{prefix}.ambiguous_iris"] = outcome.ambiguous_iris
    if outcome.possible_links_count is not None:
        updates[f"{prefix}.possible_links_count"] = outcome.possible_links_count
    if outcome.likely_links_count is not None:
        updates[f"{prefix}.likely_links_count"] = outcome.likely_links_count
    for criterion, score in outcome.disambiguation_scores.items():
        updates[f"{prefix}.disambiguation_score.{criterion}"] = score
    if outcome.deciding_factor is not None:
        updates[f"{prefix}.deciding_factor"] = outcome.deciding_factor
    if outcome.reference_search_phase is not None:
        updates[f"{prefix}.reference_search_phase"] = outcome.reference_search_phase
    for criterion, contribution in outcome.weighted_score_contributions.items():
        updates[f"{prefix}.weighted_score_contribution.{criterion}"] = contribution
    for entity_metadata in outcome.linked_entities:
        entity_prefix = f"{prefix}.{entity_metadata.entity_type}"
        updates[f"{entity_prefix}.iri"] = entity_metadata.iri
        if entity_metadata.country is not None:
            updates[f"{entity_prefix}.country"] = entity_metadata.country
        for key, value in entity_metadata.matching.items():
            updates[f"{entity_prefix}.match_{key}"] = value
        for criterion, score in entity_metadata.disambiguation_scores.items():
            updates[f"{entity_prefix}.disambiguation_score.{criterion}"] = score
        if entity_metadata.search_candidate_count is not None:
            updates[f"{entity_prefix}.search_candidate_count"] = entity_metadata.search_candidate_count
    return updates


def prefixes_in_row(row: dict) -> list[str]:
    return [prefix for prefix in FIELD_PREFIXES if f"{prefix}.raw" in row]


def process_row(
    row: dict,
    camp_matcher: CampReferenceMatcher,
    geo_db_searcher: GeoDBSearch,
    disambiguator: Disambiguator,
) -> dict:
    """
    Link every address field present in `row` and return `row` augmented with
    the new "{prefix}.*" linking columns for each of them. An address field
    whose linking raises an exception is logged and gets the columns of
    exception_columns instead, so that one address cannot stop a whole run.
    """
    output_row = dict(row)
    for prefix in prefixes_in_row(row):
        try:
            outcome = link_field(row, prefix, camp_matcher, geo_db_searcher, disambiguator, keep_address=False)
            output_row.update(apply_outcome_to_row(row, prefix, outcome))
        except Exception as exception:
            if is_interruption(exception):
                raise
            logger.exception("Linking %s of %s raised an exception", prefix, _row_label(row))
            output_row.update(exception_columns(row, prefix, exception))
    return output_row


def _row_label(row: dict):
    """Names a row in log messages."""
    return row.get("filename", row.get("card_id"))


def is_interruption(exception: BaseException) -> bool:
    """
    Whether `exception` stems from the run being interrupted (Ctrl+C reaching
    a duckdb query) rather than from linking an address.
    """
    return isinstance(exception, duckdb.InterruptException)


def exception_columns(row: dict, prefix: str, exception: BaseException) -> dict:
    """The "{prefix}.*" columns of an address field whose linking raised `exception`."""
    return {
        f"{prefix}.iri": None,
        f"{prefix}.entity_type": None,
        f"{prefix}.tags": [],
        f"{prefix}.parsing_status": _clean_optional_str(row.get(f"{prefix}.status")),
        f"{prefix}.search_status": SearchStatus.EXCEPTION,
        f"{prefix}.disambiguation_status": DisambiguationStatus.EXCEPTION,
        f"{prefix}.exception": f"{type(exception).__name__}: {exception}",
    }


def is_exception_row(row: dict) -> bool:
    """Whether linking any address field of an output row raised an exception."""
    return any(
        row.get(f"{prefix}.search_status") == SearchStatus.EXCEPTION
        for prefix in prefixes_in_row(row)
    )


def _default_significance_threshold() -> float:
    # A module level function rather than a lambda keeps the Disambiguator picklable.
    return DEFAULT_SIGNIFICANCE_THRESHOLD


def build_camp_matcher(camps_reference_path: str | Path = DEFAULT_CAMPS_REFERENCE_PATH) -> CampReferenceMatcher:
    logger.info("Loading camp/ghetto reference matcher...")
    camp_matcher = CampReferenceMatcher(camps_reference_path)
    camp_matcher.initialize()
    return camp_matcher


def build_geo_db_searcher(
    search_index_path: str,
    search_cache_db: str = ":memory:",
    geo_db_path: str = "geo.duckdb",
) -> GeoDBSearch:
    logger.info("Initializing GeoDB searcher...")
    search_index = TantivySearchIndex(search_index_path)
    geo_db_searcher = GeoDBSearch(
        search_cache_db,
        search_index=search_index,
        prune_score_threshold=SEARCH_PRUNE_SCORE_THRESHOLD,
        geo_db_path=geo_db_path,
        topk=50
    )
    geo_db_searcher.initialize()
    return geo_db_searcher


def build_disambiguator(geo_db_path: str = "geo.duckdb") -> Disambiguator:
    significance_theresholds = defaultdict(_default_significance_threshold)
    for k, v in SIGNIFICANCE_THRESHOLDS.items():
        significance_theresholds[k] = v
    return Disambiguator(
        significance_thresholds=significance_theresholds,
        primary_priority=list(DISAMBIGUATION_FACTOR_PRIMARY_PRIORITY),
        secondary_priority=list(DISAMBIGUATION_FACTOR_SECONDARY_PRIORITY),
        score_weights=dict(WEIGHTED_DISAMBIGUATION_FACTORS),
        population_rounding_factor=DISAMBIGUATION_POPULATION_ROUNDING_FACTOR,
        min_population_order_of_magnitude=MIN_POPULATION_ORDER_OF_MAGNITUDE,
        score_prune_thresholds=dict(DISAMBIGUATION_SCORE_PRUNE_THRESHOLDS),
        geo_db_path=geo_db_path,
        parent_child_max_distance_km_by_admin_level=dict(PARENT_CHILD_MAX_DISTANCE_KM_BY_ADMIN_LEVEL),
    )


# Worker processes linking addresses in parallel (see LinkingPool): enough to
# keep processing matches and disambiguating while others search, but few
# enough not to compete over the cores the searches likely use (8 linked
# about 3x faster than 1, while 32 were slower than 8)
DEFAULT_NUM_WORKERS = ((os.process_cpu_count() or 6) * 3) // 4

# Tasks submitted to a LinkingPool ahead of the result it yields next, per
# worker: enough to keep every worker busy while an address that takes long
# holds up the results queued behind it, while bounding what is held in memory
LINKING_POOL_TASKS_IN_FLIGHT_PER_WORKER = 8

# The linking steps of a LinkingPool worker process (see _initialize_worker)
_worker_steps: Optional[tuple[CampReferenceMatcher, GeoDBSearch, Disambiguator]] = None


class _ParentLoggerHandler(logging.Handler):
    """Hands the records forwarded by the worker processes over to the loggers of the main process."""
    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def _initialize_worker(pickled_steps: bytes, log_queue, log_level: int) -> None:
    """
    Initializer of a LinkingPool worker process: its copy of the linking
    steps, and its logging forwarded to the main process.
    """
    global _worker_steps
    root_logger = logging.getLogger()
    root_logger.handlers[:] = [logging.handlers.QueueHandler(log_queue)]
    root_logger.setLevel(log_level)
    _worker_steps = pickle.loads(pickled_steps)


def _link_field_in_worker(row: dict) -> "LinkingOutcome":
    return link_field(row, row["prefix"], *_worker_steps)


def _process_row_in_worker(row: dict) -> dict:
    return process_row(row, *_worker_steps)


def _apply_in_worker(function: Callable, row: dict):
    return function(row, *_worker_steps)


def _wait_for_all_workers(barrier) -> None:
    """
    Task holding a worker until every other worker runs one too, so that
    one such task per worker has every worker started and initialized (see
    LinkingPool.__init__).
    """
    barrier.wait()


class LinkingPool:
    """
    Links addresses in parallel worker processes. The linking steps are
    initialized once by the caller, in the main process, and each worker gets
    a copy of them: the smaller data they hold as is, while the larger tables
    (e.g. the regional terms, the camp reference index) are loaded by each
    copy from the cache files the original's initialize wrote, and the
    tantivy index and the duckdb connections, which only refer to files on
    disk, are reopened by each copy (see their __getstate__). Each worker
    then links one address at a time, its log records being forwarded to the
    loggers of the main process. Creating the pool waits for every worker to
    be initialized, so that linking time does not include it.

    Workers are spawned rather than forked, the main process holding threads
    (tantivy's, duckdb's) a fork would not carry over. The steps are copied
    as they are when the pool is created: changes to them (or to the code,
    e.g. by autoreload) only reach a new pool.
    """
    def __init__(
        self,
        camp_matcher: CampReferenceMatcher,
        geo_db_searcher: GeoDBSearch,
        disambiguator: Disambiguator,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        self.num_workers = num_workers
        context = multiprocessing.get_context("spawn")
        self._log_queue = context.Queue()
        self._log_listener = logging.handlers.QueueListener(self._log_queue, _ParentLoggerHandler())
        self._log_listener.start()
        # pickled once here rather than once per worker
        pickled_steps = pickle.dumps((camp_matcher, geo_db_searcher, disambiguator))
        logger.info(
            "Starting %d linking worker processes (linking steps copied as %.1f MB)",
            num_workers, len(pickled_steps) / 1e6)
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=context, initializer=_initialize_worker,
            initargs=(pickled_steps, self._log_queue, logging.getLogger().getEffectiveLevel()))
        # workers are otherwise only started as tasks arrive, and initialized
        # while the first addresses wait on them
        with context.Manager() as manager:
            barrier = manager.Barrier(num_workers)
            waiting = [self._executor.submit(_wait_for_all_workers, barrier) for _ in range(num_workers)]
            for future in waiting:
                future.result()
        logger.info("Started %d linking worker processes", num_workers)

    def _map(self, function: Callable, items: Iterable, on_error: Optional[Callable] = None) -> Iterator:
        """
        `function` applied to every item by the workers, yielding the results
        in the order of `items`, which may be a lazy iterable too large to hold
        in memory (see LINKING_POOL_TASKS_IN_FLIGHT_PER_WORKER). With
        `on_error`, an item whose task raised an exception (e.g. its result
        could not be sent back) yields on_error(item, exception) instead,
        unless the pool itself broke (e.g. a worker was killed).
        """
        def result(item, future):
            try:
                return future.result()
            except Exception as exception:
                if on_error is None or isinstance(exception, BrokenExecutor):
                    raise
                return on_error(item, exception)

        max_in_flight = self.num_workers * LINKING_POOL_TASKS_IN_FLIGHT_PER_WORKER
        pending = deque()
        for item in items:
            pending.append((item, self._executor.submit(function, item)))
            if len(pending) >= max_in_flight:
                yield result(*pending.popleft())
        while pending:
            yield result(*pending.popleft())

    def link_fields(self, rows: Iterable[dict], on_error: Optional[Callable] = None) -> Iterator["LinkingOutcome"]:
        """link_field for each row, of the address field named by its "prefix"."""
        return self._map(_link_field_in_worker, rows, on_error)

    def process_rows(self, rows: Iterable[dict], on_error: Optional[Callable] = None) -> Iterator[dict]:
        """process_row for each row."""
        return self._map(_process_row_in_worker, rows, on_error)

    def map(self, function: Callable, rows: Iterable[dict], on_error: Optional[Callable] = None) -> Iterator:
        """
        function(row, camp_matcher, geo_db_searcher, disambiguator) for each
        row, with the workers' linking steps; `function` must be picklable
        (e.g. defined at module level).
        """
        return self._map(partial(_apply_in_worker, function), rows, on_error)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._log_listener.stop()
        self._log_queue.close()

    def __enter__(self) -> "LinkingPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Stats:
    def __init__(self):
        self.status_counts = {
            "location_unspecified" : 0,
            "unlinked" : 0,
            "linked" : 0,
            "exception" : 0
        }
        self.search_status_counts = defaultdict(int)
        self.disambiguation_status_counts = defaultdict(int)
        self.tag_counts = defaultdict(int)
        self.observed_addresses = 0

    def observe_row(self, row : dict):
        for prefix in prefixes_in_row(row):
            raw_addr = row[f"{prefix}.raw"]
            if raw_addr is None or raw_addr == "":
                continue
            tags = row[f"{prefix}.tags"]
            iri = row[f"{prefix}.iri"]
            if row[f"{prefix}.search_status"] == SearchStatus.EXCEPTION:
                status = "exception"
            elif iri is None:
                if "location_unspecified" in tags:
                    status = "location_unspecified"
                else:
                    status = "unlinked"
            else:
                status = "linked"
            self.status_counts[status] += 1
            self.observed_addresses += 1
            self.search_status_counts[ row[f"{prefix}.search_status"]] += 1
            for tag in tags:
                self.tag_counts[tag] += 1
            self.disambiguation_status_counts[row[f"{prefix}.disambiguation_status"]] += 1

    def summary(self) -> str:
        tables = [
            self._format_table("Status", self.status_counts),
            self._format_table("Search status", self.search_status_counts),
            self._format_table("Disambiguation status", self.disambiguation_status_counts),
            self._format_table("Tag", self.tag_counts),
        ]
        lines = [f"Observed addresses: {self.observed_addresses:,}", ""]
        for i in range(0, len(tables), 2):
            pair = tables[i:i + 2]
            widths = [max(len(line) for line in table) for table in pair]
            height = max(len(table) for table in pair)
            for j in range(height):
                cells = [
                    (table[j] if j < len(table) else "").ljust(width)
                    for table, width in zip(pair, widths)
                ]
                lines.append("    ".join(cells).rstrip())
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_table(self, title: str, counts: dict) -> list[str]:
        total = self.observed_addresses
        rows = [
            (str(key), f"{count:,}", f"{count / total:.1%}" if total else "-")
            for key, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
        key_width = max([len(title)] + [len(r[0]) for r in rows])
        count_width = max([len("Count"), len("9,999,999")] + [len(r[1]) for r in rows])
        pct_width = max([len("%"), len("100.0%")] + [len(r[2]) for r in rows])
        header = f"{title:<{key_width}}  {'Count':>{count_width}}  {'%':>{pct_width}}"
        lines = [header, "-" * len(header)]
        for key, count, pct in rows:
            lines.append(f"{key:<{key_width}}  {count:>{count_width}}  {pct:>{pct_width}}")
        return lines




# The log file main writes in its output directory
LOG_FILE_NAME = "entity_linking.log"

# How often main logs the Stats summary of the rows linked so far
DEFAULT_STATS_INTERVAL_SECONDS = 30


def _row_key(row: dict) -> str:
    """
    Identifies an input row among those of its file, also once linked (which
    only adds columns to it), for resuming a run: its ids and raw addresses.
    """
    return json.dumps(
        [row.get("filename"), row.get("card_id"), row.get("address_id")]
        + [row.get(f"{prefix}.raw") for prefix in FIELD_PREFIXES],
        ensure_ascii=False)


def _read_jsonl(path: Path, log_invalid: bool) -> Iterator[dict]:
    """The records of a jsonl file, skipping the lines that are not valid json."""
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if log_invalid:
                    logger.error("Skipping invalid json at line %d of %s: %s", line_number, path, e)


def _load_linked_keys(output_path: Path, rewrite: bool, stats: Optional[Stats] = None) -> Counter:
    """
    The _row_key of the rows a previous run already linked into
    `output_path`, leaving out those whose linking raised an exception, so
    they are linked again, and invalid lines, e.g. one a run interrupted
    while writing left truncated. With `rewrite`, the file is rewritten
    without them, so that resuming appends neither after a truncated line nor
    a second copy of a row linked again; `stats`, if given, observes the rows kept.
    """
    linked_keys = Counter()
    if not output_path.exists():
        return linked_keys
    kept_lines = []
    invalid_lines = exception_rows = 0
    needs_rewrite = False
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                needs_rewrite = True
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if is_exception_row(row):
                exception_rows += 1
                continue
            linked_keys[_row_key(row)] += 1
            if stats is not None:
                stats.observe_row(row)
            if not line.endswith("\n"):
                line += "\n"
                needs_rewrite = True
            kept_lines.append(line)
    if rewrite and (needs_rewrite or invalid_lines or exception_rows):
        if invalid_lines:
            logger.warning("Dropping %d invalid line(s) from %s", invalid_lines, output_path)
        if exception_rows:
            logger.info("Dropping %d row(s) of %s whose linking raised an exception, to link them again",
                        exception_rows, output_path)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as f:
            f.writelines(kept_lines)
        os.replace(temporary_path, output_path)
    return linked_keys


@dataclass
class _InputFile:
    path: Path
    output_path: Path
    rows: int
    # rows not yet linked by a previous run
    pending_rows: int


def _discover_input_files(input_path: Path, output_dir: Path, file_pattern: str) -> list[_InputFile]:
    """
    The input files, each with the output file mirroring its path under
    `output_dir` and the rows left to link in it.
    """
    if input_path.is_dir():
        resolved_output_dir = output_dir.resolve()
        paths = sorted(
            path for path in input_path.rglob(file_pattern)
            if path.is_file() and not path.resolve().is_relative_to(resolved_output_dir))
        relative_paths = [path.relative_to(input_path) for path in paths]
    else:
        paths = [input_path]
        relative_paths = [Path(input_path.name)]
    input_files = []
    for path, relative_path in zip(paths, relative_paths):
        output_path = output_dir / relative_path
        try:
            linked_keys = _load_linked_keys(output_path, rewrite=False)
            rows = pending_rows = 0
            for row in _read_jsonl(path, log_invalid=False):
                rows += 1
                key = _row_key(row)
                if linked_keys[key] > 0:
                    linked_keys[key] -= 1
                else:
                    pending_rows += 1
        except Exception:
            logger.exception("Skipping input file %s, which could not be read", path)
            continue
        input_files.append(_InputFile(path, output_path, rows, pending_rows))
    return input_files


def _iter_pending_rows(input_files: list[_InputFile], stats: Stats, destinations: deque) -> Iterator[dict]:
    """
    The rows of the input files not yet linked by a previous run, appending
    the _InputFile of each to `destinations` as it is yielded. Their output
    files are first cleaned (see _load_linked_keys), the rows they already
    hold being observed by `stats`.
    """
    for input_file in input_files:
        try:
            input_file.output_path.parent.mkdir(parents=True, exist_ok=True)
            linked_keys = _load_linked_keys(input_file.output_path, rewrite=True, stats=stats)
            if input_file.pending_rows == 0:
                continue
            for row in _read_jsonl(input_file.path, log_invalid=True):
                key = _row_key(row)
                if linked_keys[key] > 0:
                    linked_keys[key] -= 1
                    continue
                destinations.append(input_file)
                yield row
        except Exception:
            logger.exception("Error reading input file %s, skipping its remaining rows", input_file.path)


def _exception_row(row: dict, exception: BaseException) -> dict:
    """The output row of an input row whose linking failed outside of process_row."""
    if is_interruption(exception):
        raise exception
    logger.error("Linking the row of %s failed in its worker process", _row_label(row), exc_info=exception)
    output_row = dict(row)
    for prefix in prefixes_in_row(row):
        output_row.update(exception_columns(row, prefix, exception))
    return output_row


def _linking_config() -> dict:
    """The module level settings of the linking steps, logged at the start of a run."""
    return {
        "SEARCH_PRUNE_SCORE_THRESHOLD": SEARCH_PRUNE_SCORE_THRESHOLD,
        "DEFAULT_SIGNIFICANCE_THRESHOLD": DEFAULT_SIGNIFICANCE_THRESHOLD,
        "SIGNIFICANCE_THRESHOLDS": SIGNIFICANCE_THRESHOLDS,
        "MIN_POPULATION_ORDER_OF_MAGNITUDE": MIN_POPULATION_ORDER_OF_MAGNITUDE,
        "PARENT_CHILD_MAX_DISTANCE_KM_BY_ADMIN_LEVEL": PARENT_CHILD_MAX_DISTANCE_KM_BY_ADMIN_LEVEL,
        "WEIGHTED_DISAMBIGUATION_FACTORS": WEIGHTED_DISAMBIGUATION_FACTORS,
        "DISAMBIGUATION_FACTOR_PRIMARY_PRIORITY": DISAMBIGUATION_FACTOR_PRIMARY_PRIORITY,
        "DISAMBIGUATION_FACTOR_SECONDARY_PRIORITY": DISAMBIGUATION_FACTOR_SECONDARY_PRIORITY,
        "DISAMBIGUATION_POPULATION_ROUNDING_FACTOR": DISAMBIGUATION_POPULATION_ROUNDING_FACTOR,
        "DISAMBIGUATION_SCORE_PRUNE_THRESHOLDS": DISAMBIGUATION_SCORE_PRUNE_THRESHOLDS,
    }


def main(argv=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", type=str, help="A parsed-address jsonl file, or a directory of jsonl files")
    arg_parser.add_argument(
        "output_dir", type=str,
        help="Directory the linked files are written to, mirroring their paths in input_path, along with "
             f"{LOG_FILE_NAME}; a run interrupted in it is resumed")
    arg_parser.add_argument("-P", "--file-pattern", type=str, default="*.jsonl",
                            help="Pattern of the input files, searched recursively when input_path is a directory")
    arg_parser.add_argument("--search-cache-db", type=str, default=":memory:")
    arg_parser.add_argument("--search-index-path", type=str, default=".geo_db_search_index/tantivy_index")
    arg_parser.add_argument("--geo-db-path", type=str, default="geo.duckdb")
    arg_parser.add_argument("--camps-reference", type=str, default=str(DEFAULT_CAMPS_REFERENCE_PATH))
    arg_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    arg_parser.add_argument("--stats-interval", type=float, default=DEFAULT_STATS_INTERVAL_SECONDS,
                            help="Seconds between logs of the statistics of the rows linked so far")
    args = arg_parser.parse_args(argv)

    input_path = Path(args.input_path)
    assert input_path.exists(), f"{input_path} does not exist"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(output_dir / LOG_FILE_NAME, mode="a"), logging.StreamHandler()],
        force=True,
    )

    logger.info("Starting entity linking run with arguments:\n%s", json.dumps(vars(args), indent=2))
    logger.info("Linking config:\n%s", json.dumps(_linking_config(), indent=2, default=str))

    discovery_start = time.monotonic()
    input_files = _discover_input_files(input_path, output_dir, args.file_pattern)
    total_rows = sum(input_file.rows for input_file in input_files)
    pending_rows = sum(input_file.pending_rows for input_file in input_files)
    logger.info(
        "Discovered %d input files with %d rows in %.1fs: %d rows already linked, %d rows to link",
        len(input_files), total_rows, time.monotonic() - discovery_start, total_rows - pending_rows, pending_rows)
    if pending_rows == 0:
        logger.info("Nothing left to link")
        return

    camp_matcher = build_camp_matcher(args.camps_reference)
    geo_db_searcher = build_geo_db_searcher(args.search_index_path, args.search_cache_db, args.geo_db_path)
    disambiguator = build_disambiguator(args.geo_db_path)

    # Rows of previous runs included, from the output files
    stats = Stats()
    destinations = deque()
    exception_rows = 0
    current_file: Optional[_InputFile] = None
    out_f = None
    try:
        with LinkingPool(camp_matcher, geo_db_searcher, disambiguator, args.num_workers) as linking_pool, \
                tqdm_logging_redirect(total=pending_rows, desc="Linking rows") as progress_bar:
            output_rows = linking_pool.process_rows(
                _iter_pending_rows(input_files, stats, destinations), on_error=_exception_row)
            last_stats_time = time.monotonic()
            for output_row in output_rows:
                input_file = destinations.popleft()
                if input_file is not current_file:
                    if out_f is not None:
                        out_f.close()
                        logger.info("Finished linking %s", current_file.path)
                    current_file = input_file
                    logger.info(
                        "Linking %d of the %d rows of %s into %s",
                        input_file.pending_rows, input_file.rows, input_file.path, input_file.output_path)
                    out_f = input_file.output_path.open("a", encoding="utf-8")
                out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                stats.observe_row(output_row)
                if is_exception_row(output_row):
                    exception_rows += 1
                progress_bar.update(1)
                if time.monotonic() - last_stats_time >= args.stats_interval:
                    out_f.flush()
                    logger.info(
                        "Statistics so far (%d rows with exceptions this run):\n%s",
                        exception_rows, stats.summary())
                    last_stats_time = time.monotonic()
            if current_file is not None:
                logger.info("Finished linking %s", current_file.path)
    finally:
        if out_f is not None:
            out_f.close()
        geo_db_searcher.finalize()
        disambiguator.close()
        logger.info("Final statistics (%d rows with exceptions this run):\n%s", exception_rows, stats.summary())
    if exception_rows:
        logger.warning("%d rows had exceptions; run again on the same output directory to retry them", exception_rows)
    logger.info("Entity linking complete")


if __name__ == "__main__":
    main()
