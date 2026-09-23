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
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Iterable, Optional

import pandas as pd
from tqdm.auto import tqdm

from modules.address_tagging import tag_address
from modules.camp_search import DEFAULT_CAMPS_REFERENCE_PATH, CampReferenceMatcher
from modules.geo_db_search import GeoDBSearch, TantivySearchIndex
from modules.geo_disambiguation import DISAMBIGUATION_FACTOR_PRIORITY, Disambiguator
from modules.entity_linking_eval_metrics import normalize_iri
from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import AddressProcessingData, BZKFieldName, LinkedAddress, MatchedEntity, MatchedName, RawEntity

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
    GEO_DB_FUZZY_MATCH = "GEO_DB_FUZZY_MATCH"
    GEO_DB_PARTIAL_WORD_MATCH = "GEO_DB_PARTIAL_WORD_MATCH"


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

    AMBIGUOUS = "AMBIGUOUS"
    NO_CANDIDATES = "NO_CANDIDATES"


# DISAMBIGUATION_FACTOR_PRIORITY factors that reflect the entity's own type
# match or agreement with sibling entities in the same address, as opposed to
# generic ranking criteria (population, country, preferred name, ...).
_CONTEXT_DISAMBIGUATION_FACTORS = {
    "child_parent_likelihood",
    "entity_types_matching_fuzzy",
    "entity_types_matching",
}


def _resolved_disambiguation_status(address: AddressProcessingData) -> str:
    """
    Distinguish why disambiguation was able to settle on a single candidate:
    see DisambiguationStatus.UNAMBIGUOUS/DISAMBIGUATED_BY_CONTEXT/DISAMBIGUATED_HEURISTICALLY.
    Only meaningful when address.linked_to is not None.
    """
    possible_links = address.possible_links or ()
    if len(possible_links) <= 1:
        return DisambiguationStatus.UNAMBIGUOUS
    # possible_links is sorted best-first; since address.linked_to is not
    # None, the best candidate is not tied with the runner-up (see
    # Disambiguator.disambiguate), so they differ on some factor. The first
    # (highest-priority) factor where they differ is what the choice hinged on.
    best, runner_up = possible_links[0].scores, possible_links[1].scores
    for factor in DISAMBIGUATION_FACTOR_PRIORITY:
        if best.get(factor, 0.0) != runner_up.get(factor, 0.0):
            if factor in _CONTEXT_DISAMBIGUATION_FACTORS:
                return DisambiguationStatus.DISAMBIGUATED_BY_CONTEXT
            return DisambiguationStatus.DISAMBIGUATED_HEURISTICALLY
    return DisambiguationStatus.DISAMBIGUATED_HEURISTICALLY


def _geo_db_search_status(matched_name: MatchedName) -> str:
    if matched_name.matching_method == "pre_linked":
        return SearchStatus.PRE_LINKED_DURING_PARSING
    if matched_name.is_abbreviation_match:
        return SearchStatus.GEO_DB_ABBREVIATION_MATCH
    if matched_name.is_phonetic_match:
        return SearchStatus.GEO_DB_PHONETIC_MATCH
    if matched_name.is_partial_word_match:
        return SearchStatus.GEO_DB_PARTIAL_WORD_MATCH
    if matched_name.cleaned_edit_distance == 0:
        return SearchStatus.GEO_DB_EXACT_MATCH
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
    return {
        entity_type: _clean_optional_str(row.get(f"{prefix}.{entity_type}.text"))
        for entity_type in ENTITY_TYPE_COLUMNS
    }


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


def _build_entities(
    entity_texts: dict[str, Optional[str]], above_city_text: Optional[str], pre_linked_iris: dict[str, str]
) -> list[RawEntity]:
    entities = [
        RawEntity.with_parsed(
            entity_type=GeographicalEntityType[entity_type], raw_text=text,
            pre_linked_iri=pre_linked_iris.get(entity_type)
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
            entities.append(RawEntity.with_parsed(entity_type=GeographicalEntityType.AboveCity, raw_text=above_city_text))
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
    total_time : Optional[float] = None


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
) -> LinkingOutcome:
    """
    Run the full linking pipeline for a single address field of a single row.
    """
    start = time.monotonic()
    def _elapsed():
        nonlocal start
        return time.monotonic() - start
    raw_address = _clean_optional_str(row.get(f"{prefix}.raw"))
    entity_texts = _extract_entity_texts(row, prefix)
    above_city_text = _clean_optional_str(row.get(f"{prefix}.AboveCity.text"))
    pre_linked_iris = _pre_linked_iris(row, prefix)
    entities = _build_entities(entity_texts, above_city_text, pre_linked_iris)
    address = AddressProcessingData(
        card_id=row.get("card_id"),
        id=str(row.get("address_id", row.get("filename"))),
        full_address=raw_address,
        bzk_field_name=FIELD_PREFIXES[prefix],
        entities=entities,
    )

    pre_linked = _pre_linked_finest_entity(pre_linked_iris, entity_texts)
    if pre_linked is not None:
        entity_type, iri = pre_linked
        return LinkingOutcome(
            iri=iri, entity_type=entity_type, tags=(),
            search_status=SearchStatus.PRE_LINKED_DURING_PARSING,
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            linked_entities=(LinkedEntityMetadata(entity_type=entity_type, iri=iri),),
            total_time=_elapsed()
        )

    
    address_tags = tag_address(raw_address or entity_texts.get("City"))
    if "location_unspecified" in address_tags:
        reasons = sorted(address_tags - {"location_unspecified"})
        return LinkingOutcome(
            iri=None, entity_type=None, tags=("unresolved", *sorted(address_tags)),
            search_status=f"{SearchStatus.NO_LOCATION} ({', '.join(reasons)})" if reasons else SearchStatus.NO_LOCATION,
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            total_time=_elapsed()
        )
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
        camp_tags = set(camp_match.tags)
        camp_tags.update(extra_tags)
        camp_tags = tuple(sorted(camp_tags))
        return LinkingOutcome(
            iri=camp_match.iri, entity_type="Camp", tags=camp_tags,
            search_status=f"{SearchStatus.CAMP_REFERENCE}",
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            linked_entities=(LinkedEntityMetadata(entity_type="Camp", iri=camp_match.iri),),
            total_time=_elapsed()
        )
    
    if len(entities) == 0:
        return LinkingOutcome(
            iri=None, entity_type=None, tags=("unresolved", *extra_tags),
            search_status=SearchStatus.NO_ENTITIES,
            disambiguation_status=DisambiguationStatus.NOT_APPLICABLE,
            total_time=_elapsed()
        )

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
    address = disambiguator.disambiguate(address)
    possible_links_count = len(address.possible_links or ())
    likely_links_count = len(address.likely_links or ())

    if address.linked_to is not None:
        linked_address = address.linked_to
        linked_entities = _linked_entities_metadata(linked_address, search_candidate_counts)
        finest_type = linked_address.finest_grain_entity.entity_type.name
        finest_iri = next(m.iri for m in linked_entities if m.entity_type == finest_type)
        return LinkingOutcome(
            iri=finest_iri,
            entity_type=finest_type,
            tags=extra_tags,
            search_status=_geo_db_search_status(linked_address.finest_grain_entity.linked_to),
            disambiguation_status=_resolved_disambiguation_status(address),
            linked_entities=linked_entities,
            possible_links_count=possible_links_count,
            likely_links_count=likely_links_count,
            disambiguation_scores=dict(linked_address.scores),
            total_time=_elapsed()
        )
    if likely_links_count > 1:
        first_candidate = address.likely_links[0]
        ambiguous_iris = [
            normalize_iri(candidate.finest_grain_entity.linked_to.geographical_name.entity.iri)
            for candidate in address.likely_links
        ]
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
            total_time=_elapsed()
        )
    return LinkingOutcome(
        iri=None, entity_type=None, tags=("unresolved", *extra_tags),
        search_status=SearchStatus.GEO_DB_NO_CANDIDATES,
        disambiguation_status=DisambiguationStatus.NO_CANDIDATES,
        possible_links_count=possible_links_count, likely_links_count=likely_links_count,
        total_time=_elapsed()
    )


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
    for entity_metadata in outcome.linked_entities:
        entity_prefix = f"{prefix}.{entity_metadata.entity_type}"
        updates[f"{entity_prefix}_iri"] = entity_metadata.iri
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
    the new "{prefix}.*" linking columns for each of them.
    """
    output_row = dict(row)
    for prefix in prefixes_in_row(row):
        outcome = link_field(row, prefix, camp_matcher, geo_db_searcher, disambiguator)
        output_row.update(apply_outcome_to_row(row, prefix, outcome))
    return output_row


def _iter_input_rows(input_path: Path, file_pattern: str) -> Iterable[dict]:
    files = sorted(input_path.glob(file_pattern)) if input_path.is_dir() else [input_path]
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main(argv=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", type=str, help="A parsed-address jsonl file, or a directory of jsonl files")
    arg_parser.add_argument("output_path", type=str)
    arg_parser.add_argument("-P", "--file-pattern", type=str, default="*.jsonl")
    arg_parser.add_argument("--search-cache-db", type=str, default=":memory:")
    arg_parser.add_argument("--search-index-path", type=str, default=".geo_db_search_index/tantivy_index")
    arg_parser.add_argument("--geo-db-path", type=str, default="geo.duckdb")
    arg_parser.add_argument("--camps-reference", type=str, default=str(DEFAULT_CAMPS_REFERENCE_PATH))
    args = arg_parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Loading camp/ghetto reference matcher...")
    camp_matcher = CampReferenceMatcher(args.camps_reference)
    camp_matcher.initialize()

    logging.info("Initializing GeoDB searcher...")
    search_index = TantivySearchIndex(args.search_index_path)
    geo_db_searcher = GeoDBSearch(args.search_cache_db, search_index=search_index, geo_db_path=args.geo_db_path)
    geo_db_searcher.initialize()
    disambiguator = Disambiguator()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            for row in tqdm(_iter_input_rows(input_path, args.file_pattern), desc="Linking addresses"):
                output_row = process_row(row, camp_matcher, geo_db_searcher, disambiguator)
                out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    finally:
        geo_db_searcher.finalize()


if __name__ == "__main__":
    main()
