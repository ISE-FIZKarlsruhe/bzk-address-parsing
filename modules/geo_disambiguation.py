

from os import name
import pprint
from typing import Literal, Optional, NamedTuple

from modules.pipeline.geographical_entity import GeographicalBranch, GeographicalEntityType, GeographicalName
from modules.geo_db_search import (
    ABOVE_CITY_ENTITY_TYPES, ENTITY_LINKING_LOGGER, distance_to_geometry_km, geometries_distance_km,
    is_entity_type_ruled_out, is_regional_term_entity, partial_match_query_span, score_name_similarity)
from modules.pipeline.linked_data import AddressSpan, BZKFieldName, AddressProcessingData, LinkedEntity, MatchedEntity, MatchedName, LinkedAddress, RawEntity
import uuid
from modules.pipeline.storage.encoding_util import decode_from_dict
import dataclasses
import duckdb
import logging
import threading
from collections import defaultdict
from modules.pipeline.linked_data import AnnotatedScore
from modules.pipeline.storage.frozendict import FrozenDict
import itertools
import math

def _is_admin_code_null(code : Optional[str]) -> bool:
    return code is None or code == "" or all(c == "0" for c in code)


DISAMBIGUATION_FACTOR_PRIORITY = [
    "weighted_score",
    "child_parent_likelihood",
    "fuzzy_similarity_score",
    "entity_types_matching_preferred",
    "population_order_of_magnitude",
    "country_likelihood_rank", # general rank of country likelihood based on observation
    "phonetic_score",
    "entity_types_matching",
    "is_preferred_name",
    "population_count"
]

# Higher value means more impact in the decision
WEIGHTED_DISAMBIGUATION_FACTORS = {
    "child_parent_likelihood" : 2,
    "phonetic_score" : 2,
    "fuzzy_similarity_score" : 2,
    "entity_types_matching_preferred" : 1
}

NA_SCORE = AnnotatedScore(0.0, "Not applicable")

# Distance from a child to the closest point of a parent's estimated geometry
# (see RegionGeometry) at and beyond which the child is considered entirely
# outside the parent, for the parents whose admin codes do not describe them
# (see _score_parent_child_by_geometry)
GEOMETRY_PARENT_CHILD_MAX_DISTANCE_KM = 50

# Weight, relative to the other entities (weighing 1), of a missed word entity
# (see RawEntity.is_missed_word) in the average of a candidate address' scores.
# Being only a guess, it should establish a preference between candidates that
# match it and candidates that don't (or match it poorly), without outweighing
# the entities identified by parsing.
MISSED_WORD_ENTITY_WEIGHT = 0.5

# "entity_types_matching_preferred" for Unknown entities (see
# _score_individual_match), which match any entity type but, coming from
# single-word addresses (e.g. "Polen"), are most likely a Country, then a
# City or a Neighborhood, and least likely any other coarser type. A
# Neighborhood is on par with a City, as when matching a City entity (see
# "Neighborhood instead of city match"), since a former town now part of a
# city (e.g. "Elberfeld" in Wuppertal) should not lose to a namesake city
# abroad; entity_types_matching still ranks it slightly lower. A match with
# several possible entity types gets the best of their scores.
UNKNOWN_ENTITY_TYPE_PREFERENCE = {
    GeographicalEntityType.Country: AnnotatedScore(1.0, "Unknown entity type matches country"),
    GeographicalEntityType.City: AnnotatedScore(0.75, "Unknown entity type matches city"),
    GeographicalEntityType.Neighborhood: AnnotatedScore(0.75, "Unknown entity type matches neighborhood"),
    GeographicalEntityType.State: AnnotatedScore(0.25, "Unknown entity type matches state"),
    GeographicalEntityType.Region: AnnotatedScore(0.25, "Unknown entity type matches region"),
    GeographicalEntityType.District: AnnotatedScore(0.25, "Unknown entity type matches district"),
}

class ScoredMatch(NamedTuple):
    match: MatchedName
    scores: dict[str, AnnotatedScore]

def _score_dict_to_tuple(scores : dict[str, AnnotatedScore | float], priority : list[str]) -> tuple:
    if isinstance(scores, ScoredMatch):
        scores = scores.scores
    def floatify(score : AnnotatedScore | float) -> float:
        if isinstance(score, AnnotatedScore):
            return score.score
        else:
            return score
    return tuple(floatify(scores.get(factor, 0.0)) for factor in priority)

def _compare_scores(
        s1 : dict[str, AnnotatedScore | float], 
        s2 : dict[str, AnnotatedScore | float], 
        priority : list[str],
        significance_thresholds : dict[str, float]
    ) -> Literal["gt", "lt", "eq"]:
    """
    Compares two score dictionaries factor by factor in priority order: the
    first factor whose difference exceeds its significance threshold decides
    ("gt" or "lt"); "eq" if no factor does.
    """
    t1 = _score_dict_to_tuple(s1, priority)
    t2 = _score_dict_to_tuple(s2, priority)
    for factor, a, b in zip(priority, t1, t2):
        if abs(a - b) > significance_thresholds[factor]:
            if a > b:
                return "gt"
            else:
                return "lt"
    return "eq"

def _average_scores(
        scored_matches : list[ScoredMatch], weights : Optional[list[float]] = None
    ) -> dict[str, float]:
    if weights is None:
        weights = [1.0] * len(scored_matches)
    sums = defaultdict(lambda: 0.0)
    for scored_match, weight in zip(scored_matches, weights):
        for factor, score in scored_match.scores.items():
            if isinstance(score, AnnotatedScore):
                score = score.score
            sums[factor] += score * weight
    weight_sum = sum(weights)
    return {factor: score / weight_sum for factor, score in sums.items()}

def _describe_linked_address(linked_address : LinkedAddress) -> str:
    """
    Compact, human-readable summary of a candidate address for debug logging.
    Only call this behind a logger.isEnabledFor(logging.DEBUG) check.
    """
    linked_name = linked_address.finest_grain_entity.linked_to
    return (
        f"{linked_name.nfc_alt_name!r} ({linked_name.geographical_name.entity.iri}, "
        f"country {linked_name.geographical_name.entity.country.iso_code}) "
        f"scores {dict(linked_address.scores)}"
    )

def _country_iso_codes(match : MatchedName) -> tuple[str, ...]:
    """
    The countries a match lies in: its country, or every country of the
    geometry of a regional term spanning several (see RegionGeometry).
    """
    entity = match.geographical_name.entity
    if is_regional_term_entity(entity) and entity.geometry is not None:
        return entity.geometry.country_iso_codes
    return (entity.country.iso_code if entity.country is not None else None,)

class Disambiguator:
    logger = ENTITY_LINKING_LOGGER.getChild("Disambiguator")

    def __init__(
            self, 
            significance_thresholds: dict[str, float] = defaultdict(float),
            priority: list[str] = DISAMBIGUATION_FACTOR_PRIORITY,
            score_weights: list[str] = WEIGHTED_DISAMBIGUATION_FACTORS,
            population_rounding_factor: int = 10_000,
            min_population_order_of_magnitude: int = 500_000,
            score_prune_thresholds : dict[str, float] = defaultdict(float),
            geo_db_path : str = "geo.duckdb",
            geometry_parent_child_max_distance_km : float = GEOMETRY_PARENT_CHILD_MAX_DISTANCE_KM
        ):
        self.significance_thresholds = significance_thresholds
        self.priority = priority
        self.score_weights = score_weights
        self.population_rounding_factor = population_rounding_factor
        self.min_population_order_of_magnitude = min_population_order_of_magnitude
        self.score_prune_thresholds = score_prune_thresholds
        self.geo_db_path = geo_db_path
        self.geometry_parent_child_max_distance_km = geometry_parent_child_max_distance_km
        # Read only connection to the geo duckdb, only needed to look up the
        # entity of a common parent branch (see _link_to_common_parent). Opened
        # lazily, and every use (and cache miss) goes through _geo_db_lock since
        # the Disambiguator is shared across worker threads while a duckdb
        # connection is not safe to use from several threads at once. Lookups
        # are few (bounded by the distinct branches) and cached, so serializing
        # them costs little.
        self._geo_db_connection : Optional[duckdb.DuckDBPyConnection] = None
        self._geo_db_lock = threading.Lock()
        self._common_parent_cache : dict[GeographicalBranch, Optional[GeographicalName]] = {}
        self.logger.info(
            "Initialized with significance thresholds %s, priority %s, population rounding factor %d, "
            "min population order of magnitude %d, score prune thresholds %s, geo db %s, "
            "geometry parent/child max distance %.0f km",
            self.significance_thresholds, self.priority, self.population_rounding_factor,
            self.min_population_order_of_magnitude, dict(self.score_prune_thresholds), self.geo_db_path,
            self.geometry_parent_child_max_distance_km)

    def __getstate__(self):
        # Neither the connection nor the lock can be pickled (e.g. when sent to
        # a worker process); each copy opens its own connection when needed.
        state = self.__dict__.copy()
        state["_geo_db_connection"] = None
        state["_geo_db_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._geo_db_lock = threading.Lock()

    def close(self):
        with self._geo_db_lock:
            if self._geo_db_connection is not None:
                self._geo_db_connection.close()
                self._geo_db_connection = None

    def _find_branch_entity(self, branch : GeographicalBranch) -> Optional[GeographicalName]:
        """
        Look up the entity representing a branch of the geographical hierarchy:
        the country itself if the branch has no admin codes, otherwise the
        (preferably current, not historical) ADM{n} division with exactly the
        branch's n admin codes. If there is no such division, the branch is
        walked up until one is found. Results are cached. Must be called while
        holding _geo_db_lock.
        """
        if branch in self._common_parent_cache:
            return self._common_parent_cache[branch]
        if self._geo_db_connection is None:
            self.logger.debug("Opening read only connection to geo db %s", self.geo_db_path)
            self._geo_db_connection = duckdb.connect(self.geo_db_path, read_only=True)
        connection = self._geo_db_connection
        level = len(branch.admin_codes)
        if level == 0:
            row = connection.execute(
                "SELECT iri FROM geographical_entities WHERE iso_country_code = ? AND classification LIKE 'A.PCL%' "
                "ORDER BY classification = 'A.PCLI' DESC, classification LIKE '%H' ASC, population DESC NULLS LAST "
                "LIMIT 1",
                [branch.country_iso_code]
            ).fetchone()
        else:
            code_conditions = " AND ".join(f"admin_codes.admin{i + 1}_code = ?" for i in range(level))
            null_next_code = (
                f" AND coalesce(ltrim(admin_codes.admin{level + 1}_code, '0'), '') = ''" if level < 5 else "")
            row = connection.execute(
                "SELECT iri FROM geographical_entities WHERE iso_country_code = ? "
                f"AND classification IN ('A.ADM{level}', 'A.ADM{level}H') AND {code_conditions}{null_next_code} "
                f"ORDER BY classification = 'A.ADM{level}' DESC, population DESC NULLS LAST LIMIT 1",
                [branch.country_iso_code, *branch.admin_codes]
            ).fetchone()
        geographical_name = None
        if row is not None:
            name_row = connection.execute(
                "SELECT * FROM geographical_names_with_entities WHERE entity.iri = ? "
                "ORDER BY is_preferred_name DESC NULLS LAST, name_id LIMIT 1",
                [row[0]]
            ).fetchone()
            if name_row is not None:
                columns = [description[0] for description in connection.description]
                geographical_name = decode_from_dict(dict(zip(columns, name_row)), GeographicalName)
        if geographical_name is None and level > 0:
            self.logger.debug("No entity found for branch %s; walking up the branch", branch)
            geographical_name = self._find_branch_entity(
                GeographicalBranch(branch.country_iso_code, branch.admin_codes[:-1]))
        self.logger.debug(
            "Branch %s resolved to %s", branch,
            geographical_name.entity.iri if geographical_name is not None else None)
        self._common_parent_cache[branch] = geographical_name
        return geographical_name

    def _branch_entity(self, branch : GeographicalBranch) -> Optional[GeographicalName]:
        # Checked without the lock first: reading a dict is thread safe, and
        # cache hits are by far the most common case.
        if branch in self._common_parent_cache:
            return self._common_parent_cache[branch]
        with self._geo_db_lock:
            return self._find_branch_entity(branch)

    def _common_branch(self, candidates : list[LinkedAddress]) -> Optional[GeographicalBranch]:
        """
        The deepest branch of the geographical hierarchy (country, then the
        leading non-null admin codes) common to the finest entity of every
        candidate, or None if they are not even in the same country.
        """
        common : Optional[GeographicalBranch] = None
        for candidate in candidates:
            entity = candidate.finest_grain_entity.linked_to.geographical_name.entity
            country_code = entity.country.iso_code if entity.country is not None else None
            if not country_code:
                return None
            codes = []
            for code in (entity.admin_codes or ()):
                if _is_admin_code_null(code):
                    break
                codes.append(code)
            if common is None:
                common = GeographicalBranch(country_code, tuple(codes))
                continue
            if common.country_iso_code != country_code:
                return None
            common_length = 0
            for a, b in zip(common.admin_codes, codes):
                if a != b:
                    break
                common_length += 1
            common = GeographicalBranch(country_code, common.admin_codes[:common_length])
        return common

    def _link_to_common_parent(
            self, address : AddressProcessingData, likely_addresses : list[LinkedAddress]
        ) -> Optional[LinkedAddress]:
        """
        For likely candidates that could not be told apart, a candidate address
        linked to the entity at the branch of the geographical hierarchy common
        to all of them (e.g. the state in Germany all of them lie in), or None
        if there is no such branch or no entity was found for it.
        """
        branch = self._common_branch(likely_addresses)
        if branch is None:
            self.logger.debug("Address %s: likely candidates share no common branch", address.id)
            return None
        geographical_name = self._branch_entity(branch)
        if geographical_name is None:
            self.logger.debug(
                "Address %s: no entity found for the likely candidates' common branch %s", address.id, branch)
            return None
        entity_types = geographical_name.entity.possible_entity_types
        entity_type = min(entity_types) if len(entity_types) > 0 else GeographicalEntityType.Country
        matched_name = MatchedName(
            geographical_name=geographical_name,
            nfc_query=address.full_address,
            nfc_alt_name=geographical_name.name,
            cleaned_query=None,
            cleaned_alt_name=None,
            cleaned_edit_distance=None,
            matching_method="common_parent",
            matching_score=0.0,
            fuzzy_score=0.0,
            phonetic_score=0.0,
            abbreviation_pattern=None,
            edit_distance=None,
            is_abbreviation_match=False,
            is_phonetic_match=False,
            is_partial_word_match=False,
        )
        linked_entity = RawEntity.with_parsed(
            raw_text=address.full_address, entity_type=entity_type
        ).link_to(matched_name, {})
        self.logger.debug(
            "Address %s: likely candidates share branch %s; linking to its entity %r (%s) of type %s",
            address.id, branch, geographical_name.name, geographical_name.entity.iri, entity_type.name)
        return LinkedAddress(
            finest_grain_entity=linked_entity,
            reference_entity=linked_entity,
            entities=(linked_entity,),
            # the candidates are tied, so any of their scores stands for all
            scores=likely_addresses[0].scores
        )

    def _drop_duplicates(self, entity : MatchedEntity, matches: list[MatchedName], bzk_field: BZKFieldName) -> list[MatchedName]:
        """
        Collapse duplicate matches that refer to the same geographical entity, keeping the one with the highest similarity score.
        """
        if matches is None or len(matches) == 0:
            return matches
        already_seen = set()
        matches = sorted(
            matches, 
            key=lambda m : _score_dict_to_tuple(self._score_individual_match(entity, m, bzk_field), self.priority), 
            reverse=True
        )
        unique_matches = []
        for match in matches:
            iri = match.geographical_name.entity.iri
            if iri in already_seen:
                self.logger.debug(
                    "Dropping duplicate match %r (%s) for %r", match.nfc_alt_name, iri, entity.raw_text)
                continue
            already_seen.add(iri)
            unique_matches.append(match)
        return unique_matches

    def _calculate_weighted_score(self, scores : dict[str, float]) -> AnnotatedScore:
        weight_sum = 0
        score_sum = 0
        for score_name, score_weight in self.score_weights.items():
            score_value = scores.get(score_name)
            if isinstance(score_value, AnnotatedScore):
                score_value = score_value.score
            if score_value is not None:
                score_contrib = score_value * score_weight
                score_sum += score_contrib
                weight_sum += score_weight
                self.logger.debug(
                    "Factor %s with weight of %.2f and score of %.2f contributes %.2f to the weighted score", 
                    score_name, score_weight, score_value, score_contrib
                )
        if weight_sum == 0:
            return AnnotatedScore(0, "No factors match")
        weighted_score = score_sum / weight_sum
        self.logger.debug("Final weighted score: %.2f / %.2f = %.2f", score_sum, weight_sum, weighted_score)
        return AnnotatedScore(weighted_score, "Weighted sum")


    def _score_individual_match(self, entity: MatchedEntity, name: MatchedName, bzk_field : BZKFieldName) -> ScoredMatch:
        scores = dict()
        if entity.entity_type != GeographicalEntityType.Unknown and  entity.entity_type in name.geographical_name.entity.possible_entity_types:
            scores["entity_types_matching_preferred"] = AnnotatedScore(1.0, "Entity type matches")
            scores["entity_types_matching"] = AnnotatedScore(1.0, "Entity type matches")
        elif entity.entity_type == GeographicalEntityType.Unknown:
            # Unknown is a wildcard: any real entity type matches it, though
            # some are preferred (see UNKNOWN_ENTITY_TYPE_PREFERENCE). Unknown
            # only comes from single-word addresses, which are rarely a
            # neighborhood on its own, so those match slightly less.
            scores["entity_types_matching_preferred"] = max(
                (UNKNOWN_ENTITY_TYPE_PREFERENCE[t] for t in name.geographical_name.entity.possible_entity_types
                 if t in UNKNOWN_ENTITY_TYPE_PREFERENCE),
                key=lambda score: score.score, default=AnnotatedScore(0.0, "Entity type does not match"))
            if all(t == GeographicalEntityType.Neighborhood for t in name.geographical_name.entity.possible_entity_types):
                scores["entity_types_matching"] = AnnotatedScore(0.8, "Unknown entity type matches neighborhood")
            else:
                scores["entity_types_matching"] = AnnotatedScore(1.0, "Unknown entity type matches any")
        else:
            scores["entity_types_matching"] = AnnotatedScore(0.0, "Entity type does not match")
            if entity.entity_type == GeographicalEntityType.Neighborhood and GeographicalEntityType.City in name.geographical_name.entity.possible_entity_types:
                scores["entity_types_matching_preferred"] = AnnotatedScore(1.0, "City instead of neighborhood match")
            elif entity.entity_type == GeographicalEntityType.City and GeographicalEntityType.Neighborhood in name.geographical_name.entity.possible_entity_types:
                scores["entity_types_matching_preferred"] = AnnotatedScore(1.0, "Neighborhood instead of city match")
            elif entity.entity_type == GeographicalEntityType.AboveCity and any(
                possible_type in ABOVE_CITY_ENTITY_TYPES for possible_type in name.geographical_name.entity.possible_entity_types
            ):
                scores["entity_types_matching_preferred"] = AnnotatedScore(1.0, "Above-city match")
            else:
                scores["entity_types_matching_preferred"] = AnnotatedScore(0.0, "Entity type does not match")
        scores["phonetic_score"] = AnnotatedScore(name.phonetic_score, "Phonetic similarity score")
        scores["is_preferred_name"] = AnnotatedScore.from_bool(name.geographical_name.is_preferred_name)
        if name.is_abbreviation_match:
            scores["fuzzy_similarity_score"] = AnnotatedScore(1.0, "Abbreviation match")
        else:
            scores["fuzzy_similarity_score"] = AnnotatedScore(name.fuzzy_score, "Fuzzy similarity score")

        if name.geographical_name.entity.country.iso_code == "DE":
            scores["country_likelihood_rank"] = AnnotatedScore(3, "Germany")
        elif name.geographical_name.entity.country.iso_code == "PL":
            scores["country_likelihood_rank"] = AnnotatedScore(3, "Poland")
        elif name.geographical_name.entity.country.iso_code == "IL" and bzk_field != BZKFieldName.VICTIM_BIRTH_PLACE:
                    scores["country_likelihood_rank"] = AnnotatedScore(3, "Israel")
        elif name.geographical_name.entity.country.continent == "EU":
            scores["country_likelihood_rank"] = AnnotatedScore(2, "Europe")
        elif name.geographical_name.entity.country.iso_code == "US" and bzk_field != BZKFieldName.VICTIM_BIRTH_PLACE:
            scores["country_likelihood_rank"] = AnnotatedScore(1, "USA")
        
        if name.geographical_name.entity.population is None:
            scores["population_count"] = AnnotatedScore(1, "Unknown")
            scores["population_order_of_magnitude"] = AnnotatedScore(0, "Unknown")
        else:
            scores["population_count"] = AnnotatedScore(round(name.geographical_name.entity.population / self.population_rounding_factor))
            if name.geographical_name.entity.population < self.min_population_order_of_magnitude:
                scores["population_order_of_magnitude"] = AnnotatedScore(0, "Population under minimum threshold")
            else:
                order_of_magnitude = round(math.log10(name.geographical_name.entity.population)) if name.geographical_name.entity.population > 0 else 0
                scores["population_order_of_magnitude"] = AnnotatedScore(order_of_magnitude, "Population order of magnitude")
        return ScoredMatch(name, scores)

    def _score_parent_child(self, parent: MatchedName, child: MatchedName) -> AnnotatedScore:
        """
        Heuristically estimate the probability that a child entity is a child of a parent entity.

        Args:
            parent (MatchedEntity): The parent entity.
            child (MatchedEntity): The child entity.
        Returns:
            Score: The probability that the child entity is a child of the parent entity.
        """
        parent_entity = parent.geographical_name.entity
        child_entity = child.geographical_name.entity
        logger = self.logger.getChild("parent_child_scoring")
        logger.debug(
            "Scoring hierarchical likelihood between parent %r (%r) and child %r (%r)", 
            parent.cleaned_alt_name, parent.geographical_name.entity.iri,
            child.cleaned_alt_name, child.geographical_name.entity.iri
        )
        if parent_entity.iri in child_entity.other_parent_iris:
            logger.debug(
                "An explicit informal relationship is set between parent %r (%r) and child %r (%r), setting score to 1", 
                parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                child.cleaned_alt_name, child.geographical_name.entity.iri
            )
            return AnnotatedScore(1, "Informal relationship")
        geometry_score = self._score_parent_child_by_geometry(parent, child)
        if geometry_score is not None and is_regional_term_entity(parent_entity):
            return geometry_score
        admin_codes_score = self._score_parent_child_by_admin_codes(parent, child)
        if geometry_score is not None and geometry_score.score > admin_codes_score.score:
            return geometry_score
        return admin_codes_score

    def _score_parent_child_by_geometry(self, parent: MatchedName, child: MatchedName) -> Optional[AnnotatedScore]:
        """
        For a parent with an estimated geometry (see RegionGeometry), how
        close the child lies to it: 1 within it, decreasing linearly to 0 at
        geometry_parent_child_max_distance_km from its closest point. None if
        the parent has no geometry or the child no location.
        """
        parent_entity = parent.geographical_name.entity
        child_entity = child.geographical_name.entity
        if parent_entity.geometry is None:
            return None
        if child_entity.geometry is not None:
            distance = geometries_distance_km(parent_entity.geometry, child_entity.geometry)
        elif child_entity.coordinates is not None and child_entity.coordinates.latitude is not None:
            distance = distance_to_geometry_km(child_entity.coordinates, parent_entity.geometry)
        else:
            return None
        score = max(0.0, 1.0 - distance / self.geometry_parent_child_max_distance_km)
        self.logger.getChild("parent_child_scoring").debug(
            "Child %r (%r) lies %.1f km from the geometry of parent %r (%r), setting score to %.2f",
            child.cleaned_alt_name, child_entity.iri, distance, parent.cleaned_alt_name, parent_entity.iri, score)
        return AnnotatedScore(score, f"{distance:.0f} km from the estimated geometry")

    def _score_parent_child_by_admin_codes(self, parent: MatchedName, child: MatchedName) -> AnnotatedScore:
        """
        How far down the administrative hierarchy (country, then admin codes
        in order) the child agrees with the parent.
        """
        parent_entity = parent.geographical_name.entity
        child_entity = child.geographical_name.entity
        logger = self.logger.getChild("parent_child_scoring")
        if min(parent_entity.possible_entity_types) <= max(child_entity.possible_entity_types):
            not_null_codes = 1
            if parent_entity.country.iso_code == child_entity.country.iso_code:
                logger.debug(
                    "Country code %r matches between parent %r (%r) and child %r (%r)",
                    parent_entity.country.iso_code,
                    parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                    child.cleaned_alt_name, child.geographical_name.entity.iri
                )
                codes_in_common = 1 
            else:
                logger.debug(
                    "Country codes %r and %r differ between parent %r (%r) and child %r (%r)",
                    parent_entity.country.iso_code, child_entity.country.iso_code,
                    parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                    child.cleaned_alt_name, child.geographical_name.entity.iri
                )
                return AnnotatedScore(0, "Different countries")
            for i, (parent_code, child_code) in enumerate(zip(parent_entity.admin_codes, child_entity.admin_codes)):
                if _is_admin_code_null(parent_code):
                    logger.debug(
                        "Admin %d code %r is considered null on the parent %r (%r); all codes match up to the parent's level, setting score to 1",
                        i, parent_code,
                        parent.cleaned_alt_name, parent.geographical_name.entity.iri
                    )
                    return AnnotatedScore(1, "Administrative Codes match up to parent's level")
                not_null_codes += 1
                if parent_code == child_code:
                    logger.debug(
                        "Admin %d codes %r match between parent %r (%r) and child %r (%r)",
                        i, parent_code,
                        parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                        child.cleaned_alt_name, child.geographical_name.entity.iri
                    )
                    codes_in_common += 1
                else:
                    score = codes_in_common / not_null_codes
                    logger.debug(
                        "Admin %d codes %r and %r differ between parent %r (%r) and child %r (%r), setting score to %.2f",
                        i, parent_code, child_code,
                        parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                        child.cleaned_alt_name, child.geographical_name.entity.iri,
                        score
                    )
                    return AnnotatedScore(score, "Administrative Code match ratio")
            logger.debug(
                "Full admin code match between parent %r (%r) and child %r (%r), setting score to 1",
                parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                child.cleaned_alt_name, child.geographical_name.entity.iri,
            )
            return AnnotatedScore(1, "Administrative Code full match")
        else:
            logger.debug(
                "Entity type mismatch between parent %r (%r) and child %r (%r): parent with entity types %r cannot possibly outrank child with entity types %r; setting score to 0",
                parent.cleaned_alt_name, parent.geographical_name.entity.iri,
                child.cleaned_alt_name, child.geographical_name.entity.iri,
                parent_entity.possible_entity_types, child_entity.possible_entity_types
            )
            return AnnotatedScore(0, "Entity type mismatch")
        
    def _is_entity_type_ruled_out(self, entity : MatchedEntity, match : MatchedName) -> bool:
        """
        Whether the match is of an entity type ruled out for the entity (see
        is_entity_type_ruled_out), left unpruned by GeoDBSearch so that it can
        still take part in _split_entity, where a split part may well be of
        that type (e.g. the river "Donau" in "Ulm/Donau").
        """
        if not is_entity_type_ruled_out(entity.entity_type, match.geographical_name.entity.possible_entity_types):
            return False
        self.logger.debug(
            "Pruned %r (%s) for %r: expected entity type %r but the possible entity types %r do not include City or Neighborhood",
            match.nfc_alt_name, match.geographical_name.entity.iri, entity.raw_text,
            entity.entity_type, match.geographical_name.entity.possible_entity_types)
        return True

    def _prune_match(self, reference_match : MatchedName, other_entity : MatchedEntity, other_match : ScoredMatch) -> bool:
        """
        Prune a match if it is unlikely to be the correct match for the given address and entity.

        Args:
            reference_match (MatchedName): The reference match.
            other_entity (MatchedEntity): The other entity.
            other_match (ScoredMatch): The other match.
        Returns:
            bool: True if the match should be pruned, False otherwise.
        """
        if self._is_entity_type_ruled_out(other_entity, other_match.match):
            return True
        if not (set(_country_iso_codes(reference_match)) & set(_country_iso_codes(other_match.match))):
            self.logger.debug(
                "Pruned %r (%s) for %r: country %s differs from reference %r country %s",
                other_match.match.nfc_alt_name, other_match.match.geographical_name.entity.iri, other_entity.raw_text,
                other_match.match.geographical_name.entity.country.iso_code,
                reference_match.nfc_alt_name, reference_match.geographical_name.entity.country.iso_code)
            return True

        if (
           other_entity.entity_type == GeographicalEntityType.Country and GeographicalEntityType.Country not in other_match.match.geographical_name.entity.possible_entity_types
        ):
            self.logger.debug(
                "Pruned %r (%s) for %r: entity is a country but match is not (types %s)",
                other_match.match.nfc_alt_name, other_match.match.geographical_name.entity.iri, other_entity.raw_text,
                other_match.match.geographical_name.entity.possible_entity_types)
            return True
        
        if other_entity.entity_type == GeographicalEntityType.Neighborhood and other_match.scores.get("child_parent_likelihood", AnnotatedScore(0, "Not applicable")).score < 0.6:
            self.logger.debug(
                "Pruned %r (%s) for neighborhood %r: child/parent likelihood %s below 0.6",
                other_match.match.nfc_alt_name, other_match.match.geographical_name.entity.iri, other_entity.raw_text,
                other_match.scores.get("child_parent_likelihood"))
            return True
        return False

    def _score_ambiguous_matches(
        self,
        address : AddressProcessingData, 
        entity : MatchedEntity,
        bzk_field : BZKFieldName
        ) -> list[LinkedAddress]:
        """
        Group possible addresses by parent entity.

        Args:
            address (LinkedAddress): The linked address to disambiguate.
            entity (MatchedEntity): The entity to group by.
        Returns:
            list[PossibleAddress]: A list of possible addresses grouped by parent entity.
        """
        
        possible_addresses = []
        if len(address.entities) == 0:
            return possible_addresses
        for match in entity.matches:
            if self._is_entity_type_ruled_out(entity, match):
                continue
            self.logger.debug(
                "Cross matching %r (%r) of type %r to other entities",
                match.nfc_alt_name, match.geographical_name.entity.iri,
                entity.entity_type.name
            )
            matched = {id(e): ScoredMatch(None, {}) for e in address.entities if e.entity_type in [GeographicalEntityType.Neighborhood, GeographicalEntityType.City, GeographicalEntityType.Region, GeographicalEntityType.State, GeographicalEntityType.Country, GeographicalEntityType.AboveCity, GeographicalEntityType.Unknown]}
            matched[id(entity)] = self._score_individual_match(entity, match, bzk_field)
            matched[id(entity)].scores["child_parent_likelihood"] = AnnotatedScore(1.0, "Reference match")
            matched[id(entity)].scores["weighted_score"] = self._calculate_weighted_score(matched[id(entity)].scores)
            finest_entity = entity
            for other_entity in address.entities:
                if other_entity is entity or not isinstance(other_entity, MatchedEntity) or other_entity.matches is None or len(other_entity.matches) == 0:
                    continue
                self.logger.debug(
                    "Cross matching %r (%r) to entities of type %r", 
                    match.nfc_alt_name, match.geographical_name.entity.iri,
                    other_entity.entity_type.name
                )
                best_other_match_scored = ScoredMatch(None, {})
                for other_match in other_entity.matches:
                    self.logger.debug(
                        "Scoring potential cross match %r (%r) of type %r against reference entity %r (%s)",
                        other_match.nfc_alt_name, other_match.geographical_name.entity.iri,
                        other_entity.entity_type.name,
                        match.nfc_alt_name, match.geographical_name.entity.iri,
                    )
                    scored_match = self._score_individual_match(other_entity, other_match, bzk_field)
                    if entity.entity_type < other_entity.entity_type:
                        scored_match.scores["child_parent_likelihood"] = self._score_parent_child(match, other_match)
                    else:
                        scored_match.scores["child_parent_likelihood"] = self._score_parent_child(other_match, match)
                    scored_match.scores["weighted_score"] = self._calculate_weighted_score(scored_match.scores)
                    self.logger.debug(
                        "Scores for other entity %r (%r) set: %r",
                        other_match.nfc_alt_name, other_match.geographical_name.entity.iri,
                        scored_match.scores
                    )
                    if self._prune_match(match, other_entity, scored_match):
                        self.logger.debug(
                            "Pruning entity %r (%r)",
                            other_match.nfc_alt_name, other_match.geographical_name.entity.iri
                        )
                        continue
                    old_scores = best_other_match_scored.scores
                    if best_other_match_scored.match == None or _score_dict_to_tuple(scored_match, self.priority) > _score_dict_to_tuple(old_scores, self.priority):
                        best_other_match_scored = ScoredMatch(other_match, scored_match.scores)
                        self.logger.debug(
                            "Best match for entity %r set to %r (%r)",
                            other_entity.entity_type.name,
                            best_other_match_scored.match.nfc_alt_name, best_other_match_scored.match.geographical_name.entity.iri
                        )
                        if other_entity.entity_type > finest_entity.entity_type:
                            finest_entity = other_entity
                if best_other_match_scored.match is None:
                    self.logger.debug("No match for entity %r", other_entity.entity_type.name)
                else:
                    self.logger.debug(
                        "Match %r (%r) with scores (%r) selected for entity %r", 
                        best_other_match_scored.match.nfc_alt_name, best_other_match_scored.match.geographical_name.entity.iri,
                        best_other_match_scored.scores,
                        other_entity.entity_type.name
                    )
                matched[id(other_entity)] = best_other_match_scored
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Cross matching complete for %r (%r) of type %r. Matches: %r",
                    match.nfc_alt_name, match.geographical_name.entity.iri,
                    entity.entity_type.name,
                    [s for s in matched.values()]
                )
            entity_weights = {
                id(e): MISSED_WORD_ENTITY_WEIGHT if e.is_missed_word else 1.0 for e in address.entities
            }
            scored_match = FrozenDict(_average_scores(
                list(matched.values()), [entity_weights[key] for key in matched]))
            self.logger.debug(
                "Final average scores for %r (%r) of type %r: %r",
                match.nfc_alt_name, match.geographical_name.entity.iri,
                entity.entity_type.name,
                scored_match
            )
            possible_addresses.append(LinkedAddress(
                finest_grain_entity=finest_entity.link_to(matched[id(finest_entity)].match, matched[id(finest_entity)].scores),
                reference_entity=entity.link_to(match, matched[id(entity)].scores),
                entities=tuple(
                    e.link_to(matched[id(e)].match, matched[id(e)].scores)
                    for e in address.entities if matched.get(id(e), ScoredMatch(None, {})).match
                ),
                scores=scored_match
            ))
        return sorted(possible_addresses, key=lambda a: _score_dict_to_tuple(a.scores, self.priority), reverse=True)

    def _split_entity(
            self, address : AddressProcessingData, entity : MatchedEntity
        ) -> list[tuple[AddressProcessingData, tuple[MatchedEntity, ...]]]:
        """
        Phase 'split entity': cross matches the partial word matches of the
        entity among themselves. Two of them matching different (non
        overlapping) parts of the entity's text (e.g. "Ulm" and "Donau" in
        "Ulm/Donau") hint that the entity names a place along with a
        qualifier (a river, a region) rather than one place under a longer
        name. For each such distinct pair of parts, the entity is split into
        one entity per part, holding the partial matches within that part,
        their fuzzy and phonetic scores recalculated against the part alone.
        The leading part keeps the entity type, while the trailing one, being
        the qualifier, is an AboveCity when the entity is finer than that.

        Returns, for each split, the address with the entity replaced by its
        parts, and those parts.
        """
        spans : dict[int, tuple[int, int]] = {}
        for i, match in enumerate(entity.matches):
            if not match.is_partial_word_match or match.cleaned_alt_name is None:
                continue
            span = partial_match_query_span(match.nfc_query, match.cleaned_alt_name)
            if span is not None:
                spans[i] = span
        splits = {}
        for i, j in itertools.combinations(spans, 2):
            m1, m2 = entity.matches[i], entity.matches[j]
            (s1, e1), (s2, e2) = sorted((spans[i], spans[j]))
            if m1.nfc_query != m2.nfc_query or e1 > s2:
                continue
            key = (m1.nfc_query, (s1, e1), (s2, e2))
            if key not in splits:
                self.logger.debug(
                    "Phase 'split entity' for %r: partial matches %r (%s) and %r (%s) match different parts "
                    "%r and %r of the query %r",
                    entity.raw_text, m1.nfc_alt_name, m1.geographical_name.entity.iri,
                    m2.nfc_alt_name, m2.geographical_name.entity.iri,
                    m1.nfc_query[s1:e1], m1.nfc_query[s2:e2], m1.nfc_query)
                splits[key] = None
        result = []
        for nfc_query, *part_spans in splits:
            parts = []
            for part_index, (start, end) in enumerate(part_spans):
                part_text = nfc_query[start:end]
                part_matches = []
                for i, span in spans.items():
                    if span[0] < start or span[1] > end:
                        continue
                    match = entity.matches[i]
                    edit_distance, fuzzy_score, phonetic_score = score_name_similarity(
                        part_text, match.nfc_alt_name, self.logger)
                    part_matches.append(dataclasses.replace(
                        match, nfc_query=part_text, edit_distance=edit_distance,
                        fuzzy_score=fuzzy_score, phonetic_score=phonetic_score))
                entity_type = entity.entity_type
                if part_index > 0 and entity_type > GeographicalEntityType.AboveCity:
                    entity_type = GeographicalEntityType.AboveCity
                # the span within the address is only known when the query
                # is the entity's text as is (e.g. not abbreviation expanded)
                span = None
                if entity.span is not None and entity.raw_text == nfc_query:
                    span = AddressSpan(entity.span.start + start, entity.span.start + end)
                parts.append(dataclasses.replace(
                    entity, address_entity_id=str(uuid.uuid4()), raw_text=part_text,
                    entity_type=entity_type, span=span, matches=tuple(part_matches)))
                self.logger.debug(
                    "Phase 'split entity' for %r: part %r (%s) with %d matches",
                    entity.raw_text, part_text, entity_type.name, len(part_matches))
            split_address = dataclasses.replace(address, entities=[
                part for e in address.entities for part in (parts if e is entity else [e])
            ])
            result.append((split_address, tuple(parts)))
        return result

    def _reference_entity_candidates(self, address : AddressProcessingData, entity : MatchedEntity) -> list[LinkedAddress]:
        """
        Every candidate address (not yet pruned) for entity as the reference
        entity, best first, including those of the entity's splits (see
        _split_entity) with any of its parts as the reference entity.
        """
        candidates = self._score_ambiguous_matches(address, entity, address.bzk_field_name)
        for split_address, parts in self._split_entity(address, entity):
            for part in parts:
                candidates.extend(self._score_ambiguous_matches(split_address, part, address.bzk_field_name))
        return sorted(candidates, key=lambda a: _score_dict_to_tuple(a.scores, self.priority), reverse=True)

    def _prune_reason(self, candidate: LinkedAddress) -> Optional[str]:
        """
        Why the candidate address is pruned based on score thresholds, or None
        if it is not.
        """
        if is_regional_term_entity(candidate.finest_grain_entity.linked_to.geographical_name.entity):
            return "finest entity is a regional term, which cannot be linked"
        score_dict = candidate.scores
        for factor, threshold in self.score_prune_thresholds.items():
            if score_dict.get(factor, 0.0) < threshold:
                return f"{factor} score {score_dict.get(factor, 0.0)} below threshold {threshold}"
        # finest_grain_entity = candidate.finest_grain_entity.linked_to.geographical_name.entity
        # if finest_grain_entity.country.continent != "EU" and finest_grain_entity.country.iso_code not in ("IL", "US"):
        #     if score_dict.get("parent_child_likelihood", 0.0) * score_dict.get("fuzzy_similarity_score", 0.0) < 0.5:
        #         return True
        return None

    def _prune(self, candidate: LinkedAddress) -> bool:
        """
        Prune possible addresses based on score thresholds.

        Args:
            candidate (LinkedAddress): The linked address to check.
        Returns:
            bool: True if the scores should be pruned, False otherwise.
        """
        reason = self._prune_reason(candidate)
        if reason is not None and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Pruned candidate %s: %s", _describe_linked_address(candidate), reason)
        return reason is not None

    def disambiguate(self, address : AddressProcessingData) -> AddressProcessingData:
        """
        Disambiguate a linked address using the Tantivy search index and the database connection.

        Args:
            address (LinkedAddress): The linked address to disambiguate.
            conn (Connection): The database connection.
            tantivy (TantivySearchIndex): The Tantivy search index.

        Returns:
            LinkedAddress: The disambiguated linked address.
        """
        unduped_entities = []
        for entity in address.entities:
            if isinstance(entity, MatchedEntity) and entity.matches is not None and len(entity.matches) > 0:
                unduped_entities.append(dataclasses.replace(entity, matches=self._drop_duplicates(entity, entity.matches, bzk_field=address.bzk_field_name)))
            else:
                unduped_entities.append(entity)
        address = dataclasses.replace(address, entities=unduped_entities)
        self.logger.debug("Disambiguating address %s (%r)", address.id, address.full_address)

        possible_addresses : list[LinkedAddress] = []

        for entity in sorted(address.entities, key=lambda e: (1 if e.entity_type == GeographicalEntityType.City else 0, e.entity_type), reverse=True):
            if not isinstance(entity, MatchedEntity) or entity.matches is None or len(entity.matches) == 0:
                self.logger.debug(
                    "Skipping %r (%s) as reference entity: no matches", entity.raw_text, entity.entity_type)
                continue
            self.logger.debug(
                "Phase 'reference entity' for address %s: entity %r (%s) with %d matches, field %s",
                address.id, entity.raw_text, entity.entity_type, len(entity.matches), address.bzk_field_name)
            result = self._reference_entity_candidates(address, entity)
            result = [r for r in result if not self._prune(r)]
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Phase 'reference entity' %r retrieved %d candidates: %s",
                    entity.raw_text, len(result), [_describe_linked_address(r) for r in result])
            possible_addresses.extend(result)
            if len(possible_addresses) > 0:
                best_address = possible_addresses[0]
                for other_address in possible_addresses[1:]:
                    if _compare_scores(best_address.scores, other_address.scores, self.priority, self.significance_thresholds) == "lt":
                        best_address = other_address
                reference_iris = set()
                reference_iris.add(best_address.finest_grain_entity.linked_to.geographical_name.entity.iri)
                likely_addresses = [best_address]
                for other_address in possible_addresses:
                    if _compare_scores(best_address.scores, other_address.scores, self.priority, self.significance_thresholds) == "eq":
                        if not other_address.finest_grain_entity.linked_to.geographical_name.entity.iri in reference_iris:
                            reference_iris.add(other_address.finest_grain_entity.linked_to.geographical_name.entity.iri)
                            likely_addresses.append(other_address)
                
                linked_to_common_parent = False
                if len(reference_iris) > 1:
                    self.logger.debug(
                        "Address %s is ambiguous: %d distinct candidates tie with the best (%s); "
                        "trying their common parent",
                        address.id, len(reference_iris), reference_iris)
                    best_address = self._link_to_common_parent(address, likely_addresses)
                    linked_to_common_parent = best_address is not None
                    if best_address is None:
                        self.logger.debug("Address %s left unlinked", address.id)
                elif self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "Address %s linked to %s", address.id, _describe_linked_address(best_address))

                return dataclasses.replace(address,
                    possible_links=possible_addresses,
                    likely_links=likely_addresses,
                    linked_to=best_address,
                    linked_to_common_parent=linked_to_common_parent
                )
        self.logger.debug("Address %s has no candidates; leaving unlinked", address.id)
        return dataclasses.replace(address,
            possible_links=tuple(),
            likely_links=tuple(),
        )
    
    