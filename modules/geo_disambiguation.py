

from os import name
import pprint
from typing import Literal, Optional, NamedTuple

from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import BZKFieldName, AddressProcessingData, LinkedEntity, MatchedEntity, MatchedName, LinkedAddress
import dataclasses
from collections import defaultdict
from modules.pipeline.linked_data import AnnotatedScore
from modules.pipeline.storage.frozendict import FrozenDict
import math

def _is_admin_code_null(code : Optional[str]) -> bool:
    return code is None or code == "" or all(c == "0" for c in code)


DISAMBIGUATION_FACTOR_PRIORITY = [
    "fuzzy_similarity_score",
    "child_parent_likelihood",
    "entity_types_matching_fuzzy",
    "population_order_of_magnitude",
    "country_likelihood_rank", # general rank of country likelihood based on observation
    "entity_types_matching",
    "is_preferred_name",
    "population_count"
]

NA_SCORE = AnnotatedScore(0.0, "Not applicable")

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
        threshold : float = 0.0
    ) -> float:
    """
    Returns true if the difference between the two score dictionaries is under the given threshold.
    """
    # TODO this logic does not work:
    #   - If a higher priority factor differs it is returned even if under the threshold
    #   - If instead we skip this factor, we may encounter a higher score on lower priority factors, which is not handled
    t1 = _score_dict_to_tuple(s1, priority)
    t2 = _score_dict_to_tuple(s2, priority)
    return all(abs(a - b) <= threshold for a, b in zip(t1, t2))

def _average_scores(scored_matches : list[ScoredMatch]) -> dict[str, float]:
    scores = defaultdict(lambda: 0.0)
    for scored_match in scored_matches:
        for factor, score in scored_match.scores.items():
            scores[factor] += score.score
    return {factor: score / len(scored_matches) for factor, score in scores.items()}

class Disambiguator:
    def __init__(
            self, 
            score_diff_threshold: float = 0.0, 
            priority: list[str] = DISAMBIGUATION_FACTOR_PRIORITY,
            population_rounding_factor: int = 10_000,
            score_prune_thresholds : dict[str, float] = defaultdict(float)
        ):
        if score_diff_threshold != 0.0:
            raise NotImplementedError("Non-zero score difference threshold is not currently working.")
        self.score_threshold = score_diff_threshold
        self.priority = priority
        self.population_rounding_factor = population_rounding_factor
        self.score_prune_thresholds = score_prune_thresholds

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
                continue
            already_seen.add(iri)
            unique_matches.append(match)
        return unique_matches

    def _score_individual_match(self, entity: MatchedEntity, name: MatchedName, bzk_field : BZKFieldName) -> ScoredMatch:
        scores = dict()
        if entity.entity_type in name.geographical_name.entity.possible_entity_types:
            scores["entity_types_matching_fuzzy"] = AnnotatedScore(1.0, "Entity type matches")
            scores["entity_types_matching"] = AnnotatedScore(1.0, "Entity type matches")
        else:
            scores["entity_types_matching"] = AnnotatedScore(0.0, "Entity type does not match")
            if entity.entity_type == GeographicalEntityType.Neighborhood and GeographicalEntityType.City in name.geographical_name.entity.possible_entity_types:
                scores["entity_types_matching_fuzzy"] = AnnotatedScore(1.0, "City instead of neighborhood match")
            elif entity.entity_type == GeographicalEntityType.City and GeographicalEntityType.Neighborhood in name.geographical_name.entity.possible_entity_types:
                scores["entity_types_matching_fuzzy"] = AnnotatedScore(1.0, "Neighborhood instead of city match")
            else:
                scores["entity_types_matching_fuzzy"] = AnnotatedScore(0.0, "Entity type does not match")
            
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
            if name.geographical_name.entity.population < 10_000:
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
        if parent_entity.iri in child_entity.other_parent_iris:
            return AnnotatedScore(1, "Informal relationship")
        if min(parent_entity.possible_entity_types) <= max(child_entity.possible_entity_types):
            not_null_codes = 1
            if parent_entity.country.iso_code == child_entity.country.iso_code:
                codes_in_common = 1 
            else:
                return AnnotatedScore(0, "Different countries")
            for parent_code, child_code in zip(parent_entity.admin_codes, child_entity.admin_codes):
                if _is_admin_code_null(parent_code):
                    break
                not_null_codes += 1
                if parent_code == child_code:
                    codes_in_common += 1
            return AnnotatedScore(codes_in_common / not_null_codes, "Administrative Code match ratio")
        else:
            return AnnotatedScore(0, "Entity type mismatch")
        
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
        if (
            reference_match.geographical_name.entity.country.iso_code != other_match.match.geographical_name.entity.country.iso_code
        ):
            return True

        if (
           other_entity.entity_type == GeographicalEntityType.Country and GeographicalEntityType.Country not in other_match.match.geographical_name.entity.possible_entity_types
        ):
            return True
        
        if other_entity.entity_type == GeographicalEntityType.Neighborhood and other_match.scores.get("child_parent_likelihood", AnnotatedScore(0, "Not applicable")).score < 0.6:
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
            matched = {id(e): ScoredMatch(None, {}) for e in address.entities if e.entity_type in [GeographicalEntityType.Neighborhood, GeographicalEntityType.City, GeographicalEntityType.Region, GeographicalEntityType.State, GeographicalEntityType.Country]}
            matched[id(entity)] = self._score_individual_match(entity, match, bzk_field)
            matched[id(entity)].scores["child_parent_likelihood"] = AnnotatedScore(1.0, "Reference match")
            finest_entity = entity
            for other_entity in address.entities:
                if other_entity is entity or not isinstance(other_entity, MatchedEntity) or other_entity.matches is None or len(other_entity.matches) == 0:
                    continue
                best_other_match_scored = ScoredMatch(None, {})
                for other_match in other_entity.matches:
                    scored_match = self._score_individual_match(other_entity, other_match, bzk_field)
                    if entity.entity_type < other_entity.entity_type:
                        scored_match.scores["child_parent_likelihood"] = self._score_parent_child(match, other_match)
                    else:
                        scored_match.scores["child_parent_likelihood"] = self._score_parent_child(other_match, match)
                    if self._prune_match(match, other_entity, scored_match):
                        continue
                    old_scores = best_other_match_scored.scores
                    if best_other_match_scored.match == None or _score_dict_to_tuple(scored_match, self.priority) > _score_dict_to_tuple(old_scores, self.priority):
                        best_other_match_scored = ScoredMatch(other_match, scored_match.scores)
                        if other_entity.entity_type > finest_entity.entity_type:
                            finest_entity = other_entity
                matched[id(other_entity)] = best_other_match_scored
            scored_match = FrozenDict(_average_scores([s for s in matched.values()]))
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
    
    def _prune(self, score_dict : dict[str, float]) -> bool:
        """
        Prune possible addresses based on score thresholds.

        Args:
            score_dict (dict[str, float]): The dictionary of scores to check.
        Returns:
            bool: True if the scores should be pruned, False otherwise.
        """
        for factor, threshold in self.score_prune_thresholds.items():
            if score_dict.get(factor, 0.0) < threshold:
                return True
        return False

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

        possible_addresses : list[LinkedAddress] = []

        for entity in sorted(address.entities, key=lambda e: (1 if e.entity_type == GeographicalEntityType.City else 0, e.entity_type), reverse=True):
            if not isinstance(entity, MatchedEntity) or entity.matches is None or len(entity.matches) == 0:
                continue
            possible_addresses.extend(self._score_ambiguous_matches(address, entity, address.bzk_field_name))
            if len(possible_addresses) > 0:
                best_address = possible_addresses[0]
                if self._prune(best_address.scores):
                    return dataclasses.replace(address,
                        possible_links=possible_addresses,
                        likely_links=tuple(),
                        linked_to=None
                    )
                reference_iris = set()
                reference_iris.add(best_address.finest_grain_entity.linked_to.geographical_name.entity.iri)
                likely_addresses = [best_address]
                for other_address in possible_addresses[1:]:
                    if _compare_scores(best_address.scores, other_address.scores, self.priority, self.score_threshold):
                        if not other_address.finest_grain_entity.linked_to.geographical_name.entity.iri in reference_iris:
                            reference_iris.add(other_address.finest_grain_entity.linked_to.geographical_name.entity.iri)
                            likely_addresses.append(other_address)
                
                if len(reference_iris) > 1:
                    best_address = None

                return dataclasses.replace(address,
                    possible_links=possible_addresses,
                    likely_links=likely_addresses,
                    linked_to=best_address
                )
        return dataclasses.replace(address,
            possible_links=tuple(),
            likely_links=tuple(),
        )
    
    