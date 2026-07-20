

from os import name
import pprint
from typing import Literal, Optional, NamedTuple

from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import BZKFieldName, AddressProcessingData, MatchedEntity, MatchedName, LinkedAddress
import dataclasses
from collections import defaultdict
from modules.pipeline.linked_data import AnnotatedScore
from modules.pipeline.storage.frozendict import FrozenDict


def _is_admin_code_null(code : Optional[str]) -> bool:
    # TODO 0+ always none?
    return code is None or code == "" or all(c == "0" for c in code)

DISAMBIGUATION_FACTOR_PRIORITY = [
    "fuzzy_similarity_score",
    "child_parent_likelihood",
    "entity_types_matching",
    "birthplace_country", # based on https://deutsche-digitale-bibliothek.atlassian.net/wiki/spaces/ArchivD/pages/49677679/Postprocessing
    "country_likelihood_rank", # general rank of country likelihood based on observation
    "is_preferred_name",
    "population_count"
]

def _tuple_abs_diff(t1 : tuple, t2 : tuple) -> float:
    """
    Absolute difference between two tuples lexicographically.
    """
    for a, b in zip(t1, t2):
        if a != b:
            return abs(a - b)
    return 0.0

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

def _score_abs_diff(s1 : dict[str, AnnotatedScore | float], s2 : dict[str, AnnotatedScore | float], priority : list[str]) -> float:
    """
    Absolute difference between two score dictionaries lexicographically.
    """
    t1 = _score_dict_to_tuple(s1, priority)
    t2 = _score_dict_to_tuple(s2, priority)
    return _tuple_abs_diff(t1, t2)

def _average_scores(scored_matches : list[ScoredMatch]) -> dict[str, float]:
    scores = defaultdict(lambda: 0.0)
    for scored_match in scored_matches:
        for factor, score in scored_match.scores.items():
            scores[factor] += score.score
    return {factor: score / len(scored_matches) for factor, score in scores.items()}

class Disambiguator:
    def __init__(
            self, 
            score_diff_threshold: float = 0.1, 
            priority: list[str] = DISAMBIGUATION_FACTOR_PRIORITY,
            population_rounding_factor: int = 10_000,
        ):
        self.score_threshold = score_diff_threshold
        self.priority = priority
        self.population_rounding_factor = population_rounding_factor
        

    def _collapse_duplicates(self, matches: list[MatchedName]) -> list[MatchedName]:
        """
        Collapse duplicate matches that refer to the same geographical entity, keeping the one with the highest similarity score.
        """
        if matches is None or len(matches) == 0:
            return matches
        already_seen = set()
        matches = sorted(matches, key=lambda m: m.raw_similarity, reverse=True)
        unique_matches = []
        for match in matches:
            iri = match.geographical_name.entity.iri
            if iri in already_seen:
                continue
            already_seen.add(iri)
            unique_matches.append(match)
        return unique_matches

    def _score_individual_match(self, address : AddressProcessingData, entity: MatchedEntity, name: MatchedName) -> ScoredMatch:
        scores = dict()
        scores["entity_types_matching"] = AnnotatedScore.from_bool(
            entity.entity_type in name.geographical_name.entity.possible_entity_types)
        scores["is_preferred_name"] = AnnotatedScore.from_bool(name.geographical_name.is_preferred_name)
        scores["fuzzy_similarity_score"] = AnnotatedScore(name.raw_similarity)
        
        # if address.bzk_field_name.is_birthplace():
            # if name.geographical_name.entity.country.iso_code == "DE":
                # scores["birthplace_country"] = Score(2, "Germany")
            # elif name.geographical_name.entity.country.iso_code == "PL":
                # scores["birthplace_country"] = Score(2, "Poland")
            # elif name.geographical_name.entity.country.continent == "EU":
                # scores["birthplace_country"] = Score(1, "Europe")

        if name.geographical_name.entity.country.iso_code == "DE":
            scores["country_likelihood_rank"] = AnnotatedScore(3, "Germany")
        elif name.geographical_name.entity.country.iso_code == "PL":
            scores["country_likelihood_rank"] = AnnotatedScore(3, "Poland")
        elif name.geographical_name.entity.country.continent == "EU":
            scores["country_likelihood_rank"] = AnnotatedScore(2, "Europe")
        elif name.geographical_name.entity.country.iso_code == "IL":
            scores["country_likelihood_rank"] = AnnotatedScore(1, "Israel")
        elif name.geographical_name.entity.country.iso_code == "US":
            scores["country_likelihood_rank"] = AnnotatedScore(1, "USA")
        
        if name.geographical_name.entity.population is None:
            scores["population_count"] = AnnotatedScore(1, "Unknown")
        else:
            scores["population_count"] = AnnotatedScore(float(name.geographical_name.entity.population / self.population_rounding_factor))
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

    def _score_ambiguous_matches(
        self,
        address : AddressProcessingData, 
        entity : MatchedEntity
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
            matched = {id(e): ScoredMatch(None, {}) for e in address.entities}
            matched[id(entity)] = self._score_individual_match(address, entity, match)
            matched[id(entity)].scores["child_parent_likelihood"] = AnnotatedScore(1.0, "Reference match")
            for other_entity in address.entities:
                if other_entity is entity or not isinstance(other_entity, MatchedEntity) or other_entity.matches is None or len(other_entity.matches) == 0:
                    continue
                best_other_entity_score = ScoredMatch(None, {})
                for other_match in other_entity.matches:
                    scores = self._score_individual_match(address, other_entity, other_match)
                    scores.scores["child_parent_likelihood"] = self._score_parent_child(other_match, match)
                    old_scores = best_other_entity_score.scores
                    if _score_dict_to_tuple(scores, self.priority) > _score_dict_to_tuple(old_scores, self.priority):
                        best_other_entity_score = ScoredMatch(other_match, scores.scores)
                matched[id(other_entity)] = best_other_entity_score
            scores = FrozenDict(_average_scores([s for s in matched.values() if s.match is not None]))
            possible_addresses.append(LinkedAddress(
                reference_entity=entity.link_to(match, matched[id(entity)].scores),
                entities=tuple(
                    e.link_to(matched[id(e)].match, matched[id(e)].scores) if matched[id(e)].match is not None else e
                    for e in address.entities
                ),
                scores=scores
            ))
        return sorted(possible_addresses, key=lambda a: _score_dict_to_tuple(a.scores, self.priority), reverse=True)
        


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
                unduped_entities.append(dataclasses.replace(entity, matches=self._collapse_duplicates(entity.matches)))
            else:
                unduped_entities.append(entity)
        address = dataclasses.replace(address, entities=unduped_entities)

        possible_addresses = []
        for entity in sorted(address.entities, key=lambda e: e.entity_type, reverse=True):
            if not isinstance(entity, MatchedEntity) or entity.matches is None or len(entity.matches) == 0:
                continue
            possible_addresses = self._score_ambiguous_matches(address, entity)
            if len(possible_addresses) > 0:
                best_address = possible_addresses[0]
                reference_iris = set()
                reference_iris.add(best_address.reference_entity.linked_to.geographical_name.entity.iri)
                likely_addresses = [best_address]
                for other_address in possible_addresses[1:]:
                    if _score_abs_diff(best_address.scores, other_address.scores, self.priority) <= self.score_threshold:
                        if not other_address.reference_entity.linked_to.geographical_name.entity.iri in reference_iris:
                            reference_iris.add(other_address.reference_entity.linked_to.geographical_name.entity.iri)
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
    
    