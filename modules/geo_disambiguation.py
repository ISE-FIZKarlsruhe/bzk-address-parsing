

from os import name
import pprint
from typing import Literal, Optional, NamedTuple

from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import BZKFieldName, LinkedAddress, MatchedEntity, MatchedName, PossibleAddress
import dataclasses
from collections import defaultdict


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

class Score(NamedTuple):
    score : int | float
    comment : Optional[str] = None

    @classmethod
    def from_bool(cls, value : bool, negate : bool = False) -> "Score":
        comment = "True" if value else "False"
        score = 1.0 if value else 0.0
        if negate:
            score = 1.0 - score
        return cls(score, comment)

NA_SCORE = Score(0.0, "Not applicable")

class ScoredMatch(NamedTuple):
    match: MatchedName
    scores: dict[str, Score]

def _score_dict_to_tuple(scores : dict[str, Score | float], priority : list[str]) -> tuple:
    if isinstance(scores, ScoredMatch):
        scores = scores.scores
    def floatify(score : Score | float) -> float:
        if isinstance(score, Score):
            return score.score
        else:
            return score
    return tuple(floatify(scores.get(factor, 0.0)) for factor in priority)

def _score_abs_diff(s1 : dict[str, Score | float], s2 : dict[str, Score | float], priority : list[str]) -> float:
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

    def _score_individual_match(self, address : LinkedAddress, entity: MatchedEntity, name: MatchedName) -> ScoredMatch:
        scores = dict()
        scores["entity_types_matching"] = Score.from_bool(
            entity.entity_type in name.geographical_name.entity.possible_entity_types)
        scores["is_preferred_name"] = Score.from_bool(name.geographical_name.is_preferred_name)
        scores["fuzzy_similarity_score"] = Score(name.raw_similarity)
        
        # if address.bzk_field_name.is_birthplace():
            # if name.geographical_name.entity.country.iso_code == "DE":
                # scores["birthplace_country"] = Score(2, "Germany")
            # elif name.geographical_name.entity.country.iso_code == "PL":
                # scores["birthplace_country"] = Score(2, "Poland")
            # elif name.geographical_name.entity.country.continent == "EU":
                # scores["birthplace_country"] = Score(1, "Europe")

        if name.geographical_name.entity.country.iso_code == "DE":
            scores["country_likelihood_rank"] = Score(3, "Germany")
        elif name.geographical_name.entity.country.iso_code == "PL":
            scores["country_likelihood_rank"] = Score(3, "Poland")
        elif name.geographical_name.entity.country.continent == "EU":
            scores["country_likelihood_rank"] = Score(2, "Europe")
        elif name.geographical_name.entity.country.iso_code == "IL":
            scores["country_likelihood_rank"] = Score(1, "Israel")
        elif name.geographical_name.entity.country.iso_code == "US":
            scores["country_likelihood_rank"] = Score(1, "USA")
        
        if name.geographical_name.entity.population is None:
            scores["population_count"] = Score(1, "Unknown")
        else:
            scores["population_count"] = Score(float(name.geographical_name.entity.population / self.population_rounding_factor))
        return ScoredMatch(name, scores)

    def _score_parent_child(self, parent: MatchedName, child: MatchedName) -> Score:
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
            return Score(1, "Informal relationship")
        if min(parent_entity.possible_entity_types) <= max(child_entity.possible_entity_types):
            not_null_codes = 1
            if parent_entity.country.iso_code == child_entity.country.iso_code:
                codes_in_common = 1 
            else:
                return Score(0, "Different countries")
            for parent_code, child_code in zip(parent_entity.admin_codes, child_entity.admin_codes):
                if _is_admin_code_null(parent_code):
                    break
                not_null_codes += 1
                if parent_code == child_code:
                    codes_in_common += 1
            return Score(codes_in_common / not_null_codes, "Administrative Code match ratio")
        else:
            return Score(0, "Entity type mismatch")

    def _score_ambiguous_matches(
        self,
        address : LinkedAddress, 
        entity : MatchedEntity
        ) -> list[PossibleAddress]:
        """
        Group possible addresses by parent entity.

        Args:
            address (LinkedAddress): The linked address to disambiguate.
            entity (MatchedEntity): The entity to group by.
        Returns:
            list[PossibleAddress]: A list of possible addresses grouped by parent entity.
        """
        
        possible_addresses = []
        if len(address.matched_entities) == 0:
            return possible_addresses
        for match in entity.matches:
            matched = {id(e): ScoredMatch(None, {}) for e in address.matched_entities}
            matched[id(entity)] = self._score_individual_match(address, entity, match)
            matched[id(entity)].scores["child_parent_likelihood"] = Score(1.0, "Reference match")
            for other_entity in address.matched_entities:
                if other_entity is entity or other_entity.matches is None or len(other_entity.matches) == 0:
                    continue
                best_other_entity_score = ScoredMatch(None, {})
                for other_match in other_entity.matches:
                    score = self._score_individual_match(address, other_entity, other_match)
                    score.scores["child_parent_likelihood"] = self._score_parent_child(other_match, match)
                    old_scores = best_other_entity_score.scores
                    if _score_dict_to_tuple(score, self.priority) > _score_dict_to_tuple(old_scores, self.priority):
                        best_other_entity_score = ScoredMatch(other_match, score.scores)
                matched[id(other_entity)] = best_other_entity_score
            score = _average_scores([s for s in matched.values() if s.match is not None])
            # TODO metadata in named tuple score is not actually used yet...
            possible_addresses.append(PossibleAddress(
                reference_match=match,
                entities=[
                    dataclasses.replace(e, linked_to=matched[id(e)])
                    for e in address.matched_entities
                ],
                score=score
            ))
        return sorted(possible_addresses, key=lambda a: _score_dict_to_tuple(a.score, self.priority), reverse=True)
        


    def disambiguate(self, address : LinkedAddress) -> LinkedAddress:
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
        for entity in address.matched_entities:
            unduped_entities.append(dataclasses.replace(entity, matches=self._collapse_duplicates(entity.matches)))
        address = dataclasses.replace(address, matched_entities=unduped_entities)

        possible_addresses = []
        for entity in sorted(address.matched_entities, key=lambda e: e.entity_type, reverse=True):
            if entity.matches is None or len(entity.matches) == 0:
                continue
            possible_addresses = self._score_ambiguous_matches(address, entity)
            if len(possible_addresses) > 0:
                best_address = possible_addresses[0]
                reference_iris = set()
                reference_iris.add(best_address.reference_match.geographical_name.entity.iri)
                likely_addresses = [best_address]
                for other_address in possible_addresses[1:]:
                    if _score_abs_diff(best_address.score, other_address.score, self.priority) <= self.score_threshold:
                        if not other_address.reference_match.geographical_name.entity.iri in reference_iris:
                            reference_iris.add(other_address.reference_match.geographical_name.entity.iri)
                            likely_addresses.append(other_address)
                
                if len(reference_iris) > 1:
                    best_address = None

                return dataclasses.replace(address,
                    possible_addresses=possible_addresses,
                    likely_addresses=likely_addresses,
                    linked_to=best_address
                )
        return dataclasses.replace(address,
            possible_addresses=[],
            likely_addresses=[],
        )
    
    