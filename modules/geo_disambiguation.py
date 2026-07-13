

from requests.compat import OrderedDict

from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import LinkedAddress, MatchedEntity, MatchedName, PossibleAddress
import dataclasses


class Disambiguator:
    def __init__(self, score_diff_threshold: float = 0.1):
        self.score_threshold = score_diff_threshold

    def _factor_entity_type(self, entity: MatchedEntity, name : MatchedName) -> float:
        """
        Heuristically estimate the probability that an entity is of a certain type.

        Args:
            entity (MatchedEntity): The entity to evaluate.
        Returns:
            float: The probability that the entity is of the specified type.
        """
        if entity.entity_type in name.geographical_name.entity.possible_entity_types:
            return 1.0
        else:
            return 0.75

    def _factor_child(self, parent: MatchedName, child: MatchedName) -> float:
        """
        Heuristically estimate the probability that a child entity is a child of a parent entity.

        Args:
            parent (MatchedEntity): The parent entity.
            child (MatchedEntity): The child entity.
        Returns:
            float: The probability that the child entity is a child of the parent entity.
        """
        parent_entity = parent.geographical_name.entity
        child_entity = child.geographical_name.entity
        if parent_entity.iri in child_entity.other_parent_iris:
            return 1.0
        if min(parent_entity.possible_entity_types) <= max(child_entity.possible_entity_types):
            not_null_codes = 1
            if parent_entity.country.iso_code == child_entity.country.iso_code:
                codes_in_common = 1 
            else:
                codes_in_common = 0
            for parent_code, child_code in zip(parent_entity.admin_codes, child_entity.admin_codes):
                if parent_code is None and child_code is None:
                    break
                not_null_codes += 1
                if parent_code == child_code:
                    codes_in_common += 1
            return codes_in_common / not_null_codes
        else:
            return 0.0
        
    def _collapse_duplicates(self, matches: list[MatchedName]) -> list[MatchedName]:
        """
        Collapse duplicate matches that refer to the same geographical entity, keeping the one with the highest similarity score.
        """
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
    
    def _score_ambiguous_matches(
        self,
        address : LinkedAddress, 
        entity : MatchedEntity
        ) -> list[PossibleAddress]:
        possible_addresses = []
        if len(address.matched_entities) == 0:
            return possible_addresses
        for match in entity.matches:
            matched = {id(e): (0.0, None) for e in address.matched_entities}
            match_score = match.raw_similarity * self._factor_entity_type(entity, match)
            matched[id(entity)] = (match_score, match)
            for other_entity in address.matched_entities:
                if other_entity is entity:
                    continue
                for other_match in other_entity.matches:
                    score = self._factor_child(match, other_match) * self._factor_entity_type(other_entity, other_match)
                    if score > 0.0:
                        score = score * other_match.raw_similarity
                        old_score, _ = matched.get(id(other_entity), (0.0, None))
                        if score > old_score:
                            matched[id(other_entity)] = (score, other_match)
            score = sum(s for s, m in matched.values()) + match.raw_similarity
            score = score / len(address.matched_entities)
            possible_addresses.append(PossibleAddress(
                entities=[
                    dataclasses.replace(e, linked_to=matched[id(e)][1]) 
                    for e in address.matched_entities
                ],
                score=score
            ))
        return sorted(possible_addresses, key=lambda a: a.score, reverse=True)

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
            possible_addresses = self._score_ambiguous_matches(address, entity)
            if len(possible_addresses) > 0:
                best_address = possible_addresses[0]
                filtered_addresses = [best_address]
                for other_address in possible_addresses[1:]:
                    if best_address.score - other_address.score <= self.score_threshold:
                        filtered_addresses.append(other_address)
                if len(filtered_addresses) > 1:
                    best_address = None

                return dataclasses.replace(address,
                    possible_addresses=filtered_addresses,
                    linked_to=best_address
                )
        return dataclasses.replace(address,
            possible_addresses=[]
        )
    
    