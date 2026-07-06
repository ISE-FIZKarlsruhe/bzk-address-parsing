

from modules.pipeline.geographical_entity import GeographicalEntityType
from modules.pipeline.linked_data import LinkedAddress, MatchedEntity, MatchedName, PossibleAddress


class Disambiguator:
    def __init__(self, score_diff_threshold: float = 0.5):
        self.score_threshold = score_diff_threshold

    def _prob_child(self, parent: MatchedName, child: MatchedName) -> float:
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
            if parent_entity.country.iso_country_code == child_entity.country.iso_country_code:
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
    
    def _score_ambiguous_matches(
        self,
        address : LinkedAddress, 
        entity : MatchedEntity
        ) -> list[PossibleAddress]:
        possible_addresses = []
        if len(address.matched_entities) <= 1:
            return possible_addresses
        for i, match in enumerate(entity.matches):
            matched = {}
            for other_entity in address.matched_entities:
                if other_entity is entity:
                    continue
                for other_match in other_entity.matches:
                    prob = self._prob_child(match, other_match)
                    if prob > 0.0:
                        n_matches_with_other_entities += 1
                        score = prob * other_match.raw_similarity
                        old_score, _ = matched.get(other_entity, (0.0, None))
                        if score > old_score:
                            matched[other_entity] = (score, other_match)
            matched_entities = [(k, v[1]) for k, v in matched.items()]
            matched_entities.sort(key=lambda x : x[0], reverse=True)
            score = sum(s for s, m in matched.values()) + match.raw_similarity
            score = score / len(address.matched_entities)
            possible_addresses.append(PossibleAddress(
                main_entity=entity,
                entities=[x[1] for x in matched_entities],
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
        entities_to_disambiguate = []
        for entity in address.matched_entities:
            if entity.entity_type == GeographicalEntityType.City:
                entities_to_disambiguate.append(entity)
        for entity in address.matched_entities:
            if entity.entity_type == GeographicalEntityType.Neighborhood:
                entities_to_disambiguate.append(entity)
        if len(entities_to_disambiguate) == 0:
            entities_to_disambiguate.append(max(address.matched_entities, key=lambda e: e.entity_type.value))
        

        for entity in sorted(address.matched_entities, key=lambda e: e.entity_type.value, reverse=True):
            possible_addresses = self._score_ambiguous_matches(address, entity)
            possible_addresses.sort(key=lambda a: a.score, reverse=True)
            if len(possible_addresses) > 0:
                best_address = possible_addresses[0]
                filtered_addresses = [best_address]
                for other_address in possible_addresses[1:]:
                    if best_address.score - other_address.score < self.score_threshold:
                        filtered_addresses.append(other_address)
                if len(filtered_addresses) > 1:
                    best_address = None

                return LinkedAddress(
                    id=address.id,
                    full_address=address.full_address,
                    bzk_field_name=address.bzk_field_name,
                    matched_entities=address.matched_entities,
                    possible_addresses=filtered_addresses,
                    linked_to=best_address
                )
        return None
    
    