

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
        mean_match_scores_with_other_entities = [0.0] * len(entity.matches)
        for i, match in enumerate(entity.matches):
            n_matches_with_other_entities = 1
            sum_score = match.raw_similarity
            for other_entity in address.matched_entities:
                if other_entity is entity:
                    continue
                for other_match in other_entity.matches:
                    prob = self._prob_child(match, other_match)
                    if prob > 0.0:
                        n_matches_with_other_entities += 1
                        sum_score += prob * other_match.raw_similarity
                


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
        
        for entity in entities_to_disambiguate:
            if len(entity.matches) == 1:
                entity.disambiguation_result = entity.matches
            elif len(entity.matches) > 1:
                for match in entity.matches:
                    pass # TODO implement scoring of ambiguous matches