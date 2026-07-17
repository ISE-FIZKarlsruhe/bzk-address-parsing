from dataclasses import dataclass
from typing import Hashable, LiteralString, Optional, Callable, NamedTuple
from enum import Enum
from modules.pipeline.geographical_entity import (
    GeographicalEntityType,
    GeographicalName
)

import pandas as pd
from functools import cached_property


class BZKFieldName(str, Enum):
    APPLICANT_CURRENT_ADDRESS = "ApplicantCurrentAddress"
    APPLICANT_BIRTH_PLACE = "ApplicantBirthPlace"
    VICTIM_CURRENT_ADDRESS = "VictimCurrentAddress"
    VICTIM_BIRTH_PLACE = "VictimBirthPlace"
    VICTIM_DEATH_PLACE = "VictimDeathPlace"

    def is_birthplace(self) -> bool:
        return self == BZKFieldName.APPLICANT_BIRTH_PLACE or self == BZKFieldName.VICTIM_BIRTH_PLACE
    
    def is_current_address(self) -> bool:
        return self == BZKFieldName.APPLICANT_CURRENT_ADDRESS or self == BZKFieldName.VICTIM_CURRENT_ADDRESS

@dataclass(frozen=True)
class MatchedName:
    geographical_name : GeographicalName
    nfc_query : str
    nfc_alt_name : str
    cleaned_queries : list[str]
    cleaned_alt_name : str
    cleaned_edit_distance : int
    matching_method : str
    abbreviation_pattern : Optional[str]
    edit_distance : int
    is_abbreviation_match : bool

    @cached_property
    def raw_similarity(self) -> float:
        max_dist = max(len(self.nfc_query), len(self.nfc_alt_name))
        if max_dist == 0:
            return 1.0
        else:
            return 1 - self.edit_distance / max_dist
    
    @cached_property
    def cleaned_similarity(self) -> float:
        max_sim = 0.0
        for clean_query, clean_alt_name, clean_distance in zip(self.cleaned_queries, self.cleaned_alt_names, self.cleaned_edit_distances):
            max_dist = max(len(clean_query), len(clean_alt_name))
            if max_dist == 0:
                return 1.0
            else:
                sim = 1 - clean_distance / max_dist
                if sim == 1.0:
                    return 1.0
                if sim > max_sim:
                    max_sim = sim
        return max_sim
            
    
    def __dict_encode__(self, default_encoder) -> dict:
        return default_encoder(self).update({
            "raw_similarity": self.raw_similarity,
            "cleaned_similarity": self.cleaned_similarity
        })

class AddressSpan(NamedTuple):
    start: int
    end: int

@dataclass(frozen=True)
class MatchedEntity:
    raw_text: str
    entity_type: GeographicalEntityType
    matches : Optional[list[MatchedName]] = None
    linked_to : Optional[MatchedName] = None
    span : Optional[AddressSpan] = None
    nearby : bool = False # This entity is near the target address but the target may not be contained in it



@dataclass(frozen=True)
class PossibleAddress:
    reference_match : MatchedName
    entities : list[MatchedEntity]
    score : float
    
    

@dataclass(frozen=True)
class LinkedAddress:
    id : str
    full_address: str
    bzk_field_name: BZKFieldName
    matched_entities: list[MatchedEntity]
    possible_addresses: Optional[list[PossibleAddress]] = None
    likely_addresses: Optional[list[PossibleAddress]] = None
    linked_to : Optional[PossibleAddress] = None
