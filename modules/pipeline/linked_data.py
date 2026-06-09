from dataclasses import dataclass
from typing import Hashable, LiteralString, Optional, Callable, NamedTuple
from enum import Enum
from modules.pipeline.geographical_entity import (
    GeographicalEntityType,
    GeographicalName
)

import pandas as pd

class BZKFieldName(str, Enum):
    APPLICANT_CURRENT_ADDRESS = "ApplicantCurrentAddress"
    APPLICANT_BIRTH_PLACE = "ApplicantBirthPlace"
    VICTIM_CURRENT_ADDRESS = "VictimCurrentAddress"
    VICTIM_BIRTH_PLACE = "VictimBirthPlace"
    VICTIM_DEATH_PLACE = "VictimDeathPlace"

@dataclass(frozen=True)
class MatchedName:
    geographical_name : GeographicalName
    query : str
    nfc_query : str
    clean_query : str
    abbreviation_pattern : Optional[str]
    nfc_alt_name : str
    clean_alt_name : str
    raw_distance : int
    cleaned_distance : int
    may_be_abbreviation : bool

    @property
    def raw_similarity(self) -> float:
        max_dist = max(len(self.nfc_query), len(self.nfc_alt_name))
        if max_dist == 0:
            return 1.0
        else:
            return 1 - self.raw_distance / max_dist
    
    @property
    def cleaned_similarity(self) -> float:
        max_dist = max(len(self.clean_query), len(self.clean_alt_name))
        if max_dist == 0:
            return 1.0
        else:
            return 1 - self.cleaned_distance / max_dist
    
    def __dict_encode__(self, default_encoder) -> dict:
        return default_encoder(self).update({
            "raw_similarity": self.raw_similarity,
            "cleaned_similarity": self.cleaned_similarity
        })

class AddressSpan(NamedTuple):
    start: int
    end: int

@dataclass(frozen=True)
class LinkedEntity:
    raw_text: str
    span : Optional[AddressSpan]
    nearby : bool # This entity is near the target address but the target may not be contained in it
    entity_type: GeographicalEntityType
    matches : Optional[list[MatchedName]]
    disambiguation_result : Optional[list[MatchedName]]
    
    @property
    def linked_to(self) -> MatchedName | None:
        if self.disambiguation_result is not None and len(self.disambiguation_result) == 1:
            return self.disambiguation_result[0]
        else:
            return None
        
    @property
    def is_resolved(self) -> bool:
        return self.linked_to is not None
    
    def __dict_encode__(self, default_encoder) -> dict:
        return default_encoder(self).update({
            "is_resolved": self.is_resolved
        })


@dataclass(frozen=True)
class LinkedAddress:
    id : str
    full_address: str
    bzk_field_name: BZKFieldName
    entities: list[LinkedEntity]
