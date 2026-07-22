from dataclasses import dataclass, field
from typing import Hashable, LiteralString, Optional, Callable, NamedTuple, Literal
from enum import Enum
from modules.pipeline.geographical_entity import (
    GeographicalEntityType,
    GeographicalName
)

import pandas as pd
from functools import cached_property
import uuid

from modules.pipeline.storage.frozendict import FrozenDict


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
    cleaned_query : str
    cleaned_alt_name : str
    cleaned_edit_distance : int
    matching_method : str
    matching_score : float
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
        cleaned_query = self.cleaned_query or ""
        cleaned_alt_name = self.cleaned_alt_name or ""
        cleaned_edit_distance = self.cleaned_edit_distance or 0
        max_dist = max(len(cleaned_query), len(cleaned_alt_name))
        if max_dist == 0:
            return 1.0
        else:
            return 1 - cleaned_edit_distance / max_dist
            
    
    def __dict_encode__(self, default_encoder) -> dict:
        data = default_encoder(self)
        data.update({
            "raw_similarity": self.raw_similarity,
            "cleaned_similarity": self.cleaned_similarity
        })
        return data
    

class AddressSpan(NamedTuple):
    start: int
    end: int

@dataclass(frozen=True)
class RawEntity:
    address_entity_id: str
    raw_text: str
    entity_type: GeographicalEntityType
    span : Optional[AddressSpan]
    nearby : Optional[str | Literal['unspecified']] # This entity is near the target address but the target may not be contained in it

    @classmethod
    def with_parsed(cls, raw_text: str, entity_type: GeographicalEntityType | str, span: Optional[AddressSpan] = None, nearby: Optional[str | Literal['unspecified']] = None) -> 'RawEntity':
        if isinstance(entity_type, str):
            entity_type = GeographicalEntityType[entity_type]
        return cls(
            address_entity_id=str(uuid.uuid4()),
            raw_text=raw_text,
            entity_type=entity_type,
            span=span,
            nearby=nearby
        )

    def with_matches(self, matches: tuple[MatchedName, ...]) -> 'MatchedEntity':
        return MatchedEntity(
            address_entity_id=self.address_entity_id,
            raw_text=self.raw_text,
            entity_type=self.entity_type,
            span=self.span,
            nearby=self.nearby,
            matches=matches
        )
    
    def link_to(self, matched_name: MatchedName, scores: FrozenDict | dict[str, float]) -> 'LinkedEntity':
        if isinstance(scores, dict):
            scores = FrozenDict(scores)
        return LinkedEntity(
            address_entity_id=self.address_entity_id,
            raw_text=self.raw_text,
            entity_type=self.entity_type,
            span=self.span,
            nearby=self.nearby,
            linked_to=matched_name,
            scores=scores
        )
    
    @classmethod
    def __dict_decode__(cls, data: dict, targs, default_decoder):
        if "matches" in data:
            return default_decoder(data, MatchedEntity)
        elif "linked_to" in data:
            return default_decoder(data, LinkedEntity)
        else:
            return default_decoder(data, cls)

@dataclass(frozen=True)
class MatchedEntity(RawEntity):
    matches : tuple[MatchedName, ...]

class AnnotatedScore(NamedTuple):
    score : int | float
    comment : Optional[str] = None

    @classmethod
    def from_bool(cls, value : bool, negate : bool = False) -> "AnnotatedScore":
        comment = "True" if value else "False"
        score = 1.0 if value else 0.0
        if negate:
            score = 1.0 - score
        return cls(score, comment)

@dataclass(frozen=True)
class LinkedEntity(RawEntity):
    linked_to : MatchedName
    scores : FrozenDict[str, AnnotatedScore]

@dataclass(frozen=True)
class LinkedAddress:
    finest_grain_entity : LinkedEntity
    reference_entity : LinkedEntity
    entities : tuple[LinkedEntity, ...]
    scores : FrozenDict[str, float]
    
    

@dataclass(frozen=True)
class AddressProcessingData:
    card_id : str
    id : str
    full_address: str
    bzk_field_name: BZKFieldName
    entities: list[RawEntity]
    possible_links: Optional[tuple[LinkedAddress, ...]] = None
    likely_links: Optional[tuple[LinkedAddress, ...]] = None
    linked_to : Optional[LinkedAddress] = None

