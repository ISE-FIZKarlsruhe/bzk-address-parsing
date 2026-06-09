from dataclasses import dataclass
from enum import Enum
from typing import LiteralString, Optional, TYPE_CHECKING, NamedTuple
import pandas as pd


@dataclass(frozen=True)
class GeographicalEntityTypeProperties:
    entity_type : LiteralString
    hierarchy_level : int # lower means higher in the hierarchy. Induces a partial order

class GeographicalEntityType(GeographicalEntityTypeProperties, Enum):
    Country = "Country", 0
    State = "State", 1
    Region = "Region", 1
    District = "District", 2
    City = "City", 3
    Neighborhood = "Neighborhood", 4
    StreetName = "StreetName", 5
    HouseNumber = "HouseNumber", 6

    def __gt__(self, other):
        if not isinstance(other, GeographicalEntityType):
            return NotImplemented
        return self.hierarchy_level > other.hierarchy_level
    
    def __lt__(self, other):
        if not isinstance(other, GeographicalEntityType):
            return NotImplemented
        return self.hierarchy_level < other.hierarchy_level

class GeographicalEntityProvider(str, Enum):
    #TODO check codes against actual URIs
    GEONAMES = "https://sws.geonames.org/"
    WIKIDATA = "https://www.wikidata.org/wiki/" 
    GND = "https://d-nb.info/gnd/"

class GeonamesAdminCodes(NamedTuple):
    admin1_code : Optional[str]
    admin2_code : Optional[str]
    admin3_code : Optional[str]
    admin4_code : Optional[str]
    admin5_code : Optional[str]

class Coordinates(NamedTuple):
    latitude : float
    longitude : float

class GeonamesClassification(NamedTuple):
    feature_class : Optional[str]
    feature_code : Optional[str]

    @classmethod
    def from_string(cls, classif : Optional[str]) -> "GeonamesClassification":
        if classif is None:
            return cls(feature_class=None, feature_code=None)
        parts = classif.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid geonames classification: {classif}")
        return cls(feature_class=parts[0], feature_code=parts[1])

@dataclass(frozen=True)
class CountryData:
    iso_country_code : str
    country_name : str
    iso_languages : list[str]
    neighboring_countries_iso_codes : list[str]


@dataclass(frozen=True)
class GeographicalEntity:
    iri : str
    provider : GeographicalEntityProvider
    name : str
    asciiname : Optional[str]
    classification : Optional[str]
    possible_entity_types : list[GeographicalEntityType]
    coordinates : Optional[Coordinates]
    population : Optional[int]
    closest_geonames_id : int
    country : Optional[CountryData]
    alternate_countries : list[CountryData]
    admin_codes : GeonamesAdminCodes
    other_parent_iris : list[str]

    @property
    def get_all_countries(self):
        if self.country is not None:
            return [self.country] + self.alternate_countries
        else:
            return self.alternate_countries

@dataclass(frozen=True)
class GeographicalName:
    alternate_name : str
    entity : GeographicalEntity
    is_preferred_name: bool
    is_short_name: Optional[bool]
    is_colloquial : Optional[bool]
    name_provider : Optional[GeographicalEntityProvider]
    isolanguage : Optional[str]