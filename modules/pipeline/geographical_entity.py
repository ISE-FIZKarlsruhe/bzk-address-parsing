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

@dataclass(frozen=True)
class GeographicalEntity:
    iri : str
    provider : GeographicalEntityProvider
    name : str
    classification : Optional[str | GeonamesClassification]
    possible_entity_types : list[GeographicalEntityType]
    coordinates : Optional[Coordinates]
    population : Optional[int]
    iso_country_code : Optional[str]
    closest_geonames_id : int
    asciiname : Optional[str]
    admin_codes : GeonamesAdminCodes
    parent_city_ids : list[int]
    parent_region_ids : list[int]
    country_name : Optional[str]

    @classmethod
    def from_db_row(cls, row : dict | 'pd.Series') -> "GeographicalEntity":
        iri = row["iri"]
        if iri.startswith(GeographicalEntityProvider.GEONAMES.value):
            provider = GeographicalEntityProvider.GEONAMES
        elif iri.startswith(GeographicalEntityProvider.WIKIDATA.value):
            provider = GeographicalEntityProvider.WIKIDATA
        else:
            raise ValueError(f"Unknown provider in IRI: {iri}")
        
        coordinates = (row.get("latitude"), row.get("longitude"))
        if pd.isna(coordinates[0]) or pd.isna(coordinates[1]):
            coordinates = None

        admin_codes = GeonamesAdminCodes(
            admin1_code=row.get("admin1_code"),
            admin2_code=row.get("admin2_code"),
            admin3_code=row.get("admin3_code"),
            admin4_code=row.get("admin4_code"),
            admin5_code=row.get("admin5_code")
        )

        geonames_class = GeonamesClassification(
            feature_class=row.get("feature_class"),
            feature_code=row.get("feature_code")
        )
        if pd.isna(geonames_class.feature_class) or pd.isna(geonames_class.feature_code):
            geonames_class = None
        name : str = row["name"]
        asciiname = row.get("asciiname")
        if asciiname is None and name.isascii():
            asciiname = name
        return cls(
            iri=iri,
            provider=provider,
            name=name,
            classification=row.get("classification") or geonames_class,
            possible_entity_types=[
                GeographicalEntityType(et) for et, v in row["entity_type_map"].items() if v
            ],
            coordinates=coordinates,
            population=row.get("population"),
            iso_country_code=row.get("country_code"),
            closest_geonames_id=row["closest_geonames_id"],
            asciiname=asciiname,
            admin_codes=admin_codes,
            parent_city_ids=row.get("parent_city_ids", []),
            parent_region_ids=row.get("parent_region_ids", []),
            country_name=row.get("country_name")
        )

@dataclass(frozen=True)
class GeographicalName:
    alternate_name : str
    entity : GeographicalEntity
    is_preferred_name: bool
    is_short_name: Optional[bool]
    is_colloquial : Optional[bool]
    name_provider : GeographicalEntityProvider
    isolanguage : Optional[str]

    @classmethod
    def from_db_row(cls, row : dict | 'pd.Series') -> "GeographicalName":
        entity = GeographicalEntity.from_db_row(row)
        return cls(
            alternate_name=row["alternate_name"],
            entity=entity,
            is_preferred_name=row["is_preferred_name"],
            is_short_name=row.get("is_short_name"),
            is_colloquial=row.get("is_colloquial"),
            name_provider=GeographicalEntityProvider(row["name_provider"]),
            isolanguage=row.get("isolanguage")
        )