from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Optional, TYPE_CHECKING, NamedTuple
import numpy as np
import pandas as pd
from urllib.parse import urlparse

@dataclass(frozen=True)
class GeographicalEntityTypeProperties:
    entity_type : str
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
    # Not a real geographical feature type: a parsing-time hint for a place
    # named alongside the city that is broader than it (e.g. "Danzig" in
    # "Putzig (Danzig)"), without knowing which of the coarser types it
    # actually is. GeoDBSearch expands it into a disjunction over all entity
    # types above City (see ABOVE_CITY_ENTITY_TYPES) when searching.
    AboveCity = "AboveCity", 1
    # Not a real geographical feature type either: a place whose type parsing
    # could not tell (e.g. a single-word address, which is most often a City
    # but sometimes a Country). It acts as a wildcard matching any entity
    # type, both when searching (see GeoDBSearch) and disambiguating (see
    # Disambiguator). Ranked like City since that is what it usually is.
    Unknown = "Unknown", 3

    def __gt__(self, other):
        if not isinstance(other, GeographicalEntityType):
            return NotImplemented
        return self.hierarchy_level > other.hierarchy_level
    
    def __ge__(self, other):
        if not isinstance(other, GeographicalEntityType):
            return NotImplemented
        return self.hierarchy_level >= other.hierarchy_level

    def __lt__(self, other):
        if not isinstance(other, GeographicalEntityType):
            return NotImplemented
        return self.hierarchy_level < other.hierarchy_level
    
    def __le__(self, other):
        if not isinstance(other, GeographicalEntityType):
            return NotImplemented
        return self.hierarchy_level <= other.hierarchy_level

class GeographicalEntityProvider(str, Enum):
    #TODO check codes against actual URIs
    GEONAMES = "https://sws.geonames.org/"
    WIKIDATA = "http://www.wikidata.org/wiki/" 
    GND = "https://d-nb.info/gnd/"

    @classmethod
    def from_iri(cls, iri: str) -> Optional["GeographicalEntityProvider"]:
        if iri is None:
            return None
        for provider in cls:
            if iri.startswith(provider.value):
                return provider
        parsed_iri = urlparse(iri)
        for provider in cls:
            parsed_provider = urlparse(provider.value)
            if parsed_iri.netloc == parsed_provider.netloc:
                return provider
        raise ValueError(f"Unknown provider for IRI: {iri}")
    
    @classmethod
    def from_namespace(cls, namespace: str) -> Optional["GeographicalEntityProvider"]:
        if namespace is None:
            return None
        try:
            return cls[namespace]
        except KeyError:
            pass
        parsed_namespace = urlparse(namespace)
        for provider in cls:
            parsed_provider = urlparse(provider.value)
            if parsed_namespace.netloc == parsed_provider.netloc:
                return provider
        raise ValueError(f"Unknown provider for namespace: {namespace}")
    
    @classmethod
    def __dict_decode__(cls, data, targs, default_decoder):
        if not isinstance(data, str):
            raise TypeError(f"Expected string serialization for GeographicalEntityProvider, got {type(data)}")
        try:
            return cls[data]
        except KeyError:
            return cls.from_iri(data)

class GeonamesAdminCodes(NamedTuple):
    admin1_code : Optional[str]
    admin2_code : Optional[str]
    admin3_code : Optional[str]
    admin4_code : Optional[str]
    admin5_code : Optional[str]

class GeographicalBranch(NamedTuple):
    """
    A path down the administrative hierarchy: a country followed by the
    leading non-null geonames admin codes (admin1, admin2, ...) shared by
    every entity within it.
    """
    country_iso_code : str
    admin_codes : tuple[str, ...]

class Coordinates(NamedTuple):
    latitude : float
    longitude : float

@dataclass(frozen=True, eq=False)
class RegionGeometry:
    """
    Estimated extent of a region with no usable boundary (e.g. "Bergstraße",
    or the Donau as named by "Neustadt an der Donau"): the locations of the
    places known to lie within it, as clusters of nearby points (see
    geo_db_search.build_regional_terms). Compared by identity, being large.
    """
    clusters : tuple[tuple[Coordinates, ...], ...]
    # iso codes of the countries the points lie in
    country_iso_codes : tuple[str, ...]

    @cached_property
    def points_radians(self) -> np.ndarray:
        """Every point, as an (n, 2) array of (latitude, longitude) in radians."""
        return np.radians([point for cluster in self.clusters for point in cluster])

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
    iso_code : str
    country_name : str
    iso_languages : tuple[str, ...]
    neighboring_countries_iso_codes : tuple[str, ...]
    continent : str
    geonames_id : int


@dataclass(frozen=True)
class GeographicalEntity:
    iri : str
    provider : GeographicalEntityProvider
    name : str
    asciiname : Optional[str]
    classification : Optional[str]
    possible_entity_types : tuple[GeographicalEntityType, ...]
    coordinates : Optional[Coordinates]
    population : Optional[int]
    geonames_id : Optional[int]
    closest_geonames_id : int
    country : Optional[CountryData]
    alternate_iso_country_codes : tuple[str, ...]
    admin_codes : GeonamesAdminCodes
    other_parent_iris : tuple[str, ...]
    # Estimated extent, for an entity whose admin codes do not describe it
    # (see geo_db_search.build_regional_terms)
    geometry : Optional[RegionGeometry] = field(default=None, compare=False, repr=False)

    @property
    def all_country_iso_codes(self) -> list[str]:
        if self.country is not None and self.country.iso_code is not None and self.country.iso_code != "":
            return [self.country.iso_code, *self.alternate_iso_country_codes]
        else:
            return list(self.alternate_iso_country_codes)

    @classmethod
    def from_dict(cls, data: dict) -> "GeographicalEntity":
        for k, v in data.items():
            if not isinstance(v, list) and pd.isna(v):
                data[k] = None
        country_data = data.pop("country", None)
        admin_codes_data = data.pop("admin_codes", None)
        coordinates = data.pop("coordinates", None)
        possible_entity_types = data.pop("possible_entity_types", [])
        possible_entity_types = [GeographicalEntityType[t] for t in possible_entity_types]
        provider = GeographicalEntityProvider.from_iri(data.get("iri"))
        return cls(
            **data,
            country=CountryData(**country_data) if country_data is not None else None,
            provider=provider,
            admin_codes=GeonamesAdminCodes(**admin_codes_data) if admin_codes_data is not None else GeonamesAdminCodes(None, None, None, None, None),
            coordinates=Coordinates(**coordinates) if coordinates is not None else None,
            possible_entity_types=possible_entity_types
        )
    
    @classmethod
    def __dict_decode__(cls, data: dict, targs, default_decoder):
        if 'provider' not in data:
            data = data.copy()
            data['provider'] = GeographicalEntityProvider.from_iri(data.get('iri'))
        return default_decoder(data, cls)

@dataclass(frozen=True)
class GeographicalName:
    name_id : int
    name : str
    entity : GeographicalEntity
    is_preferred_name: Optional[bool]
    is_short_name: Optional[bool]
    is_colloquial : Optional[bool]
    name_provider : Optional[GeographicalEntityProvider]
    isolanguage : Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "GeographicalName":
        for k, v in data.items():
            if not isinstance(v, list) and pd.isna(v):
                data[k] = None
        name_provider = GeographicalEntityProvider.from_namespace(data.pop("name_provider"))
        entity_data = data.pop("entity")
        return cls(
            **data,
            entity=GeographicalEntity.from_dict(entity_data),
            name_provider=name_provider
        )