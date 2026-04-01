"""
Classes related to matching extracted place names to place names on the database.
"""

# TODO rewrite everything
from pathlib import Path
import pandas as pd
import utils
import duckdb
import build_geonames_db
from typing import Optional, Literal, LiteralString
import contextlib
import enum
import dataclasses
import textwrap
import warnings
import textwrap

@dataclasses.dataclass
class EntityTypeProperties:
    entity_type : LiteralString
    hierarchy_level : int # lower means higher in the hierarchy. Induces a partial order
    sql_filter : str

class EntityType(EntityTypeProperties, enum.Enum):
    Country = "Country", 0, "feature_code IN ('TERR', 'PCLI', 'PCL', 'PCLF', 'LTER', 'ZN', 'PCLD', 'PCLH', 'PCLS', 'PRSH', 'PCLIX')"
    State = "State", 1, "feature_code IN ('ADM1', 'ADM1H', 'ADMDH', 'ADMD')"
    Region = "Region", 1, ("feature_code IN ('RGN', 'RGNH', 'ADM1', 'ADM1H', 'ADMDH', 'ADMD', 'ADM2', 'ADM2H', 'ADM3H', "
        "'ADM3', 'ADM4', 'ADM4H', 'ADM5')")
    District = "District", 2, "feature_code IN ('ADM1', 'ADM1H', 'ADMDH', 'ADMD', 'ADM2', 'ADM2H', 'ADM3H', 'ADM3')"
    City = "City", 3, "feature_class == 'P'"
    Neighborhood = "Neighborhood", 4, "feature_class == 'P'"

    def __gt__(self, other):
        if not isinstance(other, EntityType):
            return NotImplemented
        return self.hierarchy_level > other.hierarchy_level
    
    def __lt__(self, other):
        if not isinstance(other, EntityType):
            return NotImplemented
        return self.hierarchy_level < other.hierarchy_level
    
class GeographicNameProvider(enum.Enum):
    GEONAMES = "geonames"
    GND = "gnd"
    WIKIDATA = "wikidata" # Not used currently

@dataclasses.dataclass
class LinkedAddressPart:
    part : str
    matched_name : str
    entity_type : EntityType
    feature_class : str
    feature_code : str
    geonames_id : int
    distance : int
    match_method : Literal["levenshtein", "abbreviation_expansion"]
    name_provenance : GeographicNameProvider
    is_preferred : bool


@dataclasses.dataclass
class LinkedAddress:
    address : str
    parts : list[LinkedAddressPart]
    levenshtein_distance : int
    # neighbor distance between the country that matched the address and the country that contains the corresponding place.
    # Usually 0, however some addresses might contain cities that changed countries.
    country_distance : int = 0

def expand_abbreviation_to_pattern(part : str) -> str:
    """
    Expands an abbreviation, if it is one, to a pattern that would match the corresponding full name.
    """
    # TODO abbreviation size limit is arbitrary
    if len(part) < 4 and part.isupper():
        return "".join(f"{char}%" for char in part if char.isalpha())
    elif "." in part:
        pattern = part.replace(".", "%")
        return pattern
    return part

def build_closest_matches_query(
        entity_type : Optional[EntityType] = None,
        topk : int = 5,
        threshold : int | float = 3,                       
    ):
    if isinstance(entity_type, str):
        entity_type = EntityType[entity_type]
    elif not isinstance(entity_type, EntityType) and entity_type is not None:
        raise ValueError(f"Invalid entity type: {entity_type}")
    
    if entity_type is not None:
        entity_type_filter = entity_type.sql_filter
        if entity_type_filter is None:
            warnings.warn(f"Unknown entity type: {entity_type}. No filter will be applied.")
    else:
        entity_type_filter = None
    if entity_type_filter is not None:
        entity_type_filter = entity_type_filter + " AND "
    else:
        entity_type_filter = "-- No entity type filter"
    
    ranking_order = """
ORDER BY 
    CASE 
        WHEN raw_distance = 0 THEN 0 
        WHEN cleaned_distance = 0 THEN 1
        WHEN may_be_abbreviation THEN 2 
        ELSE 3 
    END,
    raw_distance, 
    cleaned_distance,
    CASE WHEN isPreferredName IS TRUE THEN 0 ELSE 1 END
ASC
"""

    query = f"""
WITH 
ranked_matches AS(
    SELECT 
        nfc_normalize(alternateName) AS match_name,
        nfc_normalize(?) AS query_name,
        regexp_replace(lower(strip_accents(match_name)), '[^\\w\\s]', '', 'g') AS clean_match_name,
        regexp_replace(lower(strip_accents(query_name)), '[^\\w\\s]', '', 'g') AS clean_query_name,
        levenshtein(match_name, query_name) AS raw_distance,
        levenshtein(clean_match_name, clean_query_name) AS cleaned_distance,
        CASE 
            WHEN '.' IN query_name 
            THEN clean_match_name SIMILAR TO replace(lower(strip_accents(query_name)), '.', '\\w+\\s*')
            ELSE FALSE 
        END AS may_be_abbreviation,
        ROW_NUMBER() OVER (
            PARTITION BY geonameId 
{textwrap.indent(ranking_order, ' ' * 4 * 3)}
            ) AS match_rank,
        allNames.*, simplifiedGeonames.*, countryInfo.Country
    FROM allNames NATURAL LEFT JOIN simplifiedGeonames LEFT JOIN countryInfo ON (country_code = ISO)
    WHERE 
        (least(raw_distance, cleaned_distance) <= {threshold} OR may_be_abbreviation) AND 
        {entity_type_filter}
        (
            isolanguage IS NULL OR 
            isolanguage IN countryInfo.Languages OR 
            split(isolanguage, '-')[1] IN ('en', 'de')
        )
),
ranked_entities AS (
    SELECT *,
    RANK() OVER (
{textwrap.indent(ranking_order, ' ' * 4)}
    ) AS entity_rank
    FROM ranked_matches
    WHERE match_rank = 1
)
SELECT * EXCLUDE (match_rank)
FROM ranked_entities
WHERE entity_rank <= {topk}
ORDER BY entity_rank
"""
    return query.strip()

    

class GeonamesSearch(contextlib.AbstractContextManager):
    def __init__(
            self, 
            conn : Optional[duckdb.DuckDBPyConnection] = None,
            topk : int = 5,
            threshold : int | float = 3
        ):
        self.connection = conn or build_geonames_db.open_or_init_duckdb()
        self.topk = topk
        self.threshold = threshold
    
    def find_closest_matches(
            self,
        parts : list[str],
        entity_type : Optional[EntityType] = None,
        
    ):
        query = build_closest_matches_query(entity_type, self.topk, self.threshold)
        #self.connection.execute("PREPARE closest_matches_query AS\n" + query)
        results = []
        for part in parts:
            matches = self.connection.execute(query, [part]).fetchdf()
            # query_name will be nfc normalized and may no longer match the original part
            # hence we add the original part as a separate column
            matches.insert(0, "query_part", part) 
            results.append(matches)
        #self.connection.execute("DEALLOCATE my_stmt")
        return pd.concat(results, ignore_index=True)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return super().__exit__(exc_type, exc_value, traceback)
    
    def __del__(self):
        self.close()
