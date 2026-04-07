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
import itertools
import time

@dataclasses.dataclass
class EntityTypeProperties:
    entity_type : LiteralString
    hierarchy_level : int # lower means higher in the hierarchy. Induces a partial order

class EntityType(EntityTypeProperties, enum.Enum):
    Country = "Country", 0
    State = "State", 1
    Region = "Region", 1
    District = "District", 2
    City = "City", 3
    Neighborhood = "Neighborhood", 4

    def __gt__(self, other):
        if not isinstance(other, EntityType):
            return NotImplemented
        return self.hierarchy_level > other.hierarchy_level
    
    def __lt__(self, other):
        if not isinstance(other, EntityType):
            return NotImplemented
        return self.hierarchy_level < other.hierarchy_level

# Used to materialize in memory a distilled version of the names table
# TODO filter neighborhoods?
CANDIDATE_NAMES_INIT_QUERY = """
CREATE TEMP TABLE candidate_names AS
SELECT 
    nfc_normalize(alternateName) AS nfc_alt_name,
    regexp_replace(lower(strip_accents(nfc_alt_name)), '[^\\w\\s]', '', 'g') AS clean_alt_name,
    allNames.*, 
    simplifiedGeonames.*, 
    countryInfo.Country,
    { 
        'Country' : (
            (
                feature_class = 'A' AND 
                feature_code IN ('TERR', 'PCLI', 'PCL', 'PCLF', 'LTER', 'ZN', 'PCLD', 'PCLH', 'PCLS', 'PRSH', 'PCLIX')
            ) OR (
                -- United Kingdom member states are often thought of as countries.
                feature_class = 'A' AND feature_code = 'ADM1' AND country_code = 'GB'
            )
        ),
        'State' : (
            feature_class = 'A' AND 
            feature_code IN ('ADM1', 'ADM1H', 'ADMDH', 'ADMD')
        ),
        'Region' : (
            (
                feature_class = 'A' AND 
                feature_code IN ('ADM1', 'ADM1H', 'ADMDH', 'ADMD', 'ADM2', 'ADM2H', 'ADM3H', 'ADM3', 'ADM4', 'ADM4H', 'ADM5')
            ) OR (
                feature_class = 'L' AND
                feature_code IN ('RGN', 'RGNH')
            )
        ),
        'District' : (
            feature_class = 'A' AND
            feature_code IN ('ADM1', 'ADM1H', 'ADMDH', 'ADMD', 'ADM2', 'ADM2H', 'ADM3H', 'ADM3')
        ),
        'City' : (
            feature_class = 'P' AND feature_code != 'PPLX'
        ),
        'Neighborhood' : (
            feature_class = 'P'
        )
    } AS entity_type_map
FROM allNames 
    NATURAL JOIN simplifiedGeonames 
    JOIN countryInfo ON (country_code = ISO)
WHERE
    allNames.isolanguage IS NULL OR 
    split(allNames.isolanguage, '-')[1] IN ('en', 'de', 'abbr') OR
    allNames.isolanguage IN countryInfo.Languages;

CREATE TEMP TABLE reduced_candidate_names AS
SELECT candidate_names.*
FROM candidate_names JOIN countryInfo ON (country_code = ISO)
WHERE 
    Continent = 'EU' OR country_code IN ('US', 'IL');

CREATE TEMP MACRO filter_entity_type(tbl, entity_type) AS TABLE 
    SELECT * FROM query_table(tbl) WHERE entity_type_map[entity_type];
"""

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

def abbreviation_pattern_to_regex(part : str) -> str:
    """
    Converts an abbreviation pattern to a regex pattern that can be used in SQL.
    """
    #TODO arbitrary abbreviation size limit
    initials = []
    for char in part:
        if char.isupper() and char.isalpha():
            initials.append(char.lower())
            if len(initials) > 4:
                initials = None
                break
        elif char == ".":
            continue
        else:
            initials = None
            break
    word_bound_regex = r"\w*\b\s*"
    if initials:
        return "".join(c + word_bound_regex for c in initials)
    elif "." in part:
        return part.lower().replace(".", word_bound_regex)
    else: 
        return None

def build_closest_matches_query(
        entity_type : Optional[EntityType] = None,
        topk : int = 5,
        threshold : int | float = 3,
        table : Literal["candidate_names", "reduced_candidate_names"] = "candidate_names"                  
    ):
    if isinstance(entity_type, str):
        entity_type = EntityType[entity_type]
    elif not isinstance(entity_type, EntityType) and entity_type is not None:
        raise ValueError(f"Invalid entity type: {entity_type}")
    if entity_type is not None:
        entity_type_filter = f"filter_entity_type('{table}', '{entity_type.entity_type}')"
    else:
        entity_type_filter = table
    
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

    # TODO materializing some sub queries and views might speed up the query significantly
    # TODO query size limit is arbitrary
    query = f"""
WITH
query_prep AS (
    SELECT 
        $1 AS query,
        nfc_normalize(query) AS nfc_query,
        regexp_replace(lower(strip_accents(nfc_query)), '[^\\w\\s]', '', 'g') AS clean_query,
        $2 AS country_restriction,
        $3 AS abbreviation_pattern
),
candidates AS(
    SELECT filtered.*
    FROM {entity_type_filter} AS filtered, query_prep
    WHERE 
        query_prep.country_restriction IS NULL 
        OR 
        country_code IN query_prep.country_restriction 
),
ranked_matches AS(
    SELECT 
        query_prep.*,
        nfc_alt_name,
        clean_alt_name,
        levenshtein(nfc_alt_name, nfc_query) AS raw_distance,
        levenshtein(clean_alt_name, clean_query) AS cleaned_distance,
        CASE 
            WHEN query_prep.abbreviation_pattern IS NOT NULL THEN 
                clean_alt_name SIMILAR TO query_prep.abbreviation_pattern
            ELSE FALSE
        END AS may_be_abbreviation,
        ROW_NUMBER() OVER (
            PARTITION BY geonameId 
{textwrap.indent(ranking_order, ' ' * 4 * 3)}
            ) AS match_rank,
        candidates.* EXCLUDE (nfc_alt_name, clean_alt_name)
    FROM candidates, query_prep
    WHERE 
        least(raw_distance, cleaned_distance) <= {threshold} 
        OR 
        may_be_abbreviation
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
        self.connection.execute(CANDIDATE_NAMES_INIT_QUERY)
        self.topk = topk
        self.threshold = threshold
    
    def link_entities(
            self,
        parts : list[str],
        entity_type : Optional[EntityType] = None,
        country_hints : Optional[list[list[str]]] = None
    ) -> list[pd.DataFrame]:
        # convert in case it's a series
        if not isinstance(parts, list):
            parts = list(parts)
        if country_hints is None:
            country_hints = itertools.repeat(None, len(parts))
        # No strip_accents in python. Additionally, using the exact same normalization function might avoid problems
        cleaned_strings = self.connection.execute(
            "SELECT [strip_accents(nfc_normalize(x)) FOR x IN $1] AS cleaned_parts",
            [parts]
        ).fetchone()[0]
        query = build_closest_matches_query(entity_type, self.topk, self.threshold)
        reduced_query = build_closest_matches_query(entity_type, self.topk, self.threshold, table="reduced_candidate_names")
        results = []
        query_hits = {"1st":0, "2nd":0, "3rd":0}
        for part, cleaned, country_hint in zip(parts, cleaned_strings, country_hints):
            abbreviation_regex = abbreviation_pattern_to_regex(cleaned)
            matches = []
            start = time.monotonic()
            if entity_type is None or entity_type == EntityType.Country:
                query_hits["3rd"] += 1
                matches = self.connection.execute(query, [part, None, abbreviation_regex]).fetchdf()
            else:
                # Cascade through different subsets of all names to speed up the search.
                matches = []
                if country_hint is not None:
                    matches = self.connection.execute(query, [part, country_hint, abbreviation_regex]).fetchdf()
                    if len(matches) > 0: query_hits["1st"] += 1
                if len(matches) == 0:
                    matches = self.connection.execute(reduced_query, [part, None, abbreviation_regex]).fetchdf()
                    if len(matches) > 0: query_hits["2nd"] += 1
                if len(matches) == 0:
                    matches = self.connection.execute(query, [part, None, abbreviation_regex]).fetchdf()
                    if len(matches) > 0: query_hits["3rd"] += 1
                assert isinstance(matches, pd.DataFrame)
            end = time.monotonic()
            matches["search_time"] = end - start
            results.append(matches)
        print(f"Query hits: {query_hits}")
        return results

    def link_parsed_addresses(self, addresses : pd.DataFrame | list[dict]) -> pd.DataFrame:
        if not isinstance(addresses, pd.DataFrame):
            addresses = pd.DataFrame(addresses)
        addresses = addresses[[c for c in addresses.columns if c in EntityType.__members__]]
        matches = []
        country_hints = {}
        for entity_type in EntityType:
            if entity_type.name in addresses.columns:
                target_cols = [entity_type.name, "Country"] if entity_type != EntityType.Country else ["Country"]
                targets = addresses[target_cols].reset_index(names="input_row").dropna(subset=[entity_type.name])
                nodupes = targets.drop_duplicates(subset=target_cols)
                nodupes['country_hints'] = pd.Series(country_hints.get(country) for country in nodupes["Country"])
                print(f"Country hints set for {len(nodupes['country_hints'].dropna())} / {len(nodupes)} addresses for entity type {entity_type.name}")
                nodupes = nodupes.fillna({"country_hints": None}).reset_index(drop=True)
                print(f"Starting search for entity type {entity_type.name}")
                start = time.monotonic()
                entity_matches = self.link_entities(nodupes[entity_type.name], country_hints=nodupes["country_hints"], entity_type=entity_type)
                end = time.monotonic()
                print(f"Search for entity type {entity_type.name} took {utils.format_time(end - start)} and returned {sum(len(df) for df in entity_matches)} matches")
                for idx, row in targets.iterrows():
                    match_idx = nodupes[(nodupes[target_cols].fillna("") == row[target_cols].fillna("")).all(axis=1)].index
                    if len(match_idx) != 1:
                        warnings.warn(f"Expected exactly one match for {entity_type.name}='{row[entity_type.name]}' and Country='{row['Country']}', but got {len(match_idx)}. This should not happen.")
                    else:
                        entity_matches[match_idx[0]]["input_row"] = idx
                assert all("input_row" in df.columns for df in entity_matches)
                entity_matches = pd.concat(entity_matches)
                entity_matches = entity_matches.drop(columns=["country_restriction"])
                entity_matches["entity_type"] = entity_type.name
                entity_matches.set_index(["input_row", "entity_type", "entity_rank", "geonameId"], inplace=True)
                matches.append(entity_matches)
                if entity_type == EntityType.Country:
                    for idx, match in entity_matches.iterrows():
                        if not pd.isna(match["country_code"]):
                            country = addresses.loc[idx[0], "Country"]
                            hints = country_hints.setdefault(country, [])
                            hints.append(match["country_code"])
                    print(f"Country hints set for {len(country_hints)} countries")
        return pd.concat(matches).sort_index()


    def close(self):
        self.connection.close()

    def __enter__(self):
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return super().__exit__(exc_type, exc_value, traceback)
    
    def __del__(self):
        self.close()
