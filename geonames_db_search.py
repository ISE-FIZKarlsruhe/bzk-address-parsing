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
from collections import defaultdict

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
            feature_code IN ('ADM1', 'ADM1H', 'ADMDH', 'ADMD', 'ADM2', 'ADM2H', 'ADM3H', 'ADM3', 'ADM4', 'ADM4H', 'ADM5')
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
    (
        allNames.isolanguage IS NULL OR 
        split(allNames.isolanguage, '-')[1] IN ('en', 'de', 'abbr') OR
        allNames.isolanguage IN countryInfo.Languages
    )
    AND
    alternateName IS NOT NULL AND 
    alternateName != '' AND
    clean_alt_name != '';

CREATE TEMP TABLE reduced_candidate_names AS
SELECT candidate_names.*
FROM candidate_names JOIN countryInfo ON (country_code = ISO)
WHERE 
    country_code IN ('US', 'IL', 'DE') OR 'DE' IN neighbours;

CREATE TEMP MACRO filter_entity_type(tbl, entity_type) AS TABLE 
    SELECT * FROM query_table(tbl) WHERE entity_type_map[entity_type];
"""



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
    if not initials and '.' in part:
        initials = part.split(".")
        if len(initials[-1]) == 0:
            initials = initials[:-1]
    if initials:
        initials = [x.lower() for x in initials]
        return "% ".join(initials) + "%"
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
    
    if threshold > 0:
        match_filter = f"cleaned_distance <= {threshold} OR may_be_abbreviation"
    else:
        match_filter = "clean_alt_name = clean_query"

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
                clean_alt_name LIKE query_prep.abbreviation_pattern
            ELSE FALSE
        END AS may_be_abbreviation,
        ROW_NUMBER() OVER (
            PARTITION BY geonameId 
{textwrap.indent(ranking_order, ' ' * 4 * 3)}
            ) AS match_rank,
        candidates.* EXCLUDE (nfc_alt_name, clean_alt_name)
    FROM candidates, query_prep
    WHERE {match_filter}
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


def falling_query_list(
        connection : duckdb.DuckDBPyConnection, 
        queries : list[tuple[bool, str, list]]
    ) -> tuple[pd.DataFrame, int]:
    matches = None
    query_idx = -1
    for i, (use, query, params) in enumerate(queries):
        if not use:
            continue
        matches = connection.execute(query, params).fetch_df()
        query_idx = i
        if len(matches) > 0: break
    return matches, query_idx

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
        country_hints : Optional[list[list[str]]] = None,
        fall_to_all_entities : bool = False,
    ) -> list[pd.DataFrame]:
        # convert in case it's a series
        if not isinstance(parts, list):
            parts = list(parts)
        if country_hints is None:
            country_hints = itertools.repeat(None, len(parts))
        # No strip_accents in python. Additionally, using the exact same normalization function might avoid problems
        cleaned_strings = self.connection.execute(
            "SELECT [strip_accents(nfc_normalize(x)) FOR x IN $1] AS cleaned_parts",
            [[p or "" for p in parts]]
        ).fetchone()[0]
        query = build_closest_matches_query(entity_type, self.topk, self.threshold)
        reduced_query = build_closest_matches_query(entity_type, self.topk, self.threshold, table="reduced_candidate_names")
        exact_query = build_closest_matches_query(entity_type, self.topk, 0, table="candidate_names")
        exact_reduced_query = build_closest_matches_query(entity_type, self.topk, 0, table="reduced_candidate_names")
        all_types_query = build_closest_matches_query(None, self.topk, self.threshold)
        results = []
        query_hits = defaultdict(int)
        for part, cleaned, country_hint in zip(parts, cleaned_strings, country_hints):
            if pd.isna(part) or part.strip() == "":
                results.append(pd.DataFrame())
                continue
            abbreviation_regex = abbreviation_pattern_to_regex(cleaned)
            matches = []
            start = time.monotonic()
            matches, hit_query_idx = falling_query_list(
                self.connection,
                [
                    (country_hint is not None, exact_query, [part, country_hint, abbreviation_regex]),
                    (True, exact_query, [part, None, abbreviation_regex]),
                    (entity_type != EntityType.Country, exact_reduced_query, [part, None, abbreviation_regex]),
                    (country_hint is not None, query, [part, country_hint, abbreviation_regex]),
                    (entity_type != EntityType.Country, reduced_query, [part, None, abbreviation_regex]),
                    (True, query, [part, None, abbreviation_regex]),
                    (fall_to_all_entities, all_types_query, [part, None, abbreviation_regex])
                ]
            )
            end = time.monotonic()
            assert isinstance(matches, pd.DataFrame)
            query_hits[hit_query_idx] += 1
            matches.insert(len(matches.columns), "search_time", end - start)
            results.append(matches)
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
        result = pd.concat(matches).sort_index()
        return result

    def find_parents(self, addr_matches : pd.DataFrame, match : pd.Series) -> list[pd.DataFrame]:
        """
        Finds parent matches according to geographical hierarchy for the given match.
        """
        # TODO continue later
        raise NotImplementedError()
        parents = []
        addr_matches = addr_matches[addr_matches["entity_type"] != match["entity_type"]]
        admin_cols = ["country_code"] + [f"admin{n}_code" for n in range(1, 6)]
        #possible_admin_matches = addr_matches[admin ]
        # Admin hierarchy
        admin_cols = ["country_code"] + [f"admin{n}_code" for n in range(1, 6)]
        admin_masks = [addr_matches["entity_type"] == "Country"] + [
            (addr_matches["feature_code"].str.startswith(f"ADM{n}") & (addr_matches["entity_type"] != "Country")) for n in range(1, 6)
        ]
        for i, level_i_mask in enumerate(admin_masks):
            code_cols = admin_cols[:i+1]
            match_mask = (addr_matches["feature_class"] == "A") & level_i_mask & (addr_matches[code_cols] == match[code_cols]).all(axis=1)
            parents.append(addr_matches[match_mask])
        
        parent_city_ids = match["parentCityIds"]
        if pd.isna(parent_city_ids) is not True: # isna will output a bool array when given an array
            parents.append(addr_matches[addr_matches["geonameId"].astype(str).isin(parent_city_ids)])
        parent_region_ids = match["parentRegionIds"]
        if pd.isna(parent_region_ids) is not True:
            parents.append(addr_matches[addr_matches["geonameId"].astype(str).isin(parent_region_ids)])
        return parents        

    def group_hierarchical_matches(self, matches : pd.DataFrame) -> pd.DataFrame:
        return matches.groupby("input_row").apply(self.group_address_hierarchical_matches).reset_index(level=2, drop=True)

    def group_address_hierarchical_matches(self, addr_matches : pd.DataFrame) -> pd.DataFrame:
        """
        Groups different matches of the same address for different entity types that are hierarchically dependent.
        """
        orig_index_levels = addr_matches.index.names
        addr_matches = addr_matches.reset_index()
        ungrouped_entities = set(addr_matches["geonameId"])
        matched_entity_types = [EntityType[entity_type] for entity_type in addr_matches["entity_type"].unique()]
        matched_entity_types.sort(reverse=True)
        grouped_matches : list[tuple[tuple[int, float], pd.DataFrame]] = []
        for entity_type in matched_entity_types:
            for _, match in addr_matches[addr_matches["entity_type"] == entity_type.entity_type].iterrows():
                if match["geonameId"] not in ungrouped_entities:
                    continue
                ungrouped_entities.remove(match["geonameId"])
                parents = self.find_parents(addr_matches, match)
                for parent in parents:
                    ungrouped_entities.difference_update(parent["geonameId"])
                hierarchy_group = pd.concat([match.to_frame().T, *parents])
                mean_rank = hierarchy_group["entity_rank"].mean()
                grouped_matches.append(((-len(hierarchy_group), mean_rank), hierarchy_group))
            if not ungrouped_entities:
                break
        grouped_matches.sort(key=lambda x: x[0])
        groups = []
        group_rank = 0
        last_sort_key = None
        for i, (sort_key, group) in enumerate(grouped_matches):
            group.insert(0, "group_id", i)
            if sort_key != last_sort_key:
                group_rank = i + 1
                last_sort_key = sort_key
            group.insert(1, "group_rank", group_rank)
            groups.append(group)
        result = pd.concat(groups)
        result.set_index(["group_id"] + orig_index_levels, inplace=True)
        return result

    def disambiguate(self, matches : pd.DataFrame, difference_threshold : float = 0) -> pd.DataFrame:
        """
        Returns the best match for each input row unless there are ties.
        """
        # TODO continue later
        raise NotImplementedError()
        if "group_id" in matches.index.names:
            matches_per_input = matches.groupby("input_row")
            best_per_input = matches_per_input.first()
            unambiguous = matches_per_input.transform(
                lambda group: group[(group["group_rank"] == 1) & ((group["cleaned_distance"].mean()))]
            )
            

    def close(self):
        self.connection.close()

    def __enter__(self):
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return super().__exit__(exc_type, exc_value, traceback)
    
    def __del__(self):
        self.close()
