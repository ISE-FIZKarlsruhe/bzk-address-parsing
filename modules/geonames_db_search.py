"""
Classes related to matching extracted place names to place names on the database.
"""

# TODO rewrite everything
from pathlib import Path
import pandas as pd
import modules.utils as utils
import duckdb
import modules.build_geonames_db as build_geonames_db
from typing import NamedTuple, Optional, Literal
import contextlib
import enum
import dataclasses
import textwrap
import warnings
import textwrap
import itertools
import time
from collections import defaultdict
from modules.pipeline.geographical_entity import GeographicalEntityType, GeonamesAdminCodes
import tantivy
from tqdm.auto import tqdm
import unicodedata
import re
import os

# Used to materialize in memory a distilled version of the names table
# TODO filter neighborhoods?
CANDIDATE_NAMES_INIT_QUERY = """
CREATE TABLE IF NOT EXISTS candidate_names AS
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
FROM geonames.allNames 
    NATURAL JOIN geonames.simplifiedGeonames 
    JOIN geonames.countryInfo ON (country_code = ISO)
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

CREATE TABLE IF NOT EXISTS reduced_candidate_names AS
SELECT candidate_names.*
FROM candidate_names JOIN geonames.countryInfo ON (country_code = ISO)
WHERE 
    country_code IN ('US', 'IL', 'DE') OR 'DE' IN neighbours;

CREATE TEMP MACRO filter_entity_type(tbl, entity_type) AS TABLE 
    SELECT * FROM query_table(tbl) WHERE entity_type_map[entity_type];
"""


GERMAN_ASCII_DUCKDB_MACRO = """
-- Converts to ascii but
-- rather than converting ä to a, apply german rules and convert ä to ae, etc.
-- Useful for matching against
CREATE OR REPLACE TEMP MACRO german_ascii(nfc_name) AS 
strip_accents(
    replace(
        replace(
            replace(
                replace(
                    lower(nfc_name), 
                'ä', 'ae'),
            'ö', 'oe'),
        'ü', 'ue'),
    'ß', 'ss')
);
"""

LANGUAGE_FILTERED_NAMES_PARTIAL_SQL = """
FROM geo_db.geographical_names 
    JOIN geo_db.geographical_entities USING (iri)
    JOIN geo_db.country_data ON (geo_db.geographical_entities.iso_country_code = geo_db.country_data.iso_code)
WHERE
    geo_db.geographical_names.is_preferred_name IS TRUE OR
    geo_db.geographical_names.isolanguage == '' OR
    geo_db.geographical_names.isolanguage IN ('en', 'de', 'abbr') OR
    list_bool_or([(geo_db.geographical_names.isolanguage IN lang) FOR lang IN geo_db.country_data.iso_languages]);
"""

LANGUAGE_FILTERED_NAMES_SELECT = """
SELECT
    nfc_normalize(geo_db.geographical_names.name) AS nfc_name,
    german_ascii(geo_db.geographical_names.name) AS german_ascii_name,
    regexp_replace(strip_accents(lower(nfc_name)), '[^\\w\\s]', '', 'g') AS ascii_name,
    geo_db.geographical_names.name AS name, 
    geo_db.geographical_names.iri AS iri, 
    geo_db.geographical_entities.possible_entity_types AS possible_entity_types,
    geo_db.geographical_entities.iso_country_code AS country_code,
    geo_db.geographical_entities.admin_codes AS admin_codes,
    geo_db.geographical_names.is_preferred_name AS is_preferred_name,
    geo_db.geographical_names.isolanguage AS isolanguage
""" + LANGUAGE_FILTERED_NAMES_PARTIAL_SQL

LANGUAGE_FILTERED_NAMES_COUNT = "SELECT count(*)\n" + LANGUAGE_FILTERED_NAMES_PARTIAL_SQL


def abbreviation_pattern_to_sql_regex(part : str) -> str:
    """
    Converts an abbreviation pattern to a regex pattern that can be used in SQL.
    """
    #TODO delete?
    initials = []
    for char in part:
        if char.isupper() and char.isalpha():
            initials.append(char.lower())
            #TODO arbitrary abbreviation size limit
            # matches CSR and USSR, are there other important abbreviations that would be missed?
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

def abbreviation_pattern_to_regex(part : str) -> str:
    """
    Converts an abbreviation pattern to a regex pattern.
    """
    initials = []
    for char in part:
        if char.isupper() and char.isalpha():
            initials.append(char.lower())
            #TODO arbitrary abbreviation size limit
            # matches CSR and USSR, are there other important abbreviations that would be missed?
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
        return "[a-z]* ".join(initials) + "[a-z]*"
    else:
        return None

_STRIP_PUNCTUATION_REGEX = re.compile(r'[^\w\s]')

def ascii_normalize(nfc_query : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_query.lower()
    # Strip accents replacements may differ on implementation
    # Call duckdb function to ensure the same behavior as in the database
    result = duckdb.execute("SELECT strip_accents($1)", [result]).fetchone()[0]
    result = _STRIP_PUNCTUATION_REGEX.sub("", result)
    return result

def german_normalize(nfc_query : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    Accent stripping takes into account german rules for conversion of umlauts and ß.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_query.lower()
    result = result.replace("ä", "ae")
    result = result.replace("ö", "oe")
    result = result.replace("ü", "ue")
    result = result.replace("ß", "ss")
    return ascii_normalize(result)

class TantivySearchMatch(NamedTuple):
    score : float
    german_ascii_name: str
    ascii_name: str
    iri: str


class TantivySearchResult(NamedTuple):
    nfc_query : str
    german_ascii_query: Optional[str]
    ascii_query: str
    abbreviation_pattern : Optional[str]
    matches : list[TantivySearchMatch]


class TantivySearchIndex:
    def __init__(self, index_path : str | Path, n_threads : int | Literal['auto'] = 'auto'):
        if n_threads == 'auto':
            n_threads = max(1, getattr(os, "process_cpu_count", lambda : None)() or os.cpu_count() or 8)
        self.schema = self.create_schema()
        self.index_path = Path(index_path)
        self.already_exists = self.index_path.exists()
        if not self.already_exists:
            self.index_path.mkdir(parents=True)
        self.n_threads = n_threads
        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        self.index.config_reader(num_warmers=self.n_threads)

    def populate_index(self, geo_db_connection : duckdb.DuckDBPyConnection, skip_if_exists=True):
        if skip_if_exists and self.already_exists:
            self.index.reload()
            return
        total_rows = geo_db_connection.execute(LANGUAGE_FILTERED_NAMES_COUNT).fetchone()[0]
        geo_db_connection.execute(GERMAN_ASCII_DUCKDB_MACRO)
        with self.index.writer(num_threads=self.n_threads) as writer:
            for row in tqdm(geo_db_connection.execute(LANGUAGE_FILTERED_NAMES_SELECT).fetchall(), total=total_rows, desc="Populating Tantivy index"):
                doc = tantivy.Document()
                doc.add_text("german_ascii_name", row[1])
                doc.add_text("ascii_name", row[2])
                doc.add_text("iri", row[4])

                # TODO not implemented, is it useful at all?
                # extra fields for search restriction
                for entity_type in GeographicalEntityType:
                    doc.add_boolean(entity_type.entity_type, entity_type.entity_type in row[5])
                doc.add_text("country_code", row[6] or "")
                admin_codes = row[7] or {}
                for admin_level in range(1, 6):
                    admin_code = admin_codes.get(f"admin{admin_level}_code")
                    doc.add_text(f"admin{admin_level}_code", admin_code or "")
                writer.add_document(doc)
        self.index.reload()

    def create_schema(self):
        schema_builder = tantivy.SchemaBuilder()
        text_field_options = dict( 
            # Use raw tokenizer; tantivy tokenizer is a full text search feature, 
            # we only need single term matching
            tokenizer_name = 'raw',
            index_option = 'basic'
        )
        # search keys
        schema_builder.add_text_field("german_ascii_name", stored=True, **text_field_options)
        schema_builder.add_text_field("ascii_name", stored=True, **text_field_options)
        # iri for db retrieval later
        schema_builder.add_text_field("iri", stored=True, **text_field_options)

        # extra fields for search restriction
        for entity_type in GeographicalEntityType:
            schema_builder.add_boolean_field(entity_type.entity_type, indexed=True)
        schema_builder.add_text_field("country_code", **text_field_options)
        for code in ["admin1_code", "admin2_code", "admin3_code", "admin4_code", "admin5_code"]:
            schema_builder.add_text_field(code, **text_field_options)
        return schema_builder.build()

    def _build_entity_type_restriction(
            self, entity_types : list[GeographicalEntityType]) -> tantivy.Query:
        if len(entity_types) == 1:
            return tantivy.Query.term_query(
                self.schema, entity_types[0].entity_type, True, index_option='basic')
        else:
            disjuction = []
            for entity_type in entity_types:
                disjuction.append(tantivy.Query.term_query(self.schema, entity_type.entity_type, True))
            return tantivy.Query.disjunction_max_query(disjuction)

    def _build_country_code_restriction(self, country_codes : list[str]) -> tantivy.Query:
        if len(country_codes) == 1:
            return tantivy.Query.term_query(self.schema, "country_code", country_codes[0], index_option='basic')
        else:
            disjuction = []
            for country_code in country_codes:
                disjuction.append(tantivy.Query.term_query(self.schema, "country_code", country_code))
            return tantivy.Query.disjunction_max_query(disjuction)
    
    def _build_admin_code_restriction(self, admin_codes : list[GeonamesAdminCodes]) -> tantivy.Query:
        def _admin_code_to_query(admin_code : GeonamesAdminCodes) -> tantivy.Query:
            admin_code_queries = []
            for k, v in admin_code._asdict().items():
                if v is not None and v != '':
                    admin_code_queries.append(
                        tantivy.Occur.Must, tantivy.Query.term_query(self.schema, k, v, index_option='basic'))
            if len(admin_code_queries) == 0:
                return None
            if len(admin_code_queries) == 1:
                return admin_code_queries[0][1]
            else:
                return tantivy.Query.boolean_query(admin_code_queries)
        if len(admin_codes) == 1:
            return _admin_code_to_query(admin_codes[0])
        else:
            disjuction = []
            for admin_code in admin_codes:
                admin_code_query = _admin_code_to_query(admin_code)
                if admin_code_query is not None:
                    disjuction.append(admin_code_query)
            return tantivy.Query.disjunction_max_query(disjuction)


    def _search_inner(
            self, 
            ascii_query : str, 
            german_query : Optional[str], 
            abbrev_pattern : Optional[str], 
            restrictions : list[tuple[tantivy.Occur, tantivy.Query]], 
            threshold : int, 
            limit : int
        ) -> list[TantivySearchMatch]:
        query_strings = [("ascii_name", ascii_query)]
        if german_query is not None:
            query_strings.append(("german_ascii_name", german_query))
        if threshold == 0:
            name_queries = [
                tantivy.Query.term_query(self.schema, field, query, index_option='basic')
                for field, query in query_strings
            ]
        else:
            name_queries = [
                tantivy.Query.boost_query(tantivy.Query.fuzzy_term_query(self.schema,
                    field, query, distance=threshold), 1.0)
                for field, query in query_strings
            ]
        if abbrev_pattern:
            name_queries.append(tantivy.Query.regex_query(self.schema, "ascii_name", abbrev_pattern))
            name_queries.append(tantivy.Query.regex_query(self.schema, "german_ascii_name", abbrev_pattern))
        final_name_query = tantivy.Query.disjunction_max_query(name_queries)
        if len(restrictions) > 0:
            final_query = tantivy.Query.boolean_query([(tantivy.Occur.Must, final_name_query)] + restrictions)
        else:
            final_query = final_name_query
        searcher = self.index.searcher()
        search_results = searcher.search(final_query, limit=limit)
        matches = []
        for score, doc_address in search_results.hits:
            doc = searcher.doc(doc_address)
            matches.append(TantivySearchMatch(
                score=score,
                german_ascii_name=doc.get_first("german_ascii_name"),
                ascii_name=doc.get_first("ascii_name"),
                iri=doc.get_first("iri")
            ))
        return matches

    def search(
            self, 
            query_string, 
            threshold : int, 
            limit : int = 10,
            match_abbreviations : bool = True,
            entity_types : Optional[list[GeographicalEntityType]] = None,
            country_codes : Optional[list[str]] = None,
            admin_codes : Optional[list[GeonamesAdminCodes]] = None
        ):
        nfc_query = unicodedata.normalize("NFC", query_string)
        german_query = german_normalize(nfc_query)
        ascii_query = ascii_normalize(nfc_query)
        if ascii_query == german_query:
            german_query = None
        if match_abbreviations:
            abbrev_pattern = abbreviation_pattern_to_regex(query_string)
        else: abbrev_pattern = None
        
        restrictions = []
        if entity_types is not None: 
            restrictions.append((tantivy.Occur.Must, self._build_entity_type_restriction(entity_types)))
        if country_codes is not None:
            restrictions.append((tantivy.Occur.Must, self._build_country_code_restriction(country_codes)))
        if admin_codes is not None:
            admin_code_restriction = self._build_admin_code_restriction(admin_codes)
            if admin_code_restriction is not None:
                restrictions.append((tantivy.Occur.Must, admin_code_restriction))
        for i in range(0, threshold + 1):
            matches = self._search_inner(
                ascii_query=ascii_query,
                german_query=german_query,
                abbrev_pattern=abbrev_pattern,
                restrictions=restrictions,
                threshold=i,
                limit=limit
            )
            if len(matches) > 0:
                break
        return TantivySearchResult(
            nfc_query=nfc_query,
            german_ascii_query=german_query,
            ascii_query=ascii_query,
            abbreviation_pattern=abbrev_pattern,
            matches=matches
        )

class SymSpellSearchIndex:
    def __init__(self):
        raise NotImplementedError("This class is being refactored and should not be used in its current state.")

    

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
            topk : int = 5,
            threshold : int | float = 3,
            search_cache_db : str | Literal[':memory:'] = "search_cache.duckdb"
        ):
        raise NotImplementedError("This class is being refactored and should not be used in its current state.")
        self.connection = duckdb.connect(search_cache_db)
        build_geonames_db.attach_or_init_duckdb(self.connection)
        self.connection.execute(CANDIDATE_NAMES_INIT_QUERY)
        self.topk = topk
        self.threshold = threshold
    
    def search_entities(
            self,
        parts : list[str],
        entity_type : Optional[GeographicalEntityType] = None,
        country_hints : Optional[list[list[str]]] = None,
        fall_to_all_entities : bool = False,
    ) -> list[pd.DataFrame]:
        """
        Match extracted address parts of a given type to place entities on external knowledge graphs.
        """
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
            abbreviation_regex = abbreviation_pattern_to_sql_regex(cleaned)
            matches = []
            start = time.monotonic()
            matches, hit_query_idx = falling_query_list(
                self.connection,
                [
                    (country_hint is not None, exact_query, [part, country_hint, abbreviation_regex]),
                    (True, exact_query, [part, None, abbreviation_regex]),
                    (entity_type != GeographicalEntityType.Country, exact_reduced_query, [part, None, abbreviation_regex]),
                    (country_hint is not None, query, [part, country_hint, abbreviation_regex]),
                    (entity_type != GeographicalEntityType.Country, reduced_query, [part, None, abbreviation_regex]),
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

    def search_parsed_addresses(self, addresses : pd.DataFrame | list[dict]) -> pd.DataFrame:
        """
        Match parsed addresses to place entities on external knowledge graphs.
        """
        if not isinstance(addresses, pd.DataFrame):
            addresses = pd.DataFrame(addresses)
        else:
            addresses = addresses.reset_index(drop=True)
        addresses = addresses[[c for c in addresses.columns if c in GeographicalEntityType.__members__]]
        matches = []
        country_hints = {}
        for entity_type in GeographicalEntityType:
            if entity_type.name not in addresses.columns:
                continue
            target_cols = [entity_type.name, "Country"] if entity_type != GeographicalEntityType.Country else ["Country"]
            targets = addresses[target_cols].reset_index(names="input_row").dropna(subset=[entity_type.name])
            if len(targets) == 0:
                continue
            nodupes = targets.drop_duplicates(subset=target_cols)
            nodupes['country_hints'] = pd.Series(country_hints.get(country) for country in nodupes["Country"])
            print(f"Country hints set for {len(nodupes['country_hints'].dropna())} / {len(nodupes)} addresses for entity type {entity_type.name}")
            nodupes = nodupes.fillna({"country_hints": None}).reset_index(drop=True)
            print(f"Starting search for entity type {entity_type.name}")
            start = time.monotonic()
            entity_matches = self.search_entities(nodupes[entity_type.name], country_hints=nodupes["country_hints"], entity_type=entity_type)
            end = time.monotonic()
            print(f"Search for entity type {entity_type.name} took {utils.format_time(end - start)} and returned {sum(len(df) for df in entity_matches)} matches")
            if entity_type == GeographicalEntityType.Country:
                for idx, match in enumerate(entity_matches):
                    if len(match) > 0 and not match["country_code"].isna().all():
                        country = addresses.loc[nodupes.iloc[idx]["input_row"], "Country"]
                        hints = country_hints.setdefault(country, [])
                        hints.extend(match["country_code"].dropna().unique())
                print(f"Country hints set for {len(country_hints)} countries")
            reduped_entity_matches = []
            for _, row in targets.iterrows():
                deduped_idx = nodupes[(nodupes[target_cols].fillna("") == row[target_cols].fillna("")).all(axis=1)].index
                if len(deduped_idx) != 1:
                    warnings.warn(f"Expected exactly one deduplicated index for {entity_type.name}='{row[entity_type.name]}' and Country='{row['Country']}', but got {len(deduped_idx)}. This should not happen.")
                row_matches = entity_matches[deduped_idx[0]].copy()
                row_matches["input_row"] = row["input_row"]
                reduped_entity_matches.append(row_matches)
            assert all("input_row" in df.columns for df in reduped_entity_matches)
            reduped_entity_matches = pd.concat(reduped_entity_matches)
            reduped_entity_matches = reduped_entity_matches.drop(columns=["country_restriction"])
            reduped_entity_matches["entity_type"] = entity_type.name
            reduped_entity_matches.set_index(["input_row", "entity_type", "entity_rank", "geonameId"], inplace=True)
            matches.append(reduped_entity_matches)
        result = pd.concat(matches).sort_index()
        return result

    def find_parents(self, addr_matches : pd.DataFrame, match : pd.Series) -> list[pd.DataFrame]:
        """
        Finds parent matches according to geographical hierarchy for the given match.
        """
        # Method heavily refactored with GPT-5 mini
        parents: list[pd.DataFrame] = []
        if addr_matches is None or len(addr_matches) == 0:
            return parents

        # Exclude matches with the same entity type and the match itself
        candidates = addr_matches[addr_matches["entity_type"] != match["entity_type"]].copy()
        try:
            candidates = candidates[candidates["geonameId"] != match["geonameId"]]
        except Exception:
            pass

        # Admin code columns from country + admin1..admin5
        admin_cols = ["country_code"] + [f"admin{n}_code" for n in range(1, 6)]

        # Build masks for country and admin levels 1..5
        admin_masks: list[pd.Series] = []
        admin_masks.append(candidates["entity_type"] == "Country")
        for n in range(1, 6):
            mask = (
                (candidates["feature_class"] == "A")
                & candidates["feature_code"].str.startswith(f"ADM{n}", na=False)
                & (candidates["entity_type"] != "Country")
            )
            admin_masks.append(mask)

        # For each level, pick candidates whose admin codes match the match's codes up to that level
        for i, level_mask in enumerate(admin_masks):
            code_cols = admin_cols[: i + 1]
            if len(code_cols) == 0:
                continue
            comp_mask = pd.Series(True, index=candidates.index)
            for col in code_cols:
                match_val = match.get(col, None)
                match_str = "" if pd.isna(match_val) else str(match_val)
                comp_mask &= candidates[col].fillna("").astype(str) == match_str
            match_mask = level_mask & comp_mask
            parents.append(candidates[match_mask])

        # parentCityIds and parentRegionIds are expected to be native Python
        # iterables (lists/tuples/sets) or absent. If present, match against them.
        parent_city_ids = match.get("parentCityIds", None)
        if parent_city_ids is not None and not (isinstance(parent_city_ids, float) and pd.isna(parent_city_ids)):
            try:
                ids = {str(x) for x in parent_city_ids}
            except TypeError:
                ids = {str(parent_city_ids)}
            parents.append(candidates[candidates["geonameId"].astype(str).isin(ids)])

        parent_region_ids = match.get("parentRegionIds", None)
        if parent_region_ids is not None and not (isinstance(parent_region_ids, float) and pd.isna(parent_region_ids)):
            try:
                ids = {str(x) for x in parent_region_ids}
            except TypeError:
                ids = {str(parent_region_ids)}
            parents.append(candidates[candidates["geonameId"].astype(str).isin(ids)])

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
        matched_entity_types = [GeographicalEntityType[entity_type] for entity_type in addr_matches["entity_type"].unique()]
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
