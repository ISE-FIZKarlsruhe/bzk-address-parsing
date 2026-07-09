"""
Classes related to matching extracted place names to place names on the database.
"""

# TODO rewrite everything
from pathlib import Path
import pandas as pd
import modules.utils as utils
import duckdb
import modules.build_geonames_db as build_geonames_db
from typing import Collection, Iterable, NamedTuple, Optional, Literal, Callable
import contextlib
import enum
import dataclasses
import textwrap
import warnings
import textwrap
import itertools
import time
from collections import defaultdict
from modules.pipeline.geographical_entity import CountryData, GeographicalEntity, GeographicalEntityType, GeographicalName, GeonamesAdminCodes
from modules.pipeline.linked_data import MatchedEntity, MatchedName
from modules.pipeline.linking_steps import LinkingStep
import tantivy
from tqdm.auto import tqdm
import unicodedata
import re
import os
import editdistpy
import sys
import dataclasses
import json
import pyarrow
import unidecode
import math


def levenshtein(a : str, b : str, max_distance : int) -> int:
    """
    Computes the Levenshtein distance between two strings.
    If max_distance is provided, the computation will stop if the distance exceeds max_distance.
    """
    dist = editdistpy.levenshtein.distance(a, b, max_distance=max_distance)
    if dist < 0:
        return sys.maxsize
    return dist

def similarity_and_distance(a : str, b : str, max_distance : int, distance_function : Callable[[str, str, int], int] = levenshtein):
    edit_distance = distance_function(a, b, max_distance)
    similarity = 1 - (edit_distance / max(len(a), len(b)))
    return edit_distance, similarity

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
CREATE OR REPLACE MACRO german_ascii(nfc_name) AS 
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


LANGUAGE_FILTERED_NAMES_SELECT = """
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE
    is_preferred_name IS TRUE OR
    isolanguage == '' OR
    isolanguage LIKE 'en%' OR
    isolanguage LIKE 'de%' OR
    isolanguage == 'abbr' OR
    list_bool_or([(isolanguage IN lang) FOR lang IN entity.country.iso_languages])
"""

POP_LANGUAGE_FILTERED_NAMES_SELECT = """
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE (
    is_preferred_name IS TRUE OR
    isolanguage == '' OR
    isolanguage LIKE 'en%' OR
    isolanguage LIKE 'de%' OR
    isolanguage == 'abbr' OR
    list_bool_or([(isolanguage IN lang) FOR lang IN entity.country.iso_languages])
) AND len(entity.possible_entity_types) > 0
"""

OTHER_LANGUAGE_FILTERED_NAMES_SELECT = """
SELECT *
FROM geo_db.geographical_names_with_entities
WHERE (
    is_preferred_name IS TRUE OR
    isolanguage == '' OR
    isolanguage LIKE 'en%' OR
    isolanguage LIKE 'de%' OR
    isolanguage == 'abbr' OR
    list_bool_or([(isolanguage IN lang) FOR lang IN entity.country.iso_languages])
) AND len(entity.possible_entity_types) = 0
"""

# TODO index names only for fuzzy search, and then search duckdb for all names that match the correction exactly
# This is important because otherwise we may fill our topk with the same name over and over again

WORDS_QUERY = """
WITH language_filtered AS (
""" + LANGUAGE_FILTERED_NAMES_SELECT + """
)
SELECT
    string_split(german_ascii_name, ' ') AS german_ascii_words, 
    string_split(ascii_name, ' ') AS ascii_words,
    * EXCLUDE (german_ascii_name, ascii_name)
FROM language_filtered
"""

UNIQUE_WORDS_QUERY = """
WITH
words AS (
""" + WORDS_QUERY + """
),
flattened AS (
    SELECT unnest(german_ascii_words || ascii_words) AS word FROM words
)
SELECT word, COUNT(*) AS freq
FROM flattened
WHERE word != ''
GROUP BY word
"""

UNIQUE_BIGRAMS_QUERY = """
WITH
words AS (
""" + WORDS_QUERY + """
),
bigrams AS (
    SELECT
        list_zip(words.german_ascii_words[:-1], words.german_ascii_words[2:]) AS german_ascii_bigrams,
        list_zip(words.ascii_words[:-1], words.ascii_words[2:]) AS ascii_bigrams
    FROM words
),
bigram_strings AS (
    SELECT
        array_to_string(german_ascii_bigrams, ' ') AS german_ascii_bigram_strings,
        array_to_string(ascii_bigrams, ' ') AS ascii_bigram_strings
    FROM bigrams
),
flattened AS (
    SELECT unnest([german_ascii_bigram_strings, ascii_bigram_strings]) AS bigram 
    FROM bigram_strings
)
SELECT bigram, COUNT(*) AS freq 
FROM flattened
WHERE bigram != ''
GROUP BY bigram
"""

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

def ascii_normalize(nfc_string : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_string.lower()
    # Strip accents replacements may differ on implementation
    # Call duckdb function to ensure the same behavior as in the database
    result = unidecode.unidecode(result)
    result = result.replace("-", " ")
    result = _STRIP_PUNCTUATION_REGEX.sub("", result)
    return result

_GERMAN_NORMALIZATION_REPLACEMENTS = [
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("ß", "ss")
]

# ensure the replacement keys are NFC normalized themselves
for i, (key, value) in enumerate(_GERMAN_NORMALIZATION_REPLACEMENTS):
    _GERMAN_NORMALIZATION_REPLACEMENTS[i] = (unicodedata.normalize("NFC", key), value)

def german_normalize(nfc_string : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    Accent stripping takes into account german rules for conversion of umlauts and ß.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_string.lower()
    for old, new in _GERMAN_NORMALIZATION_REPLACEMENTS:
        result = result.replace(old, new)
    return ascii_normalize(result)

def ascii_normalized_strings(nfc_string : str) -> list[str]:
    """
    Produces (if differing) two ascii normalized versions of the input string.
    One will use the german ascii normalization rules for umlauts and ß, 
    the other will use the basic ascii normalization rules.
    """
    basic_normalization = ascii_normalize(nfc_string)
    german_normalization = german_normalize(nfc_string)
    if basic_normalization == german_normalization:
        return [basic_normalization]
    else:
        return [basic_normalization, german_normalization]
    

class IndexSearchMatch(NamedTuple):
    score : float
    nfc_query : str
    german_ascii_query: Optional[str]
    ascii_query: str

class IndexSearchMatch(NamedTuple):
    score : float # score as returned by the the specific index search, meaning differs
    matched_key: str
    nfc_name: str
    levenshtein_distance: Optional[int] = None,
    retrieved_data : Optional[dict] = None


class IndexSearchResult(NamedTuple):
    nfc_query : str
    query_strings : list[str]
    abbreviation_pattern : Optional[str]
    matches : list[IndexSearchMatch]

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

    def populate_index(self, geo_db_connection : duckdb.DuckDBPyConnection, sql_query : str, skip_if_exists=True):
        if skip_if_exists and self.already_exists:
            self.index.reload()
            return
        total_rows = geo_db_connection.execute("SELECT COUNT(*) FROM (" + sql_query + ")").fetchone()[0]
        print(f"Populating Tantivy index with {total_rows} names from the geonames database...")
        with self.index.writer(num_threads=8) as writer:
            batch_iterator = geo_db_connection.execute(sql_query).to_arrow_reader(100_000)
            pbar = tqdm(total=total_rows, desc="Populating Tantivy index")
            for batch in batch_iterator:
                columns = [col.to_pylist() for col in batch.columns]
                for row_tuple in zip(*columns):
                    row = {name : value for name, value in zip(batch.column_names, row_tuple)}
                    nfc_name = unicodedata.normalize("NFC", row["name"])
                    for search_key in ascii_normalized_strings(nfc_name):
                        if search_key is None or search_key.strip() == "":
                            continue
                        doc = tantivy.Document()
                        doc.add_text("nfc_name", nfc_name)
                        doc.add_text("search_key", search_key)
                        doc.add_bytes("name_data", json.dumps(row).encode("utf-8"))

                        # extra fields for search restriction
                        entity_types = row["entity"].get("entity_types", [])
                        for entity_type in GeographicalEntityType:
                            doc.add_boolean(entity_type.entity_type, entity_type.entity_type in entity_types)
                        doc.add_text("country_code", row["entity"]["country"]["iso_code"] or "")
                        admin_codes = row["entity"].get("admin_codes", {})
                        for admin_level in range(1, 6):
                            admin_code = admin_codes.get(f"admin{admin_level}_code")
                            doc.add_text(f"admin{admin_level}_code", admin_code or "")
                        writer.add_document(doc)
                        pbar.update(1)
            pbar.close()
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
        schema_builder.add_text_field("search_key", stored=True, **text_field_options)
        # json blob with the original data
        schema_builder.add_bytes_field("name_data", stored=True, indexed=False)
        # nfc name, no char stripping
        schema_builder.add_text_field("nfc_name", stored=True, **text_field_options)

        # extra fields for search restriction
        for entity_type in GeographicalEntityType:
            schema_builder.add_boolean_field(entity_type.entity_type, indexed=True)
        schema_builder.add_text_field("country_code", **text_field_options)
        for code in ["admin1_code", "admin2_code", "admin3_code", "admin4_code", "admin5_code"]:
            schema_builder.add_text_field(code, **text_field_options)
        return schema_builder.build()

    def _build_entity_type_restriction(
            self, entity_types : Collection[GeographicalEntityType]) -> tantivy.Query:
        if len(entity_types) == 1:
            entity_type = next(iter(entity_types))
            return tantivy.Query.term_query(
                self.schema, entity_type.entity_type, True, index_option='basic')
        else:
            disjuction = []
            for entity_type in entity_types:
                disjuction.append(tantivy.Query.term_query(self.schema, entity_type.entity_type, True))
            return tantivy.Query.disjunction_max_query(disjuction)

    def _build_country_code_restriction(self, country_codes : Collection[str]) -> tantivy.Query:
        if len(country_codes) == 1:
            country_code = next(iter(country_codes))
            return tantivy.Query.term_query(self.schema, "country_code", country_code, index_option='basic')
        else:
            disjuction = []
            for country_code in country_codes:
                disjuction.append(tantivy.Query.term_query(self.schema, "country_code", country_code))
            return tantivy.Query.disjunction_max_query(disjuction)
    
    def _build_admin_code_restriction(self, admin_codes : Collection[GeonamesAdminCodes]) -> tantivy.Query:
        def _admin_code_to_query(admin_code : GeonamesAdminCodes) -> tantivy.Query:
            admin_code_queries = []
            for k, v in admin_code._asdict().items():
                if v is not None and v != '':
                    admin_code_queries.append((
                        tantivy.Occur.Must, tantivy.Query.term_query(self.schema, k, v, index_option='basic')
                    ))
            if len(admin_code_queries) == 0:
                return None
            if len(admin_code_queries) == 1:
                return admin_code_queries[0][1]
            else:
                return tantivy.Query.boolean_query(admin_code_queries)
        if len(admin_codes) == 1:
            return _admin_code_to_query(next(iter(admin_codes)))
        else:
            disjuction = []
            for admin_code in admin_codes:
                admin_code_query = _admin_code_to_query(admin_code)
                if admin_code_query is not None:
                    disjuction.append(admin_code_query)
            return tantivy.Query.disjunction_max_query(disjuction)


    def _search_inner(
            self,  
            query_strings : list[str], 
            abbrev_pattern : Optional[str], 
            restrictions : list[tuple[tantivy.Occur, tantivy.Query]], 
            distance_threshold : int, 
            similarity_threshold : float,
            limit : int
        ) -> list[IndexSearchMatch]:
        if distance_threshold == 0 or similarity_threshold == 1.0:
            name_queries = [
                tantivy.Query.term_query(self.schema, "search_key", query, index_option='basic')
                for query in query_strings
            ]
        else:
            name_queries = [
                tantivy.Query.boost_query(tantivy.Query.fuzzy_term_query(self.schema,
                    "search_key", query, distance=distance_threshold), 1.0)
                for query in query_strings
            ]
        if abbrev_pattern:
            name_queries.append(tantivy.Query.regex_query(self.schema, "search_key", abbrev_pattern))
        else:
            # If no abbreviation pattern is provided, we can determine max and min length
            # based on the similarity threshold

            # similarity = 1 - (dist(a, b) / max(len(a), len(b))))
            #   similarity >= similarity_threshold
            #   1 - (dist(a, b) / max(len(a), len(b)))) >= similarity_threshold
            #   dist(a, b) / max(len(a), len(b)) <= 1 - similarity_threshold
            #   dist(a, b) <= (1 - similarity_threshold) * max(len(a), len(b))
            # Let us assume len(a) >= len(b). 
            # Since we know len(a) we get an upper bound for distance:
            #   dist(a, b) <= (1 - similarity_threshold) * len(a)
            # since we have len(a) - len(b) <= dist(a, b)
            #   len(a) - len(b) <= (1 - similarity_threshold) * len(a)
            #   len(b) >= len(a) - (1 - similarity_threshold) * len(a)
            # Let us assume len(a) < len(b)
            # then we have len(b) - len(a) <= dist(a, b)
            #   len(b) - len(a) <= (1 - similarity_threshold) * len(b)
            #   1 - len(a)/len(b) <= 1 - similarity_threshold
            #   len(a)/len(b) >= similarity_threshold
            #   len(b) <= len(a) / similarity_threshold
            #max_upper_bound = -float("inf")
            #min_lower_bound = float("inf")
            #for query in query_strings:
            #    query_length = len(query)
            #    # similarity based bounds
            #    upper_bound = math.floor(query_length / similarity_threshold)
            #    lower_bound = math.ceil(query_length - (1 - similarity_threshold) * query_length)
            #    # distance based bounds
            #    upper_bound = min(upper_bound, query_length + distance_threshold)
            #    lower_bound = max(lower_bound, query_length - distance_threshold)
            #    max_upper_bound = max(max_upper_bound, upper_bound)
            #    min_lower_bound = min(min_lower_bound, lower_bound)
            #restrictions = restrictions + [
            #    (tantivy.Occur.Must, tantivy.Query.range_query(self.schema, "search_key_length", tantivy.FieldType.Integer, min_lower_bound, max_upper_bound))
            #]
            pass
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
            matches.append(IndexSearchMatch(
                score=score,
                matched_key=doc.get_first("search_key"),
                nfc_name=doc.get_first("nfc_name"),
                retrieved_data=json.loads(doc.get_first("name_data").decode("utf-8"))
            ))
        return matches

    def search(
            self, 
            query_string, 
            distance_threshold : int, 
            similarity_threshold : float,
            limit : int = 10,
            match_abbreviations : bool = True,
            entity_types : Optional[Collection[GeographicalEntityType]] = None,
            country_codes : Optional[Collection[str]] = None,
            admin_codes : Optional[Collection[GeonamesAdminCodes]] = None
        ):
        nfc_query = unicodedata.normalize("NFC", query_string)
        query_strings = ascii_normalized_strings(nfc_query)
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
        for i in range(0, distance_threshold + 1):
            matches = self._search_inner(
                query_strings=query_strings,
                abbrev_pattern=abbrev_pattern,
                restrictions=restrictions,
                distance_threshold=i,
                similarity_threshold=similarity_threshold,
                limit=limit
            )
            if len(matches) > 0:
                break
        return IndexSearchResult(
            nfc_query=nfc_query,
            query_strings=query_strings,
            abbreviation_pattern=abbrev_pattern,
            matches=matches
        ) # TODO fix

class SymSpellSearchIndex:
    def __init__(self):
        raise NotImplementedError("This class is being refactored and should not be used in its current state.")

# Based on observation of the distribution of countries
PRIORITY_COUNTRIES = [
    "DE", # Germany
    "US", # United States
    "IL", # Israel
    "PL", # Poland
    "FR", # France
    "RO", # Romania
    "CZ", # Czech Republic
    "HU", # Hungary
    "RU", # Russia
]

class GeoDBSearch(LinkingStep):
    def __init__(
            self, search_cache_db, 
            search_index : TantivySearchIndex, materialize_table : bool = True,
            distance_threshold : int = 2,
            similarity_threshold : float = 0.8,
            topk : int = 5,
            priority_countries : Optional[list[str]] = PRIORITY_COUNTRIES,
            geo_db_path : str = "geo.duckdb"
        ):
        self.search_cache_db_path = search_cache_db
        self.search_index = search_index
        self.materialize_table = materialize_table
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.topk = topk
        self.priority_countries = priority_countries
        self.geo_db_path = geo_db_path

    def initialize(self):
        self.connection = duckdb.connect(self.search_cache_db_path)
        self.connection.execute(f"ATTACH DATABASE '{self.geo_db_path}' AS geo_db (READ_ONLY)")
        self.connection.execute(GERMAN_ASCII_DUCKDB_MACRO)
        self.search_index.populate_index(self.connection, sql_query=POP_LANGUAGE_FILTERED_NAMES_SELECT, skip_if_exists=True)
        #self.connection.execute("""
        #CREATE TABLE IF NOT EXISTS mat_geographical_names_with_entities AS
        #SELECT * FROM geo_db.geographical_names_with_entities
        #""")
        #self.connection.execute("ALTER TABLE mat_geographical_names_with_entities ADD PRIMARY KEY (name_id)")
        #self.connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_name_id ON mat_geographical_names_with_entities(name_id)")
        return super().initialize()
    
    def finalize(self):
        self.connection.close()
        return super().finalize()

    def _search_entity(
        self, 
        entity : MatchedEntity, 
        country_codes : Optional[Collection[str]], 
        admin_codes : Optional[Collection[GeonamesAdminCodes]]
    ) -> IndexSearchResult:
        entity_type = entity.entity_type
        all_entity_types = list(GeographicalEntityType)
        if entity_type != GeographicalEntityType.Country and country_codes is None:
            country_codes = self.priority_countries
        index_matches = self.search_index.search(
            query_string=entity.raw_text,
            distance_threshold=self.distance_threshold,
            similarity_threshold=self.similarity_threshold,
            limit=self.topk,
            match_abbreviations=True,
            entity_types=[entity_type],
            country_codes=country_codes,
            admin_codes=admin_codes
        )
        if len(index_matches.matches) == 0:
            index_matches = self.search_index.search(
                query_string=entity.raw_text,
                distance_threshold=self.distance_threshold,
                similarity_threshold=self.similarity_threshold,
                limit=self.topk,
                match_abbreviations=True,
                entity_types=[entity_type]
            )
        if len(index_matches.matches) == 0 and (
                country_codes is not None or admin_codes is not None
            ):
            index_matches = self.search_index.search(
                query_string=entity.raw_text,
                distance_threshold=self.distance_threshold,
                similarity_threshold=self.similarity_threshold,
                limit=self.topk,
                match_abbreviations=True,
                entity_types=all_entity_types,
                country_codes=country_codes,
                admin_codes=admin_codes
            )
        if len(index_matches.matches) == 0:
            index_matches = self.search_index.search(
                query_string=entity.raw_text,
                distance_threshold=self.distance_threshold,
                similarity_threshold=self.similarity_threshold,
                limit=self.topk,
                match_abbreviations=True,
                entity_types=all_entity_types
            )
        # if len(index_matches.matches) == 0:
        #     index_matches = self.search_index.search(
        #         query_string=entity.raw_text,
        #         distance_threshold=self.distance_threshold,
        #         similarity_threshold=self.similarity_threshold,
        #         limit=self.topk,
        #         match_abbreviations=True
        #     )
        return index_matches
    
    def _parse_data(self, index_result : IndexSearchResult) -> Iterable[MatchedName]:
        for index_match in index_result.matches:
            is_abreviation_match = False
            if index_result.abbreviation_pattern is not None:
               is_abreviation_match = re.fullmatch(index_result.abbreviation_pattern, index_match.matched_key) is not None

            max_clean_similarity = 0.0
            clean_alt_name = None
            cleaned_edit_distance = None
            for query_string in index_result.query_strings:
                query_edit_distance, query_similarity = similarity_and_distance(query_string, index_match.matched_key, self.distance_threshold)
                if query_similarity > max_clean_similarity:
                    max_clean_similarity = query_similarity
                    clean_alt_name = query_string
                    cleaned_edit_distance = query_edit_distance
            #if (max_clean_similarity < self.similarity_threshold or cleaned_edit_distance > self.distance_threshold) and not is_abreviation_match:
            #    continue

            geographical_name=GeographicalName.from_dict(index_match.retrieved_data)
            nfc_alt_name = unicodedata.normalize("NFC", geographical_name.name)
            matched_name = MatchedName(
                geographical_name=geographical_name,
                nfc_query=index_result.nfc_query,
                nfc_alt_name=nfc_alt_name,
                cleaned_queries=index_result.query_strings,
                cleaned_alt_name=clean_alt_name,
                cleaned_edit_distance=cleaned_edit_distance,
                edit_distance = levenshtein(index_result.nfc_query, nfc_alt_name, self.distance_threshold),
                abbreviation_pattern=index_result.abbreviation_pattern,
                is_abbreviation_match=is_abreviation_match,
                matching_method="tantivy"
            )
            yield matched_name
        

    def apply(self, address):
        new_entities = []
        country_codes = set()
        admin_codes = set()
        for entity in sorted(address.matched_entities, key=lambda e: e.entity_type):
            index_result = self._search_entity(
                entity, 
                country_codes=country_codes if len(country_codes) > 0 else None, 
                admin_codes=admin_codes if len(admin_codes) > 0 else None
            )
            matched_names = list(self._parse_data(index_result))
            for matched_name in matched_names:
                country_codes.update(matched_name.geographical_name.entity.all_country_iso_codes)
                if matched_name.geographical_name.entity.admin_codes:
                    admin_codes.add(matched_name.geographical_name.entity.admin_codes)
            new_entities.append(dataclasses.replace(
                entity,
                matches=matched_names
            ))
        return dataclasses.replace(address, matched_entities=new_entities)


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
