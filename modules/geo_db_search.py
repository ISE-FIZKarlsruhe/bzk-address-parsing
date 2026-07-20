"""
Classes related to matching extracted place names to place names on the database.
"""

# TODO rewrite everything
from pathlib import Path
import pandas as pd
import modules.utils as utils
import duckdb
import modules.build_geonames_db as build_geonames_db
from typing import Collection, Iterable, NamedTuple, Optional, Literal, Callable, TYPE_CHECKING
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
from abc import ABC, abstractmethod
from modules.pipeline.storage.encoding_util import decode_from_dict


def levenshtein(a : str, b : str, max_distance : int) -> int:
    """
    Computes the Levenshtein distance between two strings.
    If max_distance is provided, the computation will stop if the distance exceeds max_distance.
    """
    # wrapper method to correct logic
    dist = editdistpy.levenshtein.distance(a, b, max_distance=max_distance)
    if dist < 0:
        return sys.maxsize
    return dist

def similarity_and_distance(a : str, b : str, max_distance : int, distance_function : Callable[[str, str, int], int] = levenshtein):
    edit_distance = distance_function(a, b, max_distance)
    similarity = 1 - (edit_distance / max(len(a), len(b)))
    return edit_distance, similarity


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

# match periods follwoing an isolated letter
_STRIP_PERIODS_ABBREV_REGEX = re.compile(r'((?<=\W\w)|(?<=^\w))\.')
_STRIP_PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
_DEDUPE_WHITESPACE_REGEX = re.compile(r'\s+')

def ascii_normalize(nfc_string : str) -> str:
    """
    Converts a string to lowercase, strips accents and punctuation.
    This is used for matching against similarly normalized names in the database.
    """
    result = nfc_string.lower()
    # Converts non ascii characters to ascii equivalents, e.g. "é" -> "e"
    result = unidecode.unidecode(result)
    # strip periods to normalize abbreviations, e.g. "U.S.A." -> "USA"
    result = _STRIP_PERIODS_ABBREV_REGEX.sub('', result)
    # replace other punctuations with spaces eg. 
    result = _STRIP_PUNCTUATION_REGEX.sub(' ', result)
    result = _DEDUPE_WHITESPACE_REGEX.sub(' ', result)
    result = result.strip()
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

def normalized_search_strings(nfc_string : str) -> list[str]:
    """
    Produces (if differing) two ascii normalized versions of the input string.
    One will use the german ascii normalization rules for umlauts and ß, 
    the other will use the basic ascii normalization rules.
    """
    # TODO maybe remove stop words?
    basic_normalization = ascii_normalize(nfc_string)
    german_normalization = german_normalize(nfc_string)
    if basic_normalization == german_normalization:
        return [basic_normalization]
    else:
        return [basic_normalization, german_normalization]
    
def abbreviation_pattern_to_regexes(part : str) -> str:
    """
    Converts an abbreviation pattern to a regex pattern.
    """
    prefixes = part.split(".")
    if len(prefixes) == 1:
        return None
    ascii_regex_pattern = "[a-z]+ ".join([ascii_normalize(x) for x in prefixes]).strip()
    german_regex_pattern = "[a-z]+ ".join([german_normalize(x) for x in prefixes]).strip()
    if ascii_regex_pattern == german_regex_pattern:
        return ascii_regex_pattern
    else:
        return f"({ascii_regex_pattern})|({german_regex_pattern})"

class IndexSearchMatch(NamedTuple):
    score : float # score as returned by the the specific index search, meaning differs
    matched_key: str
    nfc_name: str
    levenshtein_distance: Optional[int] = None,
    retrieved_data : Optional[dict] = None
    metadata : Optional[dict] = None


class IndexSearchResult(NamedTuple):
    nfc_query : str
    query_strings : list[str]
    abbreviation_pattern : Optional[str]
    matches : list[IndexSearchMatch]
    

class GeoSearchIndex(ABC):
    index_descriptor : str
    @abstractmethod
    def populate_index(self, row_retriever : Iterable[dict], skip_if_exists=True):
        pass
    @abstractmethod
    def search(
            self, 
            query_string, 
            distance_threshold : int, 
            similarity_threshold : float,
            limit : int = 10,
            expand_abbreviations : bool = True,
            entity_types : Optional[Collection[GeographicalEntityType]] = None,
            country_codes : Optional[Collection[str]] = None,
            admin_codes : Optional[Collection[GeonamesAdminCodes]] = None
        ) -> IndexSearchResult:
        pass

class TantivySearchIndex(GeoSearchIndex):
    index_descriptor = "tantivy"
    def __init__(self, index_path : str | Path, read_threads : int | Literal['auto'] = 'auto', write_threads : int = 8):
        if read_threads == 'auto':
            read_threads = (max(1, getattr(os, "process_cpu_count", lambda : None)() or os.cpu_count() or 8) * 3) // 2
        self.schema = self.create_schema()
        self.index_path = Path(index_path)
        self.already_exists = self.index_path.exists()
        if not self.already_exists:
            self.index_path.mkdir(parents=True)
        self.read_threads = read_threads
        self.write_threads = write_threads
        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        self.index.config_reader(num_warmers=self.read_threads)
        self.index.register_tokenizer(
            "whitespace",
            tantivy.TextAnalyzerBuilder(
                tantivy.Tokenizer.whitespace()
            ).build()
        )

    def populate_index(self, row_retriever : Iterable[dict], skip_if_exists=True):
        if skip_if_exists and self.already_exists:
            self.index.reload()
            return
        with self.index.writer(num_threads=self.write_threads) as writer:
            for row in row_retriever:
                nfc_name = unicodedata.normalize("NFC", row["name"])
                for search_key in normalized_search_strings(nfc_name):
                    if search_key is None or search_key.strip() == "":
                        continue
                    doc = tantivy.Document()
                    doc.add_text("nfc_name", nfc_name)
                    doc.add_text("search_key", search_key)
                    doc.add_bytes("name_data", json.dumps(row).encode("utf-8"))

                    # extra fields for search restriction
                    entity_types = row["entity"].get("possible_entity_types", [])
                    for entity_type in GeographicalEntityType:
                        doc.add_boolean(entity_type.entity_type, entity_type.entity_type in entity_types)
                    doc.add_text("country_code", row["entity"]["country"]["iso_code"] or "")
                    admin_codes = row["entity"].get("admin_codes", {})
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
        # TODO make fields fast=True
        # search keys
        schema_builder.add_text_field("search_key", stored=True, fast=True, **text_field_options)
        # TODO schema_builder.add_text_field("search_key", stored=True, fast=True, tokenizer_name='whitespace')
        # json blob with the original data
        schema_builder.add_bytes_field("name_data", stored=True, indexed=False)
        # nfc name, no char stripping
        schema_builder.add_text_field("nfc_name", stored=True, **text_field_options)

        # extra fields for search restriction
        for entity_type in GeographicalEntityType:
            schema_builder.add_boolean_field(entity_type.entity_type, fast=True, indexed=True)
        schema_builder.add_text_field("country_code", fast=True, **text_field_options)
        for code in ["admin1_code", "admin2_code", "admin3_code", "admin4_code", "admin5_code"]:
            schema_builder.add_text_field(code, fast=True, **text_field_options)
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
            limit : int,
            query_strings : list[str] = [], 
            abbrev_pattern : Optional[str] = None,
            hints : list[tuple[tantivy.Occur, tantivy.Query]] = [],
            distance_threshold : int = 0,
        ) -> list[IndexSearchMatch]:
        if distance_threshold == 0:
            name_queries = [
                tantivy.Query.term_query(self.schema, "search_key", query, index_option='basic')
                for query in query_strings
            ]
        else:
            
            name_queries = [
                tantivy.Query.fuzzy_term_query(self.schema,
                    "search_key", query, distance=distance_threshold)
                for query in query_strings
            ]
        if abbrev_pattern:
            name_queries.append(tantivy.Query.regex_query(self.schema, "search_key", abbrev_pattern))
        final_name_query = tantivy.Query.disjunction_max_query(name_queries)
        if len(hints) > 0:
            final_query = tantivy.Query.boolean_query(
                [(tantivy.Occur.Must, final_name_query)] + hints)
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
            expand_abbreviations : bool = True,
            entity_types : Optional[Collection[GeographicalEntityType]] = None,
            country_codes : Optional[Collection[str]] = None,
            admin_codes : Optional[Collection[GeonamesAdminCodes]] = None
        ):
        if similarity_threshold == 1.0:
            distance_threshold = 0
        nfc_query = unicodedata.normalize("NFC", query_string)
        query_strings = normalized_search_strings(nfc_query)
        if expand_abbreviations:
            abbrev_pattern = abbreviation_pattern_to_regexes(nfc_query)
        else: abbrev_pattern = None
        
        hints = []
        if entity_types is not None:
            hints.append((tantivy.Occur.Should, self._build_entity_type_restriction(entity_types)))
        if country_codes is not None:
            hints.append((tantivy.Occur.Should, self._build_country_code_restriction(country_codes)))
        if admin_codes is not None:
            admin_code_restriction = self._build_admin_code_restriction(admin_codes)
            if admin_code_restriction is not None:
                hints.append((tantivy.Occur.Should, admin_code_restriction))
        def _falling_queries():
            # exact matches first
            other_params = dict(hints=hints, limit=limit)
            yield self._search_inner(
                query_strings=query_strings, distance_threshold=0, **other_params)
            # abbreviation matches second
            if abbrev_pattern is not None:
                yield self._search_inner(
                    abbrev_pattern=abbrev_pattern, distance_threshold=0, **other_params)
            # fuzzy matches last
            for i in range(0, distance_threshold + 1):
                yield self._search_inner(
                    query_strings=query_strings, distance_threshold=i, **other_params)

        for query_result in _falling_queries():
            matches = query_result
            if len(matches) > 0:
                break
        return IndexSearchResult(
            nfc_query=nfc_query,
            query_strings=query_strings,
            abbreviation_pattern=abbrev_pattern,
            matches=matches
        )

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

SEARCHABLE_ENTITY_TYPES = [
    GeographicalEntityType.Country,
    GeographicalEntityType.State,
    GeographicalEntityType.Region ,
    GeographicalEntityType.District,
    GeographicalEntityType.City,
    GeographicalEntityType.Neighborhood
]

class GeoDBSearch(LinkingStep):
    def __init__(
            self, search_cache_db, 
            search_index : GeoSearchIndex, materialize_table : bool = True,
            distance_threshold : int = 2,
            similarity_threshold : float = 0.8,
            topk : int = 20,
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
        def row_retriever(sql_query : str) -> Iterable[dict]:
            total_rows = self.connection.execute("SELECT COUNT(*) FROM (" + sql_query + ")").fetchone()[0]
            print(f"Populating {self.search_index.index_descriptor} index with {total_rows} names from the geo database...")
            batch_iterator = self.connection.execute(sql_query).to_arrow_reader(10_000)
            pbar = tqdm(total=total_rows, desc="Populating search index")
            for batch in batch_iterator:
                columns = [col.to_pylist() for col in batch.columns]
                for row_tuple in zip(*columns):
                    row = {name : value for name, value in zip(batch.column_names, row_tuple)}
                    yield row
                pbar.update(len(batch))
            pbar.close()
        self.search_index.populate_index(row_retriever(POP_LANGUAGE_FILTERED_NAMES_SELECT), skip_if_exists=True)
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
        if entity_type != GeographicalEntityType.Country and country_codes is None:
            country_codes = self.priority_countries
        index_matches = self.search_index.search(
            query_string=entity.raw_text,
            distance_threshold=self.distance_threshold,
            similarity_threshold=self.similarity_threshold,
            limit=self.topk,
            expand_abbreviations=True,
            entity_types=[entity_type],
            country_codes=country_codes,
            admin_codes=admin_codes
        )
        return index_matches
    
    def _parse_data(self, index_result : IndexSearchResult) -> Iterable[MatchedName]:
        for index_match in index_result.matches:
            is_abreviation_match = False
            if index_result.abbreviation_pattern is not None:
               is_abreviation_match = re.fullmatch(index_result.abbreviation_pattern, index_match.matched_key) is not None

            max_clean_similarity = 0.0
            clean_query = None
            cleaned_edit_distance = None
            for query_string in index_result.query_strings:
                try:
                    query_edit_distance, query_similarity = similarity_and_distance(query_string, index_match.matched_key, self.distance_threshold)
                except: raise RuntimeError(f"Error calculating similarity and distance between '{query_string}' and '{index_match.matched_key}'")
                if query_similarity > max_clean_similarity:
                    max_clean_similarity = query_similarity
                    clean_query = query_string
                    cleaned_edit_distance = query_edit_distance
            if (max_clean_similarity < self.similarity_threshold or cleaned_edit_distance > self.distance_threshold) and not is_abreviation_match:
                continue

            geographical_name=decode_from_dict(index_match.retrieved_data, GeographicalName)
            nfc_alt_name = unicodedata.normalize("NFC", geographical_name.name)
            matched_name = MatchedName(
                geographical_name=geographical_name,
                nfc_query=index_result.nfc_query,
                nfc_alt_name=nfc_alt_name,
                cleaned_query=clean_query,
                cleaned_alt_name=index_match.matched_key,
                cleaned_edit_distance=cleaned_edit_distance,
                edit_distance = levenshtein(index_result.nfc_query, nfc_alt_name, self.distance_threshold),
                abbreviation_pattern=index_result.abbreviation_pattern,
                is_abbreviation_match=is_abreviation_match,
                matching_method=self.search_index.index_descriptor,
                matching_score=index_match.score
            )
            yield matched_name
        

    def apply(self, address):
        new_entities = []
        country_codes = set()
        admin_codes = set()
        for entity in sorted(address.entities, key=lambda e: e.entity_type):
            if entity.entity_type not in SEARCHABLE_ENTITY_TYPES:
                new_entities.append(entity)
                continue
            index_result = self._search_entity(
                entity, 
                country_codes=country_codes if len(country_codes) > 0 else None, 
                admin_codes=admin_codes if len(admin_codes) > 0 else None
            )
            matched_names = tuple(self._parse_data(index_result))
            for matched_name in matched_names:
                country_codes.update(matched_name.geographical_name.entity.all_country_iso_codes)
                if matched_name.geographical_name.entity.admin_codes:
                    admin_codes.add(matched_name.geographical_name.entity.admin_codes)
            new_entities.append(entity.with_matches(matched_names))
        return dataclasses.replace(address, entities=tuple(new_entities))
