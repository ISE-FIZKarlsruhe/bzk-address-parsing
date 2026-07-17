from modules.geo_db_search import GeoSearchIndex, IndexSearchMatch, IndexSearchResult, abbreviation_pattern_to_regexes, normalized_search_strings
from modules.pipeline.geographical_entity import GeographicalEntityType, GeonamesAdminCodes


import duckdb
from elasticsearch import Elasticsearch, helpers
from tqdm.auto import tqdm


import unicodedata
from typing import Collection, Optional, Iterable


class ElasticSearchClient(GeoSearchIndex):
    index_descriptor = "elastic_search"
    """
    CLAUDE reimplementation of the TantivySearchIndex using Elasticsearch as the backend.
    Elasticsearch-based reimplementation of TantivySearchIndex.

    Key mapping decisions vs. the Tantivy version:
      - Tantivy's `raw` tokenizer (exact single-term matching) -> ES `keyword` fields.
      - Tantivy's `add_bytes_field(indexed=False, stored=True)` for the JSON blob ->
        an ES sub-object with "enabled": false. ES always keeps the raw `_source`,
        so this just tells ES not to parse/index the sub-object's fields, matching
        Tantivy's "stored but not searchable" semantics. No manual json.dumps/loads
        needed - `_source` is already JSON.
      - `tantivy.Query.term_query`      -> {"term": {...}}
      - `tantivy.Query.fuzzy_term_query`-> {"fuzzy": {...}}
      - `tantivy.Query.regex_query`     -> {"regexp": {...}}
      - `tantivy.Query.disjunction_max_query` -> {"dis_max": {"queries": [...]}}
      - `tantivy.Query.boolean_query` (Occur.Must) -> {"bool": {"must": [...]}}
      - `tantivy.Query.boost_query`     -> "boost" key inside the query clause
    """

    def __init__(
        self,
        index_name: str,
        *,
        es_client: Optional[Elasticsearch] = None,
        es_hosts: Optional[list[str]] = None,
    ):
        self.index_name = index_name
        self.es = es_client or Elasticsearch(es_hosts or ["http://localhost:9201"])


    # ------------------------------------------------------------------
    # Schema / mapping
    # ------------------------------------------------------------------

    def create_settings(self) -> dict:
        return {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    # Splits purely on whitespace - no lowercasing, no stemming,
                    # no stop words, no accent folding. Case-folding and accent
                    # stripping are already handled upstream before indexing, so
                    # this analyzer intentionally does nothing beyond tokenizing
                    # on spaces.
                    "toponym_whitespace_analyzer": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": [],
                    }
                }
            },
        }

    def create_mappings(self) -> dict:
        properties = {
            # search keys - full-text field for match+fuzziness queries.
            # Tokenized on whitespace only (see toponym_whitespace_analyzer);
            # relies on upstream normalization for case/accents.
            "search_key": {
                "type": "text",
                "analyzer": "toponym_whitespace_analyzer",
            },
            # nfc name, no char stripping
            "nfc_name": {"type": "keyword"},
            # stored-but-not-indexed JSON blob with the original row data
            "name_data": {"type": "object", "enabled": False},
            "country_code": {"type": "keyword"},
        }
        for entity_type in GeographicalEntityType:
            properties[entity_type.entity_type] = {"type": "boolean"}
        for level in range(1, 6):
            properties[f"admin{level}_code"] = {"type": "keyword"}

        return {"properties": properties}

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate_index(self, row_retriever: Iterable[dict], skip_if_exists=True):
        if not self.es.ping():
            raise RuntimeError("Elasticsearch cluster is not reachable.")

        already_exists = False
        try:
            already_exists = self.es.indices.exists(index=self.index_name)
        except Exception as e:
            pass
        if skip_if_exists and already_exists:
            self.es.indices.refresh(index=self.index_name)
            return
        self.es.indices.create(
            index=self.index_name,
            settings=self.create_settings(),
            mappings=self.create_mappings(),
        )
        def _actions():
            for row in row_retriever:
                nfc_name = unicodedata.normalize("NFC", row["name"])

                search_keys = [
                    search_key
                    for search_key in normalized_search_strings(nfc_name)
                    if search_key is not None and search_key.strip() != ""
                ]
                if not search_keys:
                    continue

                doc = {
                    "nfc_name": nfc_name,
                    # multivalued: ES indexes each entry as a separate term
                    # occurrence in the same field, no array type needed
                    "search_key": search_keys,
                    "name_data": row,
                    "country_code": row["entity"]["country"]["iso_code"] or "",
                }

                entity_types = row["entity"].get("entity_types", [])
                for entity_type in GeographicalEntityType:
                    doc[entity_type.entity_type] = entity_type.entity_type in entity_types

                admin_codes = row["entity"].get("admin_codes", {})
                for admin_level in range(1, 6):
                    admin_code = admin_codes.get(f"admin{admin_level}_code")
                    doc[f"admin{admin_level}_code"] = admin_code or ""

                yield {"_index": self.index_name, "_source": doc}

        success_count = 0
        for ok, item in helpers.streaming_bulk(
            self.es,
            _actions(),
            chunk_size=2000,
            raise_on_error=False,
            max_retries=3,
        ):
            if ok:
                success_count += 1
            else:
                print(f"Failed to index document: {item}")

        self.es.indices.refresh(index=self.index_name)


    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _build_entity_type_restriction(self, entity_types: Collection["GeographicalEntityType"]) -> dict:
        if len(entity_types) == 1:
            entity_type = next(iter(entity_types))
            return {"term": {entity_type.entity_type: True}}
        else:
            return {
                "dis_max": {
                    "queries": [{"term": {et.entity_type: True}} for et in entity_types]
                }
            }

    def _build_country_code_restriction(self, country_codes: Collection[str]) -> dict:
        if len(country_codes) == 1:
            country_code = next(iter(country_codes))
            return {"term": {"country_code": country_code}}
        else:
            return {
                "dis_max": {
                    "queries": [{"term": {"country_code": cc}} for cc in country_codes]
                }
            }

    def _build_admin_code_restriction(self, admin_codes: Collection["GeonamesAdminCodes"]) -> Optional[dict]:
        def _admin_code_to_query(admin_code) -> Optional[dict]:
            must_clauses = []
            for k, v in admin_code._asdict().items():
                if v is not None and v != "":
                    must_clauses.append({"term": {k: v}})
            if len(must_clauses) == 0:
                return None
            if len(must_clauses) == 1:
                return must_clauses[0]
            return {"bool": {"must": must_clauses}}

        if len(admin_codes) == 1:
            return _admin_code_to_query(next(iter(admin_codes)))
        else:
            disjunction = []
            for admin_code in admin_codes:
                q = _admin_code_to_query(admin_code)
                if q is not None:
                    disjunction.append(q)
            if not disjunction:
                return None
            return {"dis_max": {"queries": disjunction}}

    def _search_inner(
        self,
        query_strings: list[str],
        abbrev_pattern: Optional[str],
        restrictions: list[dict],
        distance_threshold: int,
        similarity_threshold: float,
        limit: int,
    ) -> list["IndexSearchMatch"]:
        name_queries = [
            {
                "match": {
                    "search_key": {
                        "query": query,
                        "analyzer": "toponym_whitespace_analyzer",
                        "boost": 1.0,
                    }
                }
            }
            for query in query_strings
        ]
        if distance_threshold != 0:
            for query in name_queries:
                query["match"]["search_key"]["fuzziness"] = distance_threshold

        if abbrev_pattern:
            name_queries.append({"regexp": {"search_key": abbrev_pattern}})

        final_name_query = {"dis_max": {"queries": name_queries}}

        if len(restrictions) > 0:
            final_query = {"bool": {"must": [final_name_query] + restrictions}}
        else:
            final_query = final_name_query

        response = self.es.search(index=self.index_name, query=final_query, size=limit)

        matches = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            matches.append(
                IndexSearchMatch(
                    score=hit["_score"],
                    matched_key=source.get("search_key")[0], # TODO find which key actually matched?
                    nfc_name=source.get("nfc_name"),
                    retrieved_data=source.get("name_data"),
                    metadata = response
                )
            )
        return matches

    def search(
        self,
        query_string,
        distance_threshold: int,
        similarity_threshold: float,
        limit: int = 10,
        match_abbreviations: bool = True,
        entity_types: Optional[Collection["GeographicalEntityType"]] = None,
        country_codes: Optional[Collection[str]] = None,
        admin_codes: Optional[Collection["GeonamesAdminCodes"]] = None,
    ):
        nfc_query = unicodedata.normalize("NFC", query_string)
        query_strings = normalized_search_strings(nfc_query)
        if match_abbreviations:
            abbrev_pattern = abbreviation_pattern_to_regexes(query_string)
        else:
            abbrev_pattern = None

        restrictions = []
        if entity_types is not None:
            restrictions.append(self._build_entity_type_restriction(entity_types))
        if country_codes is not None:
            restrictions.append(self._build_country_code_restriction(country_codes))
        if admin_codes is not None:
            admin_code_restriction = self._build_admin_code_restriction(admin_codes)
            if admin_code_restriction is not None:
                restrictions.append(admin_code_restriction)

        for i in [distance_threshold]:
            matches = self._search_inner(
                query_strings=query_strings,
                abbrev_pattern=abbrev_pattern,
                restrictions=restrictions,
                distance_threshold=i,
                similarity_threshold=similarity_threshold,
                limit=limit,
            )
            if len(matches) > 0:
                break

        return IndexSearchResult(
            nfc_query=nfc_query,
            query_strings=query_strings,
            abbreviation_pattern=abbrev_pattern,
            matches=matches,
        )