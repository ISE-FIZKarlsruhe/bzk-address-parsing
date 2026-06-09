import duckdb
from typing import Optional, TYPE_CHECKING
from modules.pipeline.storage.storage import SynchronousStorage
from modules.pipeline.storage.encoding_util import encode_as_dict, decode_from_dict
import json
from modules.pipeline.linking_metadata import AddressLinkingMetadata
from concurrent.futures import ThreadPoolExecutor
import asyncio


_INIT_DB_SQL = """
LOAD JSON;

CREATE TABLE IF NOT EXISTS linking_metadata (
    id TEXT,
    bzk_field_name TEXT,
    full_address TEXT,
    linking_data JSON,
    applied_steps JSON[],
    is_loaded BOOLEAN,
    is_finished BOOLEAN
    PRIMARY KEY (id, bzk_field_name)
)

UPDATE linking_metadata SET is_loaded = FALSE;
"""

_INIT_GEO_ENTITIES_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS geographical_entities (
    iri TEXT PRIMARY KEY,
    name TEXT,
    asciiname TEXT,
    classification TEXT,
    possible_entity_types TEXT[],
    coordinates STRUCT(latitude REAL, longitude REAL),
    population BIGINT,
    closest_geonames_id INTEGER,
    iso_country_code TEXT,
    alternate_iso_country_codes TEXT[],
    admin_codes STRUCT(
        admin1_code TEXT, 
        admin2_code TEXT, 
        admin3_code TEXT, 
        admin4_code TEXT, 
        admin5_code TEXT
    ),
    other_parent_iris TEXT[]
);

CREATE TABLE IF NOT EXISTS country_data (
    iso_code TEXT PRIMARY KEY,
    country_name TEXT,
    iso_languages TEXT[],
    neighboring_countries_iso_codes TEXT[]
);

CREATE VIEW IF NOT EXISTS geographical_entities_with_countries AS
SELECT 
    geographical_entities.* EXCEPT (alternate_iso_country_codes, iso_country_code), 
    alternate_countries.alternate_countries AS alternate_countries,
    country_data as country 
FROM geographical_entities 
    JOIN country_data ON geographical_entities.iso_country_code = country_data.iso_code,
    LATERAL (
        SELECT LIST(alt_country) as alternate_countries FROM 
            UNNEST(geographical_entities.alternate_iso_country_codes) AS code) AS alt_country_codes
            JOIN country_data AS alt_country ON alt_country.iso_code = alt_country_codes.code
    ) alternate_countries;

CREATE VIEW IF NOT EXISTS geographical_names_with_entities AS
SELECT geographical_names.* EXCEPT (iri), geographical_entities_with_countries AS entity FROM geographical_names 
    JOIN geographical_entities_with_countries USING (iri)
    )
);
"""

def _geo_row_to_dict(row):
    # Generated using copilot
    #TODO probably can be much simplified by generalizing it
    return {
        "iri": row["iri"],
        "name": row["name"],
        "asciiname": row["asciiname"],
        "classification": row["classification"],
        "possible_entity_types": row["possible_entity_types"],
        "coordinates": {
            "latitude": row["coordinates"]["latitude"] if row["coordinates"] is not None else None,
            "longitude": row["coordinates"]["longitude"] if row["coordinates"] is not None else None
        } if row["coordinates"] is not None else None,
        "population": row["population"],
        "closest_geonames_id": row["closest_geonames_id"],
        "country": {
            "iso_country_code": row["iso_country_code"],
            "country_name": row["country_name"],
            "iso_languages": row["iso_languages"],
            "neighboring_countries_iso_codes": json.loads(row["neighboring_countries_iso_codes"])
        } if row["iso_country_code"] is not None else None,
        "alternate_countries": [
            {
                "iso_code": alt_country["iso_code"],
                "country_name": alt_country["country_name"],
                "iso_languages": alt_country["iso_languages"],
                "neighboring_countries_iso_codes": json.loads(alt_country["neighboring_countries_iso_codes"])
            } for alt_country in row["alternate_countries"]
        ],
        "admin_codes": {
            "admin1_code": row["admin_codes"]["admin1_code"] if row["admin_codes"] is not None else None,
            "admin2_code": row["admin_codes"]["admin2_code"] if row["admin_codes"] is not None else None,
            "admin3_code": row["admin_codes"]["admin3_code"] if row["admin_codes"] is not None else None,
            "admin4_code": row["admin_codes"]["admin4_code"] if row["admin_codes"] is not None else None,
            "admin5_code": row["admin_codes"]["admin5_code"] if row["admin_codes"] is not None else None
        } if row["admin_codes"] is not None else None,
        "other_parent_iris": json.loads(row["other_parent_iris"])
    }

class DuckDBStorage(SynchronousStorage):
    def __init__(
            self, 
            db_path: str,
            geographical_database_path: Optional[str]
        ):
        self.db_path = db_path
        self.geographical_database_path = geographical_database_path
        self.direct_geo_db_access = geographical_database_path is not None
        if geographical_database_path is not None:
            self._geodb_prefix = "geo_db."
        else:
            self._geodb_prefix = ""

    def initialize(self):
        super().initialize()
        self.connection = duckdb.connect(self.db_path)
        self.connection.execute(_INIT_DB_SQL)
        if self.geographical_database_path is not None:
            self.connection.execute(f"ATTACH '{self.geographical_database_path}' AS geo_db (READONLY)")
        else:
            self.connection.execute(_INIT_GEO_ENTITIES_TABLES_SQL)

    def _insert_geographical_entity(self, name_data: dict) -> None:
        """
        Insert geographical entity data into the geographical_entities table
        """
        # If we have direct access to the geographical database, 
        # we assume that the data is already there and do not insert it into processing storage.
        if self.direct_geo_db_access:
            return
        entity_data = name_data.pop("entity", {})
        country = entity_data.pop("country", None)
        alternate_countries = entity_data.pop("alternate_countries", [])
        iso_country_code = country.get("iso_country_code") if country is not None else None
        alternate_iso_country_codes = [c.get("iso_country_code") for c in alternate_countries if c is not None]
        alternate_iso_country_codes = [code for code in alternate_iso_country_codes if code is not None]
        entity_data["iso_country_code"] = iso_country_code
        entity_data["alternate_iso_country_codes"] = alternate_iso_country_codes
        name_data["iri"] = entity_data["iri"]
        self.connection.execute(
            f"""
            INSERT INTO {self._geodb_prefix}geographical_names (
                iri, alternate_name, is_preferred_name, is_short_name, is_colloquial, name_provider, isolanguage
            ) VALUES (
                :iri, :alternate_name, :is_preferred_name, :is_short_name, :is_colloquial, :name_provider, :isolanguage
            ) ON CONFLICT DO NOTHING
            """, name_data
        )
        self.connection.execute(
            f"""
            INSERT INTO {self._geodb_prefix}geographical_entities (
                iri, name, asciiname, classification, possible_entity_types, coordinates, population, closest_geonames_id, iso_country_code, alternate_iso_country_codes, admin_codes, other_parent_iris
            ) VALUES (
                :iri, :name, :asciiname, :classification, :possible_entity_types, :coordinates, :population, :closest_geonames_id, :iso_country_code, :alternate_iso_country_codes, :admin_codes, :other_parent_iris
            ) ON CONFLICT DO NOTHING
            """, entity_data
        )
        def insert_country_data(country_data: dict) -> None:
            self.connection.execute(
                f"""
                INSERT INTO {self._geodb_prefix}country_data (
                    iso_code, country_name, iso_languages, neighboring_countries_iso_codes
                ) VALUES (
                    :iso_code, :country_name, :iso_languages, :neighboring_countries_iso_codes
                ) ON CONFLICT DO NOTHING
                """, country_data
            )
        if country is not None:
            insert_country_data(country)
        for alt_country in alternate_countries:
            insert_country_data(alt_country)

    def _minimize_data_for_storage(self, data: dict):
        """
        Convert geographical data into entities for information deduplication (changes data in place)
        """
        for entity in data["entities"]:
            for match_idx, match in enumerate(entity["matches"]):
                self._insert_geographical_entity(match["geographical_name"])
                entity["matches"][match_idx]["geographical_name"] = {
                    "iri": match["geographical_name"]["iri"],
                    "alternate_name": match["geographical_name"]["alternate_name"],
                    "closest_geonames_id": match["geographical_name"]["entity"]["closest_geonames_id"]
                }
            for disambiguation_idx, disambiguation in enumerate(entity["disambiguation_result"]):
                entity["disambiguation_result"][disambiguation_idx]["geographical_name"] = {
                    "iri": disambiguation["geographical_name"]["iri"],
                    "alternate_name": disambiguation["geographical_name"]["alternate_name"],
                    "closest_geonames_id": disambiguation["geographical_name"]["entity"]["closest_geonames_id"]
                }

    def upsert_sync(self, linked_address : AddressLinkingMetadata, preserve_linking_data : bool = False) -> None:
        """
        Update or insert the linked address in the storage after applying a linking step
        """
        data = encode_as_dict(linked_address.address)
        steps_data = [json.dumps(encode_as_dict(step)) for step in linked_address.applied_steps]
        id = linked_address.address.id
        bzk_field_name = linked_address.address.bzk_field_name.value
        full_address = linked_address.address.full_address
        self._minimize_data_for_storage(data)
        json_data = json.dumps(data)
        # Comment out the line that updates linking_data if we want to preserve it
        linking_data_update_guard = "-- " if preserve_linking_data else ""
        self.connection.execute(f"""
            INSERT INTO linking_metadata (id, bzk_field_name, full_address, linking_data, applied_steps, is_loaded, is_finished)
            VALUES (:id, :bzk_field_name, :full_address, :linking_data, :applied_steps, TRUE, :is_finished)
            ON CONFLICT (id, bzk_field_name) DO UPDATE SET 
                full_address = EXCLUDED.full_address,
                {linking_data_update_guard}linking_data = EXCLUDED.linking_data,
                applied_steps = EXCLUDED.applied_steps,
                is_loaded = TRUE,
                is_finished = EXCLUDED.is_finished
        """, {
            "id": id,
            "bzk_field_name": bzk_field_name,
            "full_address": full_address,
            "linking_data": json_data,
            "applied_steps": steps_data,
            "is_finished": linked_address.finished
        })




    def _expand_data_from_storage(self, data_to_exand: list[dict]):
        """
        Resolve geographical ids to corresponding data (changes data in place)
        """
        names_to_fetch : set[tuple[str, str]] = set()
        for row_to_expand in data_to_exand:
            for entity in row_to_expand["entities"]:
                for match in enumerate(entity["matches"]):
                    names_to_fetch.add((match["geographical_name"]["iri"], match["geographical_name"]["alternate_name"]))
                for disambiguation in entity["disambiguation_result"]:
                    names_to_fetch.add((disambiguation["geographical_name"]["iri"], disambiguation["geographical_name"]["alternate_name"]))

        geographical_names_rows = self.connection.execute(f"""
            SELECT * FROM {self._geodb_prefix}geographical_names_with_entities
            WHERE (iri, alternate_name) IN (
                SELECT 
                    UNNEST(:names_to_fetch_iri) AS iri, 
                    UNNEST(:names_to_fetch_alternate_name) AS alternate_name
            )
        """, {
            "names_to_fetch_iri": [iri for iri, _ in names_to_fetch],
            "names_to_fetch_alternate_name": [alt_name for _, alt_name in names_to_fetch]
         }).fetchall()
        name_data_map = {
            (row["iri"], row["alternate_name"]): _geo_row_to_dict(row) for row in geographical_names_rows
        }
        for row_to_expand in data_to_exand:
            for entity in row_to_expand["entities"]:
                for match_idx, match in enumerate(entity["matches"]):
                    name_data = name_data_map.get((match["geographical_name"]["iri"], match["geographical_name"]["alternate_name"]))
                    entity["matches"][match_idx]["geographical_name"] = name_data
                for disambiguation_idx, disambiguation in enumerate(entity["disambiguation_result"]):
                    name_data = name_data_map.get((disambiguation["geographical_name"]["iri"], disambiguation["geographical_name"]["alternate_name"]))
                    entity["disambiguation_result"][disambiguation_idx]["geographical_name"] = name_data

        

    def fetch_pending_sync(self, n = 1) -> Optional[AddressLinkingMetadata] | list[AddressLinkingMetadata]:
        """
        Fetch pending addresses for processing, e.g. for applying a linking step
        """
        rows = self.connection.execute("""
            SELECT id, bzk_field_name, full_address, data FROM linking_metadata
            WHERE is_loaded = FALSE AND is_finished = FALSE
            LIMIT :n
        """, {"n": n}).fetchall()
        if len(rows) == 0:
            return [] if n > 1 else None
        dict_reprs = []
        for row in rows:
            dict_reprs.append(json.loads(row["data"]))
        self._expand_data_from_storage(dict_reprs)
        decoded = [decode_from_dict(d, AddressLinkingMetadata) for d in dict_reprs]
        return decoded if n > 1 else decoded[0]
        
        

    def get_pending_count_sync(self) -> int:
        """
        Get the number of pending addresses for processing, e.g. for logging or monitoring purposes
        """
        self.connection.execute("""
            SELECT COUNT(*) AS pending_count FROM linking_metadata
            WHERE is_loaded = FALSE AND is_finished = FALSE
        """).fetchone()["pending_count"]

    def get_total_count_sync(self) -> int:
        """
        Get the total number of addresses in the storage, e.g. for logging or monitoring purposes
        """
        self.connection.execute("""
            SELECT COUNT(*) AS total_count FROM linking_metadata
        """).fetchone()["total_count"]
    

    async def finalize(self) -> None:
        """
        Close the storage and release any resources, e.g. file handles or database connections
        """
        super().finalize()
        self.connection.close()