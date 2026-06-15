"""
Script and utility functions to build a DuckDB database from Geonames dumps, extending it with GND dumps.

Can be run as a standalone script or imported as module to be used in other scripts or notebooks.
To import the database in another script or notebook, 
simply import the open_or_init_duckdb function and call it to get a connection to the database, 
which will be created if it does not already exist.

Most of the tables are imported from the geonames dump files (https://download.geonames.org/export/dump/),
but the script also imports geographic names from the GND (Gemeinsame Normdatei) authority files 
(https://data.dnb.de/opendata/)

To get more information about the geoname tables, consult https://download.geonames.org/export/dump/readme.txt 
The only difference in the local database is that the admin5 code is included directly 
in the main geonames table.

The GND data is imported from the authorities-gnd-geografikum_lds.ttl file.
Only entities of type gndo:TerritorialCorporateBodyOrAdministrativeUnit that link (using owl:sameAs) 
to a geonames entity are imported to the table 'gnd'.
For these entities, their preferred and variant names are imported to the gndNames table.
Some of the entities in the authority file link to geoname ids that don't exist. Some of these, but not all,
are resolved by making a request to the geonames server which triggers a redirect to the correct geoname id 
(likely these are entities for which the id is outdated somehow).
The rest of the entities with invalid geoname ids are ignored.
"""
import contextlib
import json
from pathlib import Path
import io
import csv
import rdflib
import requests
import zipfile
import gzip
import shutil
import duckdb
from typing import IO, Generator
from tqdm.auto import tqdm
import threading
from urllib.parse import urlparse
from datetime import timedelta
import time
import warnings
from SPARQLWrapper import SPARQLWrapper, JSON, __version__ as SPARQLWrapper_version
import sys
import pprint
import textwrap
import pandas as pd

DUCK_DB_PATH = "geo.duckdb"
WIKIDATA_USER_AGENT = (
    "BZKAddressLinkingDB/0.0.1 "
    "(https://github.com/ISE-FIZKarlsruhe/bzk-address-parsing; rafael.patronilo@fiz-karlsruhe.de) "
    f"Python/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
    f"SPARQLWrapper/{SPARQLWrapper_version}"
)

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
    geonames_id INTEGER, -- Semantically differs from closest_geonames_id because it is only set on geonames entities
    iso_country_code TEXT,
    alternate_iso_country_codes TEXT[] NOT NULL,
    admin_codes STRUCT(
        admin1_code TEXT, 
        admin2_code TEXT, 
        admin3_code TEXT, 
        admin4_code TEXT, 
        admin5_code TEXT
    ),
    other_parent_iris TEXT[]
);

CREATE TABLE IF NOT EXISTS geographical_names (
    iri TEXT, -- REFERENCES geographical_entities(iri), -- Constraint has bug, see https://duckdb.org/docs/current/sql/indexes#over-eager-constraint-checking-in-foreign-keys
    name TEXT,
    is_preferred_name BOOLEAN,
    is_short_name BOOLEAN,
    is_colloquial BOOLEAN,
    name_provider TEXT NOT NULL,
    isolanguage TEXT,
    PRIMARY KEY (iri, name, name_provider, isolanguage)
);

CREATE TABLE IF NOT EXISTS country_data (
    iso_code TEXT PRIMARY KEY,
    country_name TEXT,
    continent TEXT,
    geonames_id INTEGER,
    iso_languages TEXT[],
    neighboring_countries_iso_codes TEXT[]
);

CREATE VIEW IF NOT EXISTS geographical_entities_with_countries AS
SELECT 
    geographical_entities.* EXCLUDE (alternate_iso_country_codes, iso_country_code), 
    alternate_countries.alternate_countries AS alternate_countries,
    country_data as country 
FROM geographical_entities 
    JOIN country_data ON geographical_entities.iso_country_code = country_data.iso_code,
    LATERAL (
        SELECT LIST(alt_country) as alternate_countries FROM 
            (SELECT UNNEST(geographical_entities.alternate_iso_country_codes) AS code) AS alt_country_codes
            JOIN country_data AS alt_country ON alt_country.iso_code = alt_country_codes.code
    ) alternate_countries;

CREATE VIEW IF NOT EXISTS geographical_names_with_entities AS
SELECT geographical_names.* EXCLUDE (iri), geographical_entities_with_countries AS entity 
FROM geographical_names 
    JOIN geographical_entities_with_countries USING (iri);
"""

# =================================================================================
# Generic utility functions
# =================================================================================


@contextlib.contextmanager
def duckdbpbar(connection: duckdb.DuckDBPyConnection, **kwargs):
    """Context manager to display a progress bar for long-running DuckDB queries using tqdm.
    Usage:
    with duckdbpbar(connection, desc="Running long query"):
        connection.execute("SELECT ...")
    """
    # TODO the progress bar is not very accurate
    stop_signal = threading.Event()

    def progress_thread():
        pbar = tqdm(total=100, **kwargs)
        while not stop_signal.is_set():
            progress = connection.query_progress()
            if progress >= 0.0:
                pbar.n = int(progress)
                pbar.refresh()
            stop_signal.wait(1)
        pbar.n = 100
        pbar.refresh()
        pbar.close()

    thread = threading.Thread(target=progress_thread)
    thread.start()
    try:
        yield
    finally:
        stop_signal.set()
        thread.join(1)


def extract(extension: str, data: IO[bytes], dest_path: Path):
    """
    Extracts a file from a compressed archive to a destination path.
    Compression formats supported: zip, gzip, gz, or uncompressed files.
    For zip files, the archive is expected to contain a file with the same name as
    the dest_path file name.
    """
    if extension == "zip":
        with zipfile.ZipFile(data) as zf:
            zf.extract(dest_path.name, dest_path.parent)
    elif extension in ["gz", "gzip"]:
        with gzip.open(data, "rb") as f_in:
            with open(dest_path, "wb") as f_out:
                f_out.write(f_in.read())
    else:
        with open(dest_path, "wb") as f_out:
            f_out.write(data.read())


def download_file(url, dest_path, decompress=True):
    """
    Downloads a file from a url to a destination path, with an optional decompression step.
    If decompress is True, the file will be decompressed based on its extension (zip, gzip, gz)
    and the decompressed file will be saved to dest_path.
    If decompress is False, the file will be downloaded as is to dest_path.
    If the file already exists at dest_path, the function will skip downloading and
    return immediately.
    """
    if dest_path.exists():
        return
    print(f"Downloading file from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_stream = open(dest_path, "wb") if not decompress else io.BytesIO()
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") if not decompress else io.BytesIO() as dest_stream:
        for chunk in tqdm(
            response.iter_content(None),
            total=total_size,
            unit="B",
            unit_scale=True,
        ):
            dest_stream.write(chunk)
        if decompress:
            dest_stream.seek(0)
            extension = url.split(".")[-1]
            extract(extension, dest_stream, dest_path)
    print(f"Downloaded {url} to {dest_path}")

# =================================================================================
# Specific dump file handling and database initialization functions
# =================================================================================

def decompress_and_retrieve_relevant_geonames(zipfilepath, member):
    """
    Decompresses a geonames dump file and retrieves only the relevant entries based on feature class.

    **Note**: currently all entries are considered relevant and this function simply extracts the 
    specified member from the zip file, but it can be easily modified to filter entries 
    based on feature class or other criteria if needed in the future.
    """
    if not isinstance(zipfilepath, Path):
        zipfilepath = Path(zipfilepath)
    dest_path = zipfilepath.parent / member
    if dest_path.exists():
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zipfilepath) as zf:
        print(f"Extracting and filtering {member}...")
        fileinfo = next(f for f in zf.infolist() if f.filename == member)
        pbar = tqdm(
            total=fileinfo.file_size,
            desc=f"Extracting {member}",
            unit="B",
            unit_scale=True,
        )
        with (
            zf.open(fileinfo) as src,
            open(dest_path, "w", encoding="utf-8", newline="") as dst,
        ):

            def line_yielder():
                for line in src:
                    pbar.update(len(line))
                    yield line.decode("utf-8")

            csv_reader = csv.reader(line_yielder(), delimiter="\t")
            csv_writer = csv.writer(dst, delimiter="\t")
            # row format: geonameid, name, asciiname, alternatenames, latitude, longitude, feature class, feature code, country code, cc2, admin1 code, admin2 code, admin3 code, admin4 code, population, elevation, dem, timezone, modification date
            for row in csv_reader:
                csv_writer.writerow(row)
    print(f"Extracted {member} from {zipfilepath} to {dest_path}")


def fix_csv(filename):
    """Geonames country info dump is malformed as a csv and needs to be fixed before
    importing to duckdb.
    The file starts with a comment block explaining the table
    however the comment character '#' is also used in postal code formats,
    so we cannot simply use the csv comment parameter to skip comments.
    """
    lines = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.strip().startswith("#"):
                f.write(line)


def populate_entities_from_geonames(con: duckdb.DuckDBPyConnection):
    """Imports the entities from the main table in the geonames dump into the DuckDB database.

    This includes:
    - the admin5 code column which is only available in a separate dump file and 
      needs to be imported separately to the main table.
    - informal hierarchy relations, in the form of a list column containing the IRIs of parent entities 
      that are not part of the official administrative hierarchy (i.e. not derivable from the admin1-5 codes).
    """
    print("Populating main geonames table...")
    zipfile = Path("dumps/geonames/allCountries.zip")
    download_file(
        "https://download.geonames.org/export/dump/allCountries.zip",
        zipfile,
        decompress=False,
    )
    decompress_and_retrieve_relevant_geonames(zipfile, "allCountries.txt")
    with duckdbpbar(con, desc="Importing geonames main table"):
        con.execute(
        """
        INSERT OR REPLACE INTO geographical_entities
        SELECT
            'https://sws.geonames.org/' || geonameId AS iri,
            name, asciiname, feature_class || '.' || feature_code AS classification,
            list_filter(
                [
                    'Country',
                    'State',
                    'Region',
                    'District',
                    'City',
                    'Neighborhood'
                ],
                label -> CASE label
                    WHEN 'Country'      THEN (
                        (feature_class = 'A' AND feature_code IN ('TERR','PCLI','PCL','PCLF','LTER','ZN','PCLD','PCLH','PCLS','PRSH','PCLIX'))
                        OR (feature_class = 'A' AND feature_code = 'ADM1' AND country_code = 'GB')
                    )
                    WHEN 'State'        THEN (
                        feature_class = 'A' AND feature_code IN ('ADM1','ADM1H','ADMDH','ADMD')
                    )
                    WHEN 'Region'       THEN (
                        (feature_class = 'A' AND feature_code IN ('ADM1','ADM1H','ADMDH','ADMD','ADM2','ADM2H','ADM3H','ADM3','ADM4','ADM4H','ADM5'))
                        OR (feature_class = 'L' AND feature_code IN ('RGN','RGNH'))
                    )
                    WHEN 'District'     THEN (
                        feature_class = 'A' AND feature_code IN ('ADM1','ADM1H','ADMDH','ADMD','ADM2','ADM2H','ADM3H','ADM3','ADM4','ADM4H','ADM5')
                    )
                    WHEN 'City'         THEN (
                        feature_class = 'P' AND feature_code != 'PPLX'
                    )
                    WHEN 'Neighborhood' THEN (
                        feature_class = 'P'
                    )
                    ELSE false
                END
            ) AS possible_entity_types,
            struct_pack(latitude := latitude, longitude := longitude) AS coordinates, population, 
            geonameId AS closest_geonames_id, geonameId AS geonames_id,
            country_code AS iso_country_code,
            CASE 
                WHEN cc2 = '' OR cc2 IS NULL THEN ARRAY[]::TEXT[]
                ELSE string_split(cc2, ',') 
            END AS alternate_iso_country_codes,
            struct_pack(
                admin1_code := admin1_code, admin2_code := admin2_code, 
                admin3_code := admin3_code, admin4_code := admin4_code, 
                admin5_code := NULL
            ) AS admin_codes,
            ARRAY[]::TEXT[] AS other_parent_iris
        FROM read_csv_auto(
            'dumps/geonames/allCountries.txt', 
            delim='\t', 
            header=False,
            columns={
                'geonameId': 'INTEGER', 'name': 'TEXT', 'asciiname': 'TEXT', 
                'alternatenames': 'TEXT', 'latitude': 'FLOAT', 'longitude': 'FLOAT', 
                'feature_class': 'TEXT', 'feature_code': 'TEXT', 'country_code': 'TEXT', 'cc2': 'TEXT', 
                'admin1_code': 'TEXT', 'admin2_code': 'TEXT', 'admin3_code': 'TEXT', 'admin4_code': 'TEXT', 
                'population': 'BIGINT', 'elevation': 'INTEGER', 'dem': 'INTEGER', 'timezone': 'TEXT', 
                'modification_date': 'DATE'
            }
        )
        """
        )
    with duckdbpbar(con, desc="Populating official names from main table"):
        con.execute(
        """
        INSERT INTO geographical_names
        SELECT 
            'https://sws.geonames.org/' || geonameId AS iri,
            name AS name,
            TRUE AS is_preferred_name,
            NULL AS is_short_name,
            NULL AS is_colloquial,
            'https://sws.geonames.org/' AS name_provider,
            '' AS isolanguage
        FROM read_csv_auto(
            'dumps/geonames/allCountries.txt', 
            delim='\t', 
            header=False,
            columns={
                'geonameId': 'INTEGER', 'name': 'TEXT', 'asciiname': 'TEXT', 
                'alternatenames': 'TEXT', 'latitude': 'FLOAT', 'longitude': 'FLOAT', 
                'feature_class': 'TEXT', 'feature_code': 'TEXT', 'country_code': 'TEXT', 'cc2': 'TEXT', 
                'admin1_code': 'TEXT', 'admin2_code': 'TEXT', 'admin3_code': 'TEXT', 'admin4_code': 'TEXT', 
                'population': 'BIGINT', 'elevation': 'INTEGER', 'dem': 'INTEGER', 'timezone': 'TEXT', 
                'modification_date': 'DATE'
            }
        )
        ON CONFLICT (iri, name, name_provider, isolanguage) DO UPDATE SET is_preferred_name = TRUE
        """
        )
    download_file(
        "https://download.geonames.org/export/dump/adminCode5.zip",
        Path("dumps/geonames/adminCode5.txt"),
    )
    with duckdbpbar(con, desc="Importing adminCode5"):
        con.execute(
            """
        UPDATE geographical_entities
        SET admin_codes = struct_update(admin_codes, admin5_code := admin5_table.admin5_code)
        FROM read_csv_auto(
                'dumps/geonames/adminCode5.txt', delim='\t', header=False, 
                columns={'geonameId': 'INTEGER', 'admin5_code': 'TEXT'}
            ) AS admin5_table
        WHERE geographical_entities.geonames_id = admin5_table.geonameId;
        """
        )
    download_file(
        "https://download.geonames.org/export/dump/hierarchy.zip",
        Path("dumps/geonames/hierarchy.txt"),
    )
    with duckdbpbar(con, desc="Populating informal hierarchy"):
        con.execute(
            """
        UPDATE geographical_entities
        SET other_parent_iris = ([
                ('https://sws.geonames.org/' || parent_id) for parent_id in hierarchy_table.parent_ids
            ]::TEXT[])
        FROM (
            SELECT LIST(parentId) AS parent_ids, childId
            FROM read_csv_auto(
                'dumps/geonames/hierarchy.txt', delim='\t', header=False, 
                columns={'parentId': 'INTEGER', 'childId': 'INTEGER', 'type': 'TEXT'}
            )
            WHERE type != 'ADM'
            GROUP BY childId
        ) AS hierarchy_table
        WHERE geographical_entities.geonames_id = hierarchy_table.childId;
        """
        )


def populate_names_from_geonames(con: duckdb.DuckDBPyConnection):
    """
    Populates the alternate names table from the geonames dump into the DuckDB database.

    This table contains alternate names for geonames entities, which can be used for more flexible searching.
    """
    print("Populating geonames alternate names table...")
    download_file(
        "https://download.geonames.org/export/dump/alternateNames.zip",
        Path("dumps/geonames/alternateNames.txt"),
    )
    with duckdbpbar(con, desc="Importing alternate names"):
        con.execute(
            """
        INSERT OR REPLACE INTO geographical_names
        SELECT 
            'https://sws.geonames.org/' || geonameId AS iri,
            alternateName AS name,
            isPreferredName AS is_preferred_name,
            isShortName AS is_short_name,
            isColloquial AS is_colloquial,
            'https://sws.geonames.org/' AS name_provider,
            CASE 
                WHEN isolanguage IS NULL THEN ''
                ELSE isolanguage
            END AS isolanguage
        FROM read_csv_auto(
            'dumps/geonames/alternateNames.txt', delim='\t', header=False,
            columns = {
                'alternateNameId': 'INTEGER', 'geonameId': 'INTEGER', 
                'isolanguage': 'TEXT', 'alternateName': 'TEXT',
                'isPreferredName': 'BOOLEAN', 'isShortName': 'BOOLEAN', 'isColloquial': 'BOOLEAN',
                'isHistoric': 'BOOLEAN'
            }
        )
        WHERE alternateName != '' AND alternateName IS NOT NULL
        """
        )


def populate_country_data(con):
    """
    Imports the country info table from the geonames dump into the DuckDB database.

    While the countries are contained in the main geonames table, 
    this table contains additional information about countries such as 
    languages and neighbors.
    """
    print("Populating geonames country info table...")
    download_file(
        "https://download.geonames.org/export/dump/countryInfo.txt",
        Path("dumps/geonames/countryInfo.txt"),
    )
    fix_csv("dumps/geonames/countryInfo.txt")
    with duckdbpbar(con, desc="Importing country info"):
        # TODO Preserve more data?
        con.execute(
            """
            INSERT OR REPLACE INTO country_data
            SELECT
                ISO AS iso_code,
                Country AS country_name,
                Continent AS continent,
                geonameid AS geonames_id,
                CASE 
                    WHEN Languages = '' THEN NULL
                    ELSE string_split(Languages, ',') 
                END AS iso_languages,
                CASE 
                    WHEN neighbours = '' THEN NULL
                    ELSE string_split(neighbours, ',') 
                END AS neighboring_countries_iso_codes
            FROM read_csv_auto('dumps/geonames/countryInfo.txt', delim='\t', header=False,
                columns = {
                    'ISO' : 'TEXT',
                    'ISO3' : 'TEXT',
                    'ISO_Numeric' : 'INTEGER',
                    'fips' : 'TEXT',
                    'Country' : 'TEXT',
                    'Capital' : 'TEXT',
                    'Area' : 'REAL',
                    'Population' : 'BIGINT',
                    'Continent' : 'TEXT',
                    'tld' : 'TEXT',
                    'CurrencyCode' : 'TEXT',
                    'CurrencyName' : 'TEXT',
                    'Phone' : 'TEXT',
                    'Postal_Code_Format' : 'TEXT',
                    'Postal_Code_Regex' : 'TEXT',
                    'Languages' : 'TEXT',
                    'geonameid' : 'INTEGER',
                    'neighbours' : 'TEXT',
                    'EquivalentFipsCode' : 'TEXT'
                }
            )
        """
        )


@contextlib.contextmanager
def rdf_gnd_graph():
    """
    Context manager to parse gnd turtle authority file and return a rdflib Graph object.
    Parsing the turtle file takes a long time.
    """
    print("Parsing GND RDF graph... This may take several minutes.")
    gnd = rdflib.Graph()
    gnd.parse("dumps/gnd/authorities-gnd-geografikum_lds.ttl")
    try:
        yield gnd
    finally:
        gnd.close()


def repair_gnd_geoname_ids(
    con: duckdb.DuckDBPyConnection,
    gnd_names: list[tuple[int, str, bool]],
):
    """
    Handle invalid geonameIds from GND by checking if they exist in the geonames table.
    Then attempt to fix invalid geonameIds by making a request to the geonames server, 
    which triggers a redirect to the correct geonameId for some of them.

    geonameIds that cannot be fixed are removed from both gnd_matches and gnd_names.
    """
    unavailable_geoname_ids = con.execute(
        """
        SELECT unnest(?) AS id
        EXCEPT
        SELECT geonames_id FROM geographical_entities WHERE geonames_id IS NOT NULL
    """,
        parameters=[list(set(m[0] for m in gnd_names))],
    ).fetchall()
    unavailable_geoname_ids = {k[0]: None for k in unavailable_geoname_ids}
    if not unavailable_geoname_ids:
        return gnd_names
    for k in unavailable_geoname_ids.keys():
        tqdm.write(
            f"Warning: geonameId {k} from GND does not exist in geonames table; attempting to fix using geonames server..."
        )
        # make a request to geonames to check if the geonameId is valid and to trigger a potential http redirect
        # TODO investigate what the invalid ids mean and why geonames redirects them. 
        # Have these geonames entities changed id?
        # For the few I checked, this seems to be the most likely scenario
        response = requests.get(f"https://geonames.org/{k}/", allow_redirects=True)
        redirected_id = urlparse(response.url).path.split("/")[1]
        redirected_id = int(redirected_id) if redirected_id.isdigit() else redirected_id
        if response.status_code not in range(200, 300):
            tqdm.write(
                f"ERROR: geonameId {k} from GND could not be resolved; received status code {response.status_code}"
            )
        elif isinstance(redirected_id, str):
            tqdm.write(
                f"ERROR: geonameId {k} from GND could not be resolved; redirected to {response.url} (id = {redirected_id})"
            )
        elif redirected_id == k:
            tqdm.write(
                f"ERROR: geonameId {k} from GND could not be resolved; geonames server returned the same id, is the database incomplete?"
            )
        else:
            tqdm.write(
                f"\tgeonameId {k} from GND redirected to {response.url} (id = {redirected_id}); updating match"
            )
            unavailable_geoname_ids[k] = redirected_id
    new_gnd_names = []
    for i, (geonameId, name, is_preferred) in enumerate(gnd_names):
        fixed_id = unavailable_geoname_ids.get(geonameId)
        if fixed_id is not None:
            new_gnd_names.append((fixed_id, name, is_preferred))
    return new_gnd_names


def fetch_names_from_gnd(gnd) -> Generator[tuple[str, str, bool, int], None, None]:
    """
    Fetches geonameIds and names from GND RDF graph, yielding tuples of (gndUri, name, isPreferred, geonameId).
    """
    qres = gnd.query(
    """
    SELECT ?gndUri ?nameType ?name ?geonameUri WHERE {
            ?gndUri a gndo:TerritorialCorporateBodyOrAdministrativeUnit.
            ?gndUri owl:sameAs ?geonameUri FILTER (STRSTARTS(STR(?geonameUri), 'https://sws.geonames.org/')).
            ?gndUri ?nameType ?name
            FILTER (?nameType IN (
                gndo:preferredNameForThePlaceOrGeographicName, 
                gndo:variantNameForThePlaceOrGeographicName)).
    }
    """
    )
    total_estimate = 150_000  # based on past runs with some margin for new entries
    for gndUri, name_type, name, geonameUri in tqdm(
        qres, total=total_estimate, desc="Fetching GND entities"
    ):
        try:
            name = str(name)
            geonameId = urlparse(geonameUri).path.split("/")[1]
            geonameId = int(geonameId)
            preferred = (
                str(name_type).split("#")[-1]
                == "preferredNameForThePlaceOrGeographicName"
            )
            yield (str(gndUri), name, preferred, geonameId)
        except Exception as e:
            # Many urls are malformed and would cause the entire process to fail
            print(
                f"Error processing GND query result ({(gndUri, name_type, name, geonameUri)})"
            )
            exception_info = f"{type(e).__name__}: {e}"
            print(exception_info)


def populate_names_from_gnd(db_con: duckdb.DuckDBPyConnection):
    """
    Import relevant GND entities and their names into the gnd and gndNames tables in the DuckDB database.
    """
    chunk_size = 10_000
    download_file(
        "https://data.dnb.de/opendata/authorities-gnd-geografikum_lds.ttl.gz", 
        Path("dumps/gnd/authorities-gnd-geografikum_lds.ttl")
    )
    with rdf_gnd_graph() as gnd:
        gnd_names = []

        def flush():
            nonlocal gnd_names
            gnd_names = repair_gnd_geoname_ids(db_con, gnd_names)
            with duckdbpbar(db_con, desc="Inserting GND names", leave=False):
                db_con.executemany(
                    """
                    INSERT INTO geographical_names
                    SELECT 
                        'https://sws.geonames.org/' || ? as iri, 
                        ? AS name,
                        ? AS is_preferred_name,
                        NULL AS is_short_name,
                        NULL AS is_colloquial,
                        'https://d-nb.info/gnd/' AS name_provider,
                        '' AS isolanguage
                    ON CONFLICT (iri, name, name_provider, isolanguage) 
                    DO UPDATE SET is_preferred_name = is_preferred_name OR EXCLUDED.is_preferred_name; 
                    """,
                    gnd_names,
                )
            gnd_names.clear()

        for _gndUri, name, is_preferred, geonameId in fetch_names_from_gnd(gnd):
            gnd_names.append((geonameId, name, is_preferred))
            if len(gnd_names) >= chunk_size:
                flush()
        if gnd_names:
            flush()

WIKIDATA_TARGET_CLASSES = {
        "<http://www.wikidata.org/entity/Q486972>" : ["City", "Neighborhood"], # Populated place
        "<http://www.wikidata.org/entity/Q253019>" : ["Neighborhood"], # Ortsteil
        "<http://www.wikidata.org/entity/Q262166>" : ["City"], # Municipality in Germany
        "<http://www.wikidata.org/entity/Q82794>" : ["Region"] # Region
    }



def fetch_wikidata_entities():
    endpoint_url = "https://query.wikidata.org/sparql"
    
    # Grabs only entities whose direct parent is linked to geonames.
    # This is not necessarily complete but it is unlikely
    # to miss entities and it is efficient.
    sparql_template = """
    SELECT ?id ?parentGeoname ?labelEN ?labelDE ?lat ?lon ?class WHERE {
        ?parent wdt:P17 wd:Q183.
        ?parent wdt:P1566 ?parentGeoname.
        ?id wdt:P131 ?parent.
        BIND (%(wikidata_class)s AS ?class)
        ?id wdt:P31 ?class.
        OPTIONAL { ?id wdt:P1566 ?ownGeoname. }
        FILTER(!BOUND(?ownGeoname))
        OPTIONAL {?id rdfs:label ?labelEN FILTER (LANG(?labelEN) = "en")} 
        OPTIONAL {?id rdfs:label ?labelDE FILTER (LANG(?labelDE) = "de")} 
        OPTIONAL {{
            ?id wdt:P625 ?coords.
            BIND(geof:latitude(?coords) AS ?lat)
            BIND(geof:longitude(?coords) AS ?lon)
        }}
    }
    """
    paginator_statements = """
    LIMIT %(page_size)s
    OFFSET %(offset)s
    """
    page_size = 5_000
    sparql = SPARQLWrapper(endpoint_url)
    sparql.addCustomHttpHeader("User-Agent", WIKIDATA_USER_AGENT)
    sparql.setReturnFormat(JSON)
    
    for wikidata_class, entity_types in WIKIDATA_TARGET_CLASSES.items():
        page_idx = 0
        offset = 0
        page = []
        exhausted = False
        while not exhausted:
            print(f"Querying Wikidata for class {wikidata_class}, page {page_idx} (limit {page_size}, offset {offset})...")
            query = (sparql_template + paginator_statements) % dict(wikidata_class=wikidata_class, page_size=page_size, offset=offset)
            sparql.setQuery(query)
            request_timestamp = time.monotonic()
            query_success = False
            try:
                response = sparql.query().convert()
                request_timestamp = time.monotonic()
                query_success = True
            except Exception as e:
                headers = getattr(e, "headers", {})
                print(f"Query failed for page {page_idx} class {wikidata_class}: {e}")
                print("Headers:")
                pprint.pprint(dict(headers))
                print()
                print(textwrap.dedent("""
                The WikiData SPARQL Endpoint seems to be particularly strict for bots. 
                You may want to consider running the query manually at https://query.wikidata.org/
                and downloading the results as json into dumps/wikidata/
                You can then load these using the --load-wikidata-dumps option
                These are the queries you should run:
                                      
                """))
                for wikidata_class, entity_types in WIKIDATA_TARGET_CLASSES.items():
                    print(f"### For class {wikidata_class}")
                    print(textwrap.dedent(sparql_template) % dict(wikidata_class=wikidata_class))
                    print()
                print()
                print(f"Will retry respecting rate limit...")
                retry_after = headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                
            if query_success:
                bindings = response["results"]["bindings"]
                print(f"{wikidata_class} page {page}: {len(bindings)} results")

                for row in bindings:
                    entity_id      = row["id"]["value"]            if "id"            in row else None
                    parent_geoname = row["parentGeoname"]["value"] if "parentGeoname" in row else None
                    label_en       = row["labelEN"]["value"]       if "labelEN"       in row else None
                    label_de       = row["labelDE"]["value"]       if "labelDE"       in row else None
                    lat            = row["lat"]["value"]   if "lat"   in row else None
                    lon            = row["lon"]["value"]   if "lon"   in row else None

                    page.append((
                        entity_id,
                        parent_geoname,
                        label_en,
                        label_de,
                        lat,
                        lon,
                        entity_types,   # list of type strings, e.g. ["City", "Neighborhood"]
                        wikidata_class, # source class QID
                    ))
                yield page
                if len(bindings) < page_size:
                    exhausted = True
                page_idx += 1
                offset += page_size
            elapsed = time.monotonic() - request_timestamp
            if elapsed < 60: # respect rate limit of 1 request per minute
                time.sleep(60 - elapsed)
            
def load_wikidata_entities_from_manual_dumps():
    """Loads wikidata entities from manually downloaded dumps in dumps/wikidata/"""
    page_size = 10_000
    page = []
    for dump_file in tqdm(sorted(Path("dumps/wikidata/").glob("*.json")), desc="Loading wikidata dumps"):
        with open(dump_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for row in tqdm(data, desc=f"Loading {dump_file.name}", unit=" entities"):
                entity_id      = row["id"]
                parent_geoname = row["parentGeoname"]
                cls = row["class"]
                label_en       = row.get("labelEN")
                label_de       = row.get("labelDE")
                lat            = row.get("lat")
                lon            = row.get("lon")
                entity_types   = WIKIDATA_TARGET_CLASSES.get(cls, [])
                page.append((entity_id, parent_geoname,
                    label_en, label_de, lat, lon,
                    entity_types, cls ))
                if len(page) >= page_size:
                    yield page
                    page = []
    if page:
        yield page

def populate_wikidata_entities(con, load_wikidata_dumps=False):
    """ TODO change comment
    Work in progress - There are 58k "Ortsteils" in wikidata
    that do not link to a geonames of gnd entity.
    This includes at least one place mentioned in BZK corpus 
    (Sudberg, https://www.wikidata.org/wiki/Q2362997)
    We should probaly include and link these entities as well.
    """
    iterator = None
    if load_wikidata_dumps:
        iterator = load_wikidata_entities_from_manual_dumps()
    else:
        iterator = fetch_wikidata_entities()
    for page in iterator:
        page_df = pd.DataFrame(page, columns=[
            "entity_id", "parent_geoname", "label_en", "label_de",
            "lat", "lon", "entity_types", "source_class"
        ])
        con.execute(
            """
            INSERT OR REPLACE INTO geographical_entities (
                iri, classification, name, asciiname, possible_entity_types,
                coordinates, population, closest_geonames_id, geonames_id,
                iso_country_code, alternate_iso_country_codes, admin_codes, other_parent_iris
            )
                SELECT 
                    page_df.entity_id AS iri, 
                    page_df.source_class AS classification, 
                    coalesce(page_df.label_en, page_df.label_de) AS name,
                    strip_accents(coalesce(page_df.label_en, page_df.label_de)) AS asciiname,
                    page_df.entity_types AS possible_entity_types, 
                    struct_pack(latitude := page_df.lat, longitude := page_df.lon) AS coordinates,
                    NULL AS population,
                    geonames_id AS closest_geonames_id,
                    NULL AS geonames_id,
                    iso_country_code,
                    alternate_iso_country_codes,
                    admin_codes,
                    other_parent_iris
                FROM geographical_entities 
                    JOIN page_df ON geographical_entities.geonames_id = page_df.parent_geoname;
            """
        )
        # Insert english labels
        con.execute(
            """
            INSERT OR REPLACE INTO geographical_names (
                iri, name, is_preferred_name, is_short_name, is_colloquial,
                name_provider, isolanguage
            )
            SELECT 
                page_df.entity_id AS iri,
                page_df.label_en AS name,
                NULL AS is_preferred_name,
                NULL AS is_short_name,
                NULL AS is_colloquial,
                'http://www.wikidata.org/entity/' AS name_provider,
                'en' AS isolanguage
            FROM page_df
            WHERE page_df.label_en IS NOT NULL AND page_df.label_en != ''
            """
        )
        # Insert german labels
        con.execute(
            """
            INSERT OR REPLACE INTO geographical_names (
                iri, name, is_preferred_name, is_short_name, is_colloquial,
                name_provider, isolanguage
            )
            SELECT 
                page_df.entity_id AS iri,
                page_df.label_de AS name,
                NULL AS is_preferred_name,
                NULL AS is_short_name,
                NULL AS is_colloquial,
                'http://www.wikidata.org/entity/' AS name_provider,
                'de' AS isolanguage
            FROM page_df
            WHERE page_df.label_de IS NOT NULL AND page_df.label_de != ''
            """
        )
            
    

def cleanup_dump_files():
    """
    Removes the downloaded dump files to free up disk space after the database has been built.
    """
    print("Cleaning up downloaded dump files...")
    shutil.rmtree("dumps")

def compact_db():
    db_path = Path(DUCK_DB_PATH)
    compacted_path = db_path.with_suffix(".compacted.duckdb")
    if not Path(DUCK_DB_PATH).exists():
        raise FileNotFoundError(f"DuckDB database not found at {DUCK_DB_PATH}. Cannot compact non-existent database.")
    if compacted_path.exists():
        raise FileExistsError(f"Compacted database already exists at {compacted_path}. Please remove it before compacting.")
    print(f"Compacting DuckDB database at {DUCK_DB_PATH}...")
    # Copying a whole database in duckdb compacts it: https://duckdb.org/docs/lts/operations_manual/footprint_of_duckdb/reclaiming_space
    duckdb.execute(f"""
    ATTACH '{DUCK_DB_PATH}' AS old_db;
    ATTACH '{compacted_path}' AS new_db;
    COPY FROM DATABASE old_db TO new_db;
    """)
    print(f"Compaction complete! Replacing old database with compacted version...")
    db_path.unlink()
    compacted_path.rename(DUCK_DB_PATH)

def init_duckdb(cleanup=True, update=False, load_wikidata_dumps=False):
    """
    Initializes the DuckDB database by creating the necessary tables and populating them with data 
    from the geonames and GND dumps. Fails if the database already exists.

    This function will take a long time to run (up to 30 minutes).
    """
    start = time.monotonic()
    if Path(DUCK_DB_PATH).exists() and not update:
        raise FileExistsError(
            f"DuckDB database already exists at {DUCK_DB_PATH}. Please remove it before creating a new one."
        )
    con = duckdb.connect(DUCK_DB_PATH)
    con.execute("SET enable_progress_bar=true; SET enable_progress_bar_print=false;")
    print("Creating and populating DuckDB database... (this may take up to 30 minutes)")
    if not update:
        con.execute(_INIT_GEO_ENTITIES_TABLES_SQL)
    populate_entities_from_geonames(con)
    populate_country_data(con)
    populate_names_from_geonames(con)
    populate_names_from_gnd(con)
    populate_wikidata_entities(con, load_wikidata_dumps=load_wikidata_dumps)
    if cleanup:
        cleanup_dump_files()
    end = time.monotonic()
    elapsed = end - start
    print(f"Database creation complete! (Elapsed time: {timedelta(seconds=elapsed)})")
    con.close()
    compact_db()


def open_or_init_duckdb(rebuild_views=False):
    """
    Opens a connection to the DuckDB database if it exists, otherwise initializes a new database.
    """
    if rebuild_views:
        warnings.warn("Deprecated", stacklevel=2)
    if not Path(DUCK_DB_PATH).exists():
        print(f"DuckDB database not found at {DUCK_DB_PATH}. Initializing new database...")
        init_duckdb()
    return duckdb.connect(DUCK_DB_PATH, read_only=True)

def attach_or_init_duckdb(conn, name="geo_db", rebuild_views=False):
    """
    Opens a connection to the DuckDB database if it exists, otherwise initializes a new database.
    """
    if rebuild_views:
        warnings.warn("Deprecated", stacklevel=2)
    if not Path(DUCK_DB_PATH).exists():
        print(f"DuckDB database not found at {DUCK_DB_PATH}. Initializing new database...")
        init_duckdb()
    conn.execute(f"ATTACH '{DUCK_DB_PATH}' AS {name} (READ_ONLY);")
    return conn

def main(args=None):
    """
    Main function invoked when running as a standalone script.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Build DuckDB database from Geonames and GND dumps"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing database and rebuild from scratch",
    )
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Compact existing database",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Repopulate the database",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove downloaded dump files after building the database",
    )
    parser.add_argument(
        "--load-wikidata-dumps",
        action="store_true",
        help="Load Wikidata entities from manual dump files",
    )
    args = parser.parse_args(args=args)

    duckdb_path = Path(DUCK_DB_PATH)
    delete_old_db = False

    if args.compact_only:
        compact_db()
        return
    if args.rebuild and duckdb_path.exists():
        print(f"Moving existing database at {DUCK_DB_PATH}...")
        shutil.move(DUCK_DB_PATH, f"{DUCK_DB_PATH}.old")
        delete_old_db = True
    if duckdb_path.exists() and not args.update:
        print(
            f"Database already exists at {DUCK_DB_PATH}. Use --rebuild to delete and rebuild."
        )
        if args.cleanup:
            cleanup_dump_files()
    else:
        init_duckdb(cleanup=args.cleanup, update=args.update, load_wikidata_dumps=args.load_wikidata_dumps)
    if delete_old_db:
        print(f"Deleting old database backup at {DUCK_DB_PATH}.old...")
        Path(f"{DUCK_DB_PATH}.old").unlink()

if __name__ == "__main__":
    main()
