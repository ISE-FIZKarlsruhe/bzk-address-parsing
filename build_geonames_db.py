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

DUCK_DB_PATH = "geonames.duckdb"


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


def init_geonames_table(con: duckdb.DuckDBPyConnection):
    """Imports the main table from the geonames dump into the DuckDB database.

    This includes the admin5 code column which is only available in a separate dump file and 
    needs to be imported separately to the main table."""
    print("Creating and populating main geonames table...")
    con.execute(
        """
        CREATE TABLE geonames (
            geonameId INTEGER PRIMARY KEY,
            name TEXT,
            asciiname TEXT,
            alternatenames TEXT,
            latitude REAL,
            longitude REAL,
            feature_class TEXT,
            feature_code TEXT,
            country_code TEXT,
            cc2 TEXT,
            admin1_code TEXT,
            admin2_code TEXT,
            admin3_code TEXT,
            admin4_code TEXT,
            admin5_code TEXT,
            population BIGINT,
            elevation INTEGER,
            dem INTEGER,
            timezone TEXT,
            modification_date DATE
        );
    """
    )
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
        COPY geonames(
                geonameId, name, asciiname, alternatenames, latitude, longitude, 
                feature_class, feature_code, country_code, cc2, 
                admin1_code, admin2_code, admin3_code, admin4_code, 
                population, elevation, dem, timezone, modification_date)
        FROM 'dumps/geonames/allCountries.txt' (FORMAT csv, SEP '\t', HEADER false);
        """
        )
    download_file(
        "https://download.geonames.org/export/dump/adminCode5.zip",
        Path("dumps/geonames/adminCode5.txt"),
    )
    with duckdbpbar(con, desc="Importing adminCode5"):
        con.execute(
            """
        UPDATE geonames
        SET admin5_code = admin5_table.admin5_code
        FROM read_csv_auto(
                'dumps/geonames/adminCode5.txt', delim='\t', header=False, 
                columns={'geonameId': 'INTEGER', 'admin5_code': 'TEXT'}
            ) AS admin5_table
        WHERE geonames.geonameId = admin5_table.geonameId;
        """
        )


def init_geonames_alternate_names_table(con: duckdb.DuckDBPyConnection):
    """
    Imports the alternate names table from the geonames dump into the DuckDB database.

    This table contains alternate names for geonames entities, which can be used for more flexible searching.
    """
    print("Creating and populating geonames alternate names table...")
    con.execute(
        """
    CREATE TABLE alternateNames (
        alternateNameId INTEGER PRIMARY KEY,
        geonameId INTEGER,
        isolanguage TEXT,
        alternateName TEXT,
        isPreferredName BOOLEAN,
        isShortName BOOLEAN,
        isColloquial BOOLEAN,
        isHistoric BOOLEAN,
        FOREIGN KEY (geonameId) REFERENCES geonames(geonameId)
    );
    """
    )
    download_file(
        "https://download.geonames.org/export/dump/alternateNames.zip",
        Path("dumps/geonames/alternateNames.txt"),
    )
    with duckdbpbar(con, desc="Importing alternate names"):
        con.execute(
            """
        COPY alternateNames
        FROM 'dumps/geonames/alternateNames.txt' (FORMAT csv, SEP '\t', HEADER false);
        """
        )


def init_geonames_hierarchy_table(con):
    """
    Imports the hierarchy table from the geonames dump into the DuckDB database.
    
    This table contains parent-child relationships between geonames entities. 
    This includes informal hierachy relationships such as neighborhoods of some cities. Examples:
    - https://www.geonames.org/2873589/marienfelde.html as part of https://www.geonames.org/2950159/berlin.html
    - https://www.geonames.org/5110302/brooklyn.html as part of https://www.geonames.org/5128581/new-york-city.html
    
    It also includes the official administrative hierarchy, 
    but these relations can also be derived from the admin1-5 codes in the main table.
    """
    print("Creating and populating geonames hierarchy table...")
    con.execute(
        """
        CREATE TABLE hierarchy (
            parentId INTEGER,
            childId INTEGER,
            type TEXT,
            FOREIGN KEY (parentId) REFERENCES geonames(geonameId),
            FOREIGN KEY (childId) REFERENCES geonames(geonameId)
        );
    """
    )
    download_file(
        "https://download.geonames.org/export/dump/hierarchy.zip",
        Path("dumps/geonames/hierarchy.txt"),
    )
    with duckdbpbar(con, desc="Importing geonames hierarchy"):
        con.execute(
            """
            COPY hierarchy
            FROM 'dumps/geonames/hierarchy.txt' (FORMAT csv, SEP '\t', HEADER false);
        """
        )


def init_country_info_table(con):
    """
    Imports the country info table from the geonames dump into the DuckDB database.

    While the countries are contained in the main geonames table, 
    this table contains additional information about countries such as 
    postal code format, languages, neighbors, etc.
    """
    print("Creating and populating geonames country info table...")
    con.execute(
        """
        CREATE TABLE countryInfo (
            ISO TEXT PRIMARY KEY,
            ISO3 TEXT,
            ISO_Numeric INTEGER,
            fips TEXT,
            Country TEXT,
            Capital TEXT,
            Area REAL,
            Population INTEGER,
            Continent TEXT,
            tld TEXT,
            CurrencyCode TEXT,
            CurrencyName TEXT,
            Phone TEXT,
            Postal_Code_Format TEXT,
            Postal_Code_Regex TEXT,
            Languages TEXT,
            geonameid INTEGER,
            neighbours TEXT,
            EquivalentFipsCode TEXT
        );
    """
    )
    download_file(
        "https://download.geonames.org/export/dump/countryInfo.txt",
        Path("dumps/geonames/countryInfo.txt"),
    )
    fix_csv("dumps/geonames/countryInfo.txt")
    with duckdbpbar(con, desc="Importing country info"):
        con.execute(
            """
            COPY countryInfo
            FROM 'dumps/geonames/countryInfo.txt' (FORMAT csv, SEP '\t', HEADER false);
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
    gnd_matches: list[tuple[str, int]],
    gnd_names: list[tuple[str, str, bool]],
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
        SELECT geonameId FROM geonames
    """,
        parameters=[list(set(m[1] for m in gnd_matches))],
    ).fetchall()
    _not_fixed = object()  # sentinel value to indicate geonameId has not be fixed
    unavailable_geoname_ids = {k[0]: _not_fixed for k in unavailable_geoname_ids}
    if not unavailable_geoname_ids:
        return gnd_matches, gnd_names
    for k in unavailable_geoname_ids.keys():
        tqdm.write(
            f"Warning: geonameId {k} from GND does not exist in geonames table; attempting to fix using geonames server..."
        )
        # make a request to geonames to check if the geonameId is valid and to trigger a potential http redirect
        # TODO investigate what the invalid ids mean and why geonames redirects them. Have these geonames entities changed id? Is it just closest match that geonames redirects to?
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
    gnd_uris_to_ignore = set()
    for i, (gndUri, geonameId) in enumerate(gnd_matches):
        fixed_id = unavailable_geoname_ids.get(geonameId)
        if fixed_id is _not_fixed:
            gnd_uris_to_ignore.add(gndUri)
        elif fixed_id is not None:
            gnd_matches[i] = (gndUri, fixed_id)
    if gnd_uris_to_ignore:
        gnd_matches = [m for m in gnd_matches if m[0] not in gnd_uris_to_ignore]
        gnd_names = [n for n in gnd_names if n[0] not in gnd_uris_to_ignore]
    return gnd_matches, gnd_names


def fetch_names_from_gnd(gnd) -> Generator[tuple[str, str, bool, int], None, None]:
    """
    Fetches geonameIds and names from GND RDF graph, yielding tuples of (gndUri, name, isPreferred, geonameId).
    """
    qres = gnd.query(
        """
        SELECT ?gndUri ?nameType ?name ?geonameUri WHERE {
                ?gndUri a gndo:TerritorialCorporateBodyOrAdministrativeUnit.
                ?gndUri owl:sameAs ?geonameUri FILTER (STRSTARTS(STR(?geonameUri), "https://sws.geonames.org/")).
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


def populate_gnd_tables(db_con: duckdb.DuckDBPyConnection):
    """
    Import relevant GND entities and their names into the gnd and gndNames tables in the DuckDB database.
    """
    chunk_size = 10_000
    with rdf_gnd_graph() as gnd:
        gnd_matches = []
        gnd_names = []

        def flush():
            nonlocal gnd_matches, gnd_names
            gnd_matches, gnd_names = repair_gnd_geoname_ids(
                db_con, gnd_matches, gnd_names
            )
            with duckdbpbar(db_con, desc="Inserting GND matches", leave=False):
                db_con.executemany(
                    "INSERT OR IGNORE INTO gnd (gndUri, geonameId) VALUES (?, ?)",
                    gnd_matches,
                )
            with duckdbpbar(db_con, desc="Inserting GND names", leave=False):
                db_con.executemany(
                    """
                    INSERT INTO gndNames (gndUri, name, isPreferred) VALUES (?, ?, ?)
                    ON CONFLICT (gndUri, name) 
                    DO UPDATE SET isPreferred = EXCLUDED.isPreferred 
                    WHERE NOT isPreferred AND EXCLUDED.isPreferred; 
                    """,
                    gnd_names,
                )
            gnd_matches.clear()
            gnd_names.clear()

        for gndUri, name, is_preferred, geonameId in fetch_names_from_gnd(gnd):
            gnd_matches.append((gndUri, geonameId))
            gnd_names.append((gndUri, name, is_preferred))
            if len(gnd_matches) >= chunk_size:
                flush()
        if gnd_matches:
            flush()


def init_gnd_tables(con):
    """
    Creates the gnd and gndNames tables in the DuckDB database and populates 
    them with data from the GND RDF graph.
    """
    print("Creating and populating GND tables...")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gnd (
            gndUri TEXT PRIMARY KEY,
            geonameId INTEGER,
            FOREIGN KEY (geonameId) REFERENCES geonames(geonameId)
        );
    """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gndNames (
            gndUri TEXT,
            name TEXT,
            isPreferred BOOLEAN,
            FOREIGN KEY (gndUri) REFERENCES gnd(gndUri),
            PRIMARY KEY (gndUri, name)
        );
    """
    )
    download_file(
        "https://data.dnb.de/opendata/authorities-gnd-geografikum_lds.ttl.gz",
        Path("dumps/gnd/authorities-gnd-geografikum_lds.ttl"),
    )
    populate_gnd_tables(con)
    return con

def cleanup_dump_files():
    """
    Removes the downloaded dump files to free up disk space after the database has been built.
    """
    print("Cleaning up downloaded dump files...")
    shutil.rmtree("dumps")

def init_duckdb(cleanup=True):
    """
    Initializes the DuckDB database by creating the necessary tables and populating them with data 
    from the geonames and GND dumps. Fails if the database already exists.

    This function will take a long time to run (up to 30 minutes).
    """
    start = time.monotonic()
    if Path(DUCK_DB_PATH).exists():
        raise FileExistsError(
            f"DuckDB database already exists at {DUCK_DB_PATH}. Please remove it before creating a new one."
        )
    con = duckdb.connect(DUCK_DB_PATH)
    con.execute("SET enable_progress_bar=true;")
    print("Creating and populating DuckDB database... (this may take up to 30 minutes)")
    init_geonames_table(con)
    init_geonames_alternate_names_table(con)
    init_geonames_hierarchy_table(con)
    init_country_info_table(con)
    init_gnd_tables(con)
    if cleanup:
        cleanup_dump_files()
    end = time.monotonic()
    elapsed = end - start
    print(f"Database creation complete! (Elapsed time: {timedelta(seconds=elapsed)})")
    return con


def open_or_init_duckdb():
    """
    Opens a connection to the DuckDB database if it exists, otherwise initializes a new database.
    """
    if Path(DUCK_DB_PATH).exists():
        print(f"Opening existing DuckDB database at {DUCK_DB_PATH}...")
        return duckdb.connect(DUCK_DB_PATH)
    else:
        print(f"DuckDB database not found at {DUCK_DB_PATH}. Initializing new database...")
        return init_duckdb()


def rebuild_tables(table_names):
    """
    Rebuilds the specified tables in the DuckDB database.
    Tables must be provided in order of dependency, meaning if table B can only be deleted after table A,
    then A must be listed before B in the input list.
    """
    con = duckdb.connect(DUCK_DB_PATH)
    if ("gnd" in table_names) != ("gndNames" in table_names):
        print(
            "Warning: Tables gnd and gndNames are interdependent; both tables will be rebuilt"
        )
    else:
        # Delete gndNames to avoid repeated build
        table_names = [t for t in table_names if t != "gndNames"]
    for table in table_names:
        if table not in [
            "geonames",
            "alternateNames",
            "hierarchy",
            "countryInfo",
            "gnd",
            "gndNames",
        ]:
            raise ValueError(f"Invalid table name '{table}' specified for rebuilding")
    print("Deleting current tables...")
    # Delete tables in order of dependencies to avoid issues with foreign key constraints
    # Assume input order respects dependencies (e.g. alternateNames which depends on geonames is deleted before geonames)
    for table in table_names:
        if table == "geonames":
            con.execute("DROP TABLE IF EXISTS geonames;")
        elif table == "alternateNames":
            con.execute("DROP TABLE IF EXISTS alternateNames;")
        elif table == "hierarchy":
            con.execute("DROP TABLE IF EXISTS hierarchy;")
        elif table == "countryInfo":
            con.execute("DROP TABLE IF EXISTS countryInfo;")
        elif table == "gnd" or table == "gndNames":
            con.execute("DROP TABLE IF EXISTS gndNames;")
            con.execute("DROP TABLE IF EXISTS gnd;")

    print("Rebuilding tables...")
    # Rebuild tables in reverse order of dependencies to avoid issues with foreign key constraints
    for table in reversed(table_names):
        if table == "geonames":
            init_geonames_table(con)
        elif table == "alternateNames":
            init_geonames_alternate_names_table(con)
        elif table == "hierarchy":
            init_geonames_hierarchy_table(con)
        elif table == "countryInfo":
            init_country_info_table(con)
        elif table == "gnd" or table == "gndNames":
            init_gnd_tables(con)


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
        "--cleanup",
        action="store_true",
        help="Remove downloaded dump files after building the database",
    )
    parser.add_argument(
        "--rebuild-table",
        nargs="*",
        help="Rebuild a specific table (geonames, gnd, etc.). Can be used multiple times.",
    )
    args = parser.parse_args(args=args)
    if args.rebuild and args.rebuild_table:
        parser.error("Cannot use --rebuild and --rebuild-table together")

    duckdb_path = Path(DUCK_DB_PATH)

    if args.rebuild and duckdb_path.exists():
        print(f"Removing existing database at {DUCK_DB_PATH}...")
        duckdb_path.unlink()
    if args.rebuild_table:
        if not duckdb_path.exists():
            parser.error(
                f"Database does not exist at {DUCK_DB_PATH}. Cannot rebuild tables."
            )
        rebuild_tables(args.rebuild_table)
    if duckdb_path.exists():
        print(
            f"Database already exists at {DUCK_DB_PATH}. Use --rebuild to delete and rebuild."
        )
        if args.cleanup:
            cleanup_dump_files()
    else:
        init_duckdb(cleanup=args.cleanup)


if __name__ == "__main__":
    main()
