import contextlib
from importlib.metadata import files
from pathlib import Path
import pandas as pd
import unicodedata
import utils
from io import BytesIO
import csv
import rdflib
import requests
import zipfile
import gzip
import shutil

# Non european countries (ISO 2 char codes) that are of particular relevance for the task
#because many victims emmigrated to these countries
SPECIAL_INCLUDED_COUNTRIES = [
    "US", # United States of America
    "IL" # Israel
]



# Geonames dump main columns
GEONAMES_COLUMNS = [
    "geonameId",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date"
]

_column = GEONAMES_COLUMNS.index("admin4_code")
EXTENDED_GEONAMES_COLUMNS = GEONAMES_COLUMNS.copy()
# add admin5_code column (not included in the geonames dump to avoid breaking existing scripts)
EXTENDED_GEONAMES_COLUMNS.insert(_column + 1, "admin5_code")
# Custom column to indicate the parent city for neighborhoods
EXTENDED_GEONAMES_COLUMNS.insert(_column + 2, "parent_city_id")
del _column

LOOKUP_COLUMNS = ["key", "name", "featureCode", "isPreferred", "sourceUri", "geonameId"]

# Lookup table paths
LOOKUP_TABLE_DIRECTORY = "place_lookup_tables/"
MAIN_TABLE = "main"
LOOKUP_TABLE = "lookup_table.csv"
DATA_TABLE = "data_table.csv"


#=================================================================================
# Functions and constants for downloading information from geonames, gnd and wikidata
#=================================================================================


def extract(extension : str, data : BytesIO, dest_path : Path):
    if extension == "zip":
        with zipfile.ZipFile(data) as zf:
            zf.extract(dest_path.name, dest_path.parent)
    elif extension in ["gz", "gzip"]:
        with gzip.open(data, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        with open(dest_path, 'wb') as f_out:
            f_out.write(data.read())
    

def download_file(url, dest_path):
    if dest_path.exists():
        return
    print(f"Downloading file from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    extension = url.split(".")[-1]
    extract(extension, BytesIO(response.content), dest_path)
    print(f"Downloaded {url} to {dest_path}")

def download_and_retrieve_relevant_geonames(url, member, dest_path):
    if dest_path.exists():
        return
    relevant_feature_classes = ["P", "A"] # P = populated place, A = administrative division

    print(f"Downloading archive from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    data = BytesIO(response.content)
    with zipfile.ZipFile(data) as zf:
        print(f"Extracting and filtering {member}...")
        with zf.open(member) as src, open(dest_path, "w", encoding="utf-8", newline='') as dst:
            csv_reader = csv.reader((line.decode('utf-8') for line in src), delimiter='\t')
            csv_writer = csv.writer(dst, delimiter='\t')
            for row in csv_reader: # row format: geonameid, name, asciiname, alternatenames, latitude, longitude, feature class, feature code, country code, cc2, admin1 code, admin2 code, admin3 code, admin4 code, population, elevation, dem, timezone, modification date
                if row[6] not in relevant_feature_classes: # feature class is in the 7th column (index 6)
                    continue
                #row[-1] = row[-1].rstrip("\n\r") # remove newlines from the end of the last column to avoid issues with csv writing
                csv_writer.writerow(row)
    print(f"Downloaded {url} to {dest_path}")

def retrieve_dumps():
    """
    Downloads the necessary dumps for place lookup.
    """
    download_file("https://data.dnb.de/opendata/authorities-gnd-geografikum_lds.ttl.gz", Path("dumps/gnd/authorities-gnd-geografikum_lds.ttl"))
    download_file("https://d-nb.info/standards/vocab/gnd/geographic-area-code.ttl", Path("dumps/gnd/geographic-area-code.ttl"))
    # There is no geonames dump specifically for all admin codes, so instead download the allCountries dump and filter for relevant feature classes (P and A) which include populated places and administrative divisions
    #download_and_retrieve_relevant_geonames("https://download.geonames.org/export/dump/allCountries.zip", "allCountries.txt", Path("dumps/geonames/adminAndCities.txt"))
    download_file("https://download.geonames.org/export/dump/hierarchy.zip", Path("dumps/geonames/hierarchy.txt"))
    download_file("https://download.geonames.org/export/dump/countryInfo.txt", Path("dumps/geonames/countryInfo.txt"))
    download_file("https://download.geonames.org/export/dump/adminCode5.zip", Path("dumps/geonames/adminCode5.txt"))

def normalize_for_search(text: str) -> str | None:
    text = text.strip()
    text = unicodedata.normalize('NFKD', text)
    return text

GND_GET_ALL_NAMES_QUERY = """
SELECT ?gndUri ?nameType ?name ?geonameUri WHERE {
        ?gndUri a gndo:TerritorialCorporateBodyOrAdministrativeUnit.
        ?countryUri owl:sameAs <https://sws.geonames.org/%(geonamesId)s>.
        ?countryUri gndo:geographicAreaCode ?countryAreaCode.
        ?gndUri (gndo:geographicAreaCode / skos:broader*) ?countryAreaCode.
        ?gndUri owl:sameAs ?geonameUri FILTER (STRSTARTS(STR(?geonameUri), "https://sws.geonames.org/")).
        ?gndUri ?nameType ?name
        FILTER (?nameType IN (
            gndo:preferredNameForThePlaceOrGeographicName, 
            gndo:variantNameForThePlaceOrGeographicName)).
}
"""

@contextlib.contextmanager
def _rdf_gnd_graph():
    gnd = rdflib.Graph()
    gnd.parse("dumps/gnd/authorities-gnd-geografikum_lds.ttl")
    gnd.parse("dumps/gnd/geographic-area-code.ttl")
    try:
        yield gnd
    finally:
        gnd.close()

def search_gnd(gnd, geonames_country_id):
    qres = gnd.query(GND_GET_ALL_NAMES_QUERY % {"geonamesId": geonames_country_id})
    for gndUri, name_type, name, geonameUri in qres:
        name = str(name)
        geoname_uri_parts = str(geonameUri).split("/")
        geonameId = geoname_uri_parts[-2] if geoname_uri_parts[-1] == "" else geoname_uri_parts[-1]
        preferred = str(name_type).split("#")[-1] == "preferredNameForThePlaceOrGeographicName"
        yield (name, preferred, str(gndUri), geonameId)

def read_countries_csv():
    skiprows = 0
    # Count comment lines to skip
    # Problem: opens file twice (count comments, read_csv). There might be better ways to skip the comments.
    # The pd.read_csv comment parameter DOES NOT work because it will detect comments in the middle of rows and '#' is used is postal code formats
    with open("dumps/geonames/countryInfo.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                skiprows += 1
            else: break
    return pd.read_csv("dumps/geonames/countryInfo.txt", 
                               skiprows=skiprows, sep="\t", keep_default_na=False,
                               names=["ISO","ISO3","ISO-Numeric","fips","Country","Capital","Area","Population","Continent","tld","CurrencyCode","CurrencyName","Phone","Postal Code Format","Postal Code Regex","Languages","geonameid","neighbours","EquivalentFipsCode"])

def read_admin5_csv():
    return pd.read_csv("dumps/geonames/adminCode5.txt", 
                               sep="\t",  keep_default_na=False,
                               names=["geonameId", "admin5_code"])

def read_hierarchy_csv():
    return pd.read_csv("dumps/geonames/hierarchy.txt", 
                               sep="\t",  keep_default_na=False,
                               names=["parentId", "childId", "type"])

@contextlib.contextmanager
def _open_csv_writer(path, write_header):
    if write_header is None:
        assert path.exists(), f"File {path} does not exist and no header provided for creation"
    else:
        path.touch()
    f = open(path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if write_header is not None:
        writer.writerow(write_header)
    try:
        yield writer
    finally:
        f.close()

@contextlib.contextmanager
def _open_table_files(table_name):
    table_folder = Path(LOOKUP_TABLE_DIRECTORY) / table_name
    lookup_file = table_folder / LOOKUP_TABLE
    data_file = table_folder / DATA_TABLE
    if table_folder.exists():
        lookup_writer = _open_csv_writer(lookup_file, None)
        data_writer = _open_csv_writer(data_file, None)
    else:
        table_folder.mkdir(parents=True, exist_ok=True)
        lookup_writer = _open_csv_writer(lookup_file, LOOKUP_COLUMNS)
        data_writer = _open_csv_writer(data_file, GEONAMES_COLUMNS)
    
    try:
        yield lookup_writer.__enter__(), data_writer.__enter__()
    finally:
        lookup_writer.__exit__(None, None, None)
        data_writer.__exit__(None, None, None)

class _TableFileKeeper(contextlib.AbstractContextManager):
    def __init__(self):
        self.current = None
        self.current_cm = None
        self.current_cm_inner = None

    def __enter__(self):
        return super().__enter__()
    
    def open_table_files(self, table_name):
        if self.current != table_name:
            if self.current_cm is not None:
                self.current_cm.__exit__(None, None, None)
            self.current_cm = _open_table_files(table_name)
            self.current_cm_inner = self.current_cm.__enter__()
            self.current = table_name
        return self.current_cm_inner
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.current_cm is not None:
            self.current_cm.__exit__(exc_type, exc_value, traceback)
        return super().__exit__(exc_type, exc_value, traceback)

def _label_geonames_row(row : list[str]) -> pd.Series:
    return pd.Series(row, index=GEONAMES_COLUMNS)

def _initalize_tables(countries, main_table_countries):
    with (
            open("dumps/geonames/adminAndCities.txt", "r", encoding='utf-8', errors='ignore') as geonames_file,
            _open_table_files("main") as (main_lookup_writer, main_data_writer),
            _TableFileKeeper() as country_table_keeper
        ):
        geonames_reader = csv.reader(geonames_file, delimiter='\t')
        for row in geonames_reader:
            labeled_row = _label_geonames_row(row)
            name_rows = []
            full_feature_code = f"{labeled_row['feature_class']}.{labeled_row['feature_code']}"
            for alternate_name in labeled_row["alternatenames"].split(","):
                search_key = normalize_for_search(alternate_name)
                if search_key is not None and search_key != "":
                    name_rows.append([
                        search_key, alternate_name, full_feature_code,
                        labeled_row["name"] == alternate_name, # isPreferred
                        f"https://sws.geonames.org/{labeled_row['geonameId']}/", # sourceUri
                        labeled_row['geonameId']
                    ])
            if (labeled_row["geonameId"] in countries["geonameid"] or
                labeled_row["country_code"] == "" or
                labeled_row["country_code"] in main_table_countries):
                # main table will contain places in the most relevant countries,
                #country names or places with no country
                main_lookup_writer.writerows(name_rows)
                main_data_writer.writerow(row)
            if labeled_row["country_code"] != "":
                country_lookup_writer, country_data_writer = country_table_keeper.open_table_files(labeled_row["country_code"])
                country_lookup_writer.writerows(name_rows)
                country_data_writer.writerow(row)

def _extend_lookup_with_gnd(target_countries):
    data_table_path = Path(LOOKUP_TABLE_DIRECTORY, MAIN_TABLE, DATA_TABLE)
    lookup_table_path = Path(LOOKUP_TABLE_DIRECTORY, MAIN_TABLE, LOOKUP_TABLE)
    data_table = pd.read_csv(data_table_path, keep_default_na=False)
    #orig_lookup_table = pd.read_csv(lookup_table_path, keep_default_na=False)
    
    with (
        _open_csv_writer(lookup_table_path, None) as main_lookup_writer,
        _rdf_gnd_graph() as gnd
    ):
        for _, target_country_row in target_countries.iterrows():
            #exisiting_country_name_rows = orig_lookup_table[orig_lookup_table["geonameId"] == target_country_row["geonameid"]]
            #already_used = set(exisiting_country_name_rows["key"].values)
            for name, is_preferred, gndUri, geonameId in search_gnd(gnd, target_country_row["geonameid"]):
                searchKey = normalize_for_search(name)
                main_data_row = data_table[data_table["geonameId"] == geonameId]
                if len(main_data_row) == 0:
                    continue
                main_data_row = main_data_row.iloc[0]
                full_feature_code = f"{main_data_row['feature_class']}.{main_data_row['feature_code']}"
                main_lookup_writer.writerow([
                    searchKey, name, full_feature_code, is_preferred, gndUri, geonameId
                ])

def _extend_data_tables():
    # This function extends data tables with admin5 codes and parent city for city sections
    admin5_table = read_admin5_csv()
    hierarchy_table = read_hierarchy_csv()
    for table_folder in Path(LOOKUP_TABLE_DIRECTORY).iterdir():
        data_table_path = table_folder / DATA_TABLE
        data_table = pd.read_csv(data_table_path, keep_default_na=False)
        data_table = data_table.merge(admin5_table, on="geonameId", how="left")
        pplx_rows = data_table[data_table["feature_code"] == "PPLX"]
        for rid, row in pplx_rows.iterrows():
            possible_parent_ids = hierarchy_table[hierarchy_table["childId"] == row["geonameId"]][["parentId"]]
            possible_parents = data_table.merge(possible_parent_ids, 
                                                left_on="geonameId", 
                                                right_on="parentId", how="inner")
            possible_parent_cities = possible_parents[
                (possible_parents["feature_class"] == "P") & 
                (possible_parents["feature_code"] != "PPLX")
            ]
            if len(possible_parent_cities) == 1:
                data_table.loc[rid, "parent_city_id"] = possible_parent_cities.iloc[0]["geonameId"]
        data_table.to_csv(data_table_path, index=False)



def compile_tables(cleanup=False):
    tables_path = Path(LOOKUP_TABLE_DIRECTORY)
    if True: #not tables_path.exists():
        print("Compiling place lookup tables...")
        print("Transferring dump files of place databases...")
        retrieve_dumps()
        print("Initializing lookup and data tables with geonames data (main tables and for each country)...")
        countries = read_countries_csv()
        german_speaking_countries = countries[countries["Languages"].str.contains("de")]
        european_countries = countries[countries["Continent"] == "EU"]
        main_table_countries = frozenset(
            european_countries["ISO"].tolist() + 
            german_speaking_countries["ISO"].tolist() + # redundant since german speaking countries are in europe, but to be sure not to miss any relevant countries
            SPECIAL_INCLUDED_COUNTRIES
        )
        #_initalize_tables(countries, main_table_countries)
        print("Extending lookup tables with GND for german speaking countries...")
        #_extend_lookup_with_gnd(german_speaking_countries)
        print("Extending data table with new information...")
        _extend_data_tables()
        print("Complete!")

        
            
            
        

FEATURE_CODE_HIERARCHY = [
    # countries, equivalent or above levels
    {'A.TERR', 'A.PCLI', 'A.PCL', 'A.PCLF', 'A.LTER', 'A.ZN', 'A.PCLD', 'A.PCLH', 'A.PCLS', 'A.PRSH', 'A.PCLIX'}, 
    # first-level administrative divisions (e.g. states in the US, Bundesländer in Germany) and informal divisions
    {'A.ADM1', 'A.ADM1H', 'A.ADMDH', 'A.ADMD'}, 
    # second-level administrative divisions
    {'A.ADM2', 'A.ADM2H'}, 
    # third-level administrative divisions
    {'A.ADM3H', 'A.ADM3'}, 
    # fourth-level administrative divisions
    {'A.ADM4', 'A.ADM4H'}, 
    # fifth-level administrative divisions
    {'A.ADM5'}, 
    # populated places (cities, towns, villages, etc.).
    {'P.PPLA2', 'P.PPLA3', 'P.PPLA', 'P.PPLR', 'P.PPL', 'P.PPLA5', 'P.STLMT', 'P.PPLQ', 'P.PPLS', 'P.PPLG', 'P.PPLW', 'P.PPLC', 'P.PPLH', 'P.PPLL', 'P.PPLCH', 'P.PPLF', 'P.PPLA4'}, 
    # section of populated place (neighborhoods, quarters, etc.)
    {'P.PPLX'}
]

def _to_feature_codes(key : str) -> set[str]:
        if key == "Country":
            return FEATURE_CODE_HIERARCHY[0]
        elif key == "State":
            return FEATURE_CODE_HIERARCHY[1]
        elif key in ["Region", "District"]:
            # Depending on the country, admin 1 might be a district
            return {code for level in FEATURE_CODE_HIERARCHY[1:6] for code in level}
        elif key == "City":
            return FEATURE_CODE_HIERARCHY[6]
        elif key == "Neighborhood":
            return FEATURE_CODE_HIERARCHY[7]
        else:
            raise ValueError(f"Unknown feature code key: {key}")

class PlaceLookupTable:
    def __init__(self, table_path: str | Path):
        compile_tables(table_path)
        self.table = pd.read_csv(table_path, keep_default_na=False)
        self.search_col = self.table["key"]
        feature_codes = self.table["featureCode"].unique()
        for feature_code in feature_codes:
            assert any(feature_code in level for level in FEATURE_CODE_HIERARCHY), f"Feature code {feature_code} in table is not in the defined hierarchy"
        self.table_per_level = []
        for level in FEATURE_CODE_HIERARCHY:
            self.table_per_level.append(self.table[self.table["featureCode"].isin(level)])
    
    
    
    def lookup(self, place_name : str, topk : int = 5, max_distance=3, feature_codes=None) -> pd.DataFrame:
        if isinstance(feature_codes, str):
            feature_codes = _to_feature_codes(feature_codes)
        place_name = normalize_for_search(place_name)
        matches_per_level = []
        for table in reversed(self.table_per_level): # reverse to prioritize higher levels in the hierarchy
            distances = table["key"].apply(
                lambda key: utils.levenshtein(place_name, key, case_insensitive=True, max_distance=max_distance)
                ).nsmallest(topk, keep="all")
            distances = distances[distances <= max_distance]
            matches = table.loc[distances.index].copy()
            matches["distance"] = distances
            matches_per_level.append(matches)
        def sort_key(df):
            if df.empty:
                return (float('inf'), 1)
            else:
                best_match = df.loc[df["distance"].idxmin()]
                preferred = 0 if best_match["isPreferred"] else 1
                return (best_match["distance"], preferred)
        matches_per_level.sort(key=sort_key)
        matches = pd.concat(matches_per_level)
        if feature_codes is not None:
            matches = matches[matches["featureCode"].isin(feature_codes)]
        return matches
    

