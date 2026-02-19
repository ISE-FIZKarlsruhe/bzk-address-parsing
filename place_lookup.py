import contextlib
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
import pandas as pd

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

column = GEONAMES_COLUMNS.index("admin4_code")
EXTENDED_GEONAMES_COLUMNS = GEONAMES_COLUMNS.copy()
# add admin5_code column (not included in the geonames dump to avoid breaking existing scripts)
EXTENDED_GEONAMES_COLUMNS.insert(column + 1, "admin5_code")
# Custom column to indicate the parent city for neighborhoods
EXTENDED_GEONAMES_COLUMNS.insert(column + 2, "parent_city_id")
del column

LOOKUP_COLUMNS = ["key", "name", "featureCode", "isPreferred", "sourceUri", "geonameId"]

# Lookup table paths
MAIN_LOOKUP_TABLE =  "place_lookup_tables/main_lookup_table.csv"
MAIN_DATA_TABLE = "place_lookup_tables/main_data_table.csv"
PER_CONTINENT_TABLES = "place_lookup_tables/continents/"

CONTINENT_LOOKUP_TABLE = "lookup_table.csv"
CONTINENT_DATA_TABLE = "data_table.csv"

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

_WIKIDATA_GERMAN_SPEAKING_COUNTRIES_SPARQL = """
SELECT ?wikidataId ?geonamesId ?gndId WHERE {
  ?wikidataId wdt:P31 wd:Q6256.
  FILTER NOT EXISTS {?wikidataId wdt:P31 wd:Q1221156}. # Exclude Hesse, which according to wikidata was a country from 1946 to 1949
  ?wikidataId wdt:P37 wd:Q188.
  ?wikidataId wdt:P1566 ?geonamesId. 
  OPTIONAL {?wikidataId wdt:P227 ?gndId}
}
"""

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

def read_countries_csv():
    return pd.read_csv("dumps/geonames/countryInfo.txt", 
                               comment="#", sep="\t", 
                               names=["ISO","ISO3","ISO-Numeric","fips","Country","Capital","Area(in sq km)","Population","Continent","tld","CurrencyCode","CurrencyName","Phone","Postal Code Format","Postal Code Regex","Languages","geonameid","neighbours","EquivalentFipsCode"])


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



def normalize_for_search(text: str) -> str | None:
    text = text.strip()
    text = unicodedata.normalize('NFKD', text)
    return text




GND_GET_ALL_NAMES_QUERY = """
SELECT ?gndUri ?nameType ?name ?geonameUri WHERE {
        ?gndUri a gndo:TerritorialCorporateBodyOrAdministrativeUnit.
        ?gndUri owl:sameAs ?geonameUri FILTER (STRSTARTS(STR(?geonameUri), "https://sws.geonames.org/")).
        ?gndUri gndo:geographicAreaCode ?areaCode.
        ?areaCode skos:broader+ ?countryAreaCode.
        ?countryUri gndo:geographicAreaCode ?countryAreaCode.
        ?countryUri owl:sameAs ?geonameUri FILTER (?geonameId IN %(geonameUris)s).
        ?gndUri ?nameType ?name
        FILTER (?nameType IN (
            gndo:preferredNameForThePlaceOrGeographicName, 
            gndo:variantNameForThePlaceOrGeographicName)).
}
"""

def search_gnd(geoname_country_ids):
    geoname_country_uris = ",".join(f"https://sws.geonames.org/{geonameId}/" for geonameId in geoname_country_ids)
    geoname_country_uris = "(" + geoname_country_uris + ")"
    gnd = rdflib.Graph()
    gnd.parse("dumps/gnd/authorities-gnd-geografikum_lds.ttl")
    gnd.parse("dumps/gnd/geographic-area-code.ttl")
    qres = gnd.query(GND_GET_ALL_NAMES_QUERY % {"geonameUris": geoname_country_uris})
    results = []
    for gndUri, name_type, name, geonameUri in qres:
        name = str(name)
        key = normalize_for_search(name)
        geoname_uri_parts = str(geonameUri).split("/")
        geonameId = geoname_uri_parts[-2] if geoname_uri_parts[-1] == "" else geoname_uri_parts[-1]
        preferred = str(name_type).split("#")[-1] == "preferredNameForThePlaceOrGeographicName"
        results.append((key, name, preferred, str(gndUri), int(geonameId)))
    gnd.close()
    return results

def _label_geonames_row(row):
    return pd.Series(row, index=GEONAMES_COLUMNS)

@contextlib.contextmanager
def open_continent_tables(continent):
    continent_path = Path(PER_CONTINENT_TABLES) / continent
    continent_path.mkdir(parents=True, exist_ok=True)
    write_lookup_header, write_data_header = False, False
    lookup_table_path = continent_path / CONTINENT_LOOKUP_TABLE
    data_table_path = continent_path / CONTINENT_DATA_TABLE
    if not lookup_table_path.exists():
        lookup_table_path.touch()
        write_lookup_header = True
    if not data_table_path.exists():
        data_table_path.touch()
        write_data_header = True
    main_table = open(lookup_table_path, "a", encoding="utf-8", newline="")
    main_table_writer = csv.writer(main_table)
    data_table = open(data_table_path, "a", encoding="utf-8", newline="")
    data_table_writer = csv.writer(data_table)
    try:
        if write_lookup_header:
            main_table_writer.writerow(LOOKUP_COLUMNS)
        if write_data_header:
            data_table_writer.writerow(EXTENDED_GEONAMES_COLUMNS)
        yield main_table, data_table
    finally:
        main_table.close()
        data_table.close()

def compile_tables(cleanup=False):
    main_table_path = Path(MAIN_LOOKUP_TABLE)
    continent_lookup_tables_path = Path(PER_CONTINENT_TABLES)
    main_data_table_path = Path(MAIN_DATA_TABLE)
    if not (
        main_table_path.exists() and
        continent_lookup_tables_path.exists() and
        main_data_table_path.exists()
    ):
        print("Compiling place lookup tables...")
        print("Transferring dump files of place databases...")
        retrieve_dumps()
        countries = read_countries_csv()
        german_speaking_countries = countries[countries["Languages"].str.contains("de")]
        european_countries = countries[countries["Continent"] == "EU"]
        relevant_countries = frozenset(
            european_countries["ISO"].tolist() + 
            german_speaking_countries["ISO"].tolist() + # redundant since german speaking countries are in europe, but to be sure not to miss any relevant countries
            SPECIAL_INCLUDED_COUNTRIES
        )

        print("Compiling lookup table...")
        main_table_path.parent.mkdir(parents=True, exist_ok=True)
        
        main_data_table_path.parent.mkdir(parents=True, exist_ok=True)

        main_data_rows = []
        with (
            open("dumps/geonames/adminAndCities.txt", "r", encoding='utf-8', errors='ignore') as geonames_file,
            open(main_table_path, "w", newline="", encoding="utf-8") as main_table
        ):
            main_table_writer = csv.writer(main_table)
            main_table_writer.writerow(LOOKUP_COLUMNS)
            geonames_reader = csv.reader(geonames_file, delimiter='\t')
            for row in geonames_reader:
                labeled_row = _label_geonames_row(row)
                full_feature_code = f"{labeled_row['featureClass']}.{labeled_row['featureCode']}"
                name_rows_to_write = []
                for alternate_name in labeled_row["alternateNames"].split(","):
                    search_key = normalize_for_search(alternate_name)
                    if search_key is not None and search_key != "":
                        name_rows_to_write.append([
                            search_key, alternate_name, full_feature_code,
                            labeled_row["name"] == alternate_name, # isPreferred
                            f"https://sws.geonames.org/{labeled_row['geonameId']}/", # sourceUri
                            labeled_row['geonameId']
                        ])
                if labeled_row["countryCode"] in relevant_countries:
                    # filter out places with less relevance to reduce table size
                    for name_row in name_rows_to_write:
                        main_table_writer.writerow(name_row)
                    main_data_rows.append(labeled_row)
                else:
                    with open_continent_tables(labeled_row["continent"]) as (continent_lookup_writer, continent_data_writer):
                        for name_row in name_rows_to_write:
                            continent_lookup_writer.writerow(name_row)
                        continent_data_writer.writerow(row)

            main_data_df = pd.DataFrame(main_data_rows, columns=GEONAMES_COLUMNS)
            for searchKey, name, is_preferred, gndUri, geonameId in search_gnd(german_speaking_countries["geonameid"].tolist()):
                main_data_row = main_data_df[main_data_df["geonameId"] == geonameId].iloc[0]
                full_feature_code = f"{main_data_row['feature_class']}.{main_data_row['feature_code']}"
                main_table_writer.writerow([
                    searchKey, name, full_feature_code, is_preferred,
                    gndUri, # sourceUri
                    geonameId
                ])
        
                #for gndUri, name_type, name in gnd.query(GND_GET_ALL_NAMES_QUERY % {"geonameId": geonameId}):
                #    name = str(name)
                #    name_type = str(name_type)
                #    writer.writerow([normalize_for_search(name), name, full_feature_code, name_type.split("#")[-1] == "preferredNameForThePlaceOrGeographicName", "gnd", str(gndUri), geonameId])
        #gnd.close()

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
    def __init__(self, table_path: str | Path = TABLE_PATH):
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
    

