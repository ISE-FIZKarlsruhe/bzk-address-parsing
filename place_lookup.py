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
    # There is no geonames dump specifically for all admin codes, so instead download the allCountries dump and filter for relevant feature classes (P and A) which include populated places and administrative divisions
    download_and_retrieve_relevant_geonames("https://download.geonames.org/export/dump/allCountries.zip", "allCountries.txt", Path("dumps/geonames/adminAndCities.txt"))

def normalize_for_search(text: str) -> str | None:
    text = text.strip()
    text = unicodedata.normalize('NFKD', text)
    return text

TABLE_PATH = "dumps/place_lookup_tables/place_lookup_table.csv"


GND_GET_ALL_NAMES_QUERY = """
SELECT ?gndUri ?nameType ?name WHERE {
        ?gndUri owl:sameAs <https://sws.geonames.org/%(geonameId)s> .
        ?gndUri ?nameType ?name
        FILTER (?nameType IN (
            gndo:preferredNameForThePlaceOrGeographicName, 
            gndo:variantNameForThePlaceOrGeographicName)).
}
"""

def compile_place_lookup_table(table_path : str | Path = TABLE_PATH):
    table_path = Path(table_path)
    if not table_path.exists():
        print("Compiling place lookup tables...")
        print("Transferring dump files of place databases...")
        retrieve_dumps()

        #print("Loading GND dump...")
        #gnd = rdflib.Graph()
        #gnd.parse("dumps/gnd/authorities-gnd-geografikum_lds.ttl")

        print("Compiling lookup table...")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open("dumps/geonames/adminAndCities.txt", "r", encoding='utf-8', errors='ignore') as geonames_file, open(table_path, "w", newline="", encoding="utf-8") as dest_file:
            writer = csv.writer(dest_file)
            writer.writerow(["key", "name", "featureCode", "isPreferred", "source", "sourceId", "geonameId"])
            geonames_reader = csv.reader(geonames_file, delimiter='\t')
            for row in geonames_reader:
                geonameId, name, asciiname, alternatenames, latitude, longitude, feature_class, feature_code, country_code, cc2, admin1_code, admin2_code, admin3_code, admin4_code, population, elevation, dem, timezone, modification_date = row
                if population == "" or int(population) < 1500: # filter out places with population less than 1000 to reduce noise and table size
                    continue
                full_feature_code = f"{feature_class}.{feature_code}"
                for alternate_name in alternatenames.split(","):
                    search_key = normalize_for_search(name)
                    if search_key is not None and search_key != "":
                        writer.writerow([normalize_for_search(alternate_name), alternate_name, full_feature_code, name == alternate_name, "geonames", geonameId, geonameId])
                #for gndUri, name_type, name in gnd.query(GND_GET_ALL_NAMES_QUERY % {"geonameId": geonameId}):
                #    name = str(name)
                #    name_type = str(name_type)
                #    writer.writerow([normalize_for_search(name), name, full_feature_code, name_type.split("#")[-1] == "preferredNameForThePlaceOrGeographicName", "gnd", str(gndUri), geonameId])
        #gnd.close()

class PlaceLookupTable:
    def __init__(self, table_path: str | Path = TABLE_PATH):
        compile_place_lookup_table(table_path)
        self.table = pd.read_csv(table_path, keep_default_na=False)
        self.search_col = self.table["key"]
        feature_codes = self.table["featureCode"].unique()
        self.code_hierarchy = [
            # countries, equivalent or above levels
            {code for code in feature_codes if code.startswith("A") and not code.startswith("A.ADM")},
            # first-level administrative divisions (e.g. states in the US, Bundesländer in Germany) and informal divisions
            {code for code in feature_codes if code.startswith("A.ADM1") or code.startswith("A.ADMD")},
            # second-level administrative divisions
            {code for code in feature_codes if code.startswith("A.ADM2")},
            # third-level administrative divisions
            {code for code in feature_codes if code.startswith("A.ADM3")},
            # fourth-level administrative divisions
            {code for code in feature_codes if code.startswith("A.ADM4")},
            # fifth-level administrative divisions
            {code for code in feature_codes if code.startswith("A.ADM5")},
            # populated places
            {code for code in feature_codes if code.startswith("P") and code != "P.PPLX"},
            # section of populated place
            {"P.PPLX"}
        ]
        self.table_per_level = []
        for level in self.code_hierarchy:
            self.table_per_level.append(self.table[self.table["featureCode"].isin(level)])
    
    def _get_feature_codes(self, key : str) -> pd.DataFrame:
        if key == "Country":
            return self.code_hierarchy[0]
        elif key == "State":
            return self.code_hierarchy[1]
        elif key in ["Region", "District"]:
            # Depending on the country, admin 1 might be a district
            return {code for level in self.code_hierarchy[1:6] for code in level}
        elif key == "City":
            return self.code_hierarchy[6]
        elif key == "Neighborhood":
            return self.code_hierarchy[7]
        else:
            raise ValueError(f"Unknown feature code key: {key}")
    
    def lookup(self, place_name : str, topk : int = 5, max_distance=3, feature_codes=None) -> pd.DataFrame:
        if isinstance(feature_codes, str):
            feature_codes = self._get_feature_codes(feature_codes)
        place_name = normalize_for_search(place_name)
        matches_per_level = []
        for table in reversed(self.table_per_level): # reverse to prioritize higher levels in the hierarchy
            distances = table["key"].apply(
                lambda key: utils.levenshtein(place_name, key, case_insensitive=True, max_distance=max_distance)
                ).nsmallest(topk)
            distances = distances[distances <= max_distance]
            matches = table.loc[distances.index].copy()
            matches["distance"] = distances
            matches_per_level.append(matches[matches["distance"] <= max_distance])
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
    

