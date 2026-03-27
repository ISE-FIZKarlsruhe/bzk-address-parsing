# TODO rewrite everything
from pathlib import Path
import pandas as pd
import utils


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
        #compile_tables(table_path) TODO
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
        #place_name = normalize_for_search(place_name) TODO 
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
    

