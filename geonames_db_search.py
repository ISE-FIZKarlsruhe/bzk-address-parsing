"""
Classes related to matching extracted place names to place names on the database.
"""

# TODO rewrite everything
from pathlib import Path
import pandas as pd
import utils
import duckdb
import build_geonames_db
from typing import Optional, Literal, LiteralString
import contextlib
import enum
import dataclasses
import textwrap

@dataclasses.dataclass
class TextMatchingAlgorithmProperties:
    name : LiteralString
    return_type : Literal["similarity", "distance"]
    supports_cutoff : bool = False

class TextMatchingAlgorithm(TextMatchingAlgorithmProperties, enum.Enum):
    # Text matching algorithms supported by duckdb: https://duckdb.org/docs/stable/sql/functions/text#text-similarity-functions
    LEVENSHTEIN = "levenshtein", "distance"
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein", "distance"
    HAMMING = "hamming", "distance"
    JACCARD = "jaccard", "similarity"
    JARO = "jaro_similarity", "similarity", True
    JARO_WINKLER = "jaro_winkler_similarity", "similarity", True


@dataclasses.dataclass
class SearchQueryParameters:
    topk : int = 5
    threshold : int | float = 3
    text_matching_algorithm : TextMatchingAlgorithm = TextMatchingAlgorithm.LEVENSHTEIN

    def build_query(self, filter_feature_codes : bool) -> str:
        # avoid SQL injection by validating type of the parameters before using them to construct the query
        assert isinstance(self.text_matching_algorithm, TextMatchingAlgorithm)
        assert isinstance(self.topk, int) and self.topk > 0
        assert isinstance(self.threshold, (int, float))

        if self.text_matching_algorithm.supports_cutoff:
            text_match_invocation = f"{self.text_matching_algorithm.name}(key, ?, {self.threshold})"
        else:
            text_match_invocation = f"{self.text_matching_algorithm.name}(key, ?)"

        if self.text_matching_algorithm.return_type == "distance":
            threshold_comparison_operator = "<="
            order_direction = "ASC"
        else:
            threshold_comparison_operator = ">="
            order_direction = "DESC"
        
        if filter_feature_codes:
            feature_code_filter = "AND\n    featureCode IN (?)"
        else:
            feature_code_filter = ""

        return textwrap.dedent(f"""
            SELECT *, {text_match_invocation} AS score
            FROM geonames
            WHERE 
                score {threshold_comparison_operator} {self.threshold} {feature_code_filter}
            ORDER BY score {order_direction}
            LIMIT {self.topk}
        """).strip()
        

class GeonamesSearch(contextlib.AbstractContextManager):
    def __init__(
            self, 
            conn : Optional[duckdb.DuckDBPyConnection] = None, 
            topk : int = 5, 
            threshold : int = 3,
            text_matching_algorithm : TextMatchingAlgorithm = TextMatchingAlgorithm.LEVENSHTEIN
        ):
        self.topk = topk
        self.threshold = threshold
        if isinstance(text_matching_algorithm, str):
            text_matching_algorithm = TextMatchingAlgorithm[text_matching_algorithm]
        elif not isinstance(text_matching_algorithm, TextMatchingAlgorithm):
            raise ValueError(f"Invalid text matching algorithm: {text_matching_algorithm}")
        self.distance_algorithm = text_matching_algorithm
        self.conn = conn or build_geonames_db.open_or_init_duckdb()
    
    def search_closest_matches(self, entities: list[tuple[str, str]], include_gnd_names : bool = True):
        # TODO implement search using duckdb and levenshtein distance
        pass
    
    def close(self):
        self.conn.close()

    def __enter__(self):
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return super().__exit__(exc_type, exc_value, traceback)
    
    def __del__(self):
        self.close()
    


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
    

