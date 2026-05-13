import modules.geonames_db_search as geonames_db_search
import argparse
import re
import pandas as pd
from typing import Iterable
from pathlib import Path
from tqdm.contrib.logging import tqdm_logging_redirect as tqdm
import traceback
import dataclasses
import logging


PLACE_COLS = ['ApplicantBirthPlace', 'ApplicantCurrentAddress', 'VictimBirthPlace', 'VictimDeathPlace', 'VictimCurrentAddress']
PLACE_COLS_PREFIX = {
    'ApplicantBirthPlace': 'abp',
    'ApplicantCurrentAddress': 'aca',
    'VictimBirthPlace': 'vbp',
    'VictimDeathPlace': 'vdp',
    'VictimCurrentAddress': 'vca'
}
REGEX = re.compile(r"^(?P<City>\w+)(\s*/\s*(?P<Country>\w+))?$")

@dataclasses.dataclass
class Stats:
    addresses : int = 0
    parsed : int = 0
    corrected : int = 0

class BasicRegexAddressParser:
    def __init__(self, fallback_parser = None):
        self.fallback_parser = fallback_parser
        

    def parse_addresses(self, addresses : Iterable[str]) -> list[dict[str, str]]:
        if isinstance(addresses, pd.Series):
            matches = addresses.str.extract(REGEX).to_dict(orient="records")
        else:    
            regex_matches = [REGEX.fullmatch(address) for address in addresses]
            matches = [
                match.groupdict() if match is not None else None 
                for match in regex_matches]
        if self.fallback_parser is not None:
            unmatched_addresses = [address for address, match in zip(addresses, matches) if match is None]
            fallback_matches = self.fallback_parser.parse_addresses(unmatched_addresses)
            i = 0
            for match in fallback_matches:
                while matches[i] is not None:
                    i += 1
                matches[i] = match
        return matches

def parse_and_correct(
        df : pd.DataFrame, 
        address_parser,
        search_db : geonames_db_search.GeonamesSearch, 
        stats : Stats
    ):
    """
    Parse and correct location fields.
    
    Returns a dictionary with one DataFrame per location field, containing:
    - {prefix}_raw: original raw address
    - {prefix}_city_regex, {prefix}_country_regex: regex extraction results
    - {prefix}_city_match, {prefix}_country_match: regex values where unambiguous match exists
    - {prefix}_city_corrected, {prefix}_country_corrected: corrected name from database
    - {prefix}_city_distance, {prefix}_country_distance: cleaned edit distance
    - {prefix}_city_is_abbrev, {prefix}_country_is_abbrev: whether match was via abbreviation
    """
    result_dfs = {}
    
    for field_name, prefix in PLACE_COLS_PREFIX.items():
        raw_addresses = df[field_name].fillna("").astype(str)
        stats.addresses += len(raw_addresses)
        
        # Parse addresses using regex/LLM
        parsed_list = address_parser.parse_addresses(raw_addresses)
        parsed_df = pd.DataFrame(parsed_list, index=raw_addresses.index, dtype=str)
        parsed_df = parsed_df[["City", "Country"]]
        
        # Initialize result columns for this field
        result_columns = {
            f"{prefix}_raw": raw_addresses,
            f"{prefix}_city_regex": parsed_df["City"],
            f"{prefix}_country_regex": parsed_df["Country"],
            f"{prefix}_city_match": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_country_match": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_city_corrected": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_country_corrected": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_city_final": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_country_final": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_city_edit_distance": pd.Series(dtype='Int64', index=raw_addresses.index),
            f"{prefix}_country_edit_distance": pd.Series(dtype='Int64', index=raw_addresses.index),
            f"{prefix}_city_is_abbrev": pd.Series(dtype=bool, index=raw_addresses.index),
            f"{prefix}_country_is_abbrev": pd.Series(dtype=bool, index=raw_addresses.index),
            f"{prefix}_city_in_same_country" : pd.Series(dtype='boolean', index=raw_addresses.index),
            f"{prefix}_country_status": pd.Series(dtype=object, index=raw_addresses.index),
            f"{prefix}_city_status": pd.Series(dtype=object, index=raw_addresses.index)
        }
        
        # Search and correct countries first (to use as hints for city search)
        country_hints = [None] * len(parsed_df)
        country_db_matches = search_db.link_entities(parsed_df["Country"])
        
        for i, (idx, row_matches) in enumerate(zip(parsed_df.index, country_db_matches)):
            if len(row_matches) == 1:
                match_row = row_matches.iloc[0]
                corrected_country = match_row["alternateName"]
                regex_country = parsed_df.at[idx, "Country"]
                edit_distance = int(match_row["cleaned_distance"])
                is_abbrev = bool(match_row["may_be_abbreviation"])
                if edit_distance > search_db.threshold and not is_abbrev:
                    logging.error(f"High edit distance for country match: {regex_country} -> {corrected_country} (distance={edit_distance}, abbrev={is_abbrev}, filename={df.at[idx, 'filename']})")
                    result_columns[f"{prefix}_country_status"].at[idx] = f"POST_PROCESSING_ERROR"
                else:
                    result_columns[f"{prefix}_country_match"].at[idx] = regex_country
                    if regex_country and regex_country != corrected_country:
                        stats.corrected += 1
                        result_columns[f"{prefix}_country_corrected"].at[idx] = corrected_country
                        if edit_distance == 0:
                            result_columns[f"{prefix}_country_status"].at[idx] = "EXACT_MATCH_CORRECTED"
                        elif is_abbrev:
                            result_columns[f"{prefix}_country_status"].at[idx] = "ABBREVIATION_MATCH_CORRECTED"
                        else:
                            result_columns[f"{prefix}_country_status"].at[idx] = f"FUZZY_{edit_distance}_CORRECTED"
                    else:
                        result_columns[f"{prefix}_country_status"].at[idx] = "EXACT_MATCH"
                
                result_columns[f"{prefix}_country_edit_distance"].at[idx] = edit_distance
                result_columns[f"{prefix}_country_is_abbrev"].at[idx] = is_abbrev
                country_hints[i] = match_row["country_code"]
            elif len(row_matches) > 1:
                result_columns[f"{prefix}_country_status"].at[idx] = "AMBIGUOUS"
            elif pd.isna(parsed_df.at[idx, "Country"]):
                result_columns[f"{prefix}_country_status"].at[idx] = "UNRESOLVED"
            else:
                result_columns[f"{prefix}_country_status"].at[idx] = "REGEX_NO_MATCH"
        
        # Search and correct cities using country hints
        city_db_matches = search_db.link_entities(parsed_df["City"], country_hints=country_hints)
        
        for i, (idx, row_matches) in enumerate(zip(parsed_df.index, city_db_matches)):
            if len(row_matches) == 1:
                match_row = row_matches.iloc[0]
                corrected_city = match_row["alternateName"]
                regex_city = parsed_df.at[idx, "City"]
                edit_distance = int(match_row["cleaned_distance"])
                is_abbrev = bool(match_row["may_be_abbreviation"])
                if edit_distance > search_db.threshold and not is_abbrev:
                    logging.error(f"High edit distance for city match: {regex_city} -> {corrected_city} (distance={edit_distance}, abbrev={is_abbrev}, filename={df.at[idx, 'filename']})")
                    result_columns[f"{prefix}_city_status"].at[idx] = f"POST_PROCESSING_ERROR"
                else:
                    result_columns[f"{prefix}_city_match"].at[idx] = regex_city
                    if regex_city and regex_city != corrected_city:
                        stats.corrected += 1
                        result_columns[f"{prefix}_city_corrected"].at[idx] = corrected_city
                        if edit_distance == 0:
                            result_columns[f"{prefix}_city_status"].at[idx] = "EXACT_MATCH_CORRECTED"
                        elif is_abbrev:
                            result_columns[f"{prefix}_city_status"].at[idx] = "ABBREVIATION_MATCH_CORRECTED"
                        else:
                            result_columns[f"{prefix}_city_status"].at[idx] = f"FUZZY_{edit_distance}_CORRECTED"
                    else:
                        result_columns[f"{prefix}_city_status"].at[idx] = "EXACT_MATCH"
                
                result_columns[f"{prefix}_city_edit_distance"].at[idx] = edit_distance
                result_columns[f"{prefix}_city_is_abbrev"].at[idx] = is_abbrev
                if country_hints[i] is None:
                    result_columns[f"{prefix}_city_in_same_country"].at[idx] = pd.NA
                else:
                    result_columns[f"{prefix}_city_in_same_country"].at[idx] = (match_row["country_code"] == country_hints[i])
            elif len(row_matches) > 1:
                result_columns[f"{prefix}_city_status"].at[idx] = "AMBIGUOUS"
            elif pd.isna(parsed_df.at[idx, "City"]):
                result_columns[f"{prefix}_city_status"].at[idx] = "UNRESOLVED"
            else:
                result_columns[f"{prefix}_city_status"].at[idx] = "REGEX_NO_MATCH"
        
        # Update stats
        has_data = (result_columns[f"{prefix}_city_match"].notna() | 
                   result_columns[f"{prefix}_country_match"].notna())
        stats.parsed += has_data.sum()
        
        # Create DataFrame for this field (include index to preserve row alignment)
        result_dfs[prefix] = pd.DataFrame(result_columns)
    
    return result_dfs

def main(argv=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file_directory", type=str)
    arg_parser.add_argument("-k", "--topk", type=int, default=1)
    arg_parser.add_argument("-t", "--threshold", type=int, default=3)
    arg_parser.add_argument("-P", "--file-pattern", type=str, default="1*.jsonl")
    arg_parser.add_argument("--input-chunk-size", type=int, default=100)
    arg_parser.add_argument("--output-dir", type=str, default="post-processed")
    args = arg_parser.parse_args(argv)
    
    address_parser = BasicRegexAddressParser()
    search_db = geonames_db_search.GeonamesSearch(topk=args.topk, threshold=args.threshold)

    input_dir = Path(args.input_file_directory)
    assert input_dir.is_dir(), f"{args.input_file_directory} is not a valid directory"
    output_dir = Path(args.output_dir)
    #assert not output_dir.exists(), f"{args.output_dir} already exists"
    output_dir.mkdir(parents=True, exist_ok=True)

    

    files = list(input_dir.glob(args.file_pattern))
    files.sort()
    row_counts = [len(f.read_text().splitlines()) for f in files]
    total_rows = sum(row_counts)
    stats = Stats()

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'normalize_script.log', mode='a')
        ]
    )
    
    # Output files for each location field
    output_files = {
        prefix: output_dir / f"{prefix}_locations.jsonl"
        for prefix in PLACE_COLS_PREFIX.values()
    }
    
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for i, file in enumerate(files):
            try:
                logging.info(f"Processing {file} with {row_counts[i]} rows.")
                for df in pd.read_json(file, lines=True, chunksize=args.input_chunk_size):
                    result_dfs = parse_and_correct(df, address_parser, search_db, stats)
                    pbar.update(len(df))
                    # Write each field's results to its respective output file
                    for prefix, result_df in result_dfs.items():
                        result_df.to_json(output_files[prefix], orient="records", lines=True, mode="a")
                logging.info(f"Finished processing {file}. Stats so far: {stats}")
            except Exception as e:
                if str(e) == "Query interrupted": raise
                logging.exception(f"Error processing {file}: {e}")
                try: pbar.update(row_counts[i])
                except: pass

if __name__ == "__main__":
    main()