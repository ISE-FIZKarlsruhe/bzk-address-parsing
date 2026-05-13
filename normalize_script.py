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
from collections import OrderedDict

PLACE_COLS = ['ApplicantBirthPlace', 'ApplicantCurrentAddress', 'VictimBirthPlace', 'VictimDeathPlace', 'VictimCurrentAddress']
PLACE_COLS_PREFIX = {
    'ApplicantBirthPlace': 'abp',
    'ApplicantCurrentAddress': 'aca',
    'VictimBirthPlace': 'vbp',
    'VictimDeathPlace': 'vdp',
    'VictimCurrentAddress': 'vca'
}
REGEX = re.compile(r"^(?P<City>\w+)(\s*/\s*(?P<Country>\w+))?$")


class Stats:
    """
    Track statistics for address parsing and matching across all location fields and entity types.
    Maintains a DataFrame with counts for each (field_prefix, entity_type) combination.
    """
    def __init__(self):
        # Create MultiIndex with all combinations of (prefix, entity_type)
        prefixes = list(PLACE_COLS_PREFIX.values())
        entity_types = ['City', 'Country']
        index = pd.MultiIndex.from_product([prefixes, entity_types], names=['field', 'entity_type'])
        
        # Initialize DataFrame with columns for each status and summary columns
        self.statuses = [
            'EXACT_MATCH', 'EXACT_MATCH_CORRECTED', 'ABBREVIATION_MATCH_CORRECTED',
            'AMBIGUOUS', 'UNRESOLVED', 'REGEX_NO_MATCH', 'POST_PROCESSING_ERROR'
        ]
        columns = ['input_rows', 'input_fields_with_address', 'final_count'] + self.statuses
        self.df = pd.DataFrame(0, index=index, columns=columns)
    
    def update(self, prefix: str, entity_type: str, result_series: pd.Series, 
               raw_addresses: pd.Series, final_series: pd.Series):
        """
        Update statistics for a specific field and entity type.
        
        Args:
            prefix: Field prefix (e.g., 'abp', 'aca')
            entity_type: 'City' or 'Country'
            result_series: Series with status values for this entity type
            raw_addresses: Series with raw address values
            final_series: Series with final corrected values
        """
        # Count status values
        status_counts = result_series.value_counts()
        for status in status_counts.index:
            self.df.at[(prefix, entity_type), status] = status_counts[status]
        
        # Count input rows (total rows processed)
        self.df.at[(prefix, entity_type), 'input_rows'] = len(raw_addresses)
        
        # Count input fields with an address present (non-empty)
        self.df.at[(prefix, entity_type), 'input_fields_with_address'] = (raw_addresses.astype(str).str.len() > 0).sum()
        
        # Count rows with a final value (non-null)
        self.df.at[(prefix, entity_type), 'final_count'] = final_series.notna().sum()
    
    def __str__(self):
        """
        Format the statistics DataFrame with both absolute counts and percentages.
        Percentages are calculated relative to input_rows for status columns,
        and relative to input_fields_with_address for final_count.
        """
        # Create a new DataFrame with multi-level columns (value, percentage)
        result_rows = []
        totals_df = self.df.copy()
        totals_df.loc[("TOTAL", "City"), :] = 0
        totals_df.loc[("TOTAL", "Country"), :] = 0
        totals_df.loc[("TOTAL", "TOTAL"), :] = 0

        for prefix in self.df.index.get_level_values('field').unique():
            totals_df.loc[(prefix, "TOTAL"), :] = 0
            for entity_type in self.df.index.get_level_values('entity_type').unique():
                totals_df.loc[(prefix, "TOTAL"), :] += self.df.loc[(prefix, entity_type), :]
                totals_df.loc[("TOTAL", entity_type), :] += self.df.loc[(prefix, entity_type), :]
                totals_df.loc[("TOTAL", "TOTAL"), :] += self.df.loc[(prefix, entity_type), :]
            
        for (field, entity_type), row in totals_df.iterrows():
            result_row = OrderedDict()
            input_rows = row['input_rows']
            input_fields_with_address = row['input_fields_with_address']
            
            # Process each status column - calculate percentage of input_rows
            for status in self.df.columns[3:]:  # Exclude summary columns
                count = row[status]
                status_upper_bound = input_rows if status == "NO_ADDRESS" else input_fields_with_address
                if status_upper_bound > 0:
                    pct = (count / status_upper_bound)
                    result_row[('Status Counts', status, 'Count')] = f"{count:.0f}"
                    result_row[('Status Counts', status, '%')] = f"{pct:.1%}"
                else:
                    result_row[('Status Counts', status, 'Count')] = f"{count:.0f}"
                    result_row[('Status Counts', status, '%')] = "N/A"
            
            # Process summary columns
            # input_rows - no percentage needed
            result_row[('Summary', 'input_rows', 'Count')] = f"{input_rows:.0f}"
            result_row[('Summary', 'input_rows', '%')] = "—"
            
            # input_fields_with_address - percentage of input_rows
            count = row['input_fields_with_address']
            if input_rows > 0:
                pct = (count / input_rows)
                result_row[('Summary', 'input_fields_with_address', 'Count')] = f"{count:.0f}"
                result_row[('Summary', 'input_fields_with_address', '%')] = f"{pct:.1%}"
            else:
                result_row[('Summary', 'input_fields_with_address', 'Count')] = f"{count:.0f}"
                result_row[('Summary', 'input_fields_with_address', '%')] = "N/A"
            
            # final_count - percentage of input_fields_with_address (where it makes sense)
            count = row['final_count']
            if input_fields_with_address > 0:
                pct = (count / input_fields_with_address)
                result_row[('Summary', 'final_count', 'Count')] = f"{count:.0f}"
                result_row[('Summary', 'final_count', '%')] = f"{pct:.1%}"
            else:
                result_row[('Summary', 'final_count', 'Count')] = f"{count:.0f}"
                result_row[('Summary', 'final_count', '%')] = "N/A"
            
            result_rows.append(((field, entity_type), result_row))
        
        # Create output DataFrame with multi-level columns
        display_df = pd.DataFrame([row for _, row in result_rows],
                                columns=pd.MultiIndex.from_tuples(result_rows[0][1].keys(), names=['field', 'entity_type', '']),
                                index=pd.MultiIndex.from_tuples([idx for idx, _ in result_rows],
                                                                  names=['field', 'entity_type'])
        )
        display_df.sort_index(inplace=True)
        return display_df.to_string()


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
                        result_columns[f"{prefix}_country_corrected"].at[idx] = corrected_country
                        result_columns[f"{prefix}_country_final"].at[idx] = corrected_country
                        if edit_distance == 0:
                            result_columns[f"{prefix}_country_status"].at[idx] = "EXACT_MATCH_CORRECTED"
                        elif is_abbrev:
                            result_columns[f"{prefix}_country_status"].at[idx] = "ABBREVIATION_MATCH_CORRECTED"
                        else:
                            result_columns[f"{prefix}_country_status"].at[idx] = f"FUZZY_{edit_distance}_CORRECTED"
                    else:
                        result_columns[f"{prefix}_country_status"].at[idx] = "EXACT_MATCH"
                        result_columns[f"{prefix}_country_final"].at[idx] = regex_country
                
                result_columns[f"{prefix}_country_edit_distance"].at[idx] = edit_distance
                result_columns[f"{prefix}_country_is_abbrev"].at[idx] = is_abbrev
                country_hints[i] = match_row["country_code"]
            elif len(row_matches) > 1:
                result_columns[f"{prefix}_country_status"].at[idx] = "AMBIGUOUS"
            elif raw_addresses.at[idx] == "":
                result_columns[f"{prefix}_city_status"].at[idx] = "NO_ADDRESS"
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
                        result_columns[f"{prefix}_city_corrected"].at[idx] = corrected_city
                        result_columns[f"{prefix}_city_final"].at[idx] = corrected_city
                        if edit_distance == 0:
                            result_columns[f"{prefix}_city_status"].at[idx] = "EXACT_MATCH_CORRECTED"
                        elif is_abbrev:
                            result_columns[f"{prefix}_city_status"].at[idx] = "ABBREVIATION_MATCH_CORRECTED"
                        else:
                            result_columns[f"{prefix}_city_status"].at[idx] = f"FUZZY_{edit_distance}_CORRECTED"
                    else:
                        result_columns[f"{prefix}_city_status"].at[idx] = "EXACT_MATCH"
                        result_columns[f"{prefix}_city_final"].at[idx] = regex_city
                
                result_columns[f"{prefix}_city_edit_distance"].at[idx] = edit_distance
                result_columns[f"{prefix}_city_is_abbrev"].at[idx] = is_abbrev
                if country_hints[i] is None:
                    result_columns[f"{prefix}_city_in_same_country"].at[idx] = pd.NA
                else:
                    result_columns[f"{prefix}_city_in_same_country"].at[idx] = (match_row["country_code"] == country_hints[i])
            elif len(row_matches) > 1:
                result_columns[f"{prefix}_city_status"].at[idx] = "AMBIGUOUS"
            elif raw_addresses.at[idx] == "":
                result_columns[f"{prefix}_city_status"].at[idx] = "NO_ADDRESS"
            elif pd.isna(parsed_df.at[idx, "City"]):
                result_columns[f"{prefix}_city_status"].at[idx] = "UNRESOLVED"
            else:
                result_columns[f"{prefix}_city_status"].at[idx] = "REGEX_NO_MATCH"
        
        # Create DataFrame for this field (include index to preserve row alignment)
        result_dfs[prefix] = pd.DataFrame(result_columns)
        
        # Update statistics for both entity types
        stats.update(prefix, 'City', result_columns[f"{prefix}_city_status"], 
                    raw_addresses, result_columns[f"{prefix}_city_final"])
        stats.update(prefix, 'Country', result_columns[f"{prefix}_country_status"], 
                    raw_addresses, result_columns[f"{prefix}_country_final"])
    
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
                logging.info(f"Finished processing {file}. Stats so far:\n{stats}")
            except Exception as e:
                if str(e) == "Query interrupted": raise
                logging.exception(f"Error processing {file}: {e}")
                try: pbar.update(row_counts[i])
                except: pass
    
    # Log final statistics
    logging.info("Processing complete. Final statistics:")
    logging.info("\n" + str(stats))

if __name__ == "__main__":
    main()