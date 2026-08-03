from google_crc32c import value

import modules.geo_db_search as geo_db_search
import argparse
import re
import pandas as pd
from typing import Any
from pathlib import Path
from tqdm.contrib.logging import tqdm_logging_redirect as tqdm
import traceback
import dataclasses
import logging
from collections import OrderedDict, defaultdict
from modules.regex_patterns import regex_parse
import multiprocessing as mp
from multiprocessing.pool import Pool
PLACE_COLS = ['ApplicantBirthPlace', 'ApplicantCurrentAddress', 'VictimBirthPlace', 'VictimDeathPlace', 'VictimCurrentAddress']
PLACE_COLS_PREFIX = {
    'ApplicantBirthPlace': 'abp',
    'ApplicantCurrentAddress': 'aca',
    'VictimBirthPlace': 'vbp',
    'VictimDeathPlace': 'vdp',
    'VictimCurrentAddress': 'vca'
}

class Stats:
    def __init__(self):
        self.total_addresses_count = 0
        self.total_empty_count = 0
        self.has_match_count = 0
        self.has_link_count = 0
        self.entity_linked_count = defaultdict(int)
        self.entity_matched_count = defaultdict(int)
        self.global_patterns_matches = defaultdict(int)
        self.special_patterns_matches = defaultdict(int)

    def update(self, results : pd.DataFrame):
        self.total_addresses_count += len(results)
        self.total_empty_count += (results["raw"].isna() | (results["raw"] == "")).sum()
        has_match = pd.Series(False, index=results.index)
        has_link = pd.Series(False, index=results.index)
        for col, col_values in results.items():
            if col == 'global_regex':
                for regex, count in col_values.value_counts().items():
                    self.global_patterns_matches[regex] += count
                continue
            elif col == "raw":
                continue
            parts = col.split('.')
            entity_type = parts[0]
            if parts[1] == 'text':
                has_match = has_match | col_values.notna()
                self.entity_matched_count[entity_type] += col_values.notna().sum()
            elif parts[1] == 'geonames_id':
                has_link = has_link | col_values.notna()
                self.entity_linked_count[entity_type] += col_values.notna().sum()
            elif parts[1] == 'special_regex' and not pd.isna(value):
                for regex, count in col_values.value_counts().items():
                    self.special_patterns_matches[regex] += count
        self.has_match_count += has_match.sum()
        self.has_link_count += has_link.sum()
        for pattern in results.get('global_patterns', []):
            self.global_patterns_matches[pattern] += 1

    def __add__(self, other):
        if not isinstance(other, Stats):
            return TypeError(f"Unsupported operand type(s) for +: 'Stats' and '{type(other).__name__}'")
        result = Stats()
        result.total_addresses_count = self.total_addresses_count + other.total_addresses_count
        result.total_empty_count = self.total_empty_count + other.total_empty_count
        result.has_match_count = self.has_match_count + other.has_match_count
        result.has_link_count = self.has_link_count + other.has_link_count
        for key in set(self.entity_linked_count.keys()).union(other.entity_linked_count.keys()):
            result.entity_linked_count[key] = self.entity_linked_count[key] + other.entity_linked_count[key]
        for key in set(self.entity_matched_count.keys()).union(other.entity_matched_count.keys()):
            result.entity_matched_count[key] = self.entity_matched_count[key] + other.entity_matched_count[key]
        for key in set(self.global_patterns_matches.keys()).union(other.global_patterns_matches.keys()):
            result.global_patterns_matches[key] = self.global_patterns_matches[key] + other.global_patterns_matches[key]
        for key in set(self.special_patterns_matches.keys()).union(other.special_patterns_matches.keys()):
            result.special_patterns_matches[key] = self.special_patterns_matches[key] + other.special_patterns_matches[key]
        return result

    def summary(self) -> pd.DataFrame:
        """
        Return a summary of the statistics as a pandas DataFrame.
        """
        addresses_with_values = self.total_addresses_count - self.total_empty_count
        data = [
            pd.Series({ "Total": self.total_addresses_count, "Ratio" : pd.NA}, name="Cards"),
            pd.Series({ "Total": self.total_empty_count, "Ratio" : self.total_empty_count / self.total_addresses_count if self.total_addresses_count > 0 else pd.NA }, name="Empty"),
            pd.Series({ "Total": addresses_with_values, "Ratio" : addresses_with_values / self.total_addresses_count if self.total_addresses_count > 0 else pd.NA }, name="With Values"),
            pd.Series({ "Total": self.has_link_count, "Ratio" : self.has_link_count / addresses_with_values if addresses_with_values > 0 else pd.NA }, name="Linked"),
            pd.Series({ "Total": self.has_match_count, "Ratio" : self.has_match_count / addresses_with_values if addresses_with_values > 0 else pd.NA }, name="Matched"),
        ]
        for entity_type in set(self.entity_linked_count.keys()).union(self.entity_matched_count.keys()):
            data.append(pd.Series({ "Total": self.entity_linked_count[entity_type], "Ratio" : self.entity_linked_count[entity_type] / addresses_with_values if addresses_with_values > 0 else pd.NA }, name=f"Linked {entity_type}"))
            data.append(pd.Series({ "Total": self.entity_matched_count[entity_type], "Ratio" : self.entity_matched_count[entity_type] / addresses_with_values if addresses_with_values > 0 else pd.NA }, name=f"Matched {entity_type}"))
        for pattern, count in self.global_patterns_matches.items():
            data.append(pd.Series({ "Total": count, "Ratio" : count / addresses_with_values if addresses_with_values > 0 else pd.NA }, name=f"Global Pattern: {pattern}"))
        for pattern, count in self.special_patterns_matches.items():
            data.append(pd.Series({ "Total": count, "Ratio" : count / addresses_with_values if addresses_with_values > 0 else pd.NA }, name=f"Special Pattern: {pattern}"))
        df = pd.DataFrame(data)
        df["Total"] = df["Total"].astype("Int64")
        return df

    def summary_str(self) -> str:
        """
        Return a summary of the statistics as a string.
        """
        summary_df = self.summary().style.format(na_rep="N/A", precision=2)
        return summary_df.to_string()

class AggregateStats:
    def __init__(self):
        self.stats_per_field = {prefix: Stats() for prefix in PLACE_COLS_PREFIX.values()}

    def update(self, field_prefix: str, results: pd.DataFrame):
        if field_prefix not in self.stats_per_field:
            raise ValueError(f"Unknown field prefix: {field_prefix}")
        self.stats_per_field[field_prefix].update(results)

    def summary_str(self) -> str:
        """
        Return a summary of the aggregate statistics as a string.
        """
        sb = []
        for prefix, stats in self.stats_per_field.items():
            sb.append(f"Statistics for field '{prefix}':\n")
            sb.append(stats.summary_str())
            sb.append("\n\n")
        sb.append("Overall Statistics:\n")
        overall_stats = Stats()
        for stats in self.stats_per_field.values():
            overall_stats += stats
        sb.append(overall_stats.summary_str())
        return "".join(sb)


def flatten_dict(d, sep='.', parent_key=None):
    """
    Flatten a nested dictionary into a single-level dictionary with keys as the path joined by sep.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key is not None else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, sep=sep, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse(
        df : pd.DataFrame,
        pool : Pool,
        stats : AggregateStats
    ):
    """
    Parse and correct location fields.
    
    Returns a dictionary with one DataFrame per location field
    """
    result_dfs = {}
    
    for field_name, prefix in PLACE_COLS_PREFIX.items():
        raw_addresses = df[field_name].fillna("").astype(str)
        sanity = raw_addresses.isna().sum()
        # Parse addresses using regex
        parsed_results = pool.map(regex_parse, raw_addresses)
        parsed_results = pool.map(flatten_dict, parsed_results)
        parsed_df = pd.DataFrame(parsed_results, index=df.index)
        parsed_df.insert(0, "raw", raw_addresses)
        if parsed_df["raw"].isna().sum() != sanity:
            logging.error(f"Sanity check failed for field {field_name}: {sanity} vs {parsed_df['raw'].isna().sum()}")
        stats.update(prefix, parsed_df)

        parsed_df = parsed_df.add_prefix(f"{prefix}.")
        parsed_df.insert(0, "filename", df["filename"])
        
        # Create DataFrame for this field (include index to preserve row alignment)
        result_dfs[prefix] = parsed_df
    
    return result_dfs

def main(argv=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file_directory", type=str)
    arg_parser.add_argument("-k", "--topk", type=int, default=1)
    arg_parser.add_argument("-t", "--threshold", type=int, default=3)
    arg_parser.add_argument("-P", "--file-pattern", type=str, default="1*.jsonl")
    arg_parser.add_argument("--input-chunk-size", type=int, default=500)
    arg_parser.add_argument("--output-dir", type=str, default="regex_parsed_output")
    args = arg_parser.parse_args(argv)

    input_dir = Path(args.input_file_directory)
    assert input_dir.is_dir(), f"{args.input_file_directory} is not a valid directory"
    output_dir = Path(args.output_dir)
    #assert not output_dir.exists(), f"{args.output_dir} already exists"
    output_dir.mkdir(parents=True, exist_ok=True)

    

    files = list(input_dir.glob(args.file_pattern))
    files.sort()
    row_counts = [len(f.read_text().splitlines()) for f in files]
    total_rows = sum(row_counts)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'parse_with_regex.log', mode='a')
        ]
    )
    
    # Output files for each location field
    output_files = {
        prefix: output_dir / f"{prefix}_locations.jsonl"
        for prefix in PLACE_COLS_PREFIX.values()
    }

    stats = AggregateStats()
    
    with tqdm(total=total_rows, desc="Processing rows") as pbar, mp.Pool() as pool:
        for i, file in enumerate(files):
            try:
                logging.info(f"Processing {file} with {row_counts[i]} rows.")
                for df in pd.read_json(file, lines=True, chunksize=args.input_chunk_size):
                    result_dfs = parse(df, pool, stats)
                    pbar.update(len(df))
                    # Write each field's results to its respective output file
                    for prefix, result_df in result_dfs.items():
                        result_df.to_json(output_files[prefix], orient="records", lines=True, mode="a")
                logging.info(f"Finished processing {file}.")
                logging.info(f"Current statistics:\n{stats.summary_str()}")
            except Exception as e:
                if str(e) == "Query interrupted": raise
                logging.exception(f"Error processing {file}: {e}")
                try: pbar.update(row_counts[i])
                except: pass
    
    # Log final statistics
    logging.info("Processing complete.")
    logging.info(f"Final statistics:\n{stats.summary_str()}")

if __name__ == "__main__":
    main()