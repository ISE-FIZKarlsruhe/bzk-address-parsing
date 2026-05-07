import geonames_db_search
import mllms
import argparse
import re
import pandas as pd
from typing import Iterable, Literal
from pathlib import Path
from tqdm.contrib.logging import tqdm_logging_redirect as tqdm
import traceback
import logging
import json

PROMPT="""
# Role

You are a German archivist handling the digitalization of German documents.

# Task

Your current task consists of annotating addresses identifying the respective components of each address. Consider the component types: HouseNumber, StreetName, Neighborhood, City, Country, Other.

## Hints:

When interpreting the addresses, take into consideration:
- Addresses will often be written in German, meaning country and city names may be in German rather than the international standard.
- Addresses in Israel will often have words in Hebrew.
- Streets in some countries are identified by their cardinal direction and a number, such as "West 5th Avenue".
- Place names in the addresses might be abbreviated.

## Example Terms:

The addresses often include terms such as:
- "straße" or its abbreviation "str." for street
Some of these terms may occur as a suffix to another word.

## Rules:

- Only extract information **explicitly present in the address**.
- If the address contains a neighborhood joined together with a city by a dash (e.g., Berlin-Marienfelde), separate them accordingly.
- Neighborhoods or boroughs should **not be classified as cities**.
- Do **not infer missing components**.
- Sometimes cities will include (often connected by a slash "/") a reference to a nearby place (eg. river, city, district or region) for the purpose of disambiguation. If this is commonly part of the city name (eg. Frankfurt am Main), include it as part of the city component. Otherwise, it it fits another component type, classify it as such. Finally, if it does not fit any component type include it in the city name anyway.
- Do not extract punctuation around the words such as commas or dashes. The exception is when the word ends in a period to mark it as an abbreviation
- If uncertain about a component type, exclude it from the output.

Format the output as a JSON object with the component types as keys.
%(examples)s
Now annotate the following address:
%(address)s
""".strip()

PLACE_COLS = ['ApplicantBirthPlace', 'ApplicantCurrentAddress', 'VictimBirthPlace', 'VictimDeathPlace', 'VictimCurrentAddress']
N_SHOTS = 15
REGEX = re.compile(r"^(?P<City>\w+)(\s*/\s*(?P<Country>\w+))?$")

class Stats:
    def __init__(self):
        index = pd.MultiIndex.from_product([PLACE_COLS, ["", "City", "Country"]], names=["field", "subfield"])
        self.df = pd.DataFrame(
            0, index=index, columns=["addresses", "parsed", "corrected"], dtype=int
        )
        self.n_cards = 0
    
    def load(self, path):
        if Path(path).exists():
            with open(path) as f:
                obj = json.load(f)
            self.n_cards = obj.pop("n_cards", 0)
            self.df = pd.DataFrame.from_dict(obj, orient="tight", dtype=int)

    def save(self, path):
        obj = self.df.to_dict(orient="tight")
        obj["n_cards"] = self.n_cards
        with open(path, "w") as f:
            json.dump(obj, f)

    def update(self, status : pd.DataFrame, n_cards : int):
        self.n_cards += n_cards
        status = status.groupby(level=1)
        for field, group in status:
            assert set(group.columns) == {"City", "Country"}, f"Unexpected columns in status: {group.columns}"
            for subfield in ["City", "Country"]:
                self.df.at[(field, subfield), "addresses"] += len(group)
                self.df.at[(field, subfield), "parsed"] += (group[subfield].isin(["matched", "corrected"])).sum()
                self.df.at[(field, subfield), "corrected"] += (group[subfield] == "corrected").sum()
            self.df.at[(field, ""), "parsed"] += (group.isin(["matched", "corrected"])).any(axis=1).sum()
            self.df.at[(field, ""), "corrected"] += (group == "corrected").any(axis=1).sum()
            self.df.at[(field, ""), "addresses"] += len(group)

    def display_stats(self) -> pd.DataFrame:
        df = self.df.copy()
        for subfield in ["", "City", "Country"]:
            df.loc[("Total", subfield), :] = df.xs(subfield, level="subfield").sum(min_count=1)
        df["parsed_ratio"] = (df["parsed"] / df["addresses"]).map(lambda x: f"{x:.0%}" if pd.notna(x) else "0%")
        df["corrected_ratio"] = (df["corrected"] / df["parsed"]).map(lambda x: f"{x:.0%}" if pd.notna(x) else "0%")
        df["address_ratio"] = (df["addresses"] / self.n_cards).map(lambda x: f"{x:.0%}" if pd.notna(x) else "0%")
        df.loc[pd.IndexSlice[:, "City"], "addresses"] = pd.NA
        df.loc[pd.IndexSlice[:, "Country"], "addresses"] = pd.NA
        df[["addresses", "parsed", "corrected"]] = df[["addresses", "parsed", "corrected"]].map(lambda x: f"{x:_.0f}" if pd.notna(x) else "")
        return df

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

def load_model(args):
    llm_model = None
    if args.llm is not None:
        examples = pd.read_csv(args.example_pool, keep_default_na=False, dtype=str, na_values=[""])
        example_strategy = mllms.HybridSimilarExamples(
            embedding_strategy=mllms.SimilarExamples(
                num_examples=N_SHOTS,
                similarity_threshold=0.35,
                embedding_model="all-MiniLM-L6-v2",
                example_addresses=examples["FullAddress"],
                example_labels=examples,
                labels_to_include=["HouseNumber", "StreetName", "Neighborhood", "City", "Country"]
            ),
            pattern_strategy=mllms.NERPatternSimilarExamples(
                example_addresses=examples["FullAddress"],
                example_labels=examples,
                num_examples=N_SHOTS,
                labels_to_include=["HouseNumber", "StreetName", "Neighborhood", "City", "Country"]
            ),
            num_examples=N_SHOTS,
            pool_size=N_SHOTS
        )
        if "Llamma" in args.llm:
            llm_class = mllms.LLMAddressParsingModel
        elif "Qwen" in args.llm:
            llm_class = mllms.QwenAddressParsingModel
        elif "Mistral" in args.llm:
            llm_class = mllms.MistralAddressParsingModel
        elif "DeepSeek" in args.llm:
            llm_class = mllms.DeepSeekAddressParsingModel
        else: raise ValueError(f"Unknown model {args.llm}")
        llm_model = llm_class(
            model=args.llm,
            example_strategy=example_strategy,
            prompt_template=PROMPT
        )
    if llm_model is not None and args.skip_regex_parsing:
        return llm_model
    else:
        return BasicRegexAddressParser(fallback_parser=llm_model)

def parse_and_correct(
        df : pd.DataFrame, 
        address_parser,
        search_db : geonames_db_search.GeonamesSearch, 
        stats : Stats
    ):
    addresses = df[PLACE_COLS].stack().dropna()
    parsed_addresses = pd.DataFrame(
        address_parser.parse_addresses(addresses), 
        index=addresses.index, dtype=str
    )[["City", "Country"]]
    db_matches = search_db.link_entities(parsed_addresses["Country"], name_only=True, entity_type=geonames_db_search.EntityType.Country)
    country_hints = [None] * len(parsed_addresses)
    status = pd.DataFrame("unmatched", index=parsed_addresses.index, columns=["City", "Country"])
    for i, idx, row_matches in zip(range(len(parsed_addresses)), parsed_addresses.index, db_matches):
        if len(row_matches) == 1:
            parsed_addresses.at[idx, "Country"] = row_matches.at[0, "nfc_alt_name"]
            if row_matches.at[0, "nfc_query"] != row_matches.at[0, "nfc_alt_name"]:
                status.at[idx, "Country"] = "corrected"
            else:
                status.at[idx, "Country"] = "matched"
            country_hints[i] = row_matches.at[0, "country_code"]
        else:
            parsed_addresses.at[idx, "Country"] = None
    db_matches = search_db.link_entities(parsed_addresses["City"], country_hints=country_hints, name_only=True, entity_type=geonames_db_search.EntityType.City)
    for idx, row_matches in zip(parsed_addresses.index, db_matches):
        if len(row_matches) == 1:
            parsed_addresses.at[idx, "City"] = row_matches.at[0, "nfc_alt_name"]
            if row_matches.at[0, "nfc_query"] != row_matches.at[0, "nfc_alt_name"]:
                status.at[idx, "City"] = "corrected"
            else:
                status.at[idx, "City"] = "matched"
        else:
            parsed_addresses.at[idx, "City"] = None
    stats.update(status, len(df))
    parsed_addresses = parsed_addresses.unstack(sort=False).swaplevel(axis=1).sort_index(axis=1)
    parsed_addresses.columns = ["_".join(a) for a in parsed_addresses.columns.to_flat_index()]
    return df[["filename"]].merge(
        parsed_addresses, 
        left_index=True, right_index=True)


def main(argv=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file_directory", type=str)
    arg_parser.add_argument("-t", "--threshold", type=int, default=2)
    arg_parser.add_argument("-P", "--file-pattern", type=str, default="1*.jsonl")
    arg_parser.add_argument("--input-chunk-size", type=int, default=100)
    arg_parser.add_argument("--pattern-type", choices=["glob", "regex"], default="glob")
    arg_parser.add_argument("--output-dir", type=str, default="post-processed")
    arg_parser.add_argument("--example-pool", type=str, default="open_data/open_data/bzkopen_addresses_train.csv")
    arg_parser.add_argument("--skip-regex-parsing", action="store_true")
    arg_parser.add_argument("--llm", type=str, default=None)
    args = arg_parser.parse_args(argv)
    
    address_parser = load_model(args)
    search_db = geonames_db_search.GeonamesSearch(threshold=args.threshold, topk=1)

    input_dir = Path(args.input_file_directory)
    assert input_dir.is_dir(), f"{args.input_file_directory} is not a valid directory"
    output_dir = Path(args.output_dir)
    #assert not output_dir.exists(), f"{args.output_dir} already exists"
    output_dir.mkdir(parents=True, exist_ok=True)

    

    files = list(input_dir.glob(args.file_pattern))
    files.sort()
    row_counts = [len(f.read_text().splitlines()) for f in files]
    total_rows = sum(row_counts)
    stats_file = output_dir / "stats.json"
    stats = Stats()
    stats.load(stats_file)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'normalize_script.log', mode='a')
        ]
    )
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for i, file in enumerate(files):
            try:
                logging.info(f"Processing {file} with {row_counts[i]} rows.")
                out_file = output_dir / file.name
                if out_file.exists():
                    logging.warning(f"Skipping {file} as output already exists")
                    pbar.update(row_counts[i])
                    continue
                for df in pd.read_json(file, lines=True, chunksize=args.input_chunk_size):
                    to_write = parse_and_correct(df, address_parser, search_db, stats)
                    pbar.update(len(df))
                    to_write.to_json(out_file, orient="records", lines=True, mode="a")
                logging.info(f"Finished processing {file}. Stats so far:\n{stats.display_stats()}")
                stats.save(stats_file)
            except Exception as e:
                if str(e) == "Query interrupted": raise
                logging.exception(f"Error processing {file}: {e}")
                try: pbar.update(row_counts[i])
                except: pass

if __name__ == "__main__":
    main()