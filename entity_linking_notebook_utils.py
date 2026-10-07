"""
Support code for entity_linking.ipynb.

The notebook is kept to function calls and variable definitions (configuration,
experiment parameters); every function, class and non-trivial block of logic it
needs lives here instead. Functions take their dependencies (searchers,
dataframes, outcomes...) as explicit arguments rather than reading notebook
globals.
"""
import fnmatch
import json
import logging
import pprint
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Awaitable, Callable, Optional

import colorlog
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

import modules.build_geonames_db as build_geonames_db
import modules.entity_linking as entity_linking
import modules.geo_db_search as geo_db_search
import modules.geo_disambiguation as geo_disambiguation
from modules.camp_search import CampReferenceMatcher
from modules.entity_linking import SearchStatus
from modules.entity_linking_eval_metrics import eval_entity_linking, normalize_iri
from modules.geo_disambiguation import Disambiguator
from modules.pipeline.linked_data import LinkedAddress, MatchedEntity
from modules.regex_patterns import regex_parse
from modules.utils import format_time
from parse_with_llms import _rename_llm_output_columns, prepare_llm
from parse_with_regex import PLACE_COLS_PREFIX, flatten_dict
from IPython.display import Markdown, display


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    # Cell output only (stdout, so Jupyter doesn't paint it with the stderr
    # background); force=True replaces handlers from previous runs of this cell
    # instead of stacking duplicates.
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(asctime)s.%(msecs)03d %(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s %(filename)s:%(lineno)d%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.basicConfig(handlers=[handler], force=True, level=logging.INFO)


def cleanup_previous_run(namespace: dict) -> None:
    """Releases the resources a previous run of the notebook left in `namespace` (its globals()), allowing rerunning."""
    if "conn" in namespace:
        namespace["conn"].close()
    if "linking_pool" in namespace:
        namespace["linking_pool"].close()
    if "disambiguator" in namespace:
        namespace["disambiguator"].close()


def cache_df(name: str, gen_function: Callable[[], pd.DataFrame], overwrite: bool = False) -> pd.DataFrame:
    cache_path = Path(f"experiments_data/entity_linking/{name}.jsonl")
    attrs_path = cache_path.with_suffix(".attrs.json")
    if overwrite or not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        df = gen_function()
        end = time.monotonic()
        elapsed = end - start
        print(f"Generation of '{name}' took {format_time(elapsed)}")
        df.attrs["generation_time"] = elapsed
        df.to_json(cache_path, orient="table")
        with open(attrs_path, "w") as f:
            json.dump(df.attrs, f)
    else:
        print(f"Retrieving cached results for '{name}' from {cache_path}...")
        with open(attrs_path, "r") as f:
            attrs: dict = json.load(f)
        if "generation_time" in attrs:
            print(f"Generation of '{name}' originally took {format_time(attrs['generation_time'])}")
        df = pd.read_json(cache_path, orient="table")
        df.attrs.update(attrs)
    return df


def print_storage_sizes(search_index_dir: str) -> None:
    print(f"DB Size: {Path(build_geonames_db.DUCK_DB_PATH).stat().st_size / 1e9:.2f} GB")
    print(f"Search index Size: {sum(x.stat().st_size for x, _, __ in Path(search_index_dir).walk()) / 1e9:.2f} GB") #TODO fix this


def load_ground_truth(path: str) -> pd.DataFrame:
    """Loads the linking ground truth, dropping addresses tagged "unresolved"."""
    gt: pd.DataFrame = pd.read_json(path, lines=True)
    return gt[~gt["tags"].apply(lambda tags: "unresolved" in tags)].reset_index(drop=True)


def index_by_address_id(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.copy()
    indexed["address_id"] = indexed["address_id"].astype(str)
    indexed.set_index("address_id", inplace=True)
    return indexed


# ---------------------------------------------------------------------------
# Parsing (regex -> LLM fallback)
# ---------------------------------------------------------------------------

# Replicates parse_with_regex.py's per-address regex parsing, "{prefix}.*"-flattened
# exactly as it would be written to a per-field output file.
def _regex_parse_row(full_address, prefix: str) -> dict:
    raw_address = full_address if isinstance(full_address, str) else ""
    parsed = flatten_dict(regex_parse(raw_address))
    parsed = {f"{prefix}.{k}": v for k, v in parsed.items()}
    parsed[f"{prefix}.raw"] = raw_address
    return parsed


def _needs_llm_fallback(row: dict, prefix: str) -> bool:
    # Same fallback condition as parse_with_llms.py: only addresses regex left
    # un/partially parsed (and non-empty) go to the LLM.
    raw = row.get(f"{prefix}.raw")
    status = row.get(f"{prefix}.status")
    return bool(raw) and status != "fully_parsed"


def regex_parse_addresses(addresses: pd.DataFrame) -> tuple[list[dict], list[bool]]:
    """Regex-parses every address, returning the parsed rows and which of them need the LLM fallback."""
    print(f"Running regex parsing on {len(addresses)} addresses...")
    regex_rows = [
        _regex_parse_row(row.FullAddress, PLACE_COLS_PREFIX[row.field])
        for row in addresses.itertuples()
    ]
    needs_llm_mask = [
        _needs_llm_fallback(regex_rows[i], PLACE_COLS_PREFIX[row.field])
        for i, row in enumerate(addresses.itertuples())
    ]
    print(f"{sum(needs_llm_mask)}/{len(regex_rows)} addresses need the LLM fallback (regex left them un/partially parsed).")
    return regex_rows, needs_llm_mask


def is_not_input_address(address: str, example_address: str) -> bool:
    """
    Few-shot example filter (see prepare_llm) for the experiments: the LLM's
    example pool (open_data/bzkopen_addresses_train.csv) overlaps the
    evaluated addresses, which must not be shown to the LLM along with their
    own labels.
    """
    return example_address != address


async def _run_llm_fallback(addresses: list[str]) -> list[dict]:
    """
    Replicates parse_with_llms.py's LLM fallback: parses `addresses` with the
    same remote model/prompt/example-selection config, and renames its output
    columns the same way (entity names -> "{Entity}.text", metadata -> "llm_metadata.*").
    Unlike parse_with_llms.py, few-shot examples exclude the address itself
    (is_not_input_address).
    """
    llm_parser, llm_config = prepare_llm(example_filter=is_not_input_address)
    print(f"Using LLM parser {llm_config['model']} for {len(addresses)} addresses...")
    parsed_results = await llm_parser.parse_addresses(addresses)
    results = []
    for parsed in parsed_results:
        renamed = {_rename_llm_output_columns(k): v for k, v in parsed.items()}
        renamed["status"] = "llm_error" if not pd.isna(renamed.get("llm_metadata.error")) else "llm_parsed"
        renamed["llm_parser"] = llm_config["model"]
        results.append(renamed)
    return results


async def build_parsed_addresses(addresses: pd.DataFrame, regex_rows: list[dict], needs_llm_mask: list[bool]) -> pd.DataFrame:
    """
    Runs the full regex -> LLM fallback chain over every address in
    `addresses`, returning one "{prefix}.*"-flattened row per address,
    ready to feed directly into entity_linking.link_field.
    """
    llm_indices = [i for i, needs_llm in enumerate(needs_llm_mask) if needs_llm]
    llm_addresses = [addresses.iloc[i].FullAddress for i in llm_indices]
    llm_results = await _run_llm_fallback(llm_addresses) if llm_addresses else []
    llm_results_by_index = dict(zip(llm_indices, llm_results))

    merged_rows = []
    for i, row in enumerate(addresses.itertuples()):
        prefix = PLACE_COLS_PREFIX[row.field]
        if i in llm_results_by_index:
            raw_address = row.FullAddress if isinstance(row.FullAddress, str) else ""
            record = {f"{prefix}.{k}": v for k, v in llm_results_by_index[i].items()}
            record[f"{prefix}.raw"] = raw_address
        else:
            record = regex_rows[i]
        record["card_id"] = row.card_id
        record["address_id"] = row.address_id
        record["field"] = row.field
        record["prefix"] = prefix
        merged_rows.append(record)
    return pd.DataFrame(merged_rows)


async def _load_or_build_cached_parsed_addresses(
    build: Callable[[], Awaitable[pd.DataFrame]], method: str, cache_path: Path, overwrite: bool
) -> pd.DataFrame:
    """Awaits `build()` and caches its result to `cache_path`, or reads it back from there if already cached."""
    if overwrite or not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        parsed_addresses = await build()
        elapsed = time.monotonic() - start
        print(f"Parsing {len(parsed_addresses)} addresses ({method}) took {format_time(elapsed)}")
        parsed_addresses.to_json(cache_path, orient="records", lines=True)
    else:
        print(f"Retrieving cached parsed addresses from {cache_path}...")
        parsed_addresses = pd.read_json(cache_path, orient="records", lines=True)
    return parsed_addresses


async def load_or_build_parsed_addresses(addresses: pd.DataFrame, cache_path: Path, overwrite: bool = False) -> pd.DataFrame:
    """Parses `addresses` (regex + LLM fallback), or reads the result back from `cache_path` if already cached."""
    regex_rows, needs_llm_mask = regex_parse_addresses(addresses)
    return await _load_or_build_cached_parsed_addresses(
        lambda: build_parsed_addresses(addresses, regex_rows, needs_llm_mask),
        "regex + LLM fallback", cache_path, overwrite)


# ---------------------------------------------------------------------------
# Parsing (LLM only)
# ---------------------------------------------------------------------------

async def build_llm_only_parsed_addresses(addresses: pd.DataFrame) -> pd.DataFrame:
    """
    Parses every non-empty address in `addresses` with the LLM alone (no regex
    pass), returning rows in the same "{prefix}.*"-flattened format as
    build_parsed_addresses. Empty addresses are not sent to the LLM and, as
    with the regex parser, only carry an empty "{prefix}.raw".
    """
    raw_addresses = [
        full_address if isinstance(full_address, str) else ""
        for full_address in addresses["FullAddress"]
    ]
    llm_indices = [i for i, raw_address in enumerate(raw_addresses) if raw_address]
    print(f"{len(llm_indices)}/{len(raw_addresses)} addresses are non-empty and will be parsed by the LLM.")
    llm_results = await _run_llm_fallback([raw_addresses[i] for i in llm_indices]) if llm_indices else []
    llm_results_by_index = dict(zip(llm_indices, llm_results))

    rows = []
    for i, row in enumerate(addresses.itertuples()):
        prefix = PLACE_COLS_PREFIX[row.field]
        record = {f"{prefix}.{k}": v for k, v in llm_results_by_index.get(i, {}).items()}
        record[f"{prefix}.raw"] = raw_addresses[i]
        record["card_id"] = row.card_id
        record["address_id"] = row.address_id
        record["field"] = row.field
        record["prefix"] = prefix
        rows.append(record)
    return pd.DataFrame(rows)


def _regex_parsed_address_ids(parsed_addresses: pd.DataFrame) -> set[str]:
    """(str) address_id of the `parsed_addresses` rows fully parsed by the regex parser."""
    return {
        str(row["address_id"])
        for row in parsed_addresses.to_dict(orient="records")
        if row.get(f"{row['prefix']}.status") == "fully_parsed"
    }


def _without_missing(row: dict) -> dict:
    """`row` without its missing values (columns of other prefixes or parsers, filled with NaN by the dataframe)."""
    return {k: v for k, v in row.items() if isinstance(v, (list, tuple, dict)) or not pd.isna(v)}


def merge_llm_only_parses(
    addresses: pd.DataFrame, baseline_parsed: pd.DataFrame, llm_parsed: pd.DataFrame
) -> pd.DataFrame:
    """
    The `baseline_parsed` (regex + LLM fallback) row of every address in
    `addresses`, in the same order, except that rows the regex parser fully
    parsed are replaced by their `llm_parsed` row.
    """
    regex_parsed_ids = _regex_parsed_address_ids(baseline_parsed)
    baseline_by_id = {str(row["address_id"]): row for row in baseline_parsed.to_dict(orient="records")}
    llm_by_id = {str(row["address_id"]): row for row in llm_parsed.to_dict(orient="records")}
    rows = []
    for address_id in addresses["address_id"].astype(str):
        source = llm_by_id if address_id in regex_parsed_ids else baseline_by_id
        rows.append(_without_missing(source[address_id]))
    return pd.DataFrame(rows)


async def load_or_build_llm_only_parsed_addresses(
    addresses: pd.DataFrame, cache_path: Path, baseline_cache_path: Path, overwrite: bool = False
) -> pd.DataFrame:
    """
    Parses `addresses` with the LLM only, reusing the LLM fallback parses of
    the regex + LLM run (load_or_build_parsed_addresses, cached at
    `baseline_cache_path`) so that the LLM's run-to-run variability doesn't
    add differences between both experiments: only the addresses that run
    regex parsed are parsed by the LLM here. These LLM parses are cached at
    `cache_path`, or read back from there if already cached (a cache holding
    every address also works, as only the regex parsed ones are used).
    """
    baseline_parsed = await load_or_build_parsed_addresses(addresses, baseline_cache_path)
    regex_parsed_ids = _regex_parsed_address_ids(baseline_parsed)
    regex_parsed_addresses = addresses[addresses["address_id"].astype(str).isin(regex_parsed_ids)]
    print(f"Reusing the regex + LLM run parses of {len(addresses) - len(regex_parsed_addresses)}/{len(addresses)} addresses it did not regex parse.")
    llm_parsed = await _load_or_build_cached_parsed_addresses(
        lambda: build_llm_only_parsed_addresses(regex_parsed_addresses),
        "LLM only, addresses regex parsed in the regex + LLM run", cache_path, overwrite)
    return merge_llm_only_parses(addresses, baseline_parsed, llm_parsed)


# ---------------------------------------------------------------------------
# Entity linking pipeline
# ---------------------------------------------------------------------------

def default_num_workers() -> int:
    return entity_linking.DEFAULT_NUM_WORKERS


def run_pipeline(
    rows: list[dict],
    linking_pool: entity_linking.LinkingPool,
    estimated_total_addresses: Optional[int] = None,
) -> list[entity_linking.LinkingOutcome]:
    """
    Runs entity_linking.link_field for every parsed address row in `rows`
    (the "{prefix}.*"-flattened rows build_parsed_addresses() produces), in
    the worker processes of `linking_pool`.
    """
    print(f"Running full entity linking pipeline on regex+LLM parsed addresses (using {linking_pool.num_workers} worker processes)...")
    start = time.monotonic()
    results = list(tqdm(linking_pool.link_fields(rows), total=len(rows)))
    elapsed = time.monotonic() - start
    print(f"Total pipeline time: {format_time(elapsed)}")
    if estimated_total_addresses is not None:
        print(f"Estimated time for {estimated_total_addresses:_} addresses: {format_time((elapsed / len(rows)) * estimated_total_addresses)}")
    return results


def print_outcomes(outcomes: list[entity_linking.LinkingOutcome], n: int = 10) -> None:
    for outcome in outcomes[:n]:
        pprint.pprint(outcome)


def print_slowest_addresses(outcomes: list[entity_linking.LinkingOutcome], parsed_rows: list[dict]) -> None:
    for i, outcome in sorted(enumerate(outcomes), key=lambda o: o[1].total_time, reverse=True):
        parsed_row = parsed_rows[i]
        address = None
        for c, value in parsed_row.items():
            if c.endswith("raw") and not pd.isna(value):
                address = value
        print(f"    {parsed_row['address_id']} {address} ({format_time(outcome.total_time)})")


def finest_entity(outcome: entity_linking.LinkingOutcome):
    return next((e for e in outcome.linked_entities if e.entity_type == outcome.entity_type), None)


def get_entity_for(outcome: entity_linking.LinkingOutcome, entity_type):
    for entity in outcome.linked_entities:
        if entity.entity_type == entity_type:
            return entity
    return None


class LinkStatus:
    """How a predicted IRI compares to the ground truth (see link_status)."""
    CORRECTLY_LINKED = "Correctly Linked"
    PARTIALLY_LINKED = "Partially Linked"
    INCORRECTLY_LINKED = "Incorrectly Linked"
    FAILED_TO_LINK = "Failed to link"
    NOT_A_LOCATION = "Not a location"


def link_status(pred_iri: Optional[str], true_row: pd.Series) -> str:
    """
    Categorizes a prediction against its ground-truth row, consistently with
    eval_entity_linking: a partial link is a wrong prediction that is still an
    ancestor of the true location in its full_hierarchy (some_granularity_loss),
    any other wrong prediction is incorrect (the rest of fp), a missing
    prediction is a failure where a link was expected (fn), and "not a location"
    where none was (tn).
    """
    pred_iri = normalize_iri(pred_iri)
    true_iri = None if pd.isna(true_row["iri"]) else normalize_iri(true_row["iri"])
    if pred_iri is None:
        return LinkStatus.FAILED_TO_LINK if true_iri is not None else LinkStatus.NOT_A_LOCATION
    if pred_iri == true_iri:
        return LinkStatus.CORRECTLY_LINKED
    hierarchy = true_row["full_hierarchy"]
    if isinstance(hierarchy, list) and pred_iri in [normalize_iri(h) for h in hierarchy]:
        return LinkStatus.PARTIALLY_LINKED
    return LinkStatus.INCORRECTLY_LINKED


def link_statuses(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> list[str]:
    """link_status of every outcome, row-aligned with `ground_truth`."""
    return [link_status(outcome.iri, true_row) for outcome, (_, true_row) in zip(outcomes, ground_truth.iterrows())]


# ---------------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------------

def build_disambiguated_addresses_df(addresses: pd.DataFrame, outcomes: list[entity_linking.LinkingOutcome]) -> pd.DataFrame:
    disambiguated_addresses = []
    for row, outcome in zip(addresses.itertuples(), outcomes):
        if outcome.iri is None:
            continue
        finest = finest_entity(outcome)
        disambiguated_addresses.append(dict(
            card_id=row.card_id,
            address_id=row.address_id,
            bzk_field_name=row.field,
            full_address=row.FullAddress,
            raw_address_part=finest.raw_text if finest else None,
            matched_name=finest.matching.get("matched_name") if finest else None,
            country=finest.country if finest else None,
            iri=outcome.iri,
            entity_type=outcome.entity_type,
            search_status=outcome.search_status,
            child_parent_likelihood=outcome.disambiguation_scores.get("child_parent_likelihood"),
        ))
    return pd.DataFrame(disambiguated_addresses)


def build_recovered_missed_words_df(
    parsed_rows: list[dict], outcomes: list[entity_linking.LinkingOutcome]
) -> pd.DataFrame:
    """
    Every address with words parsing left unassigned and linking recovered as
    extra entities (see entity_linking._missed_words), alongside the parsed
    entities and which of the recovered words ended up linked.
    """
    recovered = []
    for parsed_row, outcome in zip(parsed_rows, outcomes):
        parsed = entity_linking.parse_field_entities(parsed_row, parsed_row["prefix"])
        missed_words = [entity.raw_text for entity in parsed.entities if entity.is_missed_word]
        if not missed_words:
            continue
        linked_texts = {entity.raw_text for entity in outcome.linked_entities}
        recovered.append(dict(
            address_id=parsed_row["address_id"],
            parse_status = entity_linking._clean_optional_str(parsed_row.get(f"{parsed_row['prefix']}.status")),
            bzk_field_name=entity_linking.FIELD_PREFIXES[parsed_row["prefix"]],
            raw_address=parsed.raw_address,
            parsed_entities={
                entity.entity_type.name: entity.raw_text for entity in parsed.entities if not entity.is_missed_word
            },
            recovered_missed_words=missed_words,
            linked_missed_words=[word for word in missed_words if word in linked_texts],
            iri=outcome.iri,
            entity_type=outcome.entity_type,
        ))
    return pd.DataFrame(recovered)


def evaluate_outcomes(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> dict:
    # entity_linking.link_field already returns a normalized final iri and its
    # entity type directly, so no re-derivation from a nested address/entity tree
    # is needed here anymore.
    predicted_iris = [outcome.iri for outcome in outcomes]
    predicted_entity_types = [outcome.entity_type for outcome in outcomes]
    return eval_entity_linking(predicted_iris, predicted_entity_types, ground_truth)


def _get_link_entity_type(true_row: pd.Series) -> Optional[str]:
    for c in true_row.index:
        if c.endswith("_iri") and not pd.isna(true_row[c]):
            return c[:-len("_iri")]  # Remove the "_iri" suffix to get the entity type
    return None


def build_incorrect_or_missing_links_df(
    addresses: pd.DataFrame,
    outcomes: list[entity_linking.LinkingOutcome],
    parsed_rows: list[dict],
    indexed_gt: pd.DataFrame,
) -> pd.DataFrame:
    incorrect_or_missing_links = []
    for row, outcome, parsed_row in zip(addresses.itertuples(), outcomes, parsed_rows):
        true_row = indexed_gt.loc[str(row.address_id)]
        status = link_status(outcome.iri, true_row)
        true_iri = true_row["iri"]
        if pd.isna(true_iri):
            true_iri = None
            true_entity_type = None
            true_raw_text = None
        else:
            true_iri = normalize_iri(true_iri)
            true_entity_type = _get_link_entity_type(true_row)
            true_raw_text = true_row[true_entity_type]
        pred_iri = normalize_iri(outcome.iri)
        pred_entity = get_entity_for(outcome, true_entity_type) if true_entity_type is not None else None
        if pred_entity is not None:
            pred_raw_text = pred_entity.raw_text
        elif true_entity_type is not None:
            # outcome.linked_entities only carries entities GeoDBSearch actually
            # matched, so it's empty for e.g. a NO_CANDIDATES search; fall back to
            # the parsed row itself to get the raw text that was searched for
            # (and failed), keyed by "{prefix}.{entity_type}.text".
            pred_raw_text = entity_linking._clean_optional_str(
                parsed_row.get(f"{parsed_row['prefix']}.{true_entity_type}.text")
            )
        else:
            pred_raw_text = None
        if status not in (LinkStatus.CORRECTLY_LINKED, LinkStatus.NOT_A_LOCATION):
            record = dict(
                address_id=row.address_id,
                card_id=row.card_id,
                bzk_field_name=row.field,
                full_address=row.FullAddress,
                link_status=status,
                true_iri=true_iri,
                pred_iri=pred_iri,
                true_entity_type=true_entity_type,
                true_raw_text=true_raw_text,
                pred_raw_text=pred_raw_text,
                parsing_status=entity_linking._clean_optional_str(parsed_row.get(f"{parsed_row['prefix']}.status")),
                pred_status=outcome.search_status,
                true_tags=list(true_row["tags"]),
                pred_tags=list(outcome.tags),
            )
            finest = finest_entity(outcome)
            if finest is not None:
                record.update(dict(
                    pred_entity_type=outcome.entity_type,
                    pred_matched_name=finest.matching.get("matched_name"),
                    pred_country=finest.country,
                ))
            incorrect_or_missing_links.append(record)
    return pd.DataFrame(incorrect_or_missing_links)


def split_by_link_status(links: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Splits a dataframe with a "link_status" column into one dataframe per status."""
    return {
        status: group.reset_index(drop=True)
        for status, group in links.groupby("link_status", sort=False)
    }


def _aligned_table(df: pd.DataFrame) -> str:
    """Every row and column of `df` as a left-aligned plain-text table."""
    def _cell(value) -> str:
        return str(value).replace("\n", "\\n")
    columns = [str(c) for c in df.columns]
    rows = [[_cell(v) for v in row] for row in df.itertuples(index=False)]
    widths = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(columns)]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(columns, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join(lines) + "\n"


def save_result_table(df: pd.DataFrame, results_dir: Path, name: str) -> None:
    """Writes `df` to `{results_dir}/{name}.jsonl` and as an aligned table to `{results_dir}/{name}.txt`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(results_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    (results_dir / f"{name}.txt").write_text(_aligned_table(df), encoding="utf-8")
    display(Markdown(
        f"Saved {len(df)} rows to {(results_dir / name)}.jsonl\n\n"
        f"Saved {len(df)} rows to [{(results_dir / name)}.txt]({(results_dir / name)}.txt)"
    ))


# ---------------------------------------------------------------------------
# Comparison of the results tables of two runs
# ---------------------------------------------------------------------------

_RESULT_TABLE_NAMES = ["incorrectly_linked", "partially_linked", "failed_to_link"]

LINK_STATUS_EMOJIS = {
    LinkStatus.CORRECTLY_LINKED: "✅",
    LinkStatus.PARTIALLY_LINKED: "⚠️",
    LinkStatus.INCORRECTLY_LINKED: "❌",
    LinkStatus.FAILED_TO_LINK: "❌",
    LinkStatus.NOT_A_LOCATION: "❔",
}


def with_emoji(status: str) -> str:
    """`status` prefixed with its LINK_STATUS_EMOJIS visual aid."""
    return f"{LINK_STATUS_EMOJIS[status]} {status}" if status in LINK_STATUS_EMOJIS else status


def load_result_tables(results_dir: Path) -> pd.DataFrame:
    """Every row save_result_table wrote to `results_dir` (one table per link status), indexed by (str) address_id."""
    tables = []
    for name in _RESULT_TABLE_NAMES:
        path = results_dir / f"{name}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            tables.append(pd.read_json(path, orient="records", lines=True, dtype=False))
    if not tables:
        return pd.DataFrame(columns=["link_status", "pred_iri"], index=pd.Index([], name="address_id"))
    results = pd.concat(tables, ignore_index=True)
    results["address_id"] = results["address_id"].astype(str)
    return results.set_index("address_id")


def _link_results(results_dir: Path, indexed_gt: pd.DataFrame) -> pd.DataFrame:
    """
    link_status and pred_iri of every ground truth address, from the results
    tables of `results_dir`. Addresses absent from the tables were correctly
    linked (pred_iri is then the true iri) or are not a location.
    """
    results = load_result_tables(results_dir)
    rows = {}
    for address_id, true_row in indexed_gt.iterrows():
        if address_id in results.index:
            result = results.loc[address_id]
            rows[address_id] = (result["link_status"], None if pd.isna(result["pred_iri"]) else result["pred_iri"])
        elif pd.isna(true_row["iri"]):
            rows[address_id] = (LinkStatus.NOT_A_LOCATION, None)
        else:
            rows[address_id] = (LinkStatus.CORRECTLY_LINKED, normalize_iri(true_row["iri"]))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["link_status", "pred_iri"])


def link_status_transitions(
    baseline_dir: Path, other_dir: Path, indexed_gt: pd.DataFrame, labels: tuple[str, str] = ("baseline", "other")
) -> pd.DataFrame:
    """Number of addresses per (baseline link status, other link status) pair, from the results tables of both runs."""
    transitions = pd.crosstab(
        _link_results(baseline_dir, indexed_gt)["link_status"].rename(labels[0]),
        _link_results(other_dir, indexed_gt)["link_status"].rename(labels[1]),
    )
    transitions = transitions.reindex(index=_STATUS_ORDER, columns=_STATUS_ORDER, fill_value=0)
    return transitions.rename(index=with_emoji, columns=with_emoji)


def diff_link_results(
    baseline_dir: Path, other_dir: Path, indexed_gt: pd.DataFrame, labels: tuple[str, str] = ("baseline", "other")
) -> pd.DataFrame:
    """
    The addresses whose link status or predicted iri differs between the
    results tables (save_result_table) of two runs, with both runs' values
    labelled with `labels`, sorted by baseline then other link status.
    """
    baseline = _link_results(baseline_dir, indexed_gt)
    other = _link_results(other_dir, indexed_gt)
    differs = (baseline["link_status"] != other["link_status"]) | (baseline["pred_iri"].fillna("") != other["pred_iri"].fillna(""))
    diff = pd.DataFrame({
        "address_id": baseline.index,
        "full_address": indexed_gt["FullAddress"],
        f"link_status ({labels[0]})": baseline["link_status"],
        f"link_status ({labels[1]})": other["link_status"],
        "true_iri": [None if pd.isna(iri) else normalize_iri(iri) for iri in indexed_gt["iri"]],
        f"pred_iri ({labels[0]})": baseline["pred_iri"],
        f"pred_iri ({labels[1]})": other["pred_iri"],
    })[differs]
    status_rank = {status: i for i, status in enumerate(_STATUS_ORDER)}
    diff = diff.sort_values(
        [f"link_status ({labels[0]})", f"link_status ({labels[1]})"], key=lambda statuses: statuses.map(status_rank), kind="stable",
    ).reset_index(drop=True)
    for label in labels:
        diff[f"link_status ({label})"] = diff[f"link_status ({label})"].map(with_emoji)
    return diff


def load_parsed_addresses(cache_path: Path) -> pd.DataFrame:
    """The parsed addresses cached at `cache_path` by load_or_build_parsed_addresses (or its LLM-only variant)."""
    return pd.read_json(cache_path, orient="records", lines=True)


def _parsed_fields(row: dict) -> dict[str, Optional[str]]:
    """
    The fields the entity linking pipeline reads from a parsed row
    (entity_linking.parse_field_entities, so after its regex parse fixes):
    each entity type's text and, as "{EntityType}_iri", the geonames iri
    pre-linked during parsing, in GeographicalEntityType order.
    """
    parsed = entity_linking.parse_field_entities(row, row["prefix"])
    fields = {}
    for entity_type in entity_linking.ENTITY_TYPE_COLUMNS:
        fields[entity_type] = parsed.entity_texts.get(entity_type)
        fields[f"{entity_type}_iri"] = parsed.pre_linked_iris.get(entity_type)
    return fields


def add_changed_parse_fields(
    diff: pd.DataFrame,
    baseline_parsed: pd.DataFrame,
    other_parsed: pd.DataFrame,
    labels: tuple[str, str] = ("baseline", "other"),
) -> pd.DataFrame:
    """
    `diff` (see diff_link_results) with a "changed parse fields" column per
    run, labelled with `labels`: an OrderedDict of the _parsed_fields that
    differ between both runs' parses, each mapped to that run's value (None
    where it has none). Only adds columns: the rows are still those whose
    link status or pred_iri differ.
    """
    diff_ids = set(diff["address_id"].astype(str))
    rows_by_run = [
        {str(row["address_id"]): row for row in parsed.to_dict(orient="records") if str(row["address_id"]) in diff_ids}
        for parsed in (baseline_parsed, other_parsed)
    ]
    changed = {label: [] for label in labels}
    for address_id in diff["address_id"].astype(str):
        fields = [_parsed_fields(_without_missing(rows[address_id])) for rows in rows_by_run]
        changed_fields = [f for f in fields[0] if fields[0][f] != fields[1][f]]
        for label, run_fields in zip(labels, fields):
            changed[label].append(OrderedDict((f, run_fields[f]) for f in changed_fields))
    diff = diff.copy()
    position = diff.columns.get_loc("full_address") + 1
    for offset, label in enumerate(labels):
        diff.insert(position + offset, f"changed parse fields ({label})", changed[label])
    return diff


def check_differences_were_regex_parsed(diff: pd.DataFrame, baseline_parsed_addresses_path: Path) -> pd.DataFrame:
    """
    Sanity check that every address of `diff` (see diff_link_results) was
    regex parsed in the baseline run, as addresses the baseline already sent
    to the LLM fallback should get the same parse with LLM-only parsing.
    Prints the outcome and returns the offending addresses with their
    baseline parsing method (empty if the check passes).
    """
    parsed = load_parsed_addresses(baseline_parsed_addresses_path)
    parsing_methods = pd.Series(
        _parsing_methods(parsed.to_dict(orient="records")), index=parsed["address_id"].astype(str),
    )
    baseline_methods = diff["address_id"].astype(str).map(parsing_methods)
    # Filtered after assigning: assigning a Series to an empty (fully filtered)
    # dataframe would expand it to the Series' index.
    offending = diff.assign(baseline_parsing_method=baseline_methods)[
        baseline_methods != _PARSING_METHOD_NAMES["fully_parsed"]]
    if offending.empty:
        print(f"✅ All {len(diff)} differing addresses were regex parsed in the baseline run.")
    else:
        print(f"❌ {len(offending)}/{len(diff)} differing addresses were not regex parsed in the baseline run:")
    return offending


def build_unmatched_entities_df(addresses: pd.DataFrame, outcomes: list[entity_linking.LinkingOutcome]) -> pd.DataFrame:
    # LinkingOutcome only carries metadata for entities that made it into the
    # winning/likely candidate, not for every individual entity GeoDBSearch tried
    # and failed to match; the closest available equivalent is addresses for
    # which the search step ran but came up with no candidates at all.
    return pd.DataFrame([
        dict(
            address_id=row.address_id,
            bzk_field_name=row.field,
            full_address=row.FullAddress,
            search_status=outcome.search_status,
            disambiguation_status=outcome.disambiguation_status,
        )
        for row, outcome in zip(addresses.itertuples(), outcomes)
        if outcome.disambiguation_status == entity_linking.DisambiguationStatus.NO_CANDIDATES
    ])


def build_ambiguous_addresses_df(addresses: pd.DataFrame, outcomes: list[entity_linking.LinkingOutcome]) -> pd.DataFrame:
    # Disambiguation only keeps the first ambiguous candidate's full metadata
    # (see entity_linking.link_field); the other candidates are only available
    # as bare IRIs via ambiguous_iris.
    return pd.DataFrame([
        dict(
            address_id=row.address_id,
            bzk_field_name=row.field,
            full_address=row.FullAddress,
            num_candidates=outcome.likely_links_count,
            ambiguous_iris=outcome.ambiguous_iris,
            first_candidate_entity_type=outcome.entity_type,
            first_candidate_country=(finest_entity(outcome).country if finest_entity(outcome) else None),
            first_candidate_scores=outcome.disambiguation_scores,
        )
        for row, outcome in zip(addresses.itertuples(), outcomes)
        if outcome.disambiguation_status == entity_linking.DisambiguationStatus.AMBIGUOUS
    ])


def print_ambiguous_example(ambiguous_addresses: pd.DataFrame) -> None:
    if len(ambiguous_addresses) > 0:
        example = ambiguous_addresses.iloc[0]
        print(f"Example ambiguous address: {example['full_address']!r}")
        print(f"Candidates ({example['num_candidates']}): {example['ambiguous_iris']}")
        print(f"First candidate scores: {example['first_candidate_scores']}")
    else:
        print("No ambiguous addresses found.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    LinkStatus.CORRECTLY_LINKED: "tab:green",
    LinkStatus.PARTIALLY_LINKED: "tab:blue",
    LinkStatus.INCORRECTLY_LINKED: "tab:red",
    LinkStatus.FAILED_TO_LINK: "tab:orange",
    LinkStatus.NOT_A_LOCATION: "tab:gray",
}


_STATUS_ORDER = [
    LinkStatus.CORRECTLY_LINKED,
    LinkStatus.PARTIALLY_LINKED,
    LinkStatus.INCORRECTLY_LINKED,
    LinkStatus.FAILED_TO_LINK,
    LinkStatus.NOT_A_LOCATION,
]


def linking_status_counts(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> pd.Series:
    """Number of addresses per link_status, excluding those that are not a location."""
    counts = pd.Series(link_statuses(outcomes, ground_truth)).value_counts()
    counts = counts.drop(LinkStatus.NOT_A_LOCATION, errors="ignore")
    return counts.sort_index(key=lambda index: [_STATUS_ORDER.index(c) for c in index])


def print_linking_status_legend() -> None:
    print("Legend:")
    print(" - 'Correctly Linked' the final IRI predicted for the address is exactly the same as the ground truth IRI")
    print(" - 'Partially Linked' means the final IRI refers to a location that contains the address but is not as\n"
          "    granular as possible (e.g. the IRI for the country instead of the city)")
    print(" - 'Incorrectly Linked' means the final IRI refers to a different unrelated location")
    print(" - 'Failed to link' means either no IRI was found for the address or multiple IRIs were found with no way to\n"
          "    disambiguate between them")
    print(" - 'Not a location' means the predicted raw value does not contain any location that can be linked (e.g. Deportation)")


def plot_linking_status_pie(counts: pd.Series, title: str) -> None:
    colors = dict(CATEGORY_COLORS)
    pastel_colors = iter(plt.get_cmap("Pastel1").colors + plt.get_cmap("Pastel2").colors)
    for label in counts.index:
        if label not in colors:
            colors[label] = next(pastel_colors)
    counts.plot(kind='pie', title=title, autopct='%1.1f%%', ylabel='', figsize=(6, 6), colors=[colors[label] for label in counts.index]).figure.tight_layout()


# Human readable names for the classification of a linked entity (geonames
# "feature_class.feature_code", the wikidata class IRI, or the classification
# of the entities synthesized by the pipeline), as fnmatch wildcard patterns.
# Patterns are tried in order and the first match wins, so finer names come
# before the broader patterns they overlap with (P.PPLX before P.PPL*).
# Consistent with the possible_entity_types of build_geonames_db and
# build_geonames_db.WIKIDATA_TARGET_CLASSES. Listed coarsest to finest, which
# is also the order of the pie slices.
_CLASSIFICATION_NAME_PATTERNS = {
    "Country": ["A.PCL*", "A.TERR", "A.LTER", "A.ZN", "A.PRSH"],
    "State": ["A.ADM1", "A.ADM1H", "A.ADMD", "A.ADMDH"],
    "Region": ["L.RGN*", "H.STM*", "H.LK*", "H.RSV", "*/Q82794", "REGIONAL_TERM"],
    "District": ["A.ADM[2-5]", "A.ADM[2-5]H"],
    "Neighborhood": ["P.PPLX", "*/Q253019", "*/Q123705"],
    "City": ["P.PPL*", "*/Q486972", "*/Q262166"],
}
_CAMP_CLASSIFICATION_NAME = "Camp"
_OTHER_CLASSIFICATION_NAME = "Other"
_UNKNOWN_CLASSIFICATION_NAME = "No classification"
_CLASSIFICATION_NAME_ORDER = [
    *_CLASSIFICATION_NAME_PATTERNS, _CAMP_CLASSIFICATION_NAME, _OTHER_CLASSIFICATION_NAME, _UNKNOWN_CLASSIFICATION_NAME]


def classification_name(classification: Optional[str]) -> str:
    """Human readable name of an entity classification, see _CLASSIFICATION_NAME_PATTERNS."""
    if classification is None:
        return _UNKNOWN_CLASSIFICATION_NAME
    for name, patterns in _CLASSIFICATION_NAME_PATTERNS.items():
        if any(fnmatch.fnmatchcase(classification, pattern) for pattern in patterns):
            return name
    return _OTHER_CLASSIFICATION_NAME


def print_classification_names() -> None:
    """Prints which classifications (fnmatch wildcards, first match wins) each human readable name stands for."""
    print("Classification names (first matching name wins):")
    for name, patterns in _CLASSIFICATION_NAME_PATTERNS.items():
        print(f" - {name}: {', '.join(patterns)}")
    print(f" - {_CAMP_CLASSIFICATION_NAME}: linked by the camp/ghetto reference matcher")
    print(f" - {_OTHER_CLASSIFICATION_NAME}: any other classification")
    print(f" - {_UNKNOWN_CLASSIFICATION_NAME}: the linked entity has no classification")


def entity_classifications(connection, iris: list[str]) -> dict[str, Optional[str]]:
    """
    Classification of each of the given (normalized) IRIs in the GeoDB, keyed
    by normalized IRI. Entities synthesized by the pipeline (regional terms)
    are not in the GeoDB, and get the classification they are given there.
    """
    iris = {normalize_iri(iri) for iri in iris if iri is not None}
    classifications = {
        iri: "REGIONAL_TERM" for iri in iris if iri.startswith(geo_db_search.REGIONAL_TERM_IRI_PREFIX)}
    # The GeoDB does not store IRIs normalized (wikidata ones are http://)
    lookup_iris = [variant for iri in iris for variant in (iri, "http://" + iri.removeprefix("https://"))]
    rows = connection.execute(
        "SELECT iri, classification FROM geo_db.geographical_entities WHERE iri IN (SELECT unnest(?))",
        [lookup_iris]).fetchall()
    classifications.update({normalize_iri(iri): classification for iri, classification in rows})
    return classifications


def partial_link_classification_counts(
    outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame, connection
) -> pd.Series:
    """
    Number of partially linked addresses per (human readable) classification
    of the entity they were linked to, looked up in the GeoDB through
    `connection` (with geo_db attached, e.g. GeoDBSearch.connection).
    """
    partial_outcomes = [
        outcome
        for outcome, status in zip(outcomes, link_statuses(outcomes, ground_truth))
        if status == LinkStatus.PARTIALLY_LINKED
    ]
    classifications = entity_classifications(connection, [outcome.iri for outcome in partial_outcomes])
    names = [
        _CAMP_CLASSIFICATION_NAME if outcome.entity_type == "Camp"
        else classification_name(classifications.get(normalize_iri(outcome.iri)))
        for outcome in partial_outcomes
    ]
    counts = pd.Series(names, dtype=object).value_counts()
    return counts.sort_index(key=lambda index: [_CLASSIFICATION_NAME_ORDER.index(n) for n in index])


class FailedToLinkReason:
    NAME_NOT_MATCHED = "Name not matched"
    STILL_AMBIGUOUS = "Still ambiguous"


def failed_to_link_reason_counts(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> pd.Series:
    """
    Number of addresses that failed to link, split between those left with
    several candidates disambiguation could not choose from, and every other
    failure (no candidate found for the name).
    """
    reasons = [
        FailedToLinkReason.STILL_AMBIGUOUS
        if outcome.disambiguation_status == entity_linking.DisambiguationStatus.AMBIGUOUS
        else FailedToLinkReason.NAME_NOT_MATCHED
        for outcome, status in zip(outcomes, link_statuses(outcomes, ground_truth))
        if status == LinkStatus.FAILED_TO_LINK
    ]
    counts = pd.Series(reasons, dtype=object).value_counts()
    order = [FailedToLinkReason.NAME_NOT_MATCHED, FailedToLinkReason.STILL_AMBIGUOUS]
    return counts.sort_index(key=lambda index: [order.index(r) for r in index])


def _pie_label_with_count(total: int) -> Callable[[float], str]:
    return lambda pct: f"{pct:.1f}%\n({round(pct * total / 100)})"


def plot_partial_and_failed_link_pies(
    partial_link_counts: pd.Series, failed_to_link_counts: pd.Series, title: Optional[str] = None
) -> None:
    """Side by side pies: partial links by linked entity classification (left), failed links by reason (right)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, counts, ax_title, colormap in (
        (left, partial_link_counts, f"{LinkStatus.PARTIALLY_LINKED}: linked up to", "Blues_r"),
        (right, failed_to_link_counts, f"{LinkStatus.FAILED_TO_LINK}: reason", "Oranges_r"),
    ):
        if counts.empty:
            ax.set_title(ax_title)
            ax.text(0.5, 0.5, "No addresses", ha="center", va="center")
            ax.axis("off")
            continue
        colors = plt.get_cmap(colormap)([0.2 + 0.6 * i / max(1, len(counts)) for i in range(len(counts))])
        counts.plot(
            kind="pie", ax=ax, title=f"{ax_title}\n({counts.sum()} addresses)", ylabel="",
            autopct=_pie_label_with_count(counts.sum()), colors=colors)
    if title:
        fig.suptitle(title)
    fig.tight_layout()


# Link statuses the pies of plot_pies_by_link_status are split by
_PIE_LINK_STATUSES = [LinkStatus.CORRECTLY_LINKED, LinkStatus.INCORRECTLY_LINKED]
# Categories of the pies of plot_pies_by_link_status, each in a fixed order
# that gives every category the same color in every pie it appears in (see
# _pie_category_styles). The disambiguation factors are shared by the deciding
# factor and the weighted score contribution pies.
_DISAMBIGUATION_FACTOR_ORDER = [
    entity_linking.UNAMBIGUOUS_DECIDING_FACTOR,
    *geo_disambiguation.comparison_step_labels(
        geo_disambiguation.DISAMBIGUATION_FACTOR_PRIMARY_PRIORITY,
        geo_disambiguation.DISAMBIGUATION_FACTOR_SECONDARY_PRIORITY),
    entity_linking.COMMON_PARENT_DECIDING_FACTOR,
]
# Phases of TantivySearchIndex.search in the order they are tried (with the
# phonetic and edit distance 1 matches of the same phase told apart), then the
# matches obtained without a text search (see MatchedName.search_phase)
_SEARCH_PHASE_ORDER = [
    "exact", "abbreviation", geo_db_search.PHONETIC_SEARCH_PHASE, geo_db_search.FUZZY_DISTANCE_1_SEARCH_PHASE,
    "fuzzy(distance=2)", "partial_word", "regional_term", "pre_linked", "common_parent",
]
# Human readable names of the "{prefix}.status" of a parsed address
_PARSING_METHOD_NAMES = {"fully_parsed": "Regex", "llm_parsed": "LLM", "llm_error": "LLM (failed)"}
_PARSING_METHOD_ORDER = list(_PARSING_METHOD_NAMES.values())
# Categorical slots, in fixed order; past the last one, categories reuse the
# slots' hues with the next hatch (hue x texture composite encoding)
_PIE_CATEGORY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_PIE_CATEGORY_HATCHES = ["", "//", "\\\\", "xx"]


def _pie_category_styles(categories: list[str], order: list[str]) -> dict[str, tuple[str, str]]:
    """
    (color, hatch) of each category, by its position in `order` (then, for
    categories not in it, after it), so that it does not depend on which
    categories a given pie shows.
    """
    order = order + sorted(c for c in categories if c not in order)
    styles = {}
    for category in categories:
        index = order.index(category)
        slot, hatch = index % len(_PIE_CATEGORY_COLORS), index // len(_PIE_CATEGORY_COLORS)
        styles[category] = (_PIE_CATEGORY_COLORS[slot], _PIE_CATEGORY_HATCHES[hatch % len(_PIE_CATEGORY_HATCHES)])
    return styles


def _counts_by_link_status(values: list[Optional[str]], statuses: list[str], order: list[str]) -> pd.DataFrame:
    """
    Number of correctly and incorrectly linked addresses (columns) per value
    (rows, in `order`, then any value not in it) of `values`, aligned with
    their `statuses`, skipping None values.
    """
    counts_by_status = defaultdict(lambda: defaultdict(int))
    for value, status in zip(values, statuses):
        if status in _PIE_LINK_STATUSES and value is not None:
            counts_by_status[status][value] += 1
    counts = pd.DataFrame(
        {status: pd.Series(counts_by_status[status], dtype=int) for status in _PIE_LINK_STATUSES}
    ).fillna(0).astype(int)
    counts = counts.sort_index(key=lambda index: [order.index(v) if v in order else len(order) for v in index])
    counts.attrs["category_order"] = order
    return counts


def _incorrectly_linked_by_category(
    addresses: pd.DataFrame,
    outcomes: list[entity_linking.LinkingOutcome],
    ground_truth: pd.DataFrame,
    values: list[Optional[str]],
    category_name: str,
    order: list[str],
) -> pd.DataFrame:
    """
    Every incorrectly linked address with its value (in a `category_name`
    column) of `values`, the category it falls in in the incorrectly linked pie
    of plot_pies_by_link_status (see _counts_by_link_status), sorted by
    category in `order`, skipping None values like the pie does.
    """
    records = []
    for row, outcome, value, status in zip(
        addresses.itertuples(), outcomes, values, link_statuses(outcomes, ground_truth)
    ):
        if status != LinkStatus.INCORRECTLY_LINKED or value is None:
            continue
        finest = finest_entity(outcome)
        records.append({
            category_name: value,
            "address_id": row.address_id,
            "card_id": row.card_id,
            "bzk_field_name": row.field,
            "full_address": row.FullAddress,
            "true_iri": None if pd.isna(row.iri) else normalize_iri(row.iri),
            "pred_iri": normalize_iri(outcome.iri),
            "pred_entity_type": outcome.entity_type,
            "pred_raw_text": finest.raw_text if finest else None,
            "pred_matched_name": finest.matching.get("matched_name") if finest else None,
            "pred_country": finest.country if finest else None,
        })
    samples = pd.DataFrame(records, columns=[
        category_name, "address_id", "card_id", "bzk_field_name", "full_address", "true_iri", "pred_iri",
        "pred_entity_type", "pred_raw_text", "pred_matched_name", "pred_country",
    ])
    rank = {v: i for i, v in enumerate(order)}
    return samples.sort_values(
        category_name, key=lambda column: column.map(lambda v: rank.get(v, len(order))), kind="stable"
    ).reset_index(drop=True)


def deciding_factor_counts(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Number of correctly and incorrectly linked addresses per factor the
    disambiguation between their candidates hinged on (see
    entity_linking._deciding_factor), for the addresses linked through GeoDBSearch.
    """
    return _counts_by_link_status(
        [outcome.deciding_factor for outcome in outcomes], link_statuses(outcomes, ground_truth),
        _DISAMBIGUATION_FACTOR_ORDER)


def incorrectly_linked_by_deciding_factor(
    addresses: pd.DataFrame, outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """The incorrectly linked addresses counted by deciding_factor_counts, with their deciding factor."""
    return _incorrectly_linked_by_category(
        addresses, outcomes, ground_truth, [outcome.deciding_factor for outcome in outcomes],
        "deciding_factor", _DISAMBIGUATION_FACTOR_ORDER)


def weighted_score_contributions(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Average contribution of each weighted factor (rows) to the weighted score
    of the linked candidate (see LinkingOutcome.weighted_score_contributions),
    for the correctly and incorrectly linked addresses (columns) whose
    disambiguation hinged on the weighted score.
    """
    contributions_by_status = defaultdict(list)
    for outcome, status in zip(outcomes, link_statuses(outcomes, ground_truth)):
        if status in _PIE_LINK_STATUSES and outcome.deciding_factor == "weighted_score":
            contributions_by_status[status].append(outcome.weighted_score_contributions)
    means = pd.DataFrame({
        status: pd.DataFrame(contributions_by_status[status], dtype=float).fillna(0.0).mean()
        for status in _PIE_LINK_STATUSES
    }).fillna(0.0)
    order = _DISAMBIGUATION_FACTOR_ORDER
    means = means.sort_index(key=lambda index: [order.index(v) if v in order else len(order) for v in index])
    means.attrs["category_order"] = order
    means.attrs["address_counts"] = {status: len(contributions_by_status[status]) for status in _PIE_LINK_STATUSES}
    return means


def _finest_search_phase(outcome: entity_linking.LinkingOutcome) -> Optional[str]:
    """
    The phase of GeoDBSearch (see MatchedName.search_phase) that retrieved the
    match of the finest entity of the linked candidate, the one its IRI is
    that of ("common_parent" for one linked to the common parent of tied
    candidates), or None when it was not linked through GeoDBSearch.
    """
    finest = finest_entity(outcome)
    return finest.matching.get("search_phase") if finest is not None else None


def finest_search_phase_counts(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Number of correctly and incorrectly linked addresses per phase of
    GeoDBSearch that retrieved the match of the linked candidate's finest
    entity (see _finest_search_phase).
    """
    return _counts_by_link_status(
        [_finest_search_phase(outcome) for outcome in outcomes], link_statuses(outcomes, ground_truth),
        _SEARCH_PHASE_ORDER)


def incorrectly_linked_by_finest_search_phase(
    addresses: pd.DataFrame, outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """The incorrectly linked addresses counted by finest_search_phase_counts, with their search phase."""
    return _incorrectly_linked_by_category(
        addresses, outcomes, ground_truth, [_finest_search_phase(outcome) for outcome in outcomes],
        "finest_search_phase", _SEARCH_PHASE_ORDER)


def _parsing_methods(parsed_rows: list[dict]) -> list[Optional[str]]:
    """Human readable name of the method that parsed each of `parsed_rows` (see _PARSING_METHOD_NAMES)."""
    methods = []
    for parsed_row in parsed_rows:
        status = entity_linking._clean_optional_str(parsed_row.get(f"{parsed_row['prefix']}.status"))
        methods.append(_PARSING_METHOD_NAMES.get(status, status))
    return methods


def parsing_method_counts(
    outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame, parsed_rows: list[dict]
) -> pd.DataFrame:
    """
    Number of correctly and incorrectly linked addresses per method that
    parsed them (regex, or the LLM fallback), from the parsed rows the
    outcomes were linked from (see run_pipeline).
    """
    return _counts_by_link_status(
        _parsing_methods(parsed_rows), link_statuses(outcomes, ground_truth), _PARSING_METHOD_ORDER)


def incorrectly_linked_by_parsing_method(
    addresses: pd.DataFrame,
    outcomes: list[entity_linking.LinkingOutcome],
    ground_truth: pd.DataFrame,
    parsed_rows: list[dict],
) -> pd.DataFrame:
    """The incorrectly linked addresses counted by parsing_method_counts, with their parsing method."""
    return _incorrectly_linked_by_category(
        addresses, outcomes, ground_truth, _parsing_methods(parsed_rows), "parsing_method", _PARSING_METHOD_ORDER)


def _plot_pie_row(axes, values: pd.DataFrame, styles: dict[str, tuple[str, str]], min_labeled_pct: float) -> None:
    """One pie per link status column of `values` (see plot_pies_by_link_status) on `axes`."""
    is_count = pd.api.types.is_integer_dtype(values.values.dtype)
    address_counts = values.attrs.get("address_counts", values.sum().to_dict())
    for ax, status in zip(axes, values.columns):
        status_values = values[status][values[status] > 0]
        total = status_values.sum()
        subtitle = f"{address_counts[status]} addresses"
        if not is_count:
            subtitle += f", average total {total:.3f}"
        ax.set_title(f"{status}\n({subtitle})")
        if total == 0:
            ax.text(0.5, 0.5, "No addresses", ha="center", va="center")
            ax.axis("off")
            continue
        label = (
            _pie_label_with_count(total) if is_count
            else lambda pct: f"{pct:.1f}%\n({pct * total / 100:.3f})")
        wedges, _, _ = ax.pie(
            status_values, colors=[styles[c][0] for c in status_values.index],
            autopct=lambda pct: label(pct) if pct >= min_labeled_pct else "",
            startangle=90, counterclock=False, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=2), textprops=dict(color="#0b0b0b", fontsize=9))
        for wedge, category in zip(wedges, status_values.index):
            wedge.set_hatch(styles[category][1])
        ax.axis("equal")


def plot_pies_by_link_status(
    values: pd.DataFrame | dict[str, pd.DataFrame], title: str, min_labeled_pct: float = 5.0
) -> None:
    """
    Side by side pies of the rows of `values`, one per link status column:
    address counts (as built by deciding_factor_counts,
    finest_search_phase_counts and parsing_method_counts) or average
    contributions (as built by weighted_score_contributions). Several of these
    tables, keyed by a title for each, are stacked in rows of the same figure
    sharing one legend. Each category keeps the same color in every pie (see
    _pie_category_styles); slices under `min_labeled_pct` are left to the legend.
    """
    rows = values if isinstance(values, dict) else {None: values}
    categories = list(dict.fromkeys(c for row_values in rows.values() for c in row_values.index))
    order = list(dict.fromkeys(c for row_values in rows.values() for c in row_values.attrs.get("category_order", [])))
    styles = _pie_category_styles(categories, order)
    n_columns = max(len(row_values.columns) for row_values in rows.values())
    fig = plt.figure(figsize=(6 * n_columns, 6 * len(rows) + 0.3 * ((len(styles) + 3) // 4)), layout="constrained")
    subfigures = fig.subfigures(len(rows), 1, squeeze=False)[:, 0]
    for subfigure, (row_title, row_values) in zip(subfigures, rows.items()):
        if row_title is not None:
            subfigure.suptitle(row_title, fontweight="bold")
        _plot_pie_row(subfigure.subplots(1, n_columns, squeeze=False)[0], row_values, styles, min_labeled_pct)
    fig.legend(
        handles=[
            Patch(facecolor=color, hatch=hatch, edgecolor="white", label=category)
            for category, (color, hatch) in styles.items()
        ],
        loc="outside lower center", ncol=min(4, len(styles)), frameon=False)
    fig.suptitle(title, fontsize="x-large")
    # shown right away, so that several of these in one cell interleave with what it displays
    plt.show()


def country_occurrences(outcomes: list[entity_linking.LinkingOutcome]) -> pd.Series:
    occurrences = defaultdict(int)
    for outcome in outcomes:
        finest = finest_entity(outcome)
        if finest is not None and finest.country is not None:
            occurrences[finest.country] += 1
    return pd.Series(occurrences).sort_values(ascending=False)


def plot_country_pie(occurrences: pd.Series, title: str, top_n: int = 5) -> None:
    top = pd.concat([occurrences.iloc[:top_n], pd.Series({"Other": occurrences.iloc[top_n:].sum()})])
    top.plot(kind='pie', title=title, autopct='%1.1f%%', ylabel='', figsize=(6, 6), colormap='Set3').figure.tight_layout()


def display_country_occurrences(occurrences: pd.Series) -> None:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(occurrences.to_frame("Count").style.bar(subset=["Count"]).format("{:,.0f}"))


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _entity_iri(entity) -> Optional[str]:
    return normalize_iri(entity.linked_to.geographical_name.entity.iri)


def _describe_camp_match(match) -> str:
    return f"{match.label!r} -> {normalize_iri(match.iri)} (wikidata {match.wikidata_iri}, tags {sorted(match.tags)})"


def _display_block(lines: list[str]) -> None:
    """Displays `lines` as one preformatted output block (keeps the column alignment)."""
    display(Markdown("\n\n".join(lines)))


class DisambiguationErrorExplainer:
    """
    Explains, for a single address, why the full entity linking pipeline
    (entity_linking.link_field) produced no link or a different link than the
    ground truth. See explain().
    """
    # Ground-truth columns above City that the parser merges into AboveCity.
    _ABOVE_CITY_GT_COLUMNS = ["District", "Region", "State", "Country"]

    def __init__(
        self,
        indexed_gt: pd.DataFrame,
        indexed_parsed: pd.DataFrame,
        geo_db_searcher: geo_db_search.GeoDBSearch,
        camp_matcher: CampReferenceMatcher,
        disambiguator: Disambiguator,
    ):
        self.indexed_gt = indexed_gt
        self.indexed_parsed = indexed_parsed
        self.geo_db_searcher = geo_db_searcher
        self.camp_matcher = camp_matcher
        self.disambiguator = disambiguator
        # Ground-truth columns that name geographical entities, in the same terms as
        # the parsed "{prefix}.{EntityType}.text" columns (the ground truth has no
        # AboveCity, which the parser uses for any of the coarser types above).
        self._gt_entity_columns = [
            t for t in entity_linking.ENTITY_TYPE_COLUMNS if t in indexed_gt.columns and t != "AboveCity"
        ]

    def explain(self, address_id, verbose: bool = False) -> None:
        """
        Given an address_id, explains why the full entity linking pipeline
        (entity_linking.link_field) produced no link or a different link than
        the ground truth for it:

        1. Before linking: the ground truth, and how the parse result differs
           from its annotation.
        2. Linking: link_field runs on the cached regex/LLM-parsed row exactly
           as in the pipeline; with verbose=True its debug logs walk
           through each step (parsed entities, pre-linked ids, tagging, camp
           matching, GeoDBSearch and disambiguation).
        3. After linking: the outcome compared against the ground truth, at the
           step where linking stopped; for GeoDBSearch + disambiguation, the
           winning candidate's per-factor scores against the true candidate's,
           in DISAMBIGUATION_FACTOR_PRIORITY order, to surface the deciding factor
           (see geo_disambiguation.comparison_steps).
        """
        current_level = geo_db_search.ENTITY_LINKING_LOGGER.level
        if verbose:
            geo_db_search.ENTITY_LINKING_LOGGER.setLevel(logging.DEBUG)
        try:
            self._explain(address_id)
        finally:
            geo_db_search.ENTITY_LINKING_LOGGER.setLevel(current_level)  # Reset to default level

    def _explain(self, address_id) -> None:
        true_iri, true_entity_type, true_raw_text = self._ground_truth(address_id)
        parsed_row = self.indexed_parsed.loc[str(address_id)]
        prefix = parsed_row["prefix"]
        sb = []
        sb.append(f"Address {address_id!r} ({parsed_row.get(f'{prefix}.raw')!r}), field {entity_linking.FIELD_PREFIXES[prefix]}")
        if true_iri is None:
            sb.append(f"  ground truth: not a linkable location")
        else:
            sb.append(f"  ground truth: {true_entity_type} {true_raw_text!r} -> {true_iri}")
        differences = self._parse_differences(address_id, parsed_row, prefix)
        if differences:
            sb.append("  parse result differences from the ground-truth annotation:")
            sb.append(f"    {'entity type':<13} {'parsed':<30} {'ground truth':<30}")
            for entity_type, parsed_text, true_text in differences:
                marker = "  <- linked entity" if entity_type == true_entity_type else ""
                sb.append(f"    {entity_type:<13} {parsed_text!r:<30} {true_text!r:<30}{marker}")
        else:
            sb.append("  parse result agrees with the ground-truth annotation")
        missed_word_entities = [
            entity for entity in entity_linking.parse_field_entities(parsed_row.to_dict(), prefix).entities
            if entity.is_missed_word
        ]
        if missed_word_entities:
            sb.append("  entities added from words parsing left unassigned (missed word recovery):")
            for entity in missed_word_entities:
                sb.append(f"    {entity.entity_type.name:<13} {entity.raw_text!r}")
        _display_block(sb)
        sb.clear()

        outcome = self._link(address_id)

        pred_iri = normalize_iri(outcome.iri)
        sb.append(f"  predicted:    {outcome.entity_type} -> {pred_iri}")
        sb.append(f"  search status: {outcome.search_status}, disambiguation status: {outcome.disambiguation_status}, tags: {list(outcome.tags)}")
        _display_block(sb)
        sb.clear()
        self._display_address_entities(outcome.address)
        if pred_iri == true_iri:
            sb.append("✅ The pipeline already links this address correctly.")
        else:
            self._explain_outcome(sb, outcome, true_iri, true_entity_type, true_raw_text)
        _display_block(sb)

    def _link(self, address_id) -> entity_linking.LinkingOutcome:
        """Runs link_field on the cached parsed row of the address exactly as in the pipeline, keeping the address."""
        parsed_row = self.indexed_parsed.loc[str(address_id)]
        row = {**parsed_row.to_dict(), "address_id": str(address_id)}
        return entity_linking.link_field(
            row, parsed_row["prefix"], self.camp_matcher, self.geo_db_searcher, self.disambiguator, keep_address=True
        )

    def disambiguation_diagram(
            self,
            address_id,
            iris: Optional[list[str]] = None,
            entities: Optional[tuple[str, str]] = None,
            top: Optional[int] = None,
            show: bool = True,
        ) -> Optional[str]:
        """
        A mermaid (block-beta) diagram of the disambiguation of an address:
        the steps of Disambiguator.disambiguate it went through, then the
        cross matching (see Disambiguator._score_ambiguous_matches) of two of
        its linkable entities as a table, with a card for the search match of
        each row and column:

        - rows: the matches of the reference entity, each the reference match
          of a candidate address (the last column: its best candidate left
          after pruning, if any);
        - columns: the matches of another entity, each cell holding the
          scores of the column match cross matched against the row match;
          grey when pruned (see Disambiguator._prune_match), outlined blue
          when the best unpruned one of its row (the one the candidate takes);
        - green: the chosen combination (the linked address); orange: the
          deciding factor, against the best candidate of another IRI (the
          runner-up), whose row it also outlines.

        iris restricts the rows and columns to the matches of these IRIs (the
        cross matching still runs over all of them), as the table of an
        address with many matches is otherwise unreadable. entities is the
        raw text of the (row, column) entities; by default, the reference
        entity of the most likely candidate and its companion closest to it
        in the order disambiguate goes through the entities in.

        top, applied after iris, keeps only that many rows, those of the best
        ranked candidates (rows left without one last), and that many columns,
        those of the best cross match against any row kept (unpruned first),
        each in that order.

        Displays the diagram if show, and returns its source; None if the
        address has fewer than two entities with search matches.
        """
        outcome = self._link(address_id)
        address = outcome.address
        linkable = [
            e for e in (address.entities if address is not None else [])
            if isinstance(e, MatchedEntity) and e.matches
        ]
        if len(linkable) < 2:
            display(Markdown(
                f"Address {address_id!r} has {len(linkable)} entity with search matches: nothing to cross match "
                f"(search status {outcome.search_status})."))
            return None
        # the order disambiguate takes the entities as reference entity in
        linkable.sort(
            key=lambda e: (1 if e.entity_type == geo_disambiguation.GeographicalEntityType.City else 0, e.entity_type),
            reverse=True)
        _, likely = self._most_likely_candidate(address)
        if entities is not None:
            by_text = {e.raw_text: e for e in linkable}
            missing = [text for text in entities if text not in by_text]
            if missing:
                raise ValueError(f"No entity with search matches has the raw text {missing}; "
                                 f"those of the address are {list(by_text)}")
            row_entity, column_entity = by_text[entities[0]], by_text[entities[1]]
        else:
            reference_id = likely.reference_entity.address_entity_id if likely is not None else None
            # a split part (see Disambiguator._split_entity) is no entity of the address
            row_entity = next((e for e in linkable if e.address_entity_id == reference_id), linkable[0])
            linked_ids = {e.address_entity_id for e in likely.entities} if likely is not None else set()
            others = [e for e in linkable if e is not row_entity]
            column_entity = next((e for e in others if e.address_entity_id in linked_ids), others[0])

        wanted = None if iris is None else {normalize_iri(iri) for iri in iris}

        def shown(match) -> bool:
            return wanted is None or normalize_iri(match.geographical_name.entity.iri) in wanted

        def match_key(match) -> tuple[str, str]:
            return match.geographical_name.entity.iri, match.nfc_alt_name

        disambiguator = self.disambiguator

        def rank(candidate) -> tuple:
            return geo_disambiguation._score_dict_to_tuple(candidate.scores, disambiguator.priority)

        def best_candidate_of(row_match):
            """The best candidate address left after pruning with row_match as the reference match, if any."""
            return max((
                c for c in candidates
                if c.reference_entity.address_entity_id == row_entity.address_entity_id
                and match_key(c.reference_entity.linked_to) == match_key(row_match)
            ), key=rank, default=None)

        candidates = address.possible_links or ()
        rows = [m for m in row_entity.matches if shown(m)]
        if top is not None:
            # sorted is stable: rows without a candidate stay in search order
            rows = sorted(rows, key=lambda m: (
                (1, rank(c)) if (c := best_candidate_of(m)) is not None else (0, ())), reverse=True)[:top]
        row_candidates = {
            i: c for i, row_match in enumerate(rows) if (c := best_candidate_of(row_match)) is not None}
        # index among all the column entity's matches of each shown column
        column_indices = [j for j, m in enumerate(column_entity.matches) if shown(m)]

        # cross matching, as in Disambiguator._score_ambiguous_matches (over
        # every column match, so that the best of a row is the actual one)
        bzk_field = address.bzk_field_name
        cells: dict[tuple[int, int], tuple[dict, bool]] = {}
        best_column: dict[int, Optional[int]] = {}
        ruled_out: set[int] = set()
        current_level = geo_db_search.ENTITY_LINKING_LOGGER.level
        geo_db_search.ENTITY_LINKING_LOGGER.setLevel(logging.WARNING)
        try:
            for i, row_match in enumerate(rows):
                if disambiguator._is_entity_type_ruled_out(row_entity, row_match):
                    ruled_out.add(i)
                    continue
                best, best_scores = None, None
                for j, column_match in enumerate(column_entity.matches):
                    scored = disambiguator._score_individual_match(column_entity, column_match, bzk_field)
                    if row_entity.entity_type < column_entity.entity_type:
                        likelihood = disambiguator._score_parent_child(row_match, column_match)
                    else:
                        likelihood = disambiguator._score_parent_child(column_match, row_match)
                    scored.scores["child_parent_likelihood"] = likelihood
                    scored.scores["weighted_score"] = disambiguator._calculate_weighted_score(scored.scores)
                    pruned = disambiguator._prune_match(row_match, column_entity, scored)
                    cells[i, j] = (scored.scores, pruned)
                    if not pruned and (best is None or geo_disambiguation._score_dict_to_tuple(
                            scored.scores, disambiguator.priority) > best_scores):
                        best, best_scores = j, geo_disambiguation._score_dict_to_tuple(
                            scored.scores, disambiguator.priority)
                best_column[i] = best
        finally:
            geo_db_search.ENTITY_LINKING_LOGGER.setLevel(current_level)
        if top is not None:
            def column_rank(j: int) -> tuple:
                return max((
                    (not pruned, geo_disambiguation._score_dict_to_tuple(scores, disambiguator.priority))
                    for (_, cell_j), (scores, pruned) in cells.items() if cell_j == j
                ), default=(False, ()))
            column_indices = sorted(column_indices, key=column_rank, reverse=True)[:top]
        columns = [column_entity.matches[j] for j in column_indices]


        def finest_iri(candidate) -> str:
            return candidate.finest_grain_entity.linked_to.geographical_name.entity.iri

        linked = address.linked_to if not address.linked_to_common_parent else None
        runner_up, decided = None, None
        if linked is not None:
            runner_up = max(
                (c for c in candidates if finest_iri(c) != finest_iri(linked)), key=rank, default=None)
            if runner_up is not None:
                decided = geo_disambiguation.deciding_factor(
                    linked.scores, runner_up.scores, disambiguator.comparison_steps)
        deciding = geo_disambiguation.factor_of_label(decided[0]) if decided is not None else None

        def is_row_of(candidate, i) -> bool:
            return candidate is not None and row_candidates.get(i) is candidate

        def column_of(candidate) -> Optional[int]:
            """The column of the match the candidate links the column entity to."""
            linked_column = next(
                (e.linked_to for e in candidate.entities if e.address_entity_id == column_entity.address_entity_id),
                None)
            if linked_column is None:
                return None
            return next((j for j, m in enumerate(columns) if match_key(m) == match_key(linked_column)), None)

        def text(value) -> str:
            # mermaid entity codes, so that labels may hold any text
            return (str(value).replace("#", "#35;").replace('"', "#quot;")
                    .replace("<", "#lt;").replace(">", "#gt;"))

        def score(scores, factor) -> str:
            value = scores.get(factor)
            if value is None:
                return "—"
            value = value.score if isinstance(value, geo_disambiguation.AnnotatedScore) else value
            return f"{value:.3g}"

        def card(match) -> str:
            entity = match.geographical_name.entity
            population = f"{entity.population:,}" if entity.population is not None else "unknown"
            country = entity.country.iso_code if entity.country is not None else "—"
            return "<br/>".join([
                f"<b>{text(match.nfc_alt_name)}</b>",
                text(f"{entity.name} · {country}"),
                text(classification_name(entity.classification)),
                text(f"pop. {population}"),
                text(f"fuzzy {match.fuzzy_score:.2f} · phon. {match.phonetic_score:.2f}"),
                f"<i>{text(normalize_iri(entity.iri))}</i>",
            ])

        def factor_line(scores) -> str:
            if deciding is None or deciding in ("weighted_score", "child_parent_likelihood") or deciding not in scores:
                return ""
            return f"<br/><b>{text(deciding)} {score(scores, deciding)}</b>"

        width = len(columns) + 2
        lines = ["block-beta", f"  columns {width}"]
        # class of each styled block; the most telling of those it qualifies for
        class_precedence = ["header", "pruned", "best", "tied", "decision", "runnerup", "chosen"]
        block_classes: dict[str, str] = {}

        def style(block_id: str, name: str) -> None:
            current = block_classes.get(block_id)
            if current is None or class_precedence.index(name) > class_precedence.index(current):
                block_classes[block_id] = name

        raw_address = address.full_address
        lines.append(f'  heading["<b>Address {text(address_id)}</b>: {text(repr(raw_address))} '
                     f'({text(bzk_field.name)})"]:{width}')

        # the procedure
        entity_list = "<br/>".join(
            text(f"{e.entity_type.name} {e.raw_text!r}: {len(e.matches)} match{'es' if len(e.matches) != 1 else ''}")
            for e in linkable)
        candidate_count = len(candidates)
        if linked is not None:
            decision = text(f"linked to {linked.finest_grain_entity.linked_to.nfc_alt_name!r} "
                            f"({normalize_iri(finest_iri(linked))})")
            if decided is not None:
                label, a, b = decided
                decision += (f"<br/>over {text(repr(runner_up.finest_grain_entity.linked_to.nfc_alt_name))} "
                             f"({text(normalize_iri(finest_iri(runner_up)))})<br/>deciding factor: "
                             f"<b>{text(label)}</b> {a:.3g} vs {b:.3g}")
            elif runner_up is None:
                decision += "<br/>the only candidate IRI"
        elif address.linked_to_common_parent:
            decision = text(f"{len(address.likely_links)} candidates tie: linked to their common parent "
                            f"{address.linked_to.finest_grain_entity.linked_to.nfc_alt_name!r}")
        else:
            decision = text(f"left unlinked ({len(address.likely_links or ())} tied candidates)"
                            if candidate_count else "no candidate survives pruning: left unlinked")
        lines += [
            f"  block:steps:{width}",
            "    columns 4",
            f'    step1["<b>1. Searched entities</b><br/>{entity_list}"]',
            f'    step2["<b>2. Cross matching</b><br/>each match of the reference entity '
            f'{text(row_entity.entity_type.name)} {text(repr(row_entity.raw_text))} '
            f'against the matches of the other entities,<br/>keeping per entity the best unpruned one"]',
            f'    step3["<b>3. Candidate addresses</b><br/>{candidate_count} left after pruning"]',
            f'    step4["<b>4. Decision</b><br/>{decision}"]',
            "  end",
            "  step1 --> step2",
            "  step2 --> step3",
            "  step3 --> step4",
        ]
        style("step4", "decision")

        # the table: header row of column cards
        hidden_rows = len(row_entity.matches) - len(rows)
        hidden_columns = len(column_entity.matches) - len(columns)
        corner = (f"<b>rows</b>: reference entity<br/>{text(row_entity.entity_type.name)} "
                  f"{text(repr(row_entity.raw_text))}<br/><b>columns</b>: "
                  f"{text(column_entity.entity_type.name)} {text(repr(column_entity.raw_text))}")
        if hidden_rows or hidden_columns:
            corner += f"<br/><i>{hidden_rows} rows and {hidden_columns} columns filtered out</i>"
        lines.append(f'  corner["{corner}"]')
        for j, column_match in enumerate(columns):
            lines.append(f'  col{j}["{card(column_match)}"]')
        lines.append(f'  candhead["<b>best candidate of the row</b><br/>weighted score'
                     f'{"<br/>" + text(deciding) if deciding not in (None, "weighted_score") else ""}"]')
        style("corner", "header")
        style("candhead", "header")

        for i, row_match in enumerate(rows):
            lines.append(f'  row{i}["{card(row_match)}"]')
            for j, j_all in enumerate(column_indices):
                cell_id = f"cell{i}_{j}"
                if i in ruled_out:
                    lines.append(f'  {cell_id}[" "]')
                    style(cell_id, "pruned")
                    continue
                scores, pruned = cells[i, j_all]
                label = (f"child/parent {score(scores, 'child_parent_likelihood')}<br/>"
                         f"weighted {score(scores, 'weighted_score')}{factor_line(scores)}")
                if pruned:
                    label = "✂ pruned<br/>" + label
                    style(cell_id, "pruned")
                elif best_column.get(i) == j_all:
                    style(cell_id, "best")
                lines.append(f'  {cell_id}["{label}"]')
            candidate = row_candidates.get(i)
            cand_id = f"cand{i}"
            if i in ruled_out:
                label = "entity type ruled out"
                style(cand_id, "pruned")
            elif candidate is None:
                label = "✂ pruned"
                style(cand_id, "pruned")
            else:
                companion = next(
                    (e.linked_to for e in candidate.entities
                     if e.address_entity_id == column_entity.address_entity_id), None)
                label = f"weighted {score(candidate.scores, 'weighted_score')}"
                if deciding == "weighted_score":
                    label = f"<b>{label}</b>"
                elif deciding is not None:
                    label += f"<br/><b>{text(deciding)} {score(candidate.scores, deciding)}</b>"
                label += ("<br/>with " + text(repr(companion.nfc_alt_name)) if companion is not None
                          else f"<br/>without {text(repr(column_entity.raw_text))}")
                if best_column.get(i) is not None and best_column[i] not in column_indices and companion is not None:
                    label += " <i>(filtered out)</i>"
            lines.append(f'  {cand_id}["{label}"]')
            if linked is not None and is_row_of(linked, i):
                style(f"row{i}", "chosen")
                style(cand_id, "chosen")
                chosen_column = column_of(linked)
                if chosen_column is not None:
                    style(f"cell{i}_{chosen_column}", "chosen")
                    style(f"col{chosen_column}", "chosen")
            elif runner_up is not None and is_row_of(runner_up, i):
                style(f"row{i}", "runnerup")
                style(cand_id, "runnerup")
            elif linked is None and any(is_row_of(c, i) for c in address.likely_links or ()):
                style(f"row{i}", "tied")
                style(cand_id, "tied")

        lines.append(
            f'  legend["<b>green</b>: chosen combination · <b>blue outline</b>: best unpruned match of the row, '
            f'taken by its candidate · <b>grey</b>: pruned · <b>orange</b>: deciding factor and runner-up · '
            f'<b>yellow</b>: tied candidates · candidates may also link entities not shown"]:{width}')
        style("legend", "header")

        styles = {
            "header": "fill:#eceff1,stroke:#90a4ae,color:#263238",
            "pruned": "fill:#e0e0e0,stroke:#bdbdbd,color:#9e9e9e",
            "best": "fill:#e3f2fd,stroke:#1e88e5,stroke-width:3px,color:#0d47a1",
            "chosen": "fill:#c8e6c9,stroke:#2e7d32,stroke-width:4px,color:#1b5e20",
            "runnerup": "fill:#fff3e0,stroke:#ef6c00,stroke-width:3px,color:#e65100",
            "tied": "fill:#fff9c4,stroke:#f9a825,stroke-width:3px,color:#5d4037",
            "decision": "fill:#fff3e0,stroke:#ef6c00,stroke-width:3px,color:#263238",
        }
        for name, definition in styles.items():
            ids = [block_id for block_id, block_class in block_classes.items() if block_class == name]
            if ids:
                lines.append(f"  classDef {name} {definition}")
                lines += [f"  class {block_id} {name}" for block_id in ids]
        diagram = "\n".join(lines)
        if show:
            display(Markdown(f"```mermaid\n{diagram}\n```"))
        return diagram

    @staticmethod
    def _most_likely_candidate(address) -> tuple[Optional[str], Optional[LinkedAddress]]:
        """
        (description, candidate) of the most likely linked address: the linked
        one, or else (still ambiguous, or linked to the candidates' common
        parent, which is not one of them) the best ranked likely/possible one.
        """
        if address is None:
            return None, None
        if address.linked_to is not None and not address.linked_to_common_parent:
            return "linked address", address.linked_to
        if address.likely_links:
            return "best ranked of the tied likely addresses", address.likely_links[0]
        if address.possible_links:
            return "best ranked possible address", address.possible_links[0]
        return None, None

    def _display_address_entities(self, address) -> None:
        """
        Every entity of the address, with its raw value and what it is linked
        to in the most likely linked address (see _most_likely_candidate).
        """
        if address is None or not address.entities:
            return
        description, candidate = self._most_likely_candidate(address)
        linked_by_id = {} if candidate is None else {e.address_entity_id: e for e in candidate.entities}
        finest_id = None if candidate is None else candidate.finest_grain_entity.address_entity_id
        reference_id = None if candidate is None else candidate.reference_entity.address_entity_id
        lines = [
            f"Entities, linked as in the {description}:" if candidate is not None
            else "Entities (no candidate address to link them):",
            "",
            "| entity type | raw value | linked iri | linked name | country | |",
            "|---|---|---|---|---|---|",
        ]
        # entities the Disambiguator split from an address entity (see
        # Disambiguator._split_entity) only exist in the candidate
        address_entity_ids = {e.address_entity_id for e in address.entities}
        split_parts = [] if candidate is None else [
            e for e in candidate.entities if e.address_entity_id not in address_entity_ids]
        for entity in [*address.entities, *split_parts]:
            notes = []
            if entity in split_parts:
                notes.append("split part")
            if entity.address_entity_id == reference_id:
                notes.append("reference")
            if entity.address_entity_id == finest_id:
                notes.append("finest")
            if entity.is_missed_word:
                notes.append("missed word")
            if entity.pre_linked_iri:
                notes.append(f"pre-linked {entity.pre_linked_iri}")
            linked = linked_by_id.get(entity.address_entity_id)
            if linked is None:
                iri = name = country = "—"
            else:
                geographical_entity = linked.linked_to.geographical_name.entity
                iri = _entity_iri(linked)
                name = linked.linked_to.geographical_name.name
                country = geographical_entity.country.iso_code if geographical_entity.country is not None else "—"
            lines.append(
                f"| {entity.entity_type.name} | `{entity.raw_text!r}` | {iri} | {name} | {country} | {', '.join(notes)} |")
        display(Markdown("\n".join(lines)))

    def _camp_reference_keys_for(self, iri: str) -> dict[str, "CampMatch"]:
        """Every key of the camp/ghetto reference index that resolves to `iri` (geonames or wikidata)."""
        return {
            key: match for key, match in self.camp_matcher._index.items()
            if normalize_iri(match.iri) == iri or normalize_iri(match.wikidata_iri) == iri
        }

    def _ground_truth(self, address_id) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """(iri, entity type, raw text) of the ground-truth link, all None for a non-location."""
        true_row = self.indexed_gt.loc[str(address_id)]
        if pd.isna(true_row["iri"]):
            return None, None, None
        true_entity_type = _get_link_entity_type(true_row)
        true_raw_text = true_row.get(true_entity_type) if true_entity_type is not None else None
        return normalize_iri(true_row["iri"]), true_entity_type, true_raw_text

    def _parse_differences(self, address_id, parsed_row, prefix) -> list[tuple[str, Optional[str], Optional[str]]]:
        """
        (entity type, parsed text, ground-truth text) for every entity type on
        which the parse result disagrees with the ground-truth annotation.
        Texts are compared ignoring case and surrounding/repeated whitespace. A
        parsed AboveCity counts as agreeing when it equals any of the
        ground-truth's coarser-than-City columns that parsing left empty.
        """
        true_row = self.indexed_gt.loc[str(address_id)]
        parsed = entity_linking._extract_entity_texts(parsed_row, prefix)
        truth = {t: entity_linking._clean_optional_str(true_row.get(t)) for t in self._gt_entity_columns}

        def _norm(text):
            if text is None:
                return None
            # Numeric columns (e.g. HouseNumber) may be read back as floats.
            if isinstance(text, float) and text.is_integer():
                text = int(text)
            return entity_linking._normalize_for_dedup(str(text))

        above_city = parsed.get("AboveCity")
        above_city_type = next(
            (t for t in self._ABOVE_CITY_GT_COLUMNS
             if above_city is not None and parsed.get(t) is None and _norm(truth.get(t)) == _norm(above_city)),
            None,
        )
        differences = []
        for entity_type in self._gt_entity_columns:
            parsed_text = above_city if entity_type == above_city_type else parsed.get(entity_type)
            if _norm(parsed_text) != _norm(truth[entity_type]):
                differences.append((entity_type, parsed_text, truth[entity_type]))
        if above_city is not None and above_city_type is None:
            differences.append(("AboveCity", above_city, None))
        return differences

    def _explain_missing_true_candidate(self, sb: list[str], address, true_iri) -> None:
        """
        For a true IRI that is no candidate address' finest entity: whether
        GeoDBSearch retrieved it at all, and if so, why disambiguation dropped
        it (pruned, or never reached as a reference entity).
        """
        searched = [
            (entity, match) for entity in address.entities
            if isinstance(entity, MatchedEntity) and entity.matches
            for match in entity.matches
            if normalize_iri(match.geographical_name.entity.iri) == true_iri
        ]
        if not searched:
            sb.append(f"❌ GeoDBSearch did not retrieve the true IRI {true_iri} for any entity.")
            return
        for entity, match in searched:
            sb.append(f"  GeoDBSearch retrieved the true IRI for {entity.entity_type.name} {entity.raw_text!r} as "
                      f"{match.nfc_alt_name!r} (fuzzy {match.fuzzy_score:.3f}, phonetic {match.phonetic_score:.3f}, "
                      f"partial word match {match.is_partial_word_match}).")
        # Rescoring would repeat the disambiguation debug logs already shown
        current_level = geo_db_search.ENTITY_LINKING_LOGGER.level
        geo_db_search.ENTITY_LINKING_LOGGER.setLevel(logging.WARNING)
        try:
            for entity in {id(entity): entity for entity, _ in searched}.values():
                true_candidates = [
                    c for c in self.disambiguator._reference_entity_candidates(address, entity)
                    if _entity_iri(c.finest_grain_entity) == true_iri
                ]
                if not true_candidates:
                    sb.append(f"❌ no candidate address with {entity.raw_text!r} as the reference entity has the true IRI "
                              f"as its finest entity (cross matching linked a finer entity).")
                elif all(self.disambiguator._prune_reason(c) is not None for c in true_candidates):
                    sb.append(f"❌ the disambiguator pruned all {len(true_candidates)} candidate address(es) with the true IRI "
                              f"(reference entity {entity.raw_text!r}); the best ranked one because its "
                              f"{self.disambiguator._prune_reason(true_candidates[0])}.")
                else:
                    sb.append(f"❌ a candidate address with the true IRI survives pruning with {entity.raw_text!r} as the "
                              f"reference entity, but disambiguation settled on an earlier reference entity.")
        finally:
            geo_db_search.ENTITY_LINKING_LOGGER.setLevel(current_level)

    def _explain_geo_db_outcome(self, sb: list[str], address, true_iri, true_entity_type, true_raw_text) -> None:
        """
        Compares the GeoDBSearch + disambiguation candidates (already logged by
        link_field) against the ground truth: whether the true IRI was among
        them, and if so which factor ranked the predicted candidate above it.
        """
        if true_iri is None:
            return
        if not address.possible_links:
            self._explain_missing_true_candidate(sb, address, true_iri)
            return

        predicted = address.linked_to
        if predicted is None:
            if any(_entity_iri(c.finest_grain_entity) == true_iri for c in address.likely_links):
                sb.append(f"❌ true IRI {true_iri} is among the tied likely candidates.")
            elif any(_entity_iri(c.finest_grain_entity) == true_iri for c in address.possible_links):
                sb.append(f"❌ true IRI {true_iri} is not among the tied likely candidates "
                          f"(it is among the lower-ranked possible_links).")
            else:
                sb.append(f"❌ true IRI {true_iri} is not among the tied likely candidates "
                          f"nor among the {len(address.possible_links)} possible_links.")
                self._explain_missing_true_candidate(sb, address, true_iri)
            return

        true_candidate = next(
            (c for c in address.possible_links if _entity_iri(c.finest_grain_entity) == true_iri), None
        )
        if true_candidate is None:
            # The true IRI may still be matched as a coarser/finer entity of some
            # candidate (e.g. City vs Neighborhood granularity mismatch).
            as_other_entity = [
                (c, e) for c in address.possible_links for e in c.entities if _entity_iri(e) == true_iri
            ]
            sb.append(f"  raw text searched: {predicted.finest_grain_entity.raw_text!r} "
                  f"(ground truth: {true_entity_type} {true_raw_text!r})")
            if as_other_entity:
                c, e = as_other_entity[0]
                sb.append(f"  true IRI {true_iri} was only found as the {e.entity_type.name} of a candidate "
                      f"whose finest entity is {c.finest_grain_entity.entity_type.name} {_entity_iri(c.finest_grain_entity)} "
                      f"(granularity mismatch).")
            else:
                sb.append(f"❌ true IRI {true_iri} is not among the {len(address.possible_links)} candidate "
                          f"address(es) left after disambiguation.")
                self._explain_missing_true_candidate(sb, address, true_iri)
            return

        sb.append(f"  predicted: {_entity_iri(predicted.finest_grain_entity)} ({predicted.finest_grain_entity.linked_to.geographical_name.name})")
        sb.append(f"  true:      {true_iri} ({true_candidate.finest_grain_entity.linked_to.geographical_name.name})")
        _display_block(sb)
        sb.clear()

        table = ["| factor | predicted | true | |", "|---|--:|--:|---|"]
        decided = geo_disambiguation.deciding_factor(
            predicted.scores, true_candidate.scores, self.disambiguator.comparison_steps)
        deciding_factor = decided[0] if decided is not None else None
        for factor in self.disambiguator.priority:
            pred_score = predicted.scores.get(factor, 0.0)
            true_score = true_candidate.scores.get(factor, 0.0)

            marker = ""
            if deciding_factor is not None and geo_disambiguation.factor_of_label(deciding_factor) == factor:
                marker = "**← decides the ranking**"
                if deciding_factor != factor:
                    marker += " (once no primary factor differs significantly)"
            table.append(f"| `{factor}` | {pred_score:.3f} | {true_score:.3f} | {marker} |")
        display(Markdown("\n".join(table)))
        if deciding_factor is not None:
            sb.append(f"❌ The predicted candidate ranked higher because of '{deciding_factor}'.")
        else:
            sb.append("❌ Both candidates have identical average scores across all factors; "
                  "see the per-entity scores below for the actual tie-breaker.")

        sb.append("\n  Per-entity scores:")
        for label, candidate in [("predicted", predicted), ("true", true_candidate)]:
            sb.append(f"    {label}:")
            for entity in candidate.entities:
                entity_scores = {
                    factor: round(entity.scores[factor].score, 3)
                    for factor in self.disambiguator.priority
                    if factor in entity.scores
                }
                sb.append(f"      {entity.entity_type.name:<13} {_entity_iri(entity):<45} {entity.linked_to.geographical_name.name:<25} {entity_scores}")

    def _explain_outcome(self, sb: list[str], outcome, true_iri, true_entity_type, true_raw_text) -> None:
        """Why the step at which link_field stopped (see outcome.search_status) disagrees with the ground truth."""
        address = outcome.address
        status = outcome.search_status
        if true_entity_type is not None and not any(
            e.entity_type.name in (true_entity_type, "Unknown") for e in address.entities
        ):
            sb.append(f"❌ !! parsing did not extract a {true_entity_type} entity, which the ground truth links "
                  f"({true_raw_text!r}); later steps can at best link a coarser entity.")

        if status == SearchStatus.PRE_LINKED_DURING_PARSING:
            sb.append("❌ the id pre-linked during parsing is wrong.")
            return
        if status.startswith(SearchStatus.NO_LOCATION):
            sb.append("❌ tagging treated a linkable location as not a location.")
            return

        true_camp_keys = self._camp_reference_keys_for(true_iri) if true_iri is not None else {}
        true_camp = next(iter(true_camp_keys.values()), None)
        if status == SearchStatus.CAMP_REFERENCE:
            if true_iri is None:
                sb.append("❌ the camp/ghetto match is wrong: the ground truth says this is not a linkable location.")
            elif true_camp is not None:
                sb.append(f"❌ the true camp/ghetto is {_describe_camp_match(true_camp)}, indexed under keys {sorted(true_camp_keys)}.")
            else:
                sb.append(f"❌ the true IRI {true_iri} is not a camp/ghetto in the reference list: "
                      f"this camp match hijacked an address that should have gone through GeoDBSearch.")
            return
        if true_camp is not None:
            sb.append(f"❌ !! no camp match, but the true IRI is the camp/ghetto {_describe_camp_match(true_camp)}, "
                  f"indexed under keys {sorted(true_camp_keys)}; see the CampReferenceMatcher logs for why it was missed.")
        if status != SearchStatus.NO_ENTITIES:
            self._explain_geo_db_outcome(sb, address, true_iri, true_entity_type, true_raw_text)
