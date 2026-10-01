"""
Support code for entity_linking.ipynb.

The notebook is kept to function calls and variable definitions (configuration,
experiment parameters); every function, class and non-trivial block of logic it
needs lives here instead. Functions take their dependencies (searchers,
dataframes, outcomes...) as explicit arguments rather than reading notebook
globals.
"""
import json
import logging
import os
import pprint
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import colorlog
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

import modules.build_geonames_db as build_geonames_db
import modules.entity_linking as entity_linking
import modules.geo_db_search as geo_db_search
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
    if "search_executor" in namespace:
        namespace["search_executor"].shutdown()
    if "disambiguator" in namespace:
        namespace["disambiguator"].close()


class SerializedGeoDBSearch:
    """
    GeoDBSearch's duckdb connection and tantivy reader cannot be called
    concurrently from multiple threads. Rather than have every caller take a
    lock around geo_db_searcher.apply(), this routes every call (regardless
    of which thread issues it) through a single dedicated worker thread, so
    calls are serialized without blocking the rest of the pipeline (camp
    matching, tagging, disambiguation) from running concurrently elsewhere.
    """
    def __init__(self, wrapped: geo_db_search.GeoDBSearch):
        self._wrapped = wrapped
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geo-db-search")

    def apply(self, address):
        return self._executor.submit(self._wrapped.apply, address).result()

    def shutdown(self):
        self._executor.shutdown(wait=True)


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


async def _run_llm_fallback(addresses: list[str]) -> list[dict]:
    """
    Replicates parse_with_llms.py's LLM fallback: parses `addresses` with the
    same remote model/prompt/example-selection config, and renames its output
    columns the same way (entity names -> "{Entity}.text", metadata -> "llm_metadata.*").
    """
    llm_parser, llm_config = prepare_llm()
    print(f"Using LLM parser {llm_config['model']} as fallback for {len(addresses)} addresses...")
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


async def load_or_build_parsed_addresses(addresses: pd.DataFrame, cache_path: Path, overwrite: bool = False) -> pd.DataFrame:
    """Parses `addresses` (regex + LLM fallback), or reads the result back from `cache_path` if already cached."""
    regex_rows, needs_llm_mask = regex_parse_addresses(addresses)
    if overwrite or not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        parsed_addresses = await build_parsed_addresses(addresses, regex_rows, needs_llm_mask)
        elapsed = time.monotonic() - start
        print(f"Parsing {len(parsed_addresses)} addresses (regex + LLM fallback) took {format_time(elapsed)}")
        parsed_addresses.to_json(cache_path, orient="records", lines=True)
    else:
        print(f"Retrieving cached parsed addresses from {cache_path}...")
        parsed_addresses = pd.read_json(cache_path, orient="records", lines=True)
    return parsed_addresses


# ---------------------------------------------------------------------------
# Entity linking pipeline
# ---------------------------------------------------------------------------

def default_num_workers() -> int:
    # Many worker threads for the cheap/safe steps (camp/ghetto matching, address
    # tagging, disambiguation); GeoDBSearch itself stays serialized to 1 thread
    # via search_executor regardless of this number.
    return min(32, (os.cpu_count() or 4) * 4)


def run_pipeline(
    rows: list[dict],
    num_workers: int,
    camp_matcher: CampReferenceMatcher,
    search_executor: SerializedGeoDBSearch,
    disambiguator: Disambiguator,
    estimated_total_addresses: Optional[int] = None,
) -> list[entity_linking.LinkingOutcome]:
    """
    Runs entity_linking.link_field for every parsed address row in `rows`
    (the "{prefix}.*"-flattened rows build_parsed_addresses() produces), using
    `num_workers` worker threads. Camp/ghetto matching, address tagging and
    disambiguation all run concurrently across those threads; only the
    GeoDBSearch step is serialized (via search_executor).
    """
    def link_parsed_row(row: dict, idx):
        return entity_linking.link_field(row, row["prefix"], camp_matcher, search_executor, disambiguator), idx

    print(f"Running full entity linking pipeline on regex+LLM parsed addresses (using {num_workers} worker threads, search serialized to 1)...")
    start = time.monotonic()
    results: list[entity_linking.LinkingOutcome] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(link_parsed_row, row, idx) for idx, row in enumerate(rows)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, idx = future.result()
            results[idx] = result
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


# Coarsest to finest, to order the partial link entity types
_ENTITY_TYPE_ORDER = ["Country", "State", "Region", "District", "AboveCity", "City", "Unknown", "Neighborhood"]


def partial_link_entity_type_counts(outcomes: list[entity_linking.LinkingOutcome], ground_truth: pd.DataFrame) -> pd.Series:
    """Number of partially linked addresses per entity type they were linked up to."""
    entity_types = [
        outcome.entity_type
        for outcome, status in zip(outcomes, link_statuses(outcomes, ground_truth))
        if status == LinkStatus.PARTIALLY_LINKED
    ]
    counts = pd.Series(entity_types, dtype=object).value_counts()
    return counts.sort_index(key=lambda index: [
        _ENTITY_TYPE_ORDER.index(t) if t in _ENTITY_TYPE_ORDER else len(_ENTITY_TYPE_ORDER) for t in index])


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
    """Side by side pies: partial links by linked entity type (left), failed links by reason (right)."""
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
        search_executor: SerializedGeoDBSearch,
        camp_matcher: CampReferenceMatcher,
        disambiguator: Disambiguator,
    ):
        self.indexed_gt = indexed_gt
        self.indexed_parsed = indexed_parsed
        self.geo_db_searcher = geo_db_searcher
        self.search_executor = search_executor
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
           in DISAMBIGUATION_FACTOR_PRIORITY order, to surface the deciding factor.
        """
        current_level = geo_db_search.ENTITY_LINKING_LOGGER.level
        if verbose:
            geo_db_search.ENTITY_LINKING_LOGGER.setLevel(logging.DEBUG)
            region_branch_cache = self.geo_db_searcher.search_index._region_cluster_cache
            logging.debug(f"Clearing cache geo_db_searcher.search_index._region_cluster_cache, contents {region_branch_cache}")
            region_branch_cache.clear()
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

        row = {**parsed_row.to_dict(), "address_id": str(address_id)}
        outcome = entity_linking.link_field(
            row, prefix, self.camp_matcher, self.search_executor, self.disambiguator, keep_address=True
        )

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
        deciding_factor = None
        for factor in self.disambiguator.priority:
            pred_score = predicted.scores.get(factor, 0.0)
            true_score = true_candidate.scores.get(factor, 0.0)

            marker = ""
            if deciding_factor is None and abs(pred_score - true_score) > self.disambiguator.significance_thresholds[factor]:
                deciding_factor = factor
                marker = "**← decides the ranking**"
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
