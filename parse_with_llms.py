"""
File readapted from parse_with_regex.py with CLAUDE
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import asyncio
from tqdm.contrib.logging import tqdm_logging_redirect as tqdm

from parse_with_regex import PLACE_COLS_PREFIX, _count_lines
import modules.llms as llm_parsers

supported_entities = ["HouseNumber", "StreetName", "Neighborhood", "City", "Country"]

def _column_sort_key(c):
    if c == "filename":
        return (0, c)
    if c.endswith("raw") or c.endswith("status"):
        return (1, c)
    elif c.endswith(".text"):
        return (3, c)
    else:
        return (4, c)

def _rename_llm_output_columns(c : str) -> str:
    if c in supported_entities:
        return c + ".text"
    elif c in ["fullConversation", "error"]:
        return "llm_metadata." + c
    elif c.startswith("___"):
        return "llm_metadata." + c[3:]
    else:
        return c

def prepare_llm() -> llm_parsers.RemoteAddressParsingModel:
    model_name="Qwen/Qwen3.5-9B"
    prompt_template = llm_parsers.JsonDictPromptTemplate(Path("prompts/optuna_best/best_qwen_prompt.txt").read_text())
    supported_entities = ["HouseNumber", "StreetName", "Neighborhood", "City", "Country"]
    n_examples = 15
    embedding_model = "all-MiniLM-L6-v2"
    similarity_threshold = 0.35
    csv_read_args = dict(keep_default_na=False, dtype=str, na_values=[""])
    training_data = pd.read_csv("open_data/bzkopen_addresses_train.csv", **csv_read_args)
    pattern_similarity = llm_parsers.NERPatternSimilarExamples(
        example_addresses=training_data['FullAddress'],
        example_labels=training_data,
        labels_to_include=supported_entities,
        num_examples=n_examples,
        model_dir = "models/ner_bzk"
    )
    embedding_similarity = llm_parsers.SimilarExamples(
        embedding_model=embedding_model,
        example_addresses=training_data['FullAddress'],
        example_labels=training_data,
        labels_to_include=supported_entities,
        num_examples=n_examples,
        similarity_threshold=similarity_threshold
    )
    example_strategy = llm_parsers.HybridSimilarExamples(
        pattern_strategy=pattern_similarity,
        embedding_strategy=embedding_similarity,
        num_examples=n_examples,
        pool_size=n_examples
    )
    model = llm_parsers.RemoteAddressParsingModel(
        model_name=model_name,
        example_strategy=example_strategy,
        prompt=prompt_template
    )
    config = {
        "model" : model_name,
        "prompt" : prompt_template.template,
        "prompt_type" : prompt_template.__class__.__name__,
        "supported_entities" : supported_entities,
        "example_strategy" : {
            "type" : example_strategy.__class__.__name__,
            "n_examples" : n_examples,
            "similarity_threshold" : similarity_threshold,
            "embedding_model" : embedding_model,
        }
    }
    return model, config


def _truncate_invalid_trailing_line(path: Path):
    """Drop a truncated/corrupt trailing line left by a run that was interrupted
    mid-write, so resuming doesn't crash while loading already-processed filenames."""
    if not path.exists():
        return
    with path.open("rb") as f:
        lines = f.readlines()
    original_len = len(lines)
    while lines:
        try:
            json.loads(lines[-1])
            break
        except json.JSONDecodeError:
            lines.pop()
    if len(lines) != original_len:
        logging.warning(f"Dropping {original_len - len(lines)} corrupt trailing line(s) from {path}")
        with path.open("wb") as f:
            f.writelines(lines)


def _load_processed_filenames(path: Path) -> set:
    """Return the set of "filename" values already written to an output file."""
    if not path.exists() or _count_lines(path) == 0:
        return set()
    processed = pd.read_json(path, lines=True)
    if "filename" not in processed.columns:
        return set()
    return set(processed["filename"])


async def main(argv=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_dir", type=str, help="Output directory produced by parse_with_regex.py")
    arg_parser.add_argument("-P", "--file-pattern", type=str, default="*.jsonl")
    arg_parser.add_argument("--input-chunk-size", type=int, default=200)
    arg_parser.add_argument("--output-dir", type=str, default="llm_parsed_output")
    args = arg_parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    assert input_dir.is_dir(), f"{args.input_dir} is not a valid directory"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'parse_with_llms.log', mode='a')
        ]
    )
    # The openai client logs an INFO line per HTTP request via httpx2/httpcore;
    # only surface those loggers' warnings and errors.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Discover, per field, the per-input-file outputs produced by parse_with_regex.py
    field_files = {}
    row_counts = {}
    for prefix in PLACE_COLS_PREFIX.values():
        field_dir = input_dir / prefix
        assert field_dir.is_dir(), f"Missing field directory {field_dir}, run parse_with_regex.py first"
        files = list(field_dir.glob(args.file_pattern))
        files.sort()
        field_files[prefix] = files
        for file in files:
            row_counts[(prefix, file.name)] = len(file.read_text().splitlines())

    total_rows = sum(row_counts.values())
    error_count = 0

    llm_parser, llm_config = prepare_llm()
    logging.info(f"Using LLM parser {llm_config["model"]} with config:\n{json.dumps(llm_config, indent=2)}")

    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for prefix, files in field_files.items():
            output_field_dir = output_dir / prefix
            output_field_dir.mkdir(parents=True, exist_ok=True)
            raw_col = f"{prefix}.raw"
            status_col = f"{prefix}.status"

            for file in files:
                output_path = output_field_dir / file.name
                row_count = row_counts[(prefix, file.name)]
                processed_filenames = set()
                rows_seen = 0
                try:
                    # Resume support: rows whose filename is already present in the
                    # output were handled by a previous run and are skipped again.
                    # An interrupted previous run may have left a corrupt trailing
                    # line, so drop it before loading.
                    _truncate_invalid_trailing_line(output_path)
                    processed_filenames = _load_processed_filenames(output_path)
                    logging.info(f"Processing {file} ({prefix}) with {row_count} rows.")
                    if len(processed_filenames) > 0:
                        if len(processed_filenames) == row_count:
                            logging.info(f"All {row_count} rows already processed for {file} ({prefix}), skipping.")
                            pbar.update(row_count)
                            continue
                        else:
                            logging.info(f"Resuming {file} ({prefix}), {len(processed_filenames)} rows already processed, {row_count - len(processed_filenames)} remaining.")
                    for df in pd.read_json(file, lines=True, chunksize=args.input_chunk_size):
                        # Rows already written to output_path by a previous run are
                        # skipped; everything else is written out this run, whether
                        # or not it actually needs an LLM call, so the output
                        # contains one row per input row.
                        remaining = df[~df["filename"].isin(processed_filenames)]
                        if len(remaining) > 0:
                            needs_llm = (
                                remaining[raw_col].notna()
                                & (remaining[raw_col] != "")
                                & (remaining.get(status_col, None) != "fully_parsed")
                            )
                            todo = remaining[needs_llm]
                            passthrough = remaining[~needs_llm]

                            output_parts = []
                            if len(todo) > 0:
                                parsed_results = await llm_parser.parse_addresses(todo[raw_col].tolist())
                                result_df = pd.DataFrame(parsed_results, index=todo.index)
                                result_df.rename(columns=_rename_llm_output_columns, inplace=True)
                                result_df["raw"] = todo[raw_col]
                                if "llm_metadata.error" in result_df.columns:
                                    for idx, row in result_df.iterrows():
                                        if not pd.isna(row["llm_metadata.error"]):
                                            error_count += 1
                                            logging.error(
                                                f"LLM parsing error for {todo.at[idx, 'filename']}:\n{row['error']}" +
                                                "\n\nFull conversation:\n" + json.dumps(row.get("conversation", []), indent=2) +
                                                "\n\nMetadata:\n" + json.dumps(row.get("___example_metadata", []), indent=2) +
                                                f"\n\nTotal errors so far: {error_count}"
                                            )
                                    result_df["status"] = "llm_error"
                                else:
                                    result_df["status"] = "llm_parsed"
                                result_df["llm_parser"] = llm_config["model"]
                                result_df = result_df.add_prefix(f"{prefix}.")
                                result_df["filename"] = todo["filename"]
                                output_parts.append(result_df)
                            if len(passthrough) > 0:
                                output_parts.append(passthrough)

                            # Concatenate in original row order (todo and passthrough
                            # partition remaining's index, so sort_index restores it).
                            output_df = pd.concat(output_parts).sort_index()
                            output_df.sort_index(axis=1, key=lambda c: [_column_sort_key(col) for col in c], inplace=True)
                            output_df.to_json(output_path, orient="records", lines=True, mode="a")
                            processed_filenames.update(remaining["filename"])

                        pbar.update(len(df))
                        rows_seen += len(df)
                    logging.info(f"Finished processing {file} ({prefix}).")
                except Exception as e:
                    if str(e) == "Query interrupted": raise
                    logging.exception(f"Error processing {file} ({prefix}): {e}")
                    try: pbar.update(row_count - rows_seen)
                    except: pass

    logging.info("Processing complete.")


if __name__ == "__main__":
    asyncio.run(main())
