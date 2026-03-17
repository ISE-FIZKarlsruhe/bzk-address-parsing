"""
Run address component extraction using PatternTokenSimilarExamples (pattern-based similarity)
for the Llama-3-8B-prompt2-patternsim3shot configuration.

Mirrors Llama-3-8B-prompt2-similar3shot but replaces SimilarExamples (embedding-based)
with PatternTokenSimilarExamples (structural pattern + lexical similarity, no GPU needed
for example selection).

Predictions are saved to preds_patternsim3shot_val.json. Evaluation metrics
match those produced by compare.ipynb.
"""
import json
import time
import datetime
from pathlib import Path

from tqdm import tqdm

import pandas as pd
import torch

from mllms import (
    LlamaAddressParsingModel,
    JsonDictPromptTemplate,
    PatternTokenSimilarExamples,
)
from utils import compare_preds

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_NAME         = "Llama-3-8B-prompt2-patternsim3shot"
MODEL_NAME          = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_EXAMPLES        = 3
BATCH_SIZE          = 32
PREDS_OUTPUT_PATH   = Path("preds_patternsim3shot_v2_val.json")
METRICS_OUTPUT_PATH = Path("metrics_patternsim3shot_v2_val.txt")
ENTITIES_TO_PREDICT = ["HouseNumber", "StreetName", "City", "District", "State", "Country"]
METRICS_COLUMNS     = ["HouseNumber", "StreetName", "City", "State", "Country"]

TOTAL_BZK_CARDS = 2_000_000  # estimated total cards in the full BZK dataset

# ── Prompt (PROMPTS[2] from compare.ipynb) ────────────────────────────────────

llama_prompt2 = JsonDictPromptTemplate(
    "You are a german archivist handling the digitalization of german documents from the "
    "compensation efforts that followed the second world war. Your current task consists of annotating addresses found "
    "in the archival documents, identifying the respective components of each address. "
    "Consider the component types: " + ", ".join(ENTITIES_TO_PREDICT + ["Other"]) + ". "
    "It is essential that you remain loyal to the original text and do not add any information not "
    "explictly mentioned in the address. "
    "Addresses will most times be written in german, meaning country and city names may be in "
    "german. The addresses may include german terms such as:\n"
    " - \"burg\" or \"stadt\" for city\n"
    " - \"straße\", \"avenue\" or its abbreviation \"str.\" and \"av.\" for street.\n"
    "These terms may occur as a suffix to another word.\n"
    "Format the output as a JSON object with the component types as keys.\n%(examples)s"
    "Now annotate the following address:\n%(address)s"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_time(seconds):
    seconds = round(seconds)
    td = datetime.timedelta(seconds=seconds)
    days = td.days
    months, days = divmod(days, 30)
    years, months = divmod(months, 12)
    td = td - datetime.timedelta(days=td.days) + datetime.timedelta(days=days)
    parts = []
    if years:  parts.append(f"{years} year{'s' if years > 1 else ''}")
    if months: parts.append(f"{months} month{'s' if months > 1 else ''}")
    parts.append(str(td))
    return ", ".join(parts)


def format_metrics_block(metrics: dict, label: str = "") -> str:
    header = f"── Metrics{' — ' + label if label else ''} "
    lines = [f"\n{header:─<60}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<35} {v:.4f}")
        else:
            lines.append(f"  {k:<35} {v}")
    return "\n".join(lines)


def print_metrics(metrics: dict, label: str = ""):
    print(format_metrics_block(metrics, label))


def per_column_metrics(preds_df, labels_df, cols, metric="f1"):
    results = {}
    for col in cols:
        m = compare_preds(preds_df, labels_df, target_columns=[col])
        results[col] = m[metric]
    return results


def per_field_metrics(preds_df, labels_df, cols, metric="f1"):
    fields = labels_df["field"].unique()
    results = {}
    for field in fields:
        mask = labels_df["field"] == field
        m = compare_preds(preds_df[mask.values], labels_df[mask], target_columns=cols)
        results[field] = m[metric]
    return results

# ── Load data ─────────────────────────────────────────────────────────────────

csv_read_args = dict(keep_default_na=False, dtype=str, na_values=[""])
bzkopen_train = pd.read_csv("open_data/bzkopen_addresses_train.csv", **csv_read_args)
bzkopen_val   = pd.read_csv("open_data/bzkopen_addresses_val.csv",   **csv_read_args)
bzkopen_test  = pd.read_csv("open_data/bzkopen_addresses_test.csv",  **csv_read_args)

# Match compare.ipynb: estimated_total_addresses = 2_000_000 * addresses_per_card_total
# where addresses_per_card_total = sum of (addresses/cards) across splits
n_cards = {"train": 361, "val": 77, "test": 78}
n_addrs = {"train": len(bzkopen_train), "val": len(bzkopen_val), "test": len(bzkopen_test)}
addresses_per_card_total = sum(n_addrs[s] / n_cards[s] for s in n_cards)
ESTIMATED_TOTAL_ADDRESSES = TOTAL_BZK_CARDS * addresses_per_card_total

# ── Load or run predictions ───────────────────────────────────────────────────

if PREDS_OUTPUT_PATH.exists():
    print(f"Loading existing predictions from {PREDS_OUTPUT_PATH}...")
    with open(PREDS_OUTPUT_PATH, "r") as f:
        saved = json.load(f)
    preds    = saved["preds"]
    deltatime = saved["deltatime"]
    print(f"Loaded {len(preds)} predictions (original runtime: {format_time(deltatime)})")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building PatternTokenSimilarExamples strategy...")
    example_strategy = PatternTokenSimilarExamples(
        example_addresses=bzkopen_train["FullAddress"],
        example_labels=bzkopen_train,
        num_examples=NUM_EXAMPLES,
        labels_to_include=ENTITIES_TO_PREDICT,
    )

    print(f"Loading model {MODEL_NAME}...")
    model = LlamaAddressParsingModel(
        model_name=MODEL_NAME,
        prompt=llama_prompt2,
        example_strategy=example_strategy,
        device=device,
    )

    addresses = bzkopen_val["FullAddress"].tolist()
    print(f"Running inference on {len(addresses)} validation addresses (batch size {BATCH_SIZE})...")
    preds = []
    start = time.monotonic()
    batches = [addresses[i:i + BATCH_SIZE] for i in range(0, len(addresses), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Inference", unit="batch"):
        preds.extend(model.parse_addresses(batch))
    deltatime = time.monotonic() - start
    print(f"Done in {deltatime:.1f}s ({len(addresses)/deltatime:.1f} addr/s)")

    with open(PREDS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": CONFIG_NAME, "preds": preds, "deltatime": deltatime}, f, ensure_ascii=False)
    print(f"Predictions saved to {PREDS_OUTPUT_PATH}")

# ── Evaluate ──────────────────────────────────────────────────────────────────

preds_df = pd.DataFrame(preds)
output_lines = []

# Overall metrics (same as compare.ipynb eval())
metrics = compare_preds(preds_df, bzkopen_val, target_columns=METRICS_COLUMNS)
metrics["deltatime"]          = deltatime
metrics["rate"]               = len(bzkopen_val) / deltatime
metrics["estimatedTotalTime"] = ESTIMATED_TOTAL_ADDRESSES / metrics["rate"]
metrics["error"]              = int(preds_df["error"].notna().sum()) if "error" in preds_df.columns else 0
metrics["errorRate"]          = metrics["error"] / len(bzkopen_val)

output_lines.append(format_metrics_block(metrics, label=CONFIG_NAME))

# Per-entity-type F1
output_lines.append("\n── Per-entity F1 " + "─" * 43)
col_metrics = per_column_metrics(preds_df, bzkopen_val, ENTITIES_TO_PREDICT, metric="f1")
for col, val in col_metrics.items():
    output_lines.append(f"  {col:<35} {val:.4f}")

# Per-BZK-field F1
if "field" in bzkopen_val.columns:
    output_lines.append("\n── Per-BZK-field F1 " + "─" * 40)
    field_metrics = per_field_metrics(preds_df, bzkopen_val, METRICS_COLUMNS, metric="f1")
    for field, val in field_metrics.items():
        output_lines.append(f"  {field:<35} {val:.4f}")

# Summary row (matches compare.ipynb display columns)
summary_cols = ["accuracy", "precision", "recall", "f1", "average_similarity", "errorRate", "estimatedTotalTime"]
output_lines.append("\n── Summary " + "─" * 49)
for col in summary_cols:
    v = metrics[col]
    if col == "estimatedTotalTime":
        output_lines.append(f"  {col:<35} {format_time(v)}")
    elif isinstance(v, float):
        output_lines.append(f"  {col:<35} {v:.4f}")
    else:
        output_lines.append(f"  {col:<35} {v}")

report = "\n".join(output_lines)
print(report)

with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"\nMetrics saved to {METRICS_OUTPUT_PATH}")
