"""
Run address component extraction using HybridSimilarExamples:
  - 1 shot from SimilarExamples (sentence-embedding similarity)
  - 2 shots from PatternTokenSimilarExamples (regex pattern-based)

Predictions are saved to preds_hybrid2p1e_val.json.
Metrics are saved to metrics_hybrid2p1e_val.txt.
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
    SimilarExamples,
    PatternTokenSimilarExamples,
    HybridSimilarExamples,
)
from utils import compare_preds

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_NAME             = "Llama-3-8B-prompt2-hybrid2p1e3shot"
MODEL_NAME              = "meta-llama/Meta-Llama-3-8B-Instruct"
N_EMBEDDING             = 1
N_PATTERN               = 2
BONUS_PATTERN_THRESHOLD = 0.0   # always include pattern shots
BATCH_SIZE              = 32
PREDS_OUTPUT_PATH       = Path("preds_hybrid2p1e_v2_val.json")
METRICS_OUTPUT_PATH     = Path("metrics_hybrid2p1e_v2_val.txt")
ENTITIES_TO_PREDICT     = ["HouseNumber", "StreetName", "City", "District", "State", "Country"]
METRICS_COLUMNS         = ["HouseNumber", "StreetName", "City", "State", "Country"]

TOTAL_BZK_CARDS = 2_000_000

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


def per_column_metrics(preds_df, labels_df, cols, metric="f1"):
    return {col: compare_preds(preds_df, labels_df, target_columns=[col])[metric] for col in cols}


def per_field_metrics(preds_df, labels_df, cols, metric="f1"):
    results = {}
    for field in labels_df["field"].unique():
        mask = labels_df["field"] == field
        results[field] = compare_preds(preds_df[mask.values], labels_df[mask], target_columns=cols)[metric]
    return results

# ── Load data ─────────────────────────────────────────────────────────────────

csv_read_args = dict(keep_default_na=False, dtype=str, na_values=[""])
bzkopen_train = pd.read_csv("open_data/bzkopen_addresses_train.csv", **csv_read_args)
bzkopen_val   = pd.read_csv("open_data/bzkopen_addresses_val.csv",   **csv_read_args)
bzkopen_test  = pd.read_csv("open_data/bzkopen_addresses_test.csv",  **csv_read_args)

n_cards = {"train": 361, "val": 77, "test": 78}
n_addrs = {"train": len(bzkopen_train), "val": len(bzkopen_val), "test": len(bzkopen_test)}
addresses_per_card_total = sum(n_addrs[s] / n_cards[s] for s in n_cards)
ESTIMATED_TOTAL_ADDRESSES = TOTAL_BZK_CARDS * addresses_per_card_total

# ── Load or run predictions ───────────────────────────────────────────────────

if PREDS_OUTPUT_PATH.exists():
    print(f"Loading existing predictions from {PREDS_OUTPUT_PATH}...")
    with open(PREDS_OUTPUT_PATH, "r") as f:
        saved = json.load(f)
    preds     = saved["preds"]
    deltatime = saved["deltatime"]
    print(f"Loaded {len(preds)} predictions (original runtime: {format_time(deltatime)})")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building SimilarExamples (embedding) strategy...")
    embedding_strategy = SimilarExamples(
        example_addresses=bzkopen_train["FullAddress"],
        example_labels=bzkopen_train,
        num_examples=N_EMBEDDING,
        labels_to_include=ENTITIES_TO_PREDICT,
        device=device,
    )

    print("Building PatternTokenSimilarExamples strategy...")
    pattern_strategy = PatternTokenSimilarExamples(
        example_addresses=bzkopen_train["FullAddress"],
        example_labels=bzkopen_train,
        num_examples=N_PATTERN,
        labels_to_include=ENTITIES_TO_PREDICT,
    )

    example_strategy = HybridSimilarExamples(
        embedding_strategy=embedding_strategy,
        pattern_strategy=pattern_strategy,
        n_embedding=N_EMBEDDING,
        n_pattern=N_PATTERN,
        bonus_pattern_threshold=BONUS_PATTERN_THRESHOLD,
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

metrics = compare_preds(preds_df, bzkopen_val, target_columns=METRICS_COLUMNS)
metrics["deltatime"]          = deltatime
metrics["rate"]               = len(bzkopen_val) / deltatime
metrics["estimatedTotalTime"] = ESTIMATED_TOTAL_ADDRESSES / metrics["rate"]
metrics["error"]              = int(preds_df["error"].notna().sum()) if "error" in preds_df.columns else 0
metrics["errorRate"]          = metrics["error"] / len(bzkopen_val)

output_lines.append(format_metrics_block(metrics, label=CONFIG_NAME))

output_lines.append("\n── Per-entity F1 " + "─" * 43)
col_metrics = per_column_metrics(preds_df, bzkopen_val, ENTITIES_TO_PREDICT, metric="f1")
for col, val in col_metrics.items():
    output_lines.append(f"  {col:<35} {val:.4f}")

if "field" in bzkopen_val.columns:
    output_lines.append("\n── Per-BZK-field F1 " + "─" * 40)
    field_metrics = per_field_metrics(preds_df, bzkopen_val, METRICS_COLUMNS, metric="f1")
    for field, val in field_metrics.items():
        output_lines.append(f"  {field:<35} {val:.4f}")

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
