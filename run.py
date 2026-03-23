"""
Unified BZK address-parsing inference runner.

────────────────────────────────────────────────────────────────────────────────
QUICK EXAMPLES
────────────────────────────────────────────────────────────────────────────────

Reproduce Llama embedding-only baseline:
  uv run python run.py --strategy embedding

Hybrid (NER pattern + XLM-RoBERTa embedding) with extraction rules, Llama:
  uv run python run.py --strategy hybrid --pattern-type ner \\
      --embedding-model hm-haitham/xlm-roberta-large-address-parser \\
      --prompt-variant rules --fallback-threshold 0.92

Run on test split:
  uv run python run.py --strategy hybrid --pattern-type ner --split test

Force rerun even if cached predictions exist:
  uv run python run.py --strategy hybrid --force-rerun
"""

import argparse
import datetime
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from utils import compare_preds

# ── Parameter registry ────────────────────────────────────────────────────────

TOTAL_BZK_CARDS     = 2_000_000
ENTITIES_TO_PREDICT = ["HouseNumber", "StreetName", "City", "District", "State", "Country"]
METRICS_COLUMNS     = ["HouseNumber", "StreetName", "City", "State", "Country"]


_MODEL_ALIASES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b",
    "Qwen/Qwen3.5-9B":                     "qwen35-9b",
}
_EMBED_ALIASES = {
    "multi-qa-mpnet-base-dot-v1":                  "mpnet",
    "hm-haitham/xlm-roberta-large-address-parser": "xlmroberta",
}

# ── Prompt templates ──────────────────────────────────────────────────────────

_PROMPT_BASE = (
    "You are a german archivist handling the digitalization of german documents from the "
    "compensation efforts that followed the second world war. Your current task consists of "
    "annotating addresses found in the archival documents, identifying the respective "
    "components of each address. "
    "Consider the component types: {entities}. "
    "It is essential that you remain loyal to the original text and do not add any "
    "information not explictly mentioned in the address. "
    "Addresses will most times be written in german, meaning country and city names may be "
    "in german. The addresses may include german terms such as:\n"
    ' - "burg" or "stadt" for city\n'
    ' - "straße", "avenue" or its abbreviation "str." and "av." for street.\n'
    "These terms may occur as a suffix to another word.\n"
    "Format the output as a JSON object with the component types as keys.\n%(examples)s"
    "Now annotate the following address:\n%(address)s"
)

_PROMPT_RULES_SUFFIX = (
    "Important extraction rules:\n"
    " - Only extract a field if its value appears explicitly in the address text. "
    "Never infer or guess State or Country from the city name or your background knowledge.\n"
    " - In German addresses a slash (/) often connects a city name with a German regional "
    "disambiguation abbreviation (e.g. 'Weener/Ostfr.', 'Neuwied/Rh.', 'Sülzbach/Opf.', "
    "'Dinkelsbühl/Mfr.'). These German regional abbreviations after the slash are NOT State "
    "and NOT Country — leave both fields empty. This rule does NOT apply when an actual "
    "country name is explicitly written in the address (e.g. 'Österr.', 'England', 'Polen').\n"
    " - 'Krs.' (Kreis) and 'Bez.' (Bezirk) introduce a district qualifier, not a State.\n"
    " - If the text is not an actual address (e.g. 'KZ', 'verschollen', 'unbekannt', "
    "'gefallen'), return all fields empty.\n"
)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BZK address-parsing inference runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Inference model ────────────────────────────────────────────────────
    p.add_argument(
        "--inference-model", default="meta-llama/Meta-Llama-3-8B-Instruct",
        metavar="HF_MODEL_ID",
        help=(
            "HuggingFace model ID for the generative LLM.\n"
            "Tested values:\n"
            "  meta-llama/Meta-Llama-3-8B-Instruct  (default, decoder-only)\n"
            "  Qwen/Qwen3.5-9B                       (thinking model, use --max-new-tokens 4096)"
        ),
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=512,
        help=(
            "Maximum number of tokens to generate per address.  For thinking models "
            "Qwen3 reasoning trace consumes many tokens before the JSON "
            "answer — set to 4096.  For standard models 512 is ample.  Default: 512."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help=(
            "Number of addresses processed per GPU forward pass.  Reduce for larger models "
            "to avoid OOM.  Default: 32."
        ),
    )

    # ── Few-shot example strategy ──────────────────────────────────────────
    p.add_argument(
        "--strategy", default="hybrid",
        choices=["embedding", "pattern", "hybrid"],
        help=(
            "Few-shot example selection strategy.\n"
            "  embedding  — retrieve top-N by semantic similarity (SimilarExamples)\n"
            "  pattern    — retrieve top-N by NER structural pattern similarity\n"
            "  hybrid     — merge top-N from each pool, keep overall top-N by score\n"
            "Default: hybrid."
        ),
    )
    p.add_argument(
        "--num-examples", type=int, default=3,
        help="Number of few-shot examples to include in each prompt.  Default: 3.",
    )
    p.add_argument(
        "--pool-size", type=int, default=3,
        help=(
            "Candidates fetched from each pool in hybrid mode before merging.  "
            "Has no effect when --strategy is not hybrid.  Default: 3."
        ),
    )

    # ── Embedding model ────────────────────────────────────────────────────
    p.add_argument(
        "--embedding-model", default="multi-qa-mpnet-base-dot-v1",
        metavar="HF_MODEL_ID",
        help=(
            "Sentence encoder for embedding-based similarity.  Two options have been "
            "evaluated:\n"
            "  multi-qa-mpnet-base-dot-v1              (default, general-purpose)\n"
            "  hm-haitham/xlm-roberta-large-address-parser  (address-specific, mean pooling)"
        ),
    )

    # ── Pattern strategy ───────────────────────────────────────────────────
    p.add_argument(
        "--ner-model-dir", default="models/ner_bzk",
        metavar="DIR",
        help=(
            "Path to the trained spaCy NER model directory used for pattern-based "
            "similarity.  Train with: uv run python train_ner.py."
        ),
    )

    # ── Prompt variant ─────────────────────────────────────────────────────
    p.add_argument(
        "--prompt-variant", default="rules",
        choices=["base", "rules"],
        help=(
            "Prompt template variant."
        ),
    )

    # ── Fallback strategy ──────────────────────────────────────────────────
    p.add_argument(
        "--fallback-threshold", type=float, default=0.92,
        metavar="FLOAT",
        help=(
            "When the average retrieval score across selected examples falls below this "
            "threshold, replace them with a fixed set of curated demo examples covering "
            "structurally hard cases "
        ),
    )

    # ── Data / output ──────────────────────────────────────────────────────
    p.add_argument(
        "--split", default="val",
        choices=["val", "test"],
        help="Dataset split to run inference on.",
    )
    p.add_argument(
        "--output-dir", default=".",
        metavar="DIR",
        help="Directory where predictions and metrics are saved.",
    )
    p.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore any cached predictions file and rerun inference from scratch.",
    )

    return p.parse_args()


# ── Config name & output paths ────────────────────────────────────────────────

def build_config_name(args) -> str:
    model_alias  = _MODEL_ALIASES.get(args.inference_model, args.inference_model.split("/")[-1])
    embed_alias  = _EMBED_ALIASES.get(args.embedding_model, args.embedding_model.split("/")[-1])

    parts = [model_alias, f"prompt2"]

    if args.strategy == "embedding":
        parts.append(f"emb-{embed_alias}-{args.num_examples}shot")
    elif args.strategy == "pattern":
        parts.append(f"pat-ner-{args.num_examples}shot")
    else:  # hybrid
        parts.append(
            f"hybrid-ner-emb-{embed_alias}"
            f"-{args.num_examples}shot-pool{args.pool_size}"
        )

    if args.prompt_variant == "rules":
        parts.append("rules")
    if args.fallback_threshold > 0:
        parts.append(f"fb{args.fallback_threshold:.2f}".replace(".", ""))
    return "_".join(parts)


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
        lines.append(
            f"  {k:<35} {format_time(v)}" if k == "estimatedTotalTime"
            else f"  {k:<35} {v:.4f}" if isinstance(v, float)
            else f"  {k:<35} {v}"
        )
    return "\n".join(lines)


def per_column_metrics(preds_df, labels_df, cols, metric="f1"):
    return {
        col: compare_preds(preds_df, labels_df, target_columns=[col])[metric]
        for col in cols
    }


def per_field_metrics(preds_df, labels_df, cols, metric="f1"):
    fields = labels_df["field"].unique()
    return {
        field: compare_preds(
            preds_df[labels_df["field"].eq(field).values],
            labels_df[labels_df["field"] == field],
            target_columns=cols,
        )[metric]
        for field in fields
    }


# ── Strategy builders ─────────────────────────────────────────────────────────

def build_embedding_model(args, device):
    """Return a SentenceTransformer for the chosen embedding model."""
    from sentence_transformers import SentenceTransformer

    if args.embedding_model == "multi-qa-mpnet-base-dot-v1":
        print(f"Building embedding model: {args.embedding_model}")
        return SentenceTransformer(args.embedding_model, device=str(device))

    # Token-classifier models 
    print(f"Building sentence encoder from {args.embedding_model} (mean pooling)...")
    from sentence_transformers import models as st_models
    word_model   = st_models.Transformer(args.embedding_model)
    pooling      = st_models.Pooling(
        word_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True
    )
    return SentenceTransformer(modules=[word_model, pooling], device=str(device))


def build_example_strategy(args, train_df, device):
    """Instantiate the example-selection strategy from parsed args."""
    from mllms import (
        SimilarExamples, NERPatternSimilarExamples,
        HybridSimilarExamples, FallbackExamplesStrategy,
    )

    n = args.num_examples

    if args.strategy == "embedding":
        sent_model = build_embedding_model(args, device)
        strategy = SimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            embedding_model=sent_model,
            device=device,
        )

    elif args.strategy == "pattern":
        if not Path(args.ner_model_dir).exists():
            raise FileNotFoundError(
                f"NER model not found at '{args.ner_model_dir}'. "
                "Run `uv run python train_ner.py` first."
            )
        strategy = NERPatternSimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            model_dir=args.ner_model_dir,
        )

    else:  # hybrid
        sent_model = build_embedding_model(args, device)
        emb_strategy = SimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            embedding_model=sent_model,
            device=device,
        )
        if not Path(args.ner_model_dir).exists():
            raise FileNotFoundError(
                f"NER model not found at '{args.ner_model_dir}'. "
                "Run `uv run python train_ner.py` first."
            )
        pat_strategy = NERPatternSimilarExamples(
            example_addresses=train_df["FullAddress"],
            example_labels=train_df,
            num_examples=n,
            labels_to_include=ENTITIES_TO_PREDICT,
            model_dir=args.ner_model_dir,
        )
        strategy = HybridSimilarExamples(
            embedding_strategy=emb_strategy,
            pattern_strategy=pat_strategy,
            num_examples=n,
            pool_size=args.pool_size,
        )

    if args.fallback_threshold > 0:
        strategy = FallbackExamplesStrategy(
            primary=strategy,
            labels_to_include=ENTITIES_TO_PREDICT,
            threshold=args.fallback_threshold,
            num_examples=n,
        )

    return strategy


def build_prompt(args):
    """Return the prompt template for the chosen variant."""
    from mllms import JsonDictPromptTemplate

    entities_str = ", ".join(ENTITIES_TO_PREDICT + ["Other"])
    if args.prompt_variant == "base":
        text = _PROMPT_BASE.format(entities=entities_str)
    else:
        # Insert rules block before the format instruction
        base = _PROMPT_BASE.format(entities=entities_str)
        text = base.replace(
            "Format the output as a JSON object",
            _PROMPT_RULES_SUFFIX + "Format the output as a JSON object",
        )
    return JsonDictPromptTemplate(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    config_name = build_config_name(args)
    out_dir     = Path(args.output_dir)
    preds_path  = out_dir / f"preds_{config_name}_{args.split}.json"
    metrics_path= out_dir / f"metrics_{config_name}_{args.split}.txt"

    print(f"Config:  {config_name}")
    print(f"Split:   {args.split}")
    print(f"Outputs: {preds_path} / {metrics_path}")

    # ── Load data ──────────────────────────────────────────────────────────
    csv_args = dict(keep_default_na=False, dtype=str, na_values=[""])
    train_df = pd.read_csv("open_data/bzkopen_addresses_train.csv", **csv_args)
    val_df   = pd.read_csv("open_data/bzkopen_addresses_val.csv",   **csv_args)
    test_df  = pd.read_csv("open_data/bzkopen_addresses_test.csv",  **csv_args)
    eval_df  = val_df if args.split == "val" else test_df

    n_cards = {"train": 361, "val": 77, "test": 78}
    n_addrs = {s: len(df) for s, df in [("train", train_df), ("val", val_df), ("test", test_df)]}
    addr_per_card = sum(n_addrs[s] / n_cards[s] for s in n_cards)
    estimated_total = TOTAL_BZK_CARDS * addr_per_card

    # ── Load or run predictions ────────────────────────────────────────────
    if preds_path.exists() and not args.force_rerun:
        print(f"Loading cached predictions from {preds_path}...")
        with open(preds_path) as f:
            saved = json.load(f)
        preds, deltatime = saved["preds"], saved["deltatime"]
        print(f"Loaded {len(preds)} predictions (original runtime: {format_time(deltatime)})")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        prompt   = build_prompt(args)
        strategy = build_example_strategy(args, train_df, device)

        addresses = eval_df["FullAddress"].tolist()

        from mllms import LlamaAddressParsingModel
        model = LlamaAddressParsingModel(
            model_name=args.inference_model,
            prompt=prompt,
            example_strategy=strategy,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        batch_size = args.batch_size
        batches    = [addresses[i:i + batch_size] for i in range(0, len(addresses), batch_size)]
        print(f"Running inference on {len(addresses)} addresses (batch {batch_size})...")
        preds = []
        start = time.monotonic()
        for batch in tqdm(batches, desc="Inference", unit="batch"):
            preds.extend(model.parse_addresses(batch))

        deltatime = time.monotonic() - start
        print(f"Done in {deltatime:.1f}s ({len(addresses)/deltatime:.1f} addr/s)")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump({"config": config_name, "preds": preds, "deltatime": deltatime},
                      f, ensure_ascii=False)
        print(f"Saved predictions → {preds_path}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    preds_df     = pd.DataFrame(preds)
    output_lines = []

    metrics = compare_preds(preds_df, eval_df, target_columns=METRICS_COLUMNS)
    metrics["deltatime"]          = deltatime
    metrics["rate"]               = len(eval_df) / deltatime
    metrics["estimatedTotalTime"] = estimated_total / metrics["rate"]
    metrics["error"]              = int(preds_df["error"].notna().sum()) if "error" in preds_df.columns else 0
    metrics["errorRate"]          = metrics["error"] / len(eval_df)

    output_lines.append(format_metrics_block(metrics, label=config_name))

    output_lines.append("\n── Per-entity F1 " + "─" * 43)
    for col, val in per_column_metrics(preds_df, eval_df, ENTITIES_TO_PREDICT).items():
        output_lines.append(f"  {col:<35} {val:.4f}")

    if "field" in eval_df.columns:
        output_lines.append("\n── Per-BZK-field F1 " + "─" * 40)
        for field, val in per_field_metrics(preds_df, eval_df, METRICS_COLUMNS).items():
            output_lines.append(f"  {field:<35} {val:.4f}")

    summary_cols = ["accuracy", "precision", "recall", "f1",
                    "average_similarity", "errorRate", "estimatedTotalTime"]
    output_lines.append("\n── Summary " + "─" * 49)
    for col in summary_cols:
        v = metrics[col]
        output_lines.append(
            f"  {col:<35} {format_time(v)}" if col == "estimatedTotalTime"
            else f"  {col:<35} {v:.4f}" if isinstance(v, float)
            else f"  {col:<35} {v}"
        )

    report = "\n".join(output_lines)
    print(report)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
