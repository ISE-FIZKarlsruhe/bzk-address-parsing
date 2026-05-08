"""
Custom NER training for BZK address components using spaCy.

Fine-tunes de_core_news_sm with new entity labels:
    HouseNumber, StreetName, Neighborhood, City, District, State, Country

Usage:
    uv run python train_ner.py
    uv run python train_ner.py --n-iter 30 --output models/ner_bzk
    uv run python train_ner.py --augment --geonames-user <username>
    uv run python train_ner.py --eval-only --output models/ner_bzk
"""

import argparse
import random
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from tqdm.auto import tqdm
import re

# ── Config ───────────────────────────────────────────────────────────────────

ENTITIES = ["HouseNumber", "StreetName", "Neighborhood", "City", "District", "State", "Country"]

TRAIN_CSV = Path("open_data/bzkopen_addresses_train.csv")
VAL_CSV   = Path("open_data/bzkopen_addresses_val.csv")
BASE_MODEL = "de_core_news_sm"

CSV_READ_ARGS = dict(keep_default_na=False, dtype=str, na_values=[""])

# ── Span alignment ────────────────────────────────────────────────────────────

def find_spans(address: str, row: pd.Series) -> list[tuple[int, int, str]]:
    """
    Find character-level spans for each entity in the address string.

    Strategy:
      - Exact substring search in left-to-right order of ENTITIES.
      - Skip if value not found (e.g., expanded abbreviation not in raw text).
      - Skip if span overlaps with an already-assigned span.

    Returns a list of (start, end, label) sorted by start position.
    """
    spans: list[tuple[int, int, str]] = []
    used: list[tuple[int, int]] = []  

    for label in ENTITIES:
        value = row.get(label, "")
        if not value or not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue

        # Search left to right, skip overlapping positions
        search_from = 0
        while True:
            idx = address.find(value, search_from)
            if idx == -1:
                break  # entity value not found in address text
            end = idx + len(value)
            if not any(s < end and e > idx for s, e in used):
                spans.append((idx, end, label))
                used.append((idx, end))
                break
            search_from = idx + 1

    spans.sort(key=lambda x: x[0])
    return spans


def spacify_words(text: str) -> str:
    """Add spaces around non-alphanumeric characters to improve tokenization."""
    return "".join(f" {c} " if not c.isalnum() else c for c in text)

def df_to_ner_data(df: pd.DataFrame, label: str = "") -> list[tuple[str, dict]]:
    """Convert a DataFrame to a list of (address, {entities: [...]}) tuples."""
    raw: list[tuple[str, dict]] = []
    n_skipped = 0
    n_entity_misses = 0

    for _, row in df.iterrows():
        address = row.get("FullAddress", "")
        if not address or not isinstance(address, str):
            n_skipped += 1
            continue

        spans = find_spans(address, row)

        for ent_label in ENTITIES:
            value = row.get(ent_label, "")
            if value and isinstance(value, str) and value.strip():
                if not any(lbl == ent_label for _, _, lbl in spans):
                    n_entity_misses += 1

        if spans:
            raw.append((address, {"entities": spans}))
        else:
            n_skipped += 1

    tag = f" ({label})" if label else ""
    print(f"  {len(raw)} examples{tag} "
          f"({n_skipped} skipped, {n_entity_misses} entity values not aligned)")
    return raw


def load_ner_data(csv_path: Path) -> list[tuple[str, dict]]:
    """Load a CSV and convert to NER training format."""
    df = pd.read_csv(csv_path, **CSV_READ_ARGS)
    return df_to_ner_data(df, label=csv_path.name)


def retokenize(doc):
    with doc.retokenize() as retokenizer:
        for token in doc:
            if not token.text:
                continue
            subtokens = re.split(r"(\W)", token.text)
            subtokens = [s for s in subtokens if s != ""]
            if len(subtokens) > 1:
                retokenizer.split(token, subtokens, heads=[token] * len(subtokens))

# ── Training ──────────────────────────────────────────────────────────────────

def train(
    n_iter: int = 30,
    output_dir: Path = Path("models/ner_bzk"),
    dropout: float = 0.3,
    seed: int = 42,
    train_df : pd.DataFrame | None = None,
    augmented_train_df: pd.DataFrame | None = None,
    val_df : pd.DataFrame | None = None
):
    random.seed(seed)

    print(f"Loading base model: {BASE_MODEL}")
    nlp = spacy.load(BASE_MODEL)

    # Add / retrieve the NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for label in ENTITIES:
        ner.add_label(label)

    print("Loading training data...")
    if augmented_train_df is not None:
        if train_df is not None:
            warnings.warn("Both train_df and augmented_train_df provided; using augmented_train_df")
        orig_count = (augmented_train_df["is_augmented"] == False).sum()
        aug_count  = (augmented_train_df["is_augmented"] == True).sum()
        print(f"  Using augmented dataset: {orig_count} original + {aug_count} synthetic rows")
        train_raw = df_to_ner_data(augmented_train_df, label="augmented train")
    elif train_df is not None:
        print(f"  Using provided training DataFrame with {len(train_df)} rows")
        train_raw = df_to_ner_data(train_df, label="provided train")
    else:
        train_raw = load_ner_data(TRAIN_CSV)
    print("Loading validation data...")
    if val_df is not None:
        print(f"  Using provided validation DataFrame with {len(val_df)} rows")
        val_raw = df_to_ner_data(val_df, label="provided val")
    else:
        val_raw = load_ner_data(VAL_CSV)

    # Convert raw tuples to spaCy Example objects
    def to_examples(raw, nlp):
        examples = []
        for text, annots in raw:
            doc = nlp.make_doc(text)
            retokenize(doc)
            try:
                ex = Example.from_dict(doc, annots)
                examples.append(ex)
            except Exception as e:
                warnings.warn(f"Skipping '{text[:40]}': {e}")
        return examples

    # Freeze all pipes except NER during training
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    print(f"\nTraining for {n_iter} iterations (dropout={dropout})...")
    print(f"  Freezing pipes: {other_pipes}")

    optimizer = nlp.initialize()

    best_f1 = 0.0
    history = []

    with nlp.select_pipes(enable=["ner"]):
        sizes = compounding(4.0, 32.0, 1.001)
        for i in tqdm(range(n_iter)):
            random.shuffle(train_raw)
            losses = {}
            batches = minibatch(train_raw, size=sizes)
            for batch in batches:
                examples = []
                for text, annots in batch:
                    doc = nlp.make_doc(text)
                    retokenize(doc)
                    try:
                        examples.append(Example.from_dict(doc, annots))
                    except Exception:
                        continue
                nlp.update(examples, drop=dropout, losses=losses, sgd=optimizer)

            # Evaluate on val set every iteration
            val_examples = to_examples(val_raw, nlp)
            scores = nlp.evaluate(val_examples)
            ents_p = scores["ents_p"]
            ents_r = scores["ents_r"]
            ents_f = scores["ents_f"]

            history.append({"iter": i + 1, "loss": losses.get("ner", 0),
                            "p": ents_p, "r": ents_r, "f1": ents_f})

            marker = " *" if ents_f > best_f1 else ""
            tqdm.write(f"  iter {i+1:3d}  loss={losses.get('ner', 0):8.2f}  "
                  f"P={ents_p:.4f}  R={ents_r:.4f}  F1={ents_f:.4f}{marker}")

            if ents_f > best_f1:
                best_f1 = ents_f
                output_dir.mkdir(parents=True, exist_ok=True)
                nlp.to_disk(output_dir)

    print(f"\nBest val F1: {best_f1:.4f} — model saved to {output_dir}")
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_dir: Path = Path("models/ner_bzk"), split: str = "val"):
    csv_path = VAL_CSV if split == "val" else TRAIN_CSV
    print(f"Loading model from {model_dir}...")
    nlp = spacy.load(model_dir)

    print(f"Loading {split} data from {csv_path.name}...")
    raw = load_ner_data(csv_path)

    # Per-entity counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for text, annots in raw:
        doc = nlp(text)
        pred_spans = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        gold_spans = set(map(tuple, annots["entities"]))

        for span in pred_spans:
            if span in gold_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in gold_spans:
            if span not in pred_spans:
                fn[span[2]] += 1

    print(f"\n{'Entity':<15} {'P':>7} {'R':>7} {'F1':>7}  (TP / FP / FN)")
    print("-" * 52)
    all_tp = all_fp = all_fn = 0
    for label in ENTITIES:
        p  = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0
        r  = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"  {label:<13} {p:7.4f} {r:7.4f} {f1:7.4f}  "
              f"({tp[label]} / {fp[label]} / {fn[label]})")
        all_tp += tp[label]; all_fp += fp[label]; all_fn += fn[label]

    print("-" * 52)
    mp = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0
    mr = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0
    print(f"  {'OVERALL':<13} {mp:7.4f} {mr:7.4f} {mf:7.4f}  "
          f"({all_tp} / {all_fp} / {all_fn})")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / evaluate BZK NER model")
    parser.add_argument("--n-iter",        type=int,   default=30,                    help="Training iterations (default: 30)")
    parser.add_argument("--output",        type=Path,  default=Path("models/ner_bzk"), help="Model output directory")
    parser.add_argument("--dropout",       type=float, default=0.3,                   help="Dropout rate (default: 0.3)")
    parser.add_argument("--eval-only",     action="store_true",                       help="Skip training, just evaluate existing model")
    parser.add_argument("--split",         default="val", choices=["train", "val"],   help="Split to evaluate on (default: val)")
    parser.add_argument("--augment",       action="store_true",                       help="Augment training data via GeoNames city substitution")
    parser.add_argument("--geonames-user", default="amelgd",                          help="GeoNames API username (default: amelgd)")
    parser.add_argument("--n-augments",    type=int,   default=3,                     help="Synthetic rows per original row (default: 3)")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(model_dir=args.output, split=args.split)
    else:
        augmented_df = None
        if args.augment:
            from modules.augmentation import GeoNamesLookup, augment_dataset
            print(f"Running data augmentation (n_augments={args.n_augments}, user={args.geonames_user})...")
            bzkopen_train = pd.read_csv(TRAIN_CSV, **CSV_READ_ARGS)
            geo = GeoNamesLookup(username=args.geonames_user)
            augmented_df = augment_dataset(bzkopen_train, geo, n_augments=args.n_augments)
            orig = (augmented_df["is_augmented"] == False).sum()
            synth = (augmented_df["is_augmented"] == True).sum()
            print(f"Augmentation complete: {orig} original + {synth} synthetic = {len(augmented_df)} total rows")

        train(n_iter=args.n_iter, output_dir=args.output, dropout=args.dropout,
              augmented_train_df=augmented_df)
        print()
        evaluate(model_dir=args.output, split=args.split)
