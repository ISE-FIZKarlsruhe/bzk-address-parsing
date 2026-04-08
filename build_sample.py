"""
Build a stratified random sample (~500 records).

Two modes:

  False — labeled subset
      Source      : matched_offices_by_fullname_2.csv  (~1M labeled records)
      Stratification: Layout class × Matched Office × year bucket

  True — full dataset
      Source      : all JSONL files under BZK_DATA_DIR
      Stratification: CompensationOffice1 × year bucket
      maximum coverage without labeling bias
"""

import json
import argparse
import pandas as pd
from collections import defaultdict
from pathlib import Path

BZK_DATA_DIR = Path("/home/bzk-data")

IGNORE_CLASSES = {
    "Gerichtsurteile",
    "Gelbe-Hinweiskarte",
    "Rückseite_Weitere Namen",
    "Siehe-auch-Hinweiskarte",
}

# Fields to pull from JSONL records
RECORD_FIELDS = [
    "BZKNr",
    "CompensationOffice1",
    "ApplicantFirstName", "ApplicantLastName", "ApplicantAltFirstName", "ApplicantAltLastName",
    "ApplicantBirthName", "ApplicantBirthDate", "ApplicantBirthPlace",
    "ApplicantCurrentAddress", "ApplicantMaritalStatus",
    "VictimFirstName", "VictimLastName", "VictimAltFirstName", "VictimAltLastName",
    "VictimBirthName", "VictimBirthDate", "VictimBirthPlace",
    "VictimDeathDate", "VictimDeathPlace", "VictimCurrentAddress",
    "Heirs",
]

TARGET          = 500
MIN_PER_STRATUM = 1
RANDOM_STATE    = 42

# Year buckets (decade-aligned, equal frequency)
#   pre-1900 ~288k | 1900-1909 ~264k | 1910-1919 ~243k | 1920-1929 ~234k | 1930+ ~134k
YEAR_BINS   = [0,    1899,       1909,       1919,       1929,    9999]
YEAR_LABELS = ["pre-1900", "1900-1909", "1910-1919", "1920-1929", "1930+"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--full", action="store_true",
                    help="Use full JSONL dataset (CompensationOffice1 × year bucket) "
                         "instead of labeled CSV subset (Layout class × Matched Office × year bucket)")
parser.add_argument("--target", type=int, default=TARGET, help=f"Sample size (default: {TARGET})")
parser.add_argument("--out", type=Path, default=Path("sample_500.xlsx"), help="Output Excel path")
args = parser.parse_args()

TARGET   = args.target
out_path = args.out

# Build population dataframe 

if not args.full:
    # Labeled subset: CSV with layout class
    print("Mode: labeled subset  (matched_offices_by_fullname_2.csv)")
    df = pd.read_csv(BZK_DATA_DIR / "matched_offices_by_fullname_2.csv")
    df = df[~df["Layout class"].isin(IGNORE_CLASSES)].copy()
    df["year"] = pd.to_numeric(df["Filename"].str.split("_").str[0], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["year_bucket"] = pd.cut(df["year"], bins=YEAR_BINS, labels=YEAR_LABELS)
    strat_cols = ["Layout class", "Matched Office", "year_bucket"]
    meta_cols  = ["Layout class", "Matched Office", "Exclusive Office", "year_bucket", "year", "Filename"]

else:
    # Full dataset: scan all JSONL files
    print("Mode: full dataset  (all JSONL files)")
    rows = []
    for jsonl in sorted(BZK_DATA_DIR.glob("*.jsonl")):
        year = jsonl.stem
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                rows.append({
                    "Filename":            rec.get("filename", ""),
                    "CompensationOffice1": rec.get("CompensationOffice1", ""),
                    "year":                int(year),
                })
        print(f"  loaded {jsonl.name}  ({len(rows):,} records so far)")
    df = pd.DataFrame(rows)
    df = df[df["Filename"].str.match(r"^\d{4}_")].copy()
    df["year_bucket"] = pd.cut(df["year"], bins=YEAR_BINS, labels=YEAR_LABELS)
    strat_cols = ["CompensationOffice1", "year_bucket"]
    meta_cols  = ["CompensationOffice1", "year_bucket", "year", "Filename"]

# Stratified sampling 

strata        = df.groupby(strat_cols, observed=True)
stratum_sizes = strata.size().rename("n")

# Square-root allocation: allocate proportional to sqrt(stratum_size) instead of
# stratum_size. This compresses the ratio between large and small strata — e.g. a
# stratum 10000x larger than another gets sqrt(10000)=100x more samples instead of
# 10000x — shifting proportional share away from dominant strata toward rare ones
# without over-representing size-1 strata (sqrt(1)=1, so they are unaffected).
sqrt_sizes = stratum_sizes ** 0.5
alloc = (
    (sqrt_sizes / sqrt_sizes.sum() * TARGET)
    .clip(lower=MIN_PER_STRATUM)
    .round()
    .astype(int)
)

chunks = []
for name, g in strata:
    n = min(len(g), alloc.loc[name])
    chunks.append(g.sample(n, random_state=RANDOM_STATE))
sample_df = pd.concat(chunks).reset_index(drop=True)

print(f"\nStrata (non-empty)  : {(stratum_sizes > 0).sum():,}")
print(f"Sample size         : {len(sample_df):,}")
print()
if not args.full:
    print("Per layout class:")
    print(sample_df.groupby("Layout class").size().sort_values(ascending=False).to_string())
    print()
print(f"Per {'CompensationOffice1' if args.full else 'Matched Office'}:")
office_col = "CompensationOffice1" if args.full else "Matched Office"
print(sample_df.groupby(office_col).size().sort_values(ascending=False).to_string())
print()
print("Per year bucket:")
print(sample_df.groupby("year_bucket", observed=True).size().to_string())

# Load JSONL fields for sampled filenames 

sampled_fnames = set(sample_df["Filename"])
by_year = defaultdict(set)
for fname in sampled_fnames:
    year = fname.split("_")[0]
    by_year[year].add(fname)

records = {}
for year, fnames in sorted(by_year.items()):
    jsonl = BZK_DATA_DIR / f"{year}.jsonl"
    if not jsonl.exists():
        continue
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fname = rec.get("filename", "")
            if fname in fnames:
                records[fname] = {f: rec.get(f, "") for f in RECORD_FIELDS}

print(f"\nJSONL records found : {len(records):,} / {len(sampled_fnames):,}")

# Merge and save 

records_df = pd.DataFrame.from_dict(records, orient="index").reset_index()
records_df = records_df.rename(columns={"index": "Filename"})

result_df = sample_df.merge(records_df, on="Filename", how="left")
result_df = result_df[meta_cols + [f for f in RECORD_FIELDS if f not in meta_cols]]

result_df.to_excel(out_path, index=False)
print(f"Saved to {out_path}  ({len(result_df):,} rows × {len(result_df.columns):,} cols)")
