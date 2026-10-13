"""
Counts, in the output directory of modules/entity_linking.py, the address fields
with an EMPTY_PARSING_RESULT search status whose raw address is an empty string.

Usage: python count_empty_raw.py OUTPUT_DIR [-P PATTERN]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

PREFIXES = ["abp", "aca", "vbp", "vdp", "vca"]

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("output_dir")
arg_parser.add_argument("-P", "--file-pattern", default="*.jsonl")
args = arg_parser.parse_args()

empty_raw = Counter()        # per prefix: EMPTY_PARSING_RESULT and raw == ""
empty_parsing = Counter()    # per prefix: EMPTY_PARSING_RESULT
rows_with_empty_raw = 0
total_rows = 0
invalid_lines = 0

for path in sorted(Path(args.output_dir).rglob(args.file_pattern)):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            total_rows += 1
            row_has_empty_raw = False
            for prefix in PREFIXES:
                if row.get(f"{prefix}.search_status") != "EMPTY_PARSING_RESULT":
                    continue
                empty_parsing[prefix] += 1
                if row.get(f"{prefix}.raw") == "":
                    empty_raw[prefix] += 1
                    row_has_empty_raw = True
            rows_with_empty_raw += row_has_empty_raw

print(f"Rows: {total_rows:,}" + (f" ({invalid_lines:,} invalid lines skipped)" if invalid_lines else ""))
print(f"Rows with at least one EMPTY_PARSING_RESULT field whose raw is \"\": {rows_with_empty_raw:,}")
print()
print(f"{'Prefix':<8}{'EMPTY_PARSING_RESULT':>22}{'... with raw == \"\"':>22}")
for prefix in PREFIXES:
    if empty_parsing[prefix]:
        print(f"{prefix:<8}{empty_parsing[prefix]:>22,}{empty_raw[prefix]:>22,}")
print(f"{'Total':<8}{sum(empty_parsing.values()):>22,}{sum(empty_raw.values()):>22,}")
