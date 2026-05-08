"""
General utility classes and functions for handling the parsing of addresses.
"""
import pandas as pd
from collections import OrderedDict
import datetime
import re

SEPARATOR_CHARS = ",. -()/\\ \t"

def findall(string, sub):
    """Find all occurrences of `sub` in `string` and return their start indices."""
    indices = []
    start = 0
    while start < len(string):
        start = string.find(sub, start)
        if start == -1: break
        indices.append(start)
        start += len(sub)  # Move past the last found substring
    return indices

def merge_parts(address : str, part1 : str, part1_start : int, part2 : str, part2_start : int) -> str:
    if part1_start < part2_start:
        start = part1_start
        between = address[part1_start + len(part1) : part2_start]
        end = part2_start + len(part2)
    else:
        start = part2_start
        between = address[part2_start + len(part2) : part1_start]
        end = part1_start + len(part1)
    if all(x in SEPARATOR_CHARS for x in between):
        return address[start:end]
    else:
        return None
    
class StrictMergeParsedResultBuilder:
    """
    Build the prediction dictionary handling key conflicts by only merging if one match is fully contained in the other, otherwise keeping them separate to at least not lose information. This is a more strict version of the merging strategy that does not attempt to merge consecutive matches if they are not contained within each other, as this can lead to incorrect merges.
    """
    def __init__(self, original_address):
        self.inner = {}
        self.starts = {}
        self.original_address = original_address

    def add_part(self, label, part, start):
        if not part: # ignore None or empty
            return
        part = str(part)
        if label.startswith("___") or label in ["fullConversation", "model-fullAddress", "error"]:
            print(f"ERROR: Attempt to add reserved label: {label} for address: {self.original_address}")
            return
        if label == "fullAddress":
            label = "model-fullAddress" # avoid collision but keep the wrongfully generated field
        conflict = self.inner.get(label, None)
        if conflict is None:
            self.inner[label] = part
            self.starts[label] = start
        else:
            conflict_start = self.starts[label]
            merged = merge_parts(self.original_address, conflict, conflict_start, part, start)
            if merged is not None:
                self.inner[label] = merged
                self.starts[label] = min(start, conflict_start)
    
    def set_reserved(self, label, value):
        self.inner[label] = value

    def build(self) -> dict:
        return self.inner

class ParsedAddressResultBuilder:
    """
    Build the prediction dictionary handling key conflicts
    """
    def __init__(self, original_address, discard_ignorable_conflicts = False):
        self.inner = {}
        self.original_address = original_address
        self.discard_ignorable_conflicts = discard_ignorable_conflicts

    def _merge_components(self, conflict : str, new_component : str, separator = "___") -> str:
        if not conflict:
            return new_component
        
         # Ignore repeating matches
        if conflict == new_component:
            return conflict
        
         # Take larger match if one is contained in the other
        if conflict in new_component:
            return new_component 
        if new_component in conflict:
            return conflict
        
        # Try to merge if both are consecutive
        conflict_starts = findall(self.original_address, conflict)
        new_component_starts = findall(self.original_address, new_component)
        if conflict_starts and new_component_starts:
            for conflict_start in conflict_starts:
                for new_component_start in new_component_starts:
                    merged = merge_parts(self.original_address, conflict, conflict_start, new_component, new_component_start)
                    if merged is not None:
                        return merged
        if self.discard_ignorable_conflicts:
            if all(c in SEPARATOR_CHARS for c in conflict):
                return new_component
            elif all(c in SEPARATOR_CHARS for c in new_component):
                return conflict
        
        # Failure to solve conflict; set prediction to a combination of both components to at least not lose the information
        return conflict + separator + new_component

    
    def add_part(self, label, part):
        if not part: # ignore None or empty
            return
        part = str(part)
        if label.startswith("___") or label in ["fullConversation", "model-fullAddress", "error"]:
            print(f"ERROR: Attempt to add reserved label: {label} for address: {self.original_address}")
            return
        if label == "fullAddress":
            label = "model-fullAddress" # avoid collision but keep the wrongfully generated field
        conflict = self.inner.get(label, None)
        self.inner[label] = self._merge_components(conflict, part)

    def set_reserved(self, label, value):
        self.inner[label] = value

    def build(self) -> dict:
        return self.inner

class Aggregator:
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0
        self.max = None
        self.min = None

    def aggregate_single(self, value):
        if not pd.isna(value):
            self.sum += value
            self.count += 1
            if self.max is None or value > self.max:
                self.max = value
            if self.min is None or value < self.min:
                self.min = value
    
    def aggregate(self, values):
        if isinstance(values, pd.DataFrame):
            self.sum += values.sum().sum()
            self.count += values.count().sum()
            max_value = values.max().max()
            min_value = values.min().min()
            self.max = max_value if self.max is None else max(self.max, max_value)
            self.min = min_value if self.min is None else min(self.min, min_value)
        elif isinstance(values, pd.Series):
            self.sum += values.sum()
            self.count += values.count()
            self.max = values.max() if self.max is None else max(self.max, values.max())
            self.min = values.min() if self.min is None else min(self.min, values.min())
        else:
            for value in values:
                self.aggregate_single(value)

    @property
    def mean(self):
        return self.sum / self.count if self.count > 0 else float('nan')
    
    def get_all(self):
        return self.get(["mean", "max", "min"])

    def get(self, keys):
        result = {}
        if "mean" in keys:
            result["mean"] = self.mean
        if "max" in keys:
            result["max"] = self.max
        if "min" in keys:
            result["min"] = self.min
        return result



# Adapted Mahsa's code for levenshtein distance
def levenshtein(a: str, b: str, case_insensitive=True, max_distance=None) -> int:
    if max_distance is None: max_distance = float('inf')
    # If one of the strings is empty
    try:
        if len(a) == 0:
            return len(b)
    except TypeError:
        print(f"TypeError: a is not a string: {a}")
        raise
    if len(b) == 0:
        return len(a)
    
    if case_insensitive:
        a = a.lower()
        b = b.lower()

    # Create distance matrix (size: (len(a)+1) x (len(b)+1))
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    # Initialize first row/column
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    # Fill in matrix
    for i in range(1, len(a) + 1):
        all_exceeded = True
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
            if dp[i][j] <= max_distance:
                all_exceeded = False
        if all_exceeded:
            return max_distance + 1
    distance = dp[-1][-1]

    return distance

def compare_preds(preds : pd.DataFrame, labels : pd.DataFrame, target_columns, ignore_trash_columns = True):
    # Drop meta columns that may be included in the preds dataframe
    assert len(preds) == len(labels), f"Length mismatch between preds and labels"

    tolerance_levels = 5
    correct_with_tol = [0,] * tolerance_levels
    total_rows = 0
    prediction_count = 0
    label_count = 0
    true_positives = 0
    
    sum_levenshtein = 0
    sum_similarity = 0.0
    sum_levenshtein_match = 0
    sum_similarity_match = 0.0
    some_match_count = 0
    
    if not ignore_trash_columns:
        # labels that should not have been predicted at all
        trash_predictions = preds[[col for col in preds.columns if col not in labels.columns]].stack()
        trash_count = trash_predictions.notna().sum()
        total_rows += trash_count
        prediction_count += trash_count
        sum_levenshtein += trash_predictions.dropna().astype(str).str.len().sum()
    for col in target_columns:
        total_rows += len(labels)
        label_count += labels[col].notna().sum()
        if col not in preds.columns:
            # all missing predictions are incorrect
            sum_levenshtein += labels[col].dropna().str.len().sum()
        else:
            prediction_count += preds[col].notna().sum()
            strings_to_compare = pd.concat([preds[col].fillna(""), labels[col].fillna("")], axis=1)
            levenshtein_scores = strings_to_compare.apply(
                lambda row: levenshtein(row.iloc[0], row.iloc[1]), axis=1
            )
            max_lens = strings_to_compare.apply(lambda col: col.str.len()).max(axis=1)
            similarity = ((max_lens - levenshtein_scores) / max_lens).fillna(1.0) # nan => div by 0 => both are empty strings => similarity 1.0
            sum_levenshtein += levenshtein_scores.sum()
            sum_similarity += similarity.sum()
            sum_levenshtein_match += levenshtein_scores[similarity >= 0].sum()
            sum_similarity_match += similarity[similarity >= 0].sum()
            some_match_count += (similarity > 0).sum()
            true_positives += ((levenshtein_scores == 0) & preds[col].notna() & labels[col].notna()).sum()
            for tol in range(tolerance_levels):
                correct_with_tol[tol] += (levenshtein_scores <= tol).sum()
    results = OrderedDict()
    results["accuracy"] = correct_with_tol[0] / total_rows
    results["precision"] = true_positives / prediction_count if prediction_count > 0 else 0.0
    results["recall"] = true_positives / label_count if label_count > 0 else 0.0
    results["f1"] = 2 * results["precision"] * results["recall"] / (results["precision"] + results["recall"]) if (results["precision"] + results["recall"]) > 0 else 0.0
    for tol in range(1, tolerance_levels):
        results[f"accuracy_with_tol_{tol}"] = correct_with_tol[tol] / total_rows
    results["average_levenshtein"] = sum_levenshtein / total_rows
    results["average_similarity"] = sum_similarity / total_rows
    results["average_levenshtein_match"] = sum_levenshtein_match / some_match_count if some_match_count > 0 else 0.0
    results["average_similarity_match"] = sum_similarity_match / some_match_count if some_match_count > 0 else 0.0
    results["no_match_rate"] = 1.0 - (some_match_count / total_rows)
    return results

# Adapted code for partial levenshtein distance
def partial_levenshtein(key: str, query: str, case_insensitive=True) -> tuple[int, tuple[int, int]]:
    # If one of the strings is empty
    if len(key) == 0:
        return len(query), (0, 0)
    if len(query) == 0:
        return 0, (0, 0)
    
    if case_insensitive:
        key = key.lower()
        query = query.lower()
    # Create distance matrix (size: (len(key)+1) x (len(query)+1))
    dp = [[0] * (len(key) + 1) for _ in range(len(query) + 1)]
    start =[[0] * (len(key) + 1) for _ in range(len(query) + 1)]
    # Initialize first row/column
    for i in range(len(query) + 1):
        dp[i][0] = i
        # start[i][0] = 0
    for j in range(len(key) + 1):
        # The cost of starting later on the key string is 0 as we are looking for the best partial match
        # dp[0][j] = 0
        start[0][j] = j

    # Fill in matrix
    for i in range(1, len(query) + 1):
        for j in range(1, len(key) + 1):
            cost = 0 if query[i - 1] == key[j - 1] else 1
            dp[i][j], start[i][j] = min(
                (dp[i - 1][j] + 1, start[i - 1][j]),      # deletion
                (dp[i][j - 1] + 1, start[i][j - 1]),      # insertion
                (dp[i - 1][j - 1] + cost, start[i - 1][j - 1]) # substitution
            )
    distance_spans = [(dp[-1][j], (start[-1][j], j-1)) for j in range(len(key) + 1)]
    distance, span = min(distance_spans, key=lambda x: (x[0], x[1][0] - x[1][1])) # prefer smaller distance, then larger span
    return distance, span

def format_time(seconds, round_to_seconds=True):
    seconds = round(seconds) if round_to_seconds else seconds
    timedelta = datetime.timedelta(seconds=seconds)
    days = timedelta.days
    months, days = divmod(days, 30)
    years, months = divmod(months, 12)
    timedelta = timedelta - datetime.timedelta(days=timedelta.days) + datetime.timedelta(days=days)
    sb = []
    if years > 0:
        sb.append(f"{years} year{'s' if years > 1 else ''}")
    if months > 0:
        sb.append(f"{months} month{'s' if months > 1 else ''}")
    sb.append(f"{timedelta}")
    return ", ".join(sb)

def natural_casing(qualified_name: str, casing="UpperCamelCase") -> str:
    """Converts a qualified name (eg. in UpperCammelCase) to a standard natural casing format"""
    match casing:
        case "UpperCamelCase":
            return " ".join(map(lambda m: m.group(), re.finditer(r"[A-Z\d]*[a-z\d]*", qualified_name)))
        case _:
            raise ValueError(f"Unsupported casing: {casing}")