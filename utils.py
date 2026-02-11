"""
General utility classes and functions for handling the parsing of addresses.
"""
import pandas as pd
from collections import OrderedDict

class ParsedAddressResultBuilder:
    """
    Build the prediction dictionary handling key conflicts
    """
    def __init__(self, original_address):
        self.inner = {}
        self.original_address = original_address

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
        conflict_start = self.original_address.find(conflict[0])
        new_component_start = self.original_address.find(new_component)
        if conflict_start != -1 and new_component_start != -1:
            if conflict_start < new_component_start:
                start = conflict_start
                between = self.original_address[conflict_start + len(conflict[0]) : new_component_start]
                end = new_component_start + len(new_component)
            else:
                start = new_component_start
                between = self.original_address[new_component_start + len(new_component) : conflict_start]
                end = conflict_start + len(conflict[0])
            ignored_chars = set(",. -/\\ \t") # set of separator characters that can be ignored when checking for consecutivity
            if all(x in ignored_chars for x in between):
                return self.original_address[start:end]
            
        # Failure to solve conflict; set prediction to a combination of both components to at least not lose the information
        return conflict + separator + new_component

    
    def add_part(self, label, part):
        if not part: # ignore None or empty
            return
        part = str(part)
        if label in ["fullConversation", "model-fullAddress", "error"]:
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



# Adapted Mahsa's code for levenshtein distance
def levenshtein(a: str, b: str, case_insensitive=True) -> int:
    # If one of the strings is empty
    if len(a) == 0:
        return len(b)
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
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    distance = dp[-1][-1]

    return distance

def compare_preds(preds : pd.DataFrame, labels : pd.DataFrame, target_columns, ignore_trash_columns = True):
    # Drop meta columns that may be included in the preds dataframe
    labels = labels.astype(str)

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
            levenshtein_bounds = strings_to_compare.apply(
                lambda row: max(len(row.iloc[0]), len(row.iloc[1])), axis=1
            )
            similarity = ((levenshtein_bounds - levenshtein_scores) / levenshtein_bounds).fillna(1.0) # nan => div by 0 => both are empty strings => similarity 1.0
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
def partial_levenshtein(containing_string: str, substring: str, case_insensitive=True) -> tuple[int, int]:
    # If one of the strings is empty
    if len(containing_string) == 0:
        return len(substring), 0
    if len(substring) == 0:
        return 0, 0
    
    if case_insensitive:
        containing_string = containing_string.lower()
        substring = substring.lower()
    # Create distance matrix (size: (len(containing_string)+1) x (len(substring)+1))
    dp = [[0] * (len(containing_string) + 1) for _ in range(len(substring) + 1)]
    start =[[0] * (len(containing_string) + 1) for _ in range(len(substring) + 1)]
    # Initialize first row/column
    for i in range(len(substring) + 1):
        dp[i][0] = i
        # start[i][0] = 0
    for j in range(len(containing_string) + 1):
        # The cost of starting later on the containing string is 0 as we are looking for the best partial match
        # dp[0][j] = 0
        start[0][j] = j

    # Fill in matrix
    for i in range(1, len(substring) + 1):
        for j in range(1, len(containing_string) + 1):
            cost = 0 if substring[i - 1] == containing_string[j - 1] else 1
            dp[i][j], start[i][j] = min(
                (dp[i - 1][j] + 1, start[i - 1][j]),      # deletion
                (dp[i][j - 1] + 1, start[i][j - 1]),      # insertion
                (dp[i - 1][j - 1] + cost, start[i - 1][j - 1]) # substitution
            )
    distance, selected_start = min((dp[-1][j], start[-1][j]) for j in range(len(containing_string) + 1))
    return distance, selected_start