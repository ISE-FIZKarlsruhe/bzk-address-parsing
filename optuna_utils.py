from optuna import Trial
from collections import OrderedDict

def suggest_partial_permutation(trial : Trial, list_key : str, items : list[str]) -> list:
    result = []
    for item in items:
        include = trial.suggest_categorical(f"{list_key}_{item}_include", [True, False])
        if include:
            sort_key = trial.suggest_float(f"{list_key}_{item}_sortkey", 0, 1)
            result.append((sort_key, item))
    result.sort(key=lambda x: x[0])
    return [item for _, item in result]

def suggest_permutation(trial : Trial, items : list[str], key_prefix : str) -> list:
    sort_keys = [trial.suggest_float(f"{key_prefix}_{i}_sortkey", 0, 1) for i in range(len(items))]
    sorted_items = sorted(zip(sort_keys, items), key=lambda x: x[0])
    return [item for _, item in sorted_items]