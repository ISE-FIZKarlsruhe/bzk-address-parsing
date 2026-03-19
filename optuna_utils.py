from optuna import Trial

def suggest_partial_permutation(trial : Trial, list_key : str, items : list[str]) -> list:
    result = []
    for item in items:
        include = trial.suggest_categorical(f"{list_key}_{item}_include", [True, False])
        if include:
            sort_key = trial.suggest_float(f"{list_key}_{item}_sortkey", 0, 1)
            result.append((sort_key, item))
    # key = lambda x: x[0] to maintain original order on a tie
    result.sort(key=lambda x: x[0])
    return [item for _, item in result]

def suggest_permutation(trial : Trial, items : list[str], key_prefix : str) -> list:
    sort_keys = [
        trial.suggest_float(f"{key_prefix}_{item}_sortkey", 0, 1) 
        for item in items
    ] 
    # key = lambda x: x[0] to maintain original order on a tie
    sorted_items = sorted(zip(sort_keys, items), key=lambda x: x[0])
    return [item for _, item in sorted_items]