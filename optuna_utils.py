from optuna import Trial

def suggest_partial_permutation(trial : Trial, prefix : str, items : list[str]) -> list:
    result = []
    for item in items:
        include = trial.suggest_categorical(f"{prefix}_{item}_include", [True, False])
        if include:
            sort_key = trial.suggest_float(f"{prefix}_{item}_sortkey", 0, 1)
            result.append((sort_key, item))
    # key = lambda x: x[0] to maintain original order on a tie
    result.sort(key=lambda x: x[0])
    return [item for _, item in result]

def suggest_permutation(trial : Trial, prefix : str, items : list[str]) -> list:
    # TODO change based on https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-deal-with-permutation-as-a-parameter
    sort_keys = [
        trial.suggest_float(f"{prefix}_{item}_sortkey", 0, 1) 
        for item in items
    ] 
    # key = lambda x: x[0] to maintain original order on a tie
    sorted_items = sorted(zip(sort_keys, items), key=lambda x: x[0])
    return [item for _, item in sorted_items]