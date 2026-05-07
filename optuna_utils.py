from optuna import Trial
import warnings

def suggest_partial_permutation(trial : Trial, prefix : str, items : list[str]) -> list:
    result = []
    for item in items:
        include = trial.suggest_categorical(f"{prefix}_{item}_include", [True, False])
        if include:
            result.append(item)
    if len(result) > 1:
        result = suggest_permutation(trial, f"{prefix}", result)
    return result

def _decode(lehmer_code: list[int], items : list[str]) -> list[int]:
    """Decode Lehmer code to permutation.

    This function decodes Lehmer code represented as a list of integers to a permutation.
    Adapted from https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-deal-with-permutation-as-a-parameter
    """
    if len(items) <= 1:
        warnings.warn(f"List of {len(items)} items is permutation invariant, returning items as is.")
        return items
    all_indices = list(range(len(lehmer_code)))
    output = []
    for k in lehmer_code:
        idx = all_indices[k]
        try:
            output.append(items[idx]) 
        except IndexError:
            raise ValueError(f"Invalid Lehmer code {lehmer_code} resulted in index {idx} out of bounds for items of length {len(items)}")
        all_indices.remove(idx)
    return output

def suggest_permutation(trial : Trial, prefix : str, items : list[str]) -> list:
    # based on https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-deal-with-permutation-as-a-parameter
    sort_keys = [
        trial.suggest_int(f"{prefix}_{item}_sortkey", 0, len(items) - i - 1) 
        for i, item in enumerate(items[:-1])
    ]
    sort_keys.append(0)  # last item always gets sort key 0
    assert len(sort_keys) == len(items)
    shuffled_items = _decode(sort_keys, items)
    for item in items:
        assert item in shuffled_items, f"Original item {item} not in shuffled items {shuffled_items}. Lehmer code: {sort_keys}"
    for i, item, in enumerate(shuffled_items):
        trial.set_user_attr(f"{prefix}_{item}_index", i)
    return shuffled_items