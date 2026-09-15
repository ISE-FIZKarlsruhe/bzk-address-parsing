"""Extensible evaluation metrics for entity linking predictions."""

from typing import Callable, Optional, Sequence

import pandas as pd
from collections import defaultdict

ConfusionMatrix = dict[str, int | float]
MetricFn = Callable[[ConfusionMatrix], float]

_METRIC_REGISTRY: dict[str, MetricFn] = {}


def _normalize_iri(iri: Optional[str]) -> Optional[str]:
    """Normalize an IRI to a canonical form (https scheme, no trailing slash).

    IRIs come from different sources that disagree on scheme and trailing
    slash (eg. predicted IRIs are "https://sws.geonames.org/759955" while
    full_hierarchy entries are "http://sws.geonames.org/759955/"), so they
    must be normalized before being compared.
    """
    if iri is None:
        return None
    if iri.startswith("http://"):
        iri = "https://" + iri[len("http://"):]
    return iri.rstrip("/")


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Register a function that derives a metric from the confusion matrix dict.

    Registered metrics are computed in registration order and added to the
    dict returned by `eval_entity_linking`, so a metric can depend on an
    already-registered one (eg. f1 depends on precision and recall).
    """

    def decorator(fn: MetricFn) -> MetricFn:
        _METRIC_REGISTRY[name] = fn
        return fn

    return decorator


def eval_entity_linking(
    predicted_iris: Sequence[Optional[str]], 
    ground_truth: pd.DataFrame, 
    iri_column: str = "iri",
    full_hierarchy_columns : str = "full_hierarchy"
) -> ConfusionMatrix:
    """Compute a confusion matrix and derived metrics between predicted and true IRIs.

    `ground_truth` is the full ground truth dataframe (eg. the parsed
    bzkopen_linking_groundtruth.jsonl), row-aligned with `predicted_iris`;
    `iri_column` names the column holding the true IRI. Passing the full
    dataframe (rather than just a list of true IRIs) lets metrics that are
    added later make use of its other columns (eg. tags, field).

    A true positive is a prediction whose IRI equals the ground truth IRI.
    A false positive is any other non-null prediction (wrong IRI, or a
    prediction where none was expected). A false negative is a missing
    prediction (None) where a ground truth IRI was expected. A true negative
    is a missing prediction where none was expected either.
    """
    if len(predicted_iris) != len(ground_truth):
        raise ValueError("predicted_iris and ground_truth must have the same length")

    true_iris = ground_truth[iri_column]
    full_hierarchies = ground_truth[full_hierarchy_columns]
    tp = fp = fn = tn = 0
    granularity_loss_counts = defaultdict(int)
    granularity_score = 0.0
    for pred, true, hierarchy in zip(predicted_iris, true_iris, full_hierarchies):
        pred = _normalize_iri(pred)
        true = None if pd.isna(true) else _normalize_iri(true)
        if pred is not None:
            if pred == true:
                tp += 1
            else:
                fp += 1
        else:
            if true is not None:
                fn += 1
            else:
                tn += 1
        if isinstance(hierarchy, list):
            normalized_hierarchy = [_normalize_iri(h) for h in hierarchy]
            try:
                granularity_loss = normalized_hierarchy.index(pred)
            except ValueError:
                granularity_loss = len(normalized_hierarchy) # pred not in hierarchy
            granularity_score += (len(normalized_hierarchy) - granularity_loss) / len(normalized_hierarchy)
            granularity_loss_counts[granularity_loss] += 1



    metrics: ConfusionMatrix = dict(
        tp=tp, 
        fp=fp, 
        fn=fn, 
        tn=tn, 
        total=len(predicted_iris)
    )
    total_gt_with_values = tp + fn
    for name, metric_fn in _METRIC_REGISTRY.items():
        metrics[name] = metric_fn(metrics)
    for loss, count in granularity_loss_counts.items():
        metrics[f"granularity_loss_of_{loss}"] = count
    metrics["granularity_score"] = granularity_score / total_gt_with_values if total_gt_with_values > 0 else float("nan")
    return metrics


@register_metric("precision")
def _precision(m: ConfusionMatrix) -> float:
    denom = m["tp"] + m["fp"]
    return m["tp"] / denom if denom > 0 else float("nan")


@register_metric("recall")
def _recall(m: ConfusionMatrix) -> float:
    denom = m["tp"] + m["fn"]
    return m["tp"] / denom if denom > 0 else float("nan")


@register_metric("f1")
def _f1(m: ConfusionMatrix) -> float:
    precision, recall = m["precision"], m["recall"]
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else float("nan")


@register_metric("accuracy")
def _accuracy(m: ConfusionMatrix) -> float:
    return (m["tp"] + m["tn"]) / m["total"] if m["total"] > 0 else float("nan")
