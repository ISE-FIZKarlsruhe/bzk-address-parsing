"""Extensible evaluation metrics for entity linking predictions."""

from typing import Callable, Optional, Sequence

import pandas as pd
from collections import defaultdict

ConfusionMatrix = dict[str, int | float]
MetricFn = Callable[[ConfusionMatrix], float]

_METRIC_REGISTRY: dict[str, MetricFn] = {}


def normalize_iri(iri: Optional[str]) -> Optional[str]:
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
    predicted_entity_types: Sequence[Optional[str]],
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
    tp = fp = fn = tn = 0
    granularity_loss_counts = defaultdict(int)
    partial_link_counts = defaultdict(int)
    granularity_score = 0.0
    full_granularity_loss = 0
    some_granularity_loss = 0
    for pred, pred_entity_type, (_, true_row) in zip(predicted_iris, predicted_entity_types, ground_truth.iterrows()):
        pred = normalize_iri(pred)
        true = true_row[iri_column]
        hierarchy = true_row[full_hierarchy_columns]
        true = None if pd.isna(true) else normalize_iri(true)
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
            normalized_hierarchy = [normalize_iri(h) for h in hierarchy]
            partial_link = False
            try:
                granularity_loss = normalized_hierarchy.index(pred)
                if granularity_loss > 0:
                    partial_link = True
                    some_granularity_loss += 1
            except ValueError:
                granularity_loss = len(normalized_hierarchy) # pred not in hierarchy
                full_granularity_loss += 1
            granularity_score += (len(normalized_hierarchy) - granularity_loss) / len(normalized_hierarchy)
            granularity_loss_counts[granularity_loss] += 1
            if partial_link and pred_entity_type is not None:
                partial_link_counts[pred_entity_type] += 1




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
    metrics["full_granularity_loss"] = full_granularity_loss
    metrics["some_granularity_loss"] = some_granularity_loss
    for entity_type, count in partial_link_counts.items():
        metrics[f"partial_link_to_{entity_type}"] = count
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
