"""
Support code for disambiguation_optimization.ipynb.

Like entity_linking_notebook_utils for entity_linking.ipynb, the notebook is
kept to function calls and variable definitions (experiment parameters), and
everything else lives here. This is a module of its own, rather than part of
entity_linking_notebook_utils, since the worker processes evaluating
configurations import it: it only depends on the pipeline modules, not on
the LLM parsing stack.

Every configuration of the Disambiguator is evaluated on the same addresses,
whose GeoDB search does not depend on it. So the training addresses are
linked once up to (and including) the search (see load_or_search_fields),
and each configuration then only replays the disambiguation of the searched
addresses (see replay_field), in parallel worker processes (see
ConfigEvaluator).
"""
import dataclasses
import json
import logging
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

import pandas as pd
from IPython.display import Markdown, display
from tqdm.auto import tqdm

import modules.entity_linking as entity_linking
from modules.entity_linking import LinkingOutcome
from modules.entity_linking_eval_metrics import eval_entity_linking
from modules.geo_disambiguation import Disambiguator
from modules.pipeline.linked_data import AddressProcessingData
from modules.utils import format_time

logger = logging.getLogger("disambiguation_optimization")

# card_id prefix of the addresses of the training split (e.g. "train_0")
TRAIN_CARD_ID_PREFIX = "train_"

# Columns of an evaluation, in the order of the objective (see rank): higher
# precision, then more correctly linked addresses, then fewer incorrectly
# linked ones
OBJECTIVE_COLUMNS = ["precision", "correctly_linked", "incorrectly_linked"]
OBJECTIVE_ASCENDING = [False, False, True]
METRIC_COLUMNS = [*OBJECTIVE_COLUMNS, "partially_linked", "failed_to_link", "f1", "recall"]


# ---------------------------------------------------------------------------
# Training split
# ---------------------------------------------------------------------------

def train_split(parsed_addresses: pd.DataFrame, ground_truth: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """
    The parsed rows and the ground truth of the training addresses, row
    aligned (as are `parsed_addresses` and `ground_truth`).
    """
    if list(parsed_addresses["address_id"].astype(str)) != list(ground_truth["address_id"].astype(str)):
        raise ValueError("parsed_addresses and ground_truth are not row aligned")
    is_train = ground_truth["card_id"].astype(str).str.startswith(TRAIN_CARD_ID_PREFIX).to_numpy()
    rows = parsed_addresses[is_train].to_dict(orient="records")
    return rows, ground_truth[is_train].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Search once, replay disambiguation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchedField:
    """An address field linked by the pipeline, keeping the address its search returned."""
    row: dict
    # The outcome with the production disambiguator; final for a field whose
    # linking stopped before the search (camp, pre-linked, not a location...)
    outcome: LinkingOutcome
    # The address as GeoDBSearch returned it, None if linking stopped before
    searched_address: Optional[AddressProcessingData]


class _RecordingSearcher:
    """A GeoDBSearch keeping the address it returns."""
    def __init__(self, geo_db_searcher):
        self.geo_db_searcher = geo_db_searcher
        self.searched_address = None

    def apply(self, address: AddressProcessingData) -> AddressProcessingData:
        self.searched_address = self.geo_db_searcher.apply(address)
        return self.searched_address


class _ReplayingSearcher:
    """Stands for GeoDBSearch, returning the address it returned before."""
    def __init__(self, searched_address: AddressProcessingData):
        self.searched_address = searched_address

    def apply(self, address: AddressProcessingData) -> AddressProcessingData:
        if address.id != self.searched_address.id:
            raise ValueError(f"Replaying the search of address {self.searched_address.id} for address {address.id}")
        return self.searched_address


class _NoCampMatcher:
    """
    Stands for the CampReferenceMatcher when replaying a field: only fields
    that reached the search are replayed, i.e. whose camp match was None.
    """
    def match(self, address, tags):
        return None


def _search_field(row: dict, camp_matcher, geo_db_searcher, disambiguator) -> SearchedField:
    """Run in the LinkingPool workers (see LinkingPool.map)."""
    searcher = _RecordingSearcher(geo_db_searcher)
    outcome = entity_linking.link_field(row, row["prefix"], camp_matcher, searcher, disambiguator)
    return SearchedField(row, outcome, searcher.searched_address)


def search_fields(rows: list[dict], search_index_path: str, num_workers: int) -> list[SearchedField]:
    """Links every row with the production pipeline, keeping what its search returned (see SearchedField)."""
    geo_db_searcher = entity_linking.build_geo_db_searcher(search_index_path)
    camp_matcher = entity_linking.build_camp_matcher()
    disambiguator = entity_linking.build_disambiguator()
    try:
        with entity_linking.LinkingPool(camp_matcher, geo_db_searcher, disambiguator, num_workers) as pool:
            return list(tqdm(pool.map(_search_field, rows), total=len(rows), desc="Searching"))
    finally:
        geo_db_searcher.connection.close()
        disambiguator.close()


def load_or_search_fields(
        rows: list[dict], cache_path: Path, search_index_path: str, num_workers: int, overwrite: bool = False
    ) -> list[SearchedField]:
    """search_fields, or its result read back from `cache_path` (a pickle) if already cached."""
    cache_path = Path(cache_path)
    if overwrite or not cache_path.exists():
        start = time.monotonic()
        searched = search_fields(rows, search_index_path, num_workers)
        print(f"Searching {len(rows)} addresses took {format_time(time.monotonic() - start)}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(searched, f)
    else:
        print(f"Retrieving cached searched addresses from {cache_path}...")
        with open(cache_path, "rb") as f:
            searched = pickle.load(f)
    if [str(s.row["address_id"]) for s in searched] != [str(row["address_id"]) for row in rows]:
        raise ValueError(f"The searched addresses cached at {cache_path} are not those of `rows`; rerun with overwrite=True")
    return searched


def replay_field(searched: SearchedField, disambiguator: Disambiguator) -> LinkingOutcome:
    """The outcome of linking a field with `disambiguator`, replaying its search."""
    if searched.searched_address is None:
        return searched.outcome
    return entity_linking.link_field(
        searched.row, searched.row["prefix"], _NoCampMatcher(),
        _ReplayingSearcher(searched.searched_address), disambiguator)


# ---------------------------------------------------------------------------
# Disambiguator configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DisambiguationConfig:
    """The parameters of a Disambiguator (see Disambiguator.__init__)."""
    score_weights: dict[str, float]
    parent_child_max_distance_km_by_admin_level: dict[int, float]
    min_population_order_of_magnitude: int
    primary_priority: tuple[str, ...]
    secondary_priority: tuple[str, ...]
    # for every factor of the priorities
    significance_thresholds: dict[str, float]
    population_rounding_factor: int
    score_prune_thresholds: dict[str, float]
    geometry_parent_child_max_distance_km: float
    geo_db_path: str

    @classmethod
    def of(cls, disambiguator: Disambiguator) -> "DisambiguationConfig":
        """The configuration of a Disambiguator, e.g. the production one (see entity_linking.build_disambiguator)."""
        return cls(
            score_weights=dict(disambiguator.score_weights),
            parent_child_max_distance_km_by_admin_level=dict(disambiguator.parent_child_max_distance_km_by_admin_level),
            min_population_order_of_magnitude=disambiguator.min_population_order_of_magnitude,
            primary_priority=tuple(disambiguator.primary_priority),
            secondary_priority=tuple(disambiguator.secondary_priority),
            significance_thresholds={
                factor: disambiguator.significance_thresholds[factor] for factor in disambiguator.priority},
            population_rounding_factor=disambiguator.population_rounding_factor,
            score_prune_thresholds=dict(disambiguator.score_prune_thresholds),
            geometry_parent_child_max_distance_km=disambiguator.geometry_parent_child_max_distance_km,
            geo_db_path=disambiguator.geo_db_path,
        )

    def replace(self, **changes) -> "DisambiguationConfig":
        return dataclasses.replace(self, **changes)

    def with_order(self, order: list[tuple[str, float]]) -> "DisambiguationConfig":
        """
        This configuration comparing candidates on the factors of `order` only,
        each with its significance threshold, as primary factors (so then
        again with no threshold, see geo_disambiguation.comparison_steps).
        """
        return self.replace(
            primary_priority=tuple(factor for factor, _ in order),
            secondary_priority=(),
            significance_thresholds=dict(order),
        )

    def key(self) -> str:
        """Identifies the configuration, whatever the order its dicts were built in."""
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    def build(self) -> Disambiguator:
        # quietly: every configuration evaluated would otherwise log its parameters
        disambiguator_logger = Disambiguator.logger
        level = disambiguator_logger.level
        disambiguator_logger.setLevel(logging.WARNING)
        try:
            return Disambiguator(
                significance_thresholds=dict(self.significance_thresholds),
                primary_priority=list(self.primary_priority),
                secondary_priority=list(self.secondary_priority),
                score_weights=dict(self.score_weights),
                population_rounding_factor=self.population_rounding_factor,
                min_population_order_of_magnitude=self.min_population_order_of_magnitude,
                score_prune_thresholds=dict(self.score_prune_thresholds),
                geo_db_path=self.geo_db_path,
                geometry_parent_child_max_distance_km=self.geometry_parent_child_max_distance_km,
                parent_child_max_distance_km_by_admin_level=dict(self.parent_child_max_distance_km_by_admin_level),
            )
        finally:
            disambiguator_logger.setLevel(level)


def describe_config(config: DisambiguationConfig) -> str:
    """The parameters of a configuration, as the constants of the pipeline modules they correspond to."""
    def priority(factors):
        return "".join(f"\n    {factor!r}," for factor in factors)
    thresholds = "".join(
        f"\n    {factor!r}: {threshold!r}," for factor, threshold in config.significance_thresholds.items())
    weights = "".join(f"\n    {factor!r}: {weight!r}," for factor, weight in config.score_weights.items())
    distances = "".join(
        f"\n    {level}: {km!r}," for level, km in config.parent_child_max_distance_km_by_admin_level.items())
    return (
        "# modules/geo_disambiguation.py\n"
        f"DISAMBIGUATION_FACTOR_PRIMARY_PRIORITY = [{priority(config.primary_priority)}\n]\n"
        f"DISAMBIGUATION_FACTOR_SECONDARY_PRIORITY = [{priority(config.secondary_priority)}\n]\n"
        f"WEIGHTED_DISAMBIGUATION_FACTORS = {{{weights}\n}}\n"
        f"PARENT_CHILD_MAX_DISTANCE_KM_BY_ADMIN_LEVEL = {{{distances}\n}}\n"
        f"# Disambiguator(min_population_order_of_magnitude=...)\n"
        f"min_population_order_of_magnitude = {config.min_population_order_of_magnitude:_}\n"
        "# modules/entity_linking.py\n"
        f"SIGNIFICANCE_THRESHOLDS = {{{thresholds}\n}}"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class LinkResult(NamedTuple):
    """What evaluating a configuration needs of a LinkingOutcome."""
    iri: Optional[str]
    entity_type: Optional[str]
    deciding_factor: Optional[str]

    @classmethod
    def of(cls, outcome: LinkingOutcome) -> "LinkResult":
        return cls(outcome.iri, outcome.entity_type, outcome.deciding_factor)


def link_metrics(results: list[LinkResult], ground_truth: pd.DataFrame) -> dict:
    """
    The objective (see OBJECTIVE_COLUMNS) and other metrics of linking results
    row aligned with `ground_truth`. Correctly linked addresses are the true
    positives; incorrectly and partially linked ones (an ancestor of the true
    location, see entity_linking_notebook_utils.link_status) make up the
    false positives; addresses that failed to link are the false negatives.
    """
    metrics = eval_entity_linking([r.iri for r in results], [r.entity_type for r in results], ground_truth)
    return dict(
        f1=metrics["f1"],
        correctly_linked=metrics["tp"],
        incorrectly_linked=metrics["fp"] - metrics["some_granularity_loss"],
        partially_linked=metrics["some_granularity_loss"],
        failed_to_link=metrics["fn"],
        precision=metrics["precision"],
        recall=metrics["recall"],
    )


def rank(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Evaluations best first by the objective (see OBJECTIVE_COLUMNS), keeping their order among ties."""
    return evaluations.sort_values(OBJECTIVE_COLUMNS, ascending=OBJECTIVE_ASCENDING, kind="stable")


# The searched addresses of a ConfigEvaluator worker process
_worker_searched_fields: Optional[list[SearchedField]] = None


def _initialize_evaluation_worker(searched_fields_path: Path) -> None:
    global _worker_searched_fields
    with open(searched_fields_path, "rb") as f:
        _worker_searched_fields = pickle.load(f)


def _replay_chunk(disambiguator: Disambiguator, indices: range) -> list[LinkResult]:
    """
    Run in the ConfigEvaluator workers, each task with its own copy of the
    disambiguator (single threaded, see Disambiguator.__setstate__).
    """
    try:
        return [LinkResult.of(replay_field(_worker_searched_fields[i], disambiguator)) for i in indices]
    finally:
        disambiguator.close()


class ConfigEvaluator:
    """
    Evaluates Disambiguator configurations on the searched addresses cached at
    `searched_fields_path` (see load_or_search_fields) against `ground_truth`,
    row aligned with them. Configurations are evaluated in parallel worker
    processes, each holding the searched addresses, by chunks of
    `chunk_size` addresses; the evaluation of a configuration is remembered,
    so evaluating it again costs nothing.
    """
    def __init__(
            self, searched_fields_path: Path, ground_truth: pd.DataFrame, num_workers: int, chunk_size: int = 50
        ):
        self.ground_truth = ground_truth
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self._evaluations : dict[str, dict] = {}
        self._link_results : dict[str, list[LinkResult]] = {}
        # spawned rather than forked, as for LinkingPool
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_evaluation_worker, initargs=(Path(searched_fields_path),))

    def link_results(self, configs: dict[str, DisambiguationConfig]) -> dict[str, list[LinkResult]]:
        """The LinkResult of every address with each configuration, by name."""
        chunks = [
            range(start, min(start + self.chunk_size, len(self.ground_truth)))
            for start in range(0, len(self.ground_truth), self.chunk_size)
        ]
        pending = {}
        for name, config in configs.items():
            key = config.key()
            if key not in self._link_results and key not in pending:
                disambiguator = config.build()
                pending[key] = [self._executor.submit(_replay_chunk, disambiguator, chunk) for chunk in chunks]
        if pending:
            with tqdm(total=len(pending) * len(chunks), desc=f"Evaluating {len(pending)} configurations") as progress:
                for key, futures in pending.items():
                    results = []
                    for future in futures:
                        results.extend(future.result())
                        progress.update()
                    self._link_results[key] = results
        return {name: self._link_results[config.key()] for name, config in configs.items()}

    def evaluate(self, configs: dict[str, DisambiguationConfig]) -> pd.DataFrame:
        """The metrics (see link_metrics) of each configuration, indexed by name, in the order given."""
        link_results = self.link_results(configs)
        rows = {}
        for name, config in configs.items():
            key = config.key()
            if key not in self._evaluations:
                self._evaluations[key] = link_metrics(link_results[name], self.ground_truth)
            rows[name] = self._evaluations[key]
        return pd.DataFrame.from_dict(rows, orient="index", columns=METRIC_COLUMNS)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "ConfigEvaluator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def check_replay(evaluator: ConfigEvaluator, config: DisambiguationConfig, searched: list[SearchedField]) -> None:
    """
    Checks that replaying the disambiguation with `config` (the production
    one) links every address as the pipeline did when searching.
    """
    replayed = evaluator.link_results({"replayed": config})["replayed"]
    mismatches = [
        (s.row["address_id"], LinkResult.of(s.outcome), r)
        for s, r in zip(searched, replayed) if LinkResult.of(s.outcome) != r
    ]
    if mismatches:
        raise AssertionError(f"{len(mismatches)} addresses linked differently when replayed, e.g. {mismatches[:3]}")
    print(f"Replaying the disambiguation links all {len(searched)} addresses as the pipeline did")


def cleanup_previous_run(namespace: dict) -> None:
    """Releases the ConfigEvaluator a previous run of the notebook left in `namespace` (its globals())."""
    if "evaluator" in namespace:
        namespace["evaluator"].close()


def deciding_factor_counts(evaluator: ConfigEvaluator, configs: dict[str, DisambiguationConfig]) -> pd.DataFrame:
    """
    How many linked addresses each disambiguation factor decided (see
    LinkingOutcome.deciding_factor), with each configuration.
    """
    return pd.DataFrame({
        name: pd.Series([r.deciding_factor for r in results if r.iri is not None]).value_counts()
        for name, results in evaluator.link_results(configs).items()
    }).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageResult:
    """The configurations a stage of the optimization evaluated, and the best of them."""
    # each configuration's parameters and metrics, best first (see rank)
    evaluations: pd.DataFrame
    best_name: str
    best_config: DisambiguationConfig


def _stage_result(
        evaluator: ConfigEvaluator, configs: dict[str, DisambiguationConfig], parameters: pd.DataFrame
    ) -> StageResult:
    """Evaluates `configs`, their parameters described by the rows of `parameters`, of the same index."""
    evaluations = rank(parameters.join(evaluator.evaluate(configs)))
    best_name = evaluations.index[0]
    return StageResult(evaluations, best_name, configs[best_name])


def _weights_config(base: DisambiguationConfig, weights: dict[str, float]) -> tuple[str, DisambiguationConfig]:
    """A configuration with the given weights (a factor of weight 0 left out of the weighted score), and its name."""
    name = ", ".join(f"{factor}={weight:g}" for factor, weight in weights.items())
    return name, base.replace(score_weights={factor: weight for factor, weight in weights.items() if weight != 0})


def optimize_score_weights(evaluator: ConfigEvaluator, base: DisambiguationConfig) -> tuple[StageResult, StageResult]:
    """
    Optimizes the weights of the factors of the weighted score of `base`
    (see Disambiguator.weighted_score_contributions), in two rounds:

    1. every factor weighing 1, then each in turn weighing 2 instead (and
       the weights of `base`, for reference);
    2. the factor that did best weighing 2 in round 1 weighing 3, and the
       other factors weighing 2 except one, weighing 1 or left out (weight
       0) in turn; plus the other factors all weighing 2, and all weighing 1.

    Returns the result of each round; the best configuration of the second
    is the best of both (it includes the best of the first).
    """
    factors = list(base.score_weights)
    configs = {"current": base}
    parameters = {"current": dict(base.score_weights)}
    def add(weights):
        name, config = _weights_config(base, weights)
        configs[name] = config
        parameters[name] = weights
        return name

    all_ones = add({factor: 1 for factor in factors})
    doubled = {factor: add({other: 2 if other == factor else 1 for other in factors}) for factor in factors}
    first_round = _stage_result(evaluator, dict(configs), pd.DataFrame.from_dict(parameters, orient="index"))
    best_doubled = min(factors, key=lambda factor: first_round.evaluations.index.get_loc(doubled[factor]))

    second_names = [first_round.best_name]
    others = [factor for factor in factors if factor != best_doubled]
    second_names.append(add({factor: 3 if factor == best_doubled else 2 for factor in factors}))
    second_names.append(add({factor: 3 if factor == best_doubled else 1 for factor in factors}))
    for exception in others:
        for exception_weight in (1, 0):
            second_names.append(add({
                factor: 3 if factor == best_doubled else exception_weight if factor == exception else 2
                for factor in factors
            }))
    second_names.append(all_ones)
    second_names = list(dict.fromkeys(second_names))
    second_round = _stage_result(
        evaluator, {name: configs[name] for name in second_names},
        pd.DataFrame.from_dict({name: parameters[name] for name in second_names}, orient="index"))
    print(f"Best factor weighing 2 in round 1: {best_doubled}")
    return first_round, second_round


def scaled_distances(distances: dict[int, float], first_distance: float) -> dict[int, float]:
    """Distances by admin level scaled for the first (admin level 0) to be `first_distance`, keeping their cascade."""
    scale = first_distance / distances[min(distances)]
    return {level: max(1, round(km * scale)) for level, km in distances.items()}


def optimize_parent_child_distances(
        evaluator: ConfigEvaluator, base: DisambiguationConfig, first_distances: Iterable[float]
    ) -> StageResult:
    """
    Optimizes the parent/child max distances by admin level (see
    Disambiguator._score_parent_child_by_distance), scaling the cascade of
    `base` for its first distance to be each of `first_distances`.
    """
    configs = {}
    parameters = {}
    for first_distance in first_distances:
        distances = scaled_distances(base.parent_child_max_distance_km_by_admin_level, first_distance)
        name = " / ".join(f"{km:g}" for km in distances.values())
        configs[name] = base.replace(parent_child_max_distance_km_by_admin_level=distances)
        parameters[name] = {f"admin level {level} (km)": km for level, km in distances.items()}
    return _stage_result(evaluator, configs, pd.DataFrame.from_dict(parameters, orient="index"))


def optimize_min_population_order_of_magnitude(
        evaluator: ConfigEvaluator, base: DisambiguationConfig, min_populations: Iterable[int]
    ) -> StageResult:
    """
    Optimizes the population under which population_order_of_magnitude is 0
    (see Disambiguator._score_individual_match).
    """
    configs = {}
    parameters = {}
    for min_population in min_populations:
        name = f"{min_population:_}"
        configs[name] = base.replace(min_population_order_of_magnitude=min_population)
        parameters[name] = {"min_population_order_of_magnitude": min_population}
    return _stage_result(evaluator, configs, pd.DataFrame.from_dict(parameters, orient="index"))


@dataclass(frozen=True)
class GreedyOrderSearch:
    """The result of greedy_order_search."""
    # what each step chose, and the metrics of the order it led to
    summary: pd.DataFrame
    # every order each step evaluated, best first
    steps: list[pd.DataFrame]
    # the factors in order, each with its significance threshold
    order: list[tuple[str, float]]
    # the configuration of the best order of any step (see greedy_order_search)
    best_config: DisambiguationConfig


def _order_name(order: list[tuple[str, float]]) -> str:
    return " > ".join(f"{factor} ({threshold:g})" for factor, threshold in order)


def _greedy_step(
        evaluator: ConfigEvaluator, base: DisambiguationConfig, candidates: dict[str, tuple[dict, list[tuple[str, float]]]]
    ) -> tuple[StageResult, list[tuple[str, float]]]:
    """Evaluates candidate orders, each named and with the parameters describing it; the result and the best order."""
    configs = {name: base.with_order(order) for name, (_, order) in candidates.items()}
    parameters = pd.DataFrame.from_dict({name: params for name, (params, _) in candidates.items()}, orient="index")
    result = _stage_result(evaluator, configs, parameters)
    best_order = candidates[result.best_name][1]
    print(f"Step: {result.best_name}")
    return result, best_order


def greedy_order_search(
        evaluator: ConfigEvaluator, base: DisambiguationConfig, threshold_candidates: dict[str, Iterable[float]]
    ) -> GreedyOrderSearch:
    """
    Orders the factors of `threshold_candidates` (see
    DisambiguationConfig.with_order) by a greedy search:

    1. which factor alone decides best;
    2. then, the first being set, which factor decides best after it, and
       with which of its threshold candidates for the first;
    3. and so on, each step adding a factor at the end and setting the
       threshold of the one before, until every factor is ordered;
    4. finally, which threshold candidate is best for the last factor.

    The factor just added is given a threshold of 0 until the next step sets
    it: candidates it does not tell apart otherwise only go on to the
    comparison with no threshold. Among orders tied on the objective, a step
    keeps the first, factors and thresholds being tried in the order of
    `threshold_candidates`.

    The best configuration is that of the best order of any step, preferring
    the longest among ties: leaving the last factors out leaves candidates
    they would tell apart tied (linked to their common parent, or left
    unlinked).
    """
    factors = list(threshold_candidates)
    order : list[tuple[str, float]] = []
    steps = []
    summary = []
    step_orders = []

    def record(result, added_factor, previous_factor):
        best = result.evaluations.iloc[0]
        steps.append(result.evaluations)
        step_orders.append(order)
        summary.append({
            "step": len(steps),
            "added factor": added_factor,
            "threshold set on": previous_factor,
            "threshold": best["threshold"],
            "configurations evaluated": len(result.evaluations),
            **{column: best[column] for column in METRIC_COLUMNS},
        })

    for _ in factors:
        previous = order[-1][0] if order else None
        candidates = {}
        for factor in factors:
            if any(factor == f for f, _ in order):
                continue
            for threshold in (threshold_candidates[previous] if previous else [None]):
                candidate = [*order[:-1], (previous, threshold)] if previous else []
                candidate.append((factor, 0.0))
                candidates[_order_name(candidate)] = ({"added factor": factor, "threshold": threshold}, candidate)
        result, order = _greedy_step(evaluator, base, candidates)
        record(result, result.evaluations.iloc[0]["added factor"], previous)

    last = order[-1][0]
    candidates = {}
    for threshold in threshold_candidates[last]:
        candidate = [*order[:-1], (last, threshold)]
        candidates[_order_name(candidate)] = ({"added factor": None, "threshold": threshold}, candidate)
    result, order = _greedy_step(evaluator, base, candidates)
    record(result, None, last)

    summary = pd.DataFrame(summary).set_index("step")
    best_step = rank(summary.iloc[::-1]).index[0]
    best_order = step_orders[best_step - 1]
    print(f"Best order (step {best_step}): {_order_name(best_order)}")
    return GreedyOrderSearch(summary, steps, order, base.with_order(best_order))


@dataclass(frozen=True)
class SequentialSearch:
    """The result of a search changing one factor at a time (see optimize_significance_thresholds and move_factors_up)."""
    # what was chosen for each factor, and the metrics it led to
    summary: pd.DataFrame
    # the configurations evaluated for each factor
    steps: list[pd.DataFrame]
    best_config: DisambiguationConfig


def _priority(config: DisambiguationConfig) -> list[str]:
    return [*config.primary_priority, *config.secondary_priority]


def _objective_key(evaluation: pd.Series) -> tuple:
    """Orders evaluations by the objective (see OBJECTIVE_COLUMNS), the greater the better."""
    return tuple(
        -evaluation[column] if ascending else evaluation[column]
        for column, ascending in zip(OBJECTIVE_COLUMNS, OBJECTIVE_ASCENDING))


def optimize_significance_thresholds(
        evaluator: ConfigEvaluator, base: DisambiguationConfig, threshold_candidates: dict[str, Iterable[float]]
    ) -> SequentialSearch:
    """
    Optimizes the significance thresholds of the factors of `base`, keeping
    its order: greedily, from the first factor to the last, each factor
    tries its threshold candidates (and its current threshold, kept among
    ties) with the thresholds chosen for the factors before it.
    """
    config = base
    steps = []
    summary = []
    for factor in _priority(base):
        current = config.significance_thresholds[factor]
        configs = {}
        parameters = {}
        for threshold in dict.fromkeys([current, *threshold_candidates[factor]]):
            name = f"{factor} = {threshold:g}"
            configs[name] = config.replace(significance_thresholds={**config.significance_thresholds, factor: threshold})
            parameters[name] = {"threshold": threshold, "current": threshold == current}
        result = _stage_result(evaluator, configs, pd.DataFrame.from_dict(parameters, orient="index"))
        config = result.best_config
        steps.append(result.evaluations)
        best = result.evaluations.iloc[0]
        summary.append({
            "factor": factor, "previous threshold": current, "threshold": best["threshold"],
            **{column: best[column] for column in METRIC_COLUMNS},
        })
        print(f"{factor}: {current:g} -> {best['threshold']:g}")
    return SequentialSearch(pd.DataFrame(summary).set_index("factor"), steps, config)


def move_factors_up(evaluator: ConfigEvaluator, base: DisambiguationConfig) -> SequentialSearch:
    """
    Mutates the order of the factors of `base` (primary then secondary,
    see DisambiguationConfig): from the last factor to the first, each is
    moved up one position at a time until the objective gets worse than at
    the position before (ties go on moving up), and left at the last
    position that was not worse. The boundary between primary and secondary
    factors stays at the same position: a secondary factor moved above it
    becomes primary, pushing the last primary factor down to secondary.

    The positions above a factor are all evaluated at once, in parallel,
    though only those up to the first worse one count.
    """
    n_primary = len(base.primary_priority)
    def config_of(order):
        return base.replace(primary_priority=tuple(order[:n_primary]), secondary_priority=tuple(order[n_primary:]))

    order = _priority(base)
    current = evaluator.evaluate({"current": base}).iloc[0]
    steps = []
    summary = []
    for factor in reversed(_priority(base)):
        start = order.index(factor)
        orders = {}
        for position in range(start - 1, -1, -1):
            moved = [f for f in order if f != factor]
            moved.insert(position, factor)
            orders[f"{factor} at position {position + 1}"] = moved
        chosen = None
        if orders:
            evaluations = evaluator.evaluate({name: config_of(o) for name, o in orders.items()})
            evaluations.insert(0, "position", [o.index(factor) + 1 for o in orders.values()])
            evaluations.insert(1, "worse than the position below", False)
            previous = current
            for name, evaluation in evaluations.iterrows():
                if _objective_key(evaluation) < _objective_key(previous):
                    evaluations.loc[name, "worse than the position below"] = True
                    break
                chosen, previous = name, evaluation
            steps.append(evaluations)
        else:
            steps.append(pd.DataFrame())
        if chosen is not None:
            order = orders[chosen]
            current = previous
        summary.append({
            "factor": factor, "from position": start + 1, "to position": order.index(factor) + 1,
            **{column: current[column] for column in METRIC_COLUMNS},
        })
        print(f"{factor}: position {start + 1} -> {order.index(factor) + 1}")
    return SequentialSearch(pd.DataFrame(summary).set_index("factor"), steps, config_of(order))


def display_sequential_steps(search: SequentialSearch) -> None:
    """The configurations a sequential search evaluated for each factor."""
    for factor, evaluations in zip(search.summary.index, search.steps):
        display(Markdown(f"**{factor}**"))
        display(evaluations if len(evaluations) else Markdown("_already first_"))


def compare_stages(evaluator: ConfigEvaluator, configs: dict[str, DisambiguationConfig]) -> pd.DataFrame:
    """The metrics of the configurations each stage led to, in order, with their change from the first."""
    evaluations = evaluator.evaluate(configs)
    first = evaluations.iloc[0]
    for column in OBJECTIVE_COLUMNS:
        evaluations[f"Δ {column}"] = evaluations[column] - first[column]
    return evaluations


def display_greedy_steps(search: GreedyOrderSearch, top: int = 5) -> None:
    """The `top` orders of each step of a greedy order search."""
    for i, (evaluations, (_, row)) in enumerate(zip(search.steps, search.summary.iterrows()), start=1):
        if row["added factor"] is None:
            display(Markdown(f"**Step {i}**: threshold of the last factor, {row['threshold set on']}"))
        else:
            display(Markdown(
                f"**Step {i}**: factor added after {row['threshold set on']}, and the threshold of the latter"
                if row["threshold set on"] is not None else f"**Step {i}**: the single best factor"))
        display(evaluations.head(top))
