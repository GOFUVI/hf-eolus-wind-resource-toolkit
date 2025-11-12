"""Bootstrap-based uncertainty estimates for wind-resource metrics.

This module now mirrors the power-computation workflow used in
``scripts/generate_power_estimates.py`` by resampling the original ANN
inference records (wind speed, range probabilities, classifier flags) and by
invoking :mod:`hf_wind_resource.stats.power_pipeline` for every replica. In
doing so the bootstrap delivers confidence intervals that are faithful to the
regular resource estimation flow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import math

import numpy as np
from statistics import NormalDist

from .power import PowerCurve
from .power_pipeline import (
    HeightCorrection,
    compute_power_distribution,
    format_height_note,
    summarise_records_for_selection,
)
from .kaplan_meier import KaplanMeierSelectionCriteria
from .rmse import GlobalRmseProvider, GlobalRmseRecord
from .weibull import CensoredWeibullData, WeibullFitResult, build_censored_data_from_records
from ..preprocessing.censoring import RangeThresholds, load_range_thresholds

__all__ = [
    "BootstrapConfidenceInterval",
    "BootstrapMetricName",
    "BootstrapPowerDiagnostics",
    "BootstrapUncertaintyResult",
    "NodeBootstrapInput",
    "StratifiedBootstrapConfig",
    "compute_stratified_bootstrap_uncertainty",
]


BootstrapMetricName = str


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """Confidence interval produced by bootstrap resampling."""

    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    replicates: int
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BootstrapPowerDiagnostics:
    """Diagnostics of the power-distribution step used in the bootstrap."""

    method: str
    selection_reasons: Tuple[str, ...]
    method_notes: Tuple[str, ...]
    weibull: WeibullFitResult | None
    replicate_method_counts: Mapping[str, int]


@dataclass(frozen=True)
class NodeBootstrapInput:
    """Raw records and metadata required to run the bootstrap for a node."""

    node_id: str
    records: Sequence[Mapping[str, object]]
    height: HeightCorrection

    def __post_init__(self) -> None:
        if not isinstance(self.records, Sequence):
            raise TypeError("records must be a sequence of mappings")


@dataclass(frozen=True)
class StratifiedBootstrapConfig:
    """Configuration controlling the bootstrap procedure."""

    replicas: int = 500
    confidence_level: float = 0.95
    random_seed: int | None = None
    apply_rmse_noise: bool = True
    rmse_mode: str = "velocity"
    ci_method: str = "percentile"
    jackknife_max_samples: int = 200
    label_strategy: str = "fixed"
    resampling_mode: str = "iid"
    block_length: int = 1
    node_block_lengths: Mapping[str, int] | None = None
    air_density: float = 1.225
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    min_confidence: float = 0.5
    min_in_range_weight: float = 500.0
    tail_surrogate: float | None = None
    noise_truncation_multiplier: float = 4.0
    power_curve: PowerCurve | None = None
    km_criteria: KaplanMeierSelectionCriteria | None = None

    def __post_init__(self) -> None:
        if self.replicas < 0:
            raise ValueError("replicas must be non-negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.air_density <= 0.0:
            raise ValueError("air_density must be positive")
        if self.min_in_range_weight < 0.0:
            raise ValueError("min_in_range_weight must be non-negative")
        if self.noise_truncation_multiplier <= 0.0:
            raise ValueError("noise_truncation_multiplier must be positive")
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise ValueError("min_confidence must lie within [0, 1]")
        if self.power_curve is None:
            raise ValueError("power_curve must be provided in StratifiedBootstrapConfig")
        if self.km_criteria is None:
            raise ValueError("km_criteria must be provided in StratifiedBootstrapConfig")
        allowed_modes = {"velocity", "power", "none"}
        if self.rmse_mode not in allowed_modes:
            raise ValueError(f"rmse_mode must be one of {sorted(allowed_modes)}")
        if self.rmse_mode == "none" and self.apply_rmse_noise:
            raise ValueError("apply_rmse_noise cannot be True when rmse_mode='none'")
        if self.rmse_mode != "velocity" and self.noise_truncation_multiplier <= 0.0:
            object.__setattr__(self, "noise_truncation_multiplier", 4.0)
        allowed_ci = {"percentile", "bca", "percentile_t"}
        if self.ci_method not in allowed_ci:
            raise ValueError(f"ci_method must be one of {sorted(allowed_ci)}")
        if self.jackknife_max_samples <= 0:
            raise ValueError("jackknife_max_samples must be positive")
        allowed_label_strategies = {"fixed", "label_resample"}
        if self.label_strategy not in allowed_label_strategies:
            raise ValueError(f"label_strategy must be one of {sorted(allowed_label_strategies)}")
        allowed_resampling_modes = {"iid", "moving_block", "stationary"}
        if self.resampling_mode not in allowed_resampling_modes:
            raise ValueError(f"resampling_mode must be one of {sorted(allowed_resampling_modes)}")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        if self.node_block_lengths is not None:
            for node, value in self.node_block_lengths.items():
                if value <= 0:
                    raise ValueError(f"Block length for node '{node}' must be positive")


@dataclass(frozen=True)
class BootstrapUncertaintyResult:
    """Output bundle with metric intervals and supporting metadata."""

    node_id: str
    metrics: Mapping[BootstrapMetricName, BootstrapConfidenceInterval]
    bootstrap_means: Mapping[BootstrapMetricName, float | None]
    rmse_record: GlobalRmseRecord
    label_counts: Mapping[str, float]
    label_proportions: Mapping[str, float]
    total_samples: int
    power_diagnostics: BootstrapPowerDiagnostics | None
    notes: Tuple[str, ...] = ()


def compute_stratified_bootstrap_uncertainty(
    data: NodeBootstrapInput,
    *,
    config: StratifiedBootstrapConfig,
    rmse_provider: GlobalRmseProvider | None = None,
) -> BootstrapUncertaintyResult:
    """Compute bootstrap confidence intervals for wind metrics."""

    if rmse_provider is None:
        rmse_provider = GlobalRmseProvider()

    rmse_record = rmse_provider.get_global_rmse()
    thresholds = _resolve_thresholds(config)

    records = tuple(data.records)
    total_samples = len(records)

    ordered_records = _sort_records_by_timestamp(records)

    partitions = _partition_records(ordered_records, min_confidence=config.min_confidence)
    label_counts = {label: float(len(items)) for label, items in partitions.items()}
    label_proportions = _normalise_label_counts(label_counts, total_samples)

    notes: list[str] = []
    if total_samples == 0:
        notes.append("No samples available for this node; metrics cannot be computed.")
    if config.label_strategy == "label_resample":
        notes.append("Label resampling enabled to propagate range-class uncertainty.")
    if config.resampling_mode == "moving_block":
        block_length = _resolve_block_length(data.node_id, config)
        notes.append(f"Moving-block bootstrap enabled with block length {block_length}.")
    elif config.resampling_mode == "stationary":
        block_length = _resolve_block_length(data.node_id, config)
        notes.append(
            "Stationary bootstrap enabled with expected block length "
            f"{block_length}."
        )

    velocity_noise = config.apply_rmse_noise and config.rmse_mode == "velocity"
    power_noise = config.apply_rmse_noise and config.rmse_mode == "power"
    if config.apply_rmse_noise:
        if config.rmse_mode == "velocity":
            notes.append("RMSE perturbations applied to velocities before resampling.")
        elif config.rmse_mode == "power":
            notes.append("RMSE perturbations injected on the power density metric.")

    baseline_metrics: Dict[BootstrapMetricName, float | None]
    baseline_notes: Tuple[str, ...]
    baseline_weibull: WeibullFitResult | None
    method_used: str
    selection_reasons: Tuple[str, ...]

    if records:
        (
            baseline_metrics,
            method_used,
            selection_reasons,
            method_notes,
            power_density_notes,
            power_curve_notes,
            baseline_weibull,
            baseline_power_sigma,
        ) = _evaluate_dataset(
            ordered_records,
            height=data.height,
            config=config,
            thresholds=thresholds,
            rmse_value=rmse_record.value,
            rng=None,
            apply_noise=False,
        )
        baseline_notes = tuple(filter(None, (
            format_height_note(data.height),
            *method_notes,
            *power_density_notes,
            *power_curve_notes,
        )))
        notes.extend(baseline_notes)
    else:
        baseline_metrics = _empty_metric_set()
        baseline_notes = ()
        baseline_weibull = None
        method_used = "none"
        selection_reasons = ()
        baseline_power_sigma = None

    if config.replicas <= 0 or total_samples == 0:
        metrics = _wrap_without_bootstrap(
            baseline_metrics,
            config,
            notes,
            replicates=config.replicas,
        )
        return BootstrapUncertaintyResult(
            node_id=data.node_id,
            metrics=metrics,
            bootstrap_means={name: baseline_metrics.get(name) for name in metrics.keys()},
            rmse_record=rmse_record,
            label_counts=label_counts,
            label_proportions=label_proportions,
            total_samples=total_samples,
            power_diagnostics=BootstrapPowerDiagnostics(
                method=method_used,
                selection_reasons=selection_reasons,
                method_notes=baseline_notes,
                weibull=baseline_weibull,
                replicate_method_counts={},
            ),
            notes=tuple(notes),
        )

    rng = np.random.default_rng(config.random_seed)

    velocity_noise = config.apply_rmse_noise and config.rmse_mode == "velocity"
    power_noise = config.apply_rmse_noise and config.rmse_mode == "power"

    replicates = {name: [] for name in baseline_metrics.keys()}
    method_counter: Counter[str] = Counter()
    power_clamp_count = 0

    for _ in range(config.replicas):
        if config.resampling_mode == "iid":
            sample_records = _draw_bootstrap_sample(
                ordered_records,
                partitions,
                rng,
                strategy=config.label_strategy,
                min_confidence=config.min_confidence,
            )
        else:
            block_length = _resolve_block_length(data.node_id, config)
            sample_records = _draw_block_sample(
                ordered_records,
                rng,
                block_length=block_length,
                mode=config.resampling_mode,
            )
            sample_records = _apply_label_strategy(
                sample_records,
                rng=rng,
                strategy=config.label_strategy,
                min_confidence=config.min_confidence,
            )
        (
            sample_metrics,
            sample_method,
            _,
            _,
            _,
            _,
            _,
            sample_power_sigma,
        ) = _evaluate_dataset(
            sample_records,
            height=data.height,
            config=config,
            thresholds=thresholds,
            rmse_value=rmse_record.value,
            rng=rng,
            apply_noise=velocity_noise,
        )
        method_counter[sample_method] += 1
        if power_noise:
            sigma = sample_power_sigma if sample_power_sigma is not None else baseline_power_sigma
            if sigma is not None and sample_metrics.get("power_density") is not None:
                noise_value = float(rng.normal(loc=0.0, scale=sigma))
                raw_power = float(sample_metrics["power_density"]) + noise_value
                if raw_power < 0.0:
                    power_clamp_count += 1
                sample_metrics["power_density"] = max(raw_power, 0.0)
        for name, value in sample_metrics.items():
            if value is None or not math.isfinite(value):
                continue
            replicates[name].append(value)

    raw_replicates = {name: list(values) for name, values in replicates.items()}

    metrics, bootstrap_means, interval_notes = _build_confidence_intervals(
        baseline=baseline_metrics,
        draws=raw_replicates,
        config=config,
        partitions=partitions,
        thresholds=thresholds,
        rmse_value=rmse_record.value,
        height=data.height,
        velocity_noise=velocity_noise,
        power_clamp_count=power_clamp_count,
    )
    notes.extend(interval_notes)

    power_diag = BootstrapPowerDiagnostics(
        method=method_used,
        selection_reasons=selection_reasons,
        method_notes=baseline_notes,
        weibull=baseline_weibull,
        replicate_method_counts=dict(method_counter),
    )

    return BootstrapUncertaintyResult(
        node_id=data.node_id,
        metrics=metrics,
        bootstrap_means=bootstrap_means,
        rmse_record=rmse_record,
        label_counts=label_counts,
        label_proportions=label_proportions,
        total_samples=total_samples,
        power_diagnostics=power_diag,
        notes=tuple(notes),
    )


def _resolve_thresholds(config: StratifiedBootstrapConfig) -> RangeThresholds:
    if config.lower_threshold is None or config.upper_threshold is None:
        defaults = load_range_thresholds()
        lower = config.lower_threshold if config.lower_threshold is not None else defaults.lower
        upper = config.upper_threshold if config.upper_threshold is not None else defaults.upper
    else:
        lower = config.lower_threshold
        upper = config.upper_threshold
    return RangeThresholds(lower=lower, upper=upper)


def _normalise_label_counts(
    counts: Mapping[str, float],
    total_samples: int,
) -> Dict[str, float]:
    if total_samples <= 0:
        return {key: 0.0 for key in counts}
    return {key: float(value) / float(total_samples) for key, value in counts.items()}


def _partition_records(
    records: Sequence[Mapping[str, object]],
    *,
    min_confidence: float,
) -> Dict[str, Tuple[Mapping[str, object], ...]]:
    partitions: Dict[str, list[Mapping[str, object]]] = {
        "below": [],
        "in": [],
        "above": [],
        "uncertain": [],
    }
    for record in records:
        label = _classify_record(record, min_confidence=min_confidence)
        partitions[label].append(record)
    return {label: tuple(items) for label, items in partitions.items()}


def _draw_bootstrap_sample(
    records: Sequence[Mapping[str, object]],
    partitions: Mapping[str, Tuple[Mapping[str, object], ...]],
    rng: np.random.Generator,
    *,
    strategy: str,
    min_confidence: float,
) -> Tuple[Mapping[str, object], ...]:
    if strategy == "fixed":
        return _draw_stratified_sample(partitions, rng)
    if strategy == "label_resample":
        return _draw_label_resample_sample(records, rng, min_confidence=min_confidence)
    raise ValueError(f"Unsupported label_strategy '{strategy}'")


def _draw_stratified_sample(
    partitions: Mapping[str, Tuple[Mapping[str, object], ...]],
    rng: np.random.Generator,
) -> Tuple[Mapping[str, object], ...]:
    sample: list[Mapping[str, object]] = []
    for label in ("below", "in", "above", "uncertain"):
        bucket = partitions.get(label, ())
        if not bucket:
            continue
        indices = rng.integers(0, len(bucket), size=len(bucket))
        sample.extend(bucket[idx] for idx in indices)
    return tuple(sample)


def _draw_label_resample_sample(
    records: Sequence[Mapping[str, object]],
    rng: np.random.Generator,
    *,
    min_confidence: float,
) -> Tuple[Mapping[str, object], ...]:
    if not records:
        return ()
    size = len(records)
    sample: list[Mapping[str, object]] = []
    for _ in range(size):
        index = int(rng.integers(0, size))
        base_record = records[index]
        sample.append(_impute_record_label(base_record, rng=rng, min_confidence=min_confidence))
    return tuple(sample)


def _draw_block_sample(
    records: Sequence[Mapping[str, object]],
    rng: np.random.Generator,
    *,
    block_length: int,
    mode: str,
) -> Tuple[Mapping[str, object], ...]:
    if not records:
        return ()
    if block_length <= 1:
        indices = rng.integers(0, len(records), size=len(records))
        return tuple(records[int(idx)] for idx in indices)
    if mode == "moving_block":
        return _draw_moving_block_sample(records, rng, block_length=block_length)
    if mode == "stationary":
        return _draw_stationary_sample(records, rng, block_length=block_length)
    raise ValueError(f"Unsupported resampling_mode '{mode}'")


def _draw_moving_block_sample(
    records: Sequence[Mapping[str, object]],
    rng: np.random.Generator,
    *,
    block_length: int,
) -> Tuple[Mapping[str, object], ...]:
    size = len(records)
    sample: list[Mapping[str, object]] = []
    blocks_needed = int(math.ceil(size / float(block_length)))
    for _ in range(blocks_needed):
        start = int(rng.integers(0, size))
        for offset in range(block_length):
            index = (start + offset) % size
            sample.append(records[index])
            if len(sample) >= size:
                break
        if len(sample) >= size:
            break
    return tuple(sample[:size])


def _draw_stationary_sample(
    records: Sequence[Mapping[str, object]],
    rng: np.random.Generator,
    *,
    block_length: int,
) -> Tuple[Mapping[str, object], ...]:
    size = len(records)
    sample: list[Mapping[str, object]] = []
    p = 1.0 / float(block_length)
    current = int(rng.integers(0, size))
    sample.append(records[current])
    while len(sample) < size:
        if float(rng.random()) < p:
            current = int(rng.integers(0, size))
        else:
            current = (current + 1) % size
        sample.append(records[current])
    return tuple(sample)


def _apply_label_strategy(
    records: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator,
    strategy: str,
    min_confidence: float,
) -> Tuple[Mapping[str, object], ...]:
    if strategy == "fixed":
        return tuple(records)
    if strategy == "label_resample":
        return tuple(
            _impute_record_label(record, rng=rng, min_confidence=min_confidence) for record in records
        )
    raise ValueError(f"Unsupported label_strategy '{strategy}'")


def _sort_records_by_timestamp(records: Sequence[Mapping[str, object]]) -> Tuple[Mapping[str, object], ...]:
    if not records or "timestamp" not in records[0]:
        return tuple(records)
    return tuple(sorted(records, key=_record_timestamp_key))


def _record_timestamp_key(record: Mapping[str, object]) -> float:
    value = record.get("timestamp")
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _resolve_block_length(node_id: str, config: StratifiedBootstrapConfig) -> int:
    if config.node_block_lengths is not None and node_id in config.node_block_lengths:
        length = int(config.node_block_lengths[node_id])
    else:
        length = config.block_length
    return max(1, length)

def _classify_record(record: Mapping[str, object], *, min_confidence: float) -> str:
    prob_below = _clamp_probability(record.get("prob_range_below"))
    prob_in = _clamp_probability(record.get("prob_range_in"))
    prob_above = _clamp_probability(record.get("prob_range_above"))
    total_prob = prob_below + prob_in + prob_above
    if total_prob <= 0.0:
        total_prob = 1.0
    prob_below /= total_prob
    prob_in /= total_prob
    prob_above /= total_prob

    flag = str(record.get("range_flag") or "").strip().lower()
    flag_confident = bool(record.get("range_flag_confident"))

    if flag_confident:
        if flag == "in" and prob_in >= min_confidence:
            return "in"
        if flag == "below" and prob_below >= min_confidence:
            return "below"
        if flag == "above" and prob_above >= min_confidence:
            return "above"

    triplets = [(prob_below, "below"), (prob_in, "in"), (prob_above, "above")]
    best_prob, best_label = max(triplets, key=lambda item: item[0])
    if best_prob >= min_confidence:
        return best_label

    return "uncertain"


def _impute_record_label(
    record: Mapping[str, object],
    *,
    rng: np.random.Generator,
    min_confidence: float,
) -> Mapping[str, object]:
    entry = dict(record)
    prob_below = _clamp_probability(record.get("prob_range_below"))
    prob_in = _clamp_probability(record.get("prob_range_in"))
    prob_above = _clamp_probability(record.get("prob_range_above"))
    total_prob = prob_below + prob_in + prob_above

    if total_prob <= 0.0:
        entry["range_flag"] = "uncertain"
        entry["range_flag_confident"] = False
        entry["prob_range_below"] = 0.0
        entry["prob_range_in"] = 0.0
        entry["prob_range_above"] = 0.0
        return entry

    prob_below /= total_prob
    prob_in /= total_prob
    prob_above /= total_prob

    entry["prob_range_below"] = prob_below
    entry["prob_range_in"] = prob_in
    entry["prob_range_above"] = prob_above

    max_prob = max(prob_below, prob_in, prob_above)
    if max_prob < min_confidence:
        entry["range_flag"] = "uncertain"
        entry["range_flag_confident"] = False
        return entry

    draw = float(rng.random())
    cumulative = prob_below
    if draw < cumulative:
        label = "below"
    else:
        cumulative += prob_in
        if draw < cumulative:
            label = "in"
        else:
            label = "above"

    entry["range_flag"] = label
    entry["range_flag_confident"] = True
    entry["prob_range_below"] = 1.0 if label == "below" else 0.0
    entry["prob_range_in"] = 1.0 if label == "in" else 0.0
    entry["prob_range_above"] = 1.0 if label == "above" else 0.0
    return entry


def _evaluate_dataset(
    records: Sequence[Mapping[str, object]],
    *,
    height: HeightCorrection,
    config: StratifiedBootstrapConfig,
    thresholds: RangeThresholds,
    rmse_value: float,
    rng: np.random.Generator | None,
    apply_noise: bool,
) -> tuple[
    Dict[BootstrapMetricName, float | None],
    str,
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[str, ...],
    WeibullFitResult | None,
    float | None,
]:
    sample = _apply_noise_to_records(
        records,
        rng=rng,
        rmse_value=rmse_value,
        apply_noise=apply_noise,
        truncation_multiplier=config.noise_truncation_multiplier,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
    )

    censored = build_censored_data_from_records(
        sample,
        lower_threshold=thresholds.lower,
        upper_threshold=thresholds.upper,
        min_confidence=config.min_confidence,
    )

    summary = summarise_records_for_selection(sample, min_confidence=config.min_confidence)
    summary = dict(summary)
    summary.setdefault("total_observations", len(sample))

    tail_surrogate = config.tail_surrogate if config.tail_surrogate is not None else thresholds.upper

    (
        method,
        power_density,
        power_curve_estimate,
        weibull,
        km_result,
        selection_reasons,
        method_notes,
    ) = compute_power_distribution(
        data=censored,
        summary_row=summary,
        power_curve=config.power_curve,
        air_density=config.air_density,
        tail_surrogate=tail_surrogate,
        min_in_range=config.min_in_range_weight,
        km_criteria=config.km_criteria,
        height=height,
    )

    metrics = _compute_metric_set(censored, height=height, air_density=config.air_density)
    metrics["power_density"] = power_density.estimate_w_per_m2

    power_sigma = _estimate_power_density_sigma(
        censored,
        height=height,
        air_density=config.air_density,
        rmse_value=rmse_value,
    )

    notes_method = tuple(method_notes)
    notes_power_density = tuple(power_density.notes)
    notes_power_curve = tuple(power_curve_estimate.notes)

    return (
        metrics,
        method,
        tuple(selection_reasons),
        notes_method,
        notes_power_density,
        notes_power_curve,
        weibull,
        power_sigma,
    )


def _apply_noise_to_records(
    records: Sequence[Mapping[str, object]],
    *,
    rng: np.random.Generator | None,
    rmse_value: float,
    apply_noise: bool,
    truncation_multiplier: float,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> Tuple[Mapping[str, object], ...]:
    prepared: list[Mapping[str, object]] = []
    # Use a dedicated RNG for noise; fall back to numpy global if rng is None but preserve determinism otherwise.
    local_rng = rng if rng is not None else np.random.default_rng()

    for record in records:
        entry = dict(record)
        speed = _to_float(entry.get("pred_wind_speed"))

        if apply_noise and rmse_value > 0.0 and speed is not None:
            noise = float(local_rng.normal(loc=0.0, scale=rmse_value))
            limit = truncation_multiplier * rmse_value
            noise = float(np.clip(noise, -limit, limit))
            perturbed = max(0.0, speed + noise)
            if lower_threshold is not None:
                perturbed = max(lower_threshold, perturbed)
            if upper_threshold is not None:
                perturbed = min(upper_threshold, perturbed)
            entry["pred_wind_speed"] = perturbed
        elif speed is not None:
            clamped = speed
            if lower_threshold is not None:
                clamped = max(lower_threshold, clamped)
            if upper_threshold is not None:
                clamped = min(upper_threshold, clamped)
            if clamped != speed:
                entry["pred_wind_speed"] = clamped

        prepared.append(entry)

    return tuple(prepared)


def _compute_metric_set(
    data: CensoredWeibullData,
    *,
    height: HeightCorrection,
    air_density: float,
) -> Dict[BootstrapMetricName, float | None]:
    if data.in_count <= 0.0:
        return _empty_metric_set()

    values = np.asarray(data.in_values, dtype=float) * height.speed_scale
    weights = np.asarray(data.in_weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0 or np.sum(weights) <= 0.0:
        return _empty_metric_set()

    mean_speed = float(np.average(values, weights=weights))
    quantiles = _weighted_quantile(values, weights, [0.5, 0.9, 0.99])
    power_density = float(0.5 * air_density * np.average(np.power(np.clip(values, 0.0, None), 3), weights=weights))

    return {
        "mean_speed": mean_speed,
        "p50": quantiles[0],
        "p90": quantiles[1],
        "p99": quantiles[2],
        "power_density": power_density,
    }


def _estimate_power_density_sigma(
    data: CensoredWeibullData,
    *,
    height: HeightCorrection,
    air_density: float,
    rmse_value: float,
) -> float | None:
    if rmse_value <= 0.0 or data.in_count <= 0.0:
        return None

    values = np.asarray(data.in_values, dtype=float)
    weights = np.asarray(data.in_weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return None

    values = values * height.speed_scale
    sum_weights = np.sum(weights)
    sum_weights_sq = np.sum(np.square(weights))
    if sum_weights <= 0.0 or sum_weights_sq <= 0.0:
        return None

    derivatives = 1.5 * air_density * np.power(values, 2)
    mean_sq = np.average(np.square(derivatives), weights=weights)
    if not math.isfinite(mean_sq) or mean_sq <= 0.0:
        return None

    n_eff = (sum_weights ** 2) / sum_weights_sq
    if n_eff <= 0.0:
        return None

    return rmse_value * math.sqrt(mean_sq / n_eff)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0.0:
        return np.full(len(quantiles), np.nan)

    normalized = cumulative / total
    results = np.empty(len(quantiles), dtype=float)

    for idx, q in enumerate(quantiles):
        q = min(max(q, 0.0), 1.0)
        position = np.searchsorted(normalized, q, side="left")
        if position == 0:
            results[idx] = values[0]
        elif position >= values.size:
            results[idx] = values[-1]
        else:
            weight_low = normalized[position - 1]
            weight_high = normalized[position]
            value_low = values[position - 1]
            value_high = values[position]
            if weight_high == weight_low:
                results[idx] = value_high
            else:
                ratio = (q - weight_low) / (weight_high - weight_low)
                results[idx] = value_low + ratio * (value_high - value_low)

    return results


def _compute_jackknife_statistics(
    partitions: Mapping[str, Tuple[Mapping[str, object], ...]],
    *,
    config: StratifiedBootstrapConfig,
    thresholds: RangeThresholds,
    rmse_value: float,
    height: HeightCorrection,
) -> Dict[str, np.ndarray] | None:
    total = sum(len(bucket) for bucket in partitions.values())
    if total <= 1 or total > config.jackknife_max_samples:
        return None

    metric_names = ("mean_speed", "p50", "p90", "p99", "power_density")
    jackknife: Dict[str, list[float]] = {name: [] for name in metric_names}

    for label, bucket in partitions.items():
        entries = list(bucket)
        for idx in range(len(entries)):
            reduced_records: list[Mapping[str, object]] = []
            for lab, data_bucket in partitions.items():
                if lab == label:
                    reduced_records.extend(record for j, record in enumerate(data_bucket) if j != idx)
                else:
                    reduced_records.extend(data_bucket)
            if not reduced_records:
                return None
            metrics, _, _, _, _, _, _, _ = _evaluate_dataset(
                tuple(reduced_records),
                height=height,
                config=config,
                thresholds=thresholds,
                rmse_value=rmse_value,
                rng=None,
                apply_noise=False,
            )
            for name in metric_names:
                value = metrics.get(name)
                if value is None or not math.isfinite(value):
                    return None
                jackknife[name].append(float(value))

    return {name: np.asarray(values, dtype=float) for name, values in jackknife.items()}


def _build_confidence_intervals(
    *,
    baseline: Mapping[BootstrapMetricName, float | None],
    draws: Mapping[BootstrapMetricName, Sequence[float]],
    config: StratifiedBootstrapConfig,
    partitions: Mapping[str, Tuple[Mapping[str, object], ...]],
    thresholds: RangeThresholds,
    rmse_value: float,
    height: HeightCorrection,
    velocity_noise: bool,
    power_clamp_count: int,
) -> tuple[
    Dict[BootstrapMetricName, BootstrapConfidenceInterval],
    Dict[BootstrapMetricName, float | None],
    Tuple[str, ...],
]:
    interval_notes: list[str] = []
    arrays: Dict[str, np.ndarray] = {
        name: np.asarray(list(values), dtype=float)
        for name, values in draws.items()
    }

    method = config.ci_method
    jackknife: Dict[str, np.ndarray] | None = None

    if method in {"bca", "percentile_t"}:
        jackknife = _compute_jackknife_statistics(
            partitions,
            config=config,
            thresholds=thresholds,
            rmse_value=rmse_value,
            height=height,
        )
        if jackknife is None:
            interval_notes.append(
                f"CI method {method} requires jackknife samples; falling back to percentile intervals."
            )
            method = "percentile"

    if power_clamp_count > 0 and config.replicas > 0:
        ratio = power_clamp_count / config.replicas
        interval_notes.append(
            f"Power density truncation applied in {power_clamp_count}/{config.replicas} draws ({ratio:.1%})."
        )

    adjusted_arrays = arrays
    if velocity_noise:
        adjusted_arrays = {}
        for name, values in arrays.items():
            adjusted = values[np.isfinite(values)]
            baseline_value = baseline.get(name)
            if adjusted.size > 0 and baseline_value is not None and math.isfinite(baseline_value):
                bias = float(adjusted.mean()) - float(baseline_value)
                adjusted = adjusted - bias
                if name == "power_density":
                    adjusted = np.clip(adjusted, 0.0, None)
            adjusted_arrays[name] = adjusted
    else:
        adjusted_arrays = {name: values[np.isfinite(values)] for name, values in arrays.items()}

    if method == "percentile":
        intervals, means = _compute_percentile_intervals(
            baseline,
            adjusted_arrays,
            confidence_level=config.confidence_level,
        )
    elif method == "bca":
        intervals, means, extra = _compute_bca_intervals(
            baseline,
            adjusted_arrays,
            jackknife,
            confidence_level=config.confidence_level,
        )
        interval_notes.extend(extra)
    elif method == "percentile_t":
        intervals, means, extra = _compute_percentile_t_intervals(
            baseline,
            adjusted_arrays,
            jackknife,
            confidence_level=config.confidence_level,
        )
        interval_notes.extend(extra)
    else:  # defensive fallback
        intervals, means = _compute_percentile_intervals(
            baseline,
            adjusted_arrays,
            confidence_level=config.confidence_level,
        )

    return intervals, means, tuple(interval_notes)


def _compute_percentile_intervals(
    baseline: Mapping[BootstrapMetricName, float | None],
    arrays: Mapping[BootstrapMetricName, np.ndarray],
    *,
    confidence_level: float,
) -> tuple[Dict[BootstrapMetricName, BootstrapConfidenceInterval], Dict[BootstrapMetricName, float | None]]:
    alpha = 1.0 - confidence_level
    result: Dict[BootstrapMetricName, BootstrapConfidenceInterval] = {}
    means: Dict[BootstrapMetricName, float | None] = {}

    for name, estimate in baseline.items():
        values = np.asarray(arrays.get(name, ()), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0 or estimate is None or not math.isfinite(estimate):
            result[name] = BootstrapConfidenceInterval(
                estimate=estimate,
                lower=None,
                upper=None,
                confidence_level=confidence_level,
                replicates=values.size,
                notes=(),
            )
            means[name] = estimate
            continue

        corrected = values
        if corrected.size > 0:
            lower = float(np.quantile(corrected, alpha / 2.0, method="linear"))
            upper = float(np.quantile(corrected, 1.0 - alpha / 2.0, method="linear"))
            mean_value = float(corrected.mean())
        else:
            lower = upper = None
            mean_value = estimate
        result[name] = BootstrapConfidenceInterval(
            estimate=estimate,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            replicates=corrected.size,
            notes=(),
        )
        means[name] = mean_value

    return result, means


def _compute_bca_intervals(
    baseline: Mapping[BootstrapMetricName, float | None],
    arrays: Mapping[BootstrapMetricName, np.ndarray],
    jackknife: Dict[str, np.ndarray] | None,
    *,
    confidence_level: float,
) -> tuple[
    Dict[BootstrapMetricName, BootstrapConfidenceInterval],
    Dict[BootstrapMetricName, float | None],
    Tuple[str, ...],
]:
    nd = NormalDist()
    alpha = 1.0 - confidence_level
    lower_quant = nd.inv_cdf(alpha / 2.0)
    upper_quant = nd.inv_cdf(1.0 - alpha / 2.0)

    intervals: Dict[BootstrapMetricName, BootstrapConfidenceInterval] = {}
    means: Dict[BootstrapMetricName, float | None] = {}
    notes: list[str] = []

    for name, estimate in baseline.items():
        values = np.asarray(arrays.get(name, ()), dtype=float)
        values = values[np.isfinite(values)]
        jack = None if jackknife is None else jackknife.get(name)
        if (
            values.size == 0
            or estimate is None
            or not math.isfinite(estimate)
            or jack is None
            or jack.size <= 1
        ):
            intervals[name] = BootstrapConfidenceInterval(
                estimate=estimate,
                lower=None,
                upper=None,
                confidence_level=confidence_level,
                replicates=values.size,
                notes=("BCa fallback to percentile due to insufficient data.",),
            )
            means[name] = estimate if values.size == 0 else float(values.mean())
            notes.append(f"BCa interval for {name} unavailable; insufficient jackknife data.")
            continue

        prop = (np.sum(values < estimate) + 0.5 * np.sum(values == estimate)) / values.size
        prop = min(max(prop, 1e-10), 1.0 - 1e-10)
        z0 = nd.inv_cdf(prop)

        theta_dot = float(jack.mean())
        diff = theta_dot - jack
        numerator = np.sum(diff ** 3)
        denominator = 6.0 * (np.sum(diff ** 2) ** 1.5)
        a = numerator / denominator if denominator != 0.0 else 0.0

        def transform(z_quant: float) -> float:
            denom = 1.0 - a * (z0 + z_quant)
            if denom == 0.0:
                return float(z_quant)
            adjusted = z0 + (z0 + z_quant) / denom
            return float(nd.cdf(adjusted))

        alpha1 = transform(lower_quant)
        alpha2 = transform(upper_quant)
        alpha1 = float(np.clip(alpha1, 1e-10, 1.0 - 1e-10))
        alpha2 = float(np.clip(alpha2, 1e-10, 1.0 - 1e-10))
        lower = float(np.quantile(values, alpha1, method="linear"))
        upper = float(np.quantile(values, alpha2, method="linear"))
        intervals[name] = BootstrapConfidenceInterval(
            estimate=estimate,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            replicates=values.size,
            notes=(),
        )
        means[name] = float(values.mean())

    return intervals, means, tuple(notes)


def _compute_percentile_t_intervals(
    baseline: Mapping[BootstrapMetricName, float | None],
    arrays: Mapping[BootstrapMetricName, np.ndarray],
    jackknife: Dict[str, np.ndarray] | None,
    *,
    confidence_level: float,
) -> tuple[
    Dict[BootstrapMetricName, BootstrapConfidenceInterval],
    Dict[BootstrapMetricName, float | None],
    Tuple[str, ...],
]:
    alpha = 1.0 - confidence_level
    intervals: Dict[BootstrapMetricName, BootstrapConfidenceInterval] = {}
    means: Dict[BootstrapMetricName, float | None] = {}
    notes: list[str] = []

    for name, estimate in baseline.items():
        values = np.asarray(arrays.get(name, ()), dtype=float)
        values = values[np.isfinite(values)]
        jack = None if jackknife is None else jackknife.get(name)
        if (
            values.size == 0
            or estimate is None
            or not math.isfinite(estimate)
            or jack is None
            or jack.size <= 1
        ):
            intervals[name] = BootstrapConfidenceInterval(
                estimate=estimate,
                lower=None,
                upper=None,
                confidence_level=confidence_level,
                replicates=values.size,
                notes=("Percentile-t fallback to percentile due to insufficient jackknife data.",),
            )
            means[name] = estimate if values.size == 0 else float(values.mean())
            notes.append(f"Percentile-t interval for {name} unavailable; insufficient jackknife data.")
            continue

        theta_dot = float(jack.mean())
        variance = (jack.size - 1) / jack.size * np.sum((jack - theta_dot) ** 2)
        if variance <= 0.0:
            intervals[name] = BootstrapConfidenceInterval(
                estimate=estimate,
                lower=None,
                upper=None,
                confidence_level=confidence_level,
                replicates=values.size,
                notes=("Percentile-t fallback to percentile due to zero jackknife variance.",),
            )
            means[name] = float(values.mean())
            notes.append(f"Percentile-t interval for {name} unavailable; jackknife variance is zero.")
            continue

        s = math.sqrt(variance)
        studentized = (values - float(estimate)) / s
        t_lower = np.quantile(studentized, 1.0 - alpha / 2.0, method="linear")
        t_upper = np.quantile(studentized, alpha / 2.0, method="linear")
        lower = float(estimate) - float(t_lower) * s
        upper = float(estimate) - float(t_upper) * s
        intervals[name] = BootstrapConfidenceInterval(
            estimate=estimate,
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            replicates=values.size,
            notes=(),
        )
        means[name] = float(values.mean())

    return intervals, means, tuple(notes)


def _wrap_without_bootstrap(
    baseline: Mapping[BootstrapMetricName, float | None],
    config: StratifiedBootstrapConfig,
    notes: Iterable[str],
    *,
    replicates: int,
) -> Dict[BootstrapMetricName, BootstrapConfidenceInterval]:
    result: Dict[BootstrapMetricName, BootstrapConfidenceInterval] = {}
    note_tuple = tuple(notes)
    for name, estimate in baseline.items():
        result[name] = BootstrapConfidenceInterval(
            estimate=estimate,
            lower=None,
            upper=None,
            confidence_level=config.confidence_level,
            replicates=replicates,
            notes=note_tuple,
        )
    return result


def _empty_metric_set() -> Dict[BootstrapMetricName, float | None]:
    return {
        "mean_speed": None,
        "p50": None,
        "p90": None,
        "p99": None,
        "power_density": None,
    }


def _clamp_probability(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
