"""Statistical helpers for uncertainty modelling."""

from __future__ import annotations

from .direction import (
    DirectionErrorMetrics,
    DirectionErrorResult,
    DirectionQualitySummary,
    evaluate_direction_pairs,
)
from .empirical import BiasThresholds, assemble_per_node_metrics, compute_label_ratios, load_taxonomy_records
from .kaplan_meier import (
    KaplanMeierResult,
    KaplanMeierSelectionCriteria,
    load_kaplan_meier_selection_criteria,
    evaluate_kaplan_meier_selection,
    evaluate_step_cdf,
    run_weighted_kaplan_meier,
)
from .bootstrapping import (
    BootstrapConfidenceInterval,
    BootstrapPowerDiagnostics,
    BootstrapUncertaintyResult,
    NodeBootstrapInput,
    StratifiedBootstrapConfig,
    compute_stratified_bootstrap_uncertainty,
)
from .dependence import (
    NodeDependenceMetrics,
    compute_node_dependence_metrics,
    summarise_dependence_levels,
)
from .power import (
    PowerCurve,
    PowerCurveEstimate,
    PowerDensityEstimate,
    compute_kaplan_meier_power_density,
    compute_weibull_power_density,
    estimate_power_curve_from_kaplan_meier,
    estimate_power_curve_from_weibull,
)
from .power_pipeline import (
    HeightCorrection,
    compute_power_distribution,
    format_height_note,
    summarise_records_for_selection,
)
from .parametric import (
    ParametricComparison,
    ParametricComparisonConfig,
    ParametricModelSummary,
    evaluate_parametric_models,
)
from .seasonal import (
    AnnualSlice,
   NodeVariationSummary,
   SeasonalAnalysisResult,
   SeasonalSlice,
   compute_seasonal_analysis,
)
from .time_series import (
    EtsResult,
    SarimaConfig,
    SarimaResult,
    TimeSeriesSegment,
    load_time_series_config,
    compute_acf_pacf,
    fit_ets_seasonal,
    fit_sarima_auto,
    prepare_monthly_series,
    split_series_by_gaps,
)
from .rmse import GlobalRmseProvider, GlobalRmseRecord, NodeRmseAssessment, NodeTaxonomyEntry
from .weibull import (
    CensoredWeibullData,
    WeibullFitDiagnostics,
    WeibullFitResult,
    build_censored_data_from_records,
    compute_censored_weibull_log_likelihood,
    fit_censored_weibull,
)

__all__ = [
    "BiasThresholds",
    "CensoredWeibullData",
    "DirectionErrorMetrics",
    "DirectionErrorResult",
    "DirectionQualitySummary",
    "GlobalRmseProvider",
    "GlobalRmseRecord",
    "KaplanMeierResult",
    "KaplanMeierSelectionCriteria",
    "WeibullFitDiagnostics",
    "WeibullFitResult",
    "assemble_per_node_metrics",
    "build_censored_data_from_records",
    "BootstrapConfidenceInterval",
    "BootstrapPowerDiagnostics",
    "BootstrapUncertaintyResult",
    "compute_label_ratios",
    "compute_censored_weibull_log_likelihood",
    "compute_seasonal_analysis",
    "compute_stratified_bootstrap_uncertainty",
    "compute_power_distribution",
    "evaluate_direction_pairs",
    "evaluate_kaplan_meier_selection",
    "evaluate_step_cdf",
    "evaluate_parametric_models",
    "fit_censored_weibull",
    "load_kaplan_meier_selection_criteria",
    "load_taxonomy_records",
    "NodeBootstrapInput",
    "NodeDependenceMetrics",
    "NodeRmseAssessment",
    "NodeTaxonomyEntry",
    "NodeVariationSummary",
    "StratifiedBootstrapConfig",
    "run_weighted_kaplan_meier",
    "PowerCurve",
    "PowerCurveEstimate",
    "PowerDensityEstimate",
    "compute_kaplan_meier_power_density",
    "compute_weibull_power_density",
    "estimate_power_curve_from_kaplan_meier",
    "estimate_power_curve_from_weibull",
    "HeightCorrection",
    "format_height_note",
    "summarise_records_for_selection",
    "SeasonalAnalysisResult",
    "SeasonalSlice",
    "AnnualSlice",
    "SarimaConfig",
    "SarimaResult",
    "EtsResult",
    "TimeSeriesSegment",
    "load_time_series_config",
    "fit_sarima_auto",
    "fit_ets_seasonal",
    "prepare_monthly_series",
    "split_series_by_gaps",
    "compute_acf_pacf",
    "compute_node_dependence_metrics",
    "summarise_dependence_levels",
    "ParametricComparison",
    "ParametricComparisonConfig",
    "ParametricModelSummary",
]
