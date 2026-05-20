from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.stats import STATISTICS_SUMMARY_COLUMNS, run_statistical_tests


PAPER_SEED_SUMMARY_COLUMNS = (
    "source_experiment",
    "paper_model_id",
    "model_name",
    "metric_name",
    "n_seeds",
    "mean",
    "std",
    "min",
    "max",
    "median",
)
PPO_DQN_HIERARCHICAL_REIMPLEMENTATION = "ppo_dqn_hierarchical_reimplementation"
DEFAULT_PAPER_BENCHMARK_MODELS = (
    "best_traditional",
    "equal_weight",
    "cnn_ppo_native",
    "pgportfolio_eiie_native",
    "without_dqn_gate",
    "without_auxiliary",
    "no_pca",
)
DEFAULT_PAPER_SEED_METRICS = (
    "n_steps",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "report_sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "var",
    "cvar",
    "turnover",
    "average_turnover",
    "total_transaction_cost",
    "hit_ratio",
    "violation_rate",
)
TRADITIONAL_BASELINE_FAMILY = "traditional"
HYBRID_DQN_OPTIMIZER_ALIAS = "hybrid_dqn_optimizer_reimplementation"
HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES = (
    "hybrid_dqn_optimizer_equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity",
)
PAPER_TRAINABLE_MODEL_IDS = (
    PPO_DQN_HIERARCHICAL_REIMPLEMENTATION,
    *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
)
PAPER_TRAINABLE_REQUIRED_METADATA = (
    "algorithm_fidelity",
    "baseline_family",
    "training_algorithm",
    "rankable_in_unified_table",
)
CLOSEST_HYBRID_FIGURE_SOURCE_COLUMNS = (
    "source_experiment",
    "source_run",
    "source_path",
    "source_file",
    "paper_group_id",
    "paper_model_id",
    "model_name",
    *PAPER_TRAINABLE_REQUIRED_METADATA,
)
PAPER_MAIN_EXCLUDED_DIAGNOSTIC_STATUSES = {"partial_diagnostic", "diagnostic_shared_dqn"}
PAPER_DIAGNOSTIC_COMPARISON_COLUMNS = (
    "source_experiment",
    "source_run",
    "source_path",
    "source_file",
    "paper_group_id",
    "paper_model_id",
    "model_name",
    "seed",
    "fold_id",
    "status",
    "diagnostic_status",
    "reason",
    "rankable_in_unified_table",
)
COMPARISON_FILES = (
    "main_comparison.csv",
    "baseline_comparison.csv",
    "hpo_model_final_comparison.csv",
    "ablation_results.csv",
    "input_matrix_ablation_results.csv",
    "PCA_ablation_results.csv",
    "reward_ablation_results.csv",
    "transaction_cost_sensitivity.csv",
    "asset_universe_sensitivity.csv",
    "market_regime_results.csv",
    "seed_stability.csv",
    "rebalance_frequency_analysis.csv",
    "preference_conditioned_results.csv",
    "uncertainty_results.csv",
    "distributional_cvar_results.csv",
    "partial_rebalance_results.csv",
    "walk_forward_results.csv",
)
COMPARISON_FILE_PRIORITY = {
    "hpo_model_final_comparison.csv": 0,
    "main_comparison.csv": 10,
    "baseline_comparison.csv": 20,
}
DAILY_RETURN_FILES = (
    "hpo_model_final_daily_returns.csv",
    "daily_returns.csv",
)
DAILY_RETURN_FILE_PRIORITY = {
    "hpo_model_final_daily_returns.csv": 0,
    "daily_returns.csv": 20,
}


def aggregate_paper_results(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    benchmark_model: str | None = None,
    benchmark_models: Sequence[str] | None = None,
    seed_metric_columns: Sequence[str] | None = None,
    paper_group_id: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    runs = [Path(path).expanduser().resolve() for path in run_dirs]
    if not runs:
        raise ValueError("ERR_PAPER_AGGREGATE_NO_RUN_DIRS")
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    comparison = _collect_comparison_rows(runs)
    daily_returns = _collect_daily_returns(runs)
    if paper_group_id is not None:
        comparison["paper_group_id"] = str(paper_group_id)
        daily_returns["paper_group_id"] = str(paper_group_id)
    paper_main = _paper_main_comparison(comparison, daily_returns)
    paper_diagnostic = _paper_diagnostic_comparison(comparison, daily_returns)
    rankable_models = _rankable_models(paper_main)
    resolved_benchmarks = _resolve_benchmark_models(
        paper_main,
        benchmark_models=_benchmark_model_values(benchmark_model, benchmark_models),
    )
    paired = _paper_paired_statistics(
        daily_returns,
        rankable_models=rankable_models,
        benchmark_models=resolved_benchmarks,
        config=config,
    )
    seed_summary = _paper_seed_summary(paper_main, metric_columns=seed_metric_columns)
    closest_hybrid = _closest_hybrid_figure_source(paper_main, seed_summary)

    outputs = {
        "paper_main_comparison": target / "paper_main_comparison.csv",
        "paper_diagnostic_comparison": target / "paper_diagnostic_comparison.csv",
        "paper_paired_statistics": target / "paper_paired_statistics.csv",
        "paper_seed_summary": target / "paper_seed_summary.csv",
        "closest_hybrid_figure_source": target / "closest_hybrid_figure_source.csv",
    }
    _write_csv(paper_main, outputs["paper_main_comparison"])
    _write_csv(paper_diagnostic, outputs["paper_diagnostic_comparison"])
    _write_csv(paired, outputs["paper_paired_statistics"])
    _write_csv(seed_summary, outputs["paper_seed_summary"])
    _write_csv(closest_hybrid, outputs["closest_hybrid_figure_source"])
    return outputs


def _collect_comparison_rows(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        manifest = _manifest(run_dir)
        experiment_result = _experiment_result(run_dir)
        metrics_dir = run_dir / "metrics"
        filenames = ("hpo_model_final_comparison.csv",) if (metrics_dir / "hpo_model_final_comparison.csv").exists() else COMPARISON_FILES
        for filename in filenames:
            path = metrics_dir / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["source_run"] = _source_run(run_dir, manifest)
            frame["source_experiment"] = manifest.get("experiment_type") or "unknown"
            frame["source_path"] = str(run_dir)
            frame["source_file"] = filename
            frame["_source_file_priority"] = _comparison_file_priority(filename)
            if "seed" not in frame.columns:
                frame["seed"] = manifest.get("seed")
            fallback_model = _clean_value(manifest.get("model_name")) or _clean_value(experiment_result.get("model_name"))
            if fallback_model:
                if "model_name" not in frame.columns:
                    frame["model_name"] = fallback_model
                else:
                    missing_model = frame["model_name"].isna() | frame["model_name"].astype(str).str.strip().eq("")
                    frame.loc[missing_model, "model_name"] = fallback_model
            fallback_status = _clean_value(experiment_result.get("status")) or _clean_value(manifest.get("status"))
            if fallback_status:
                if "status" not in frame.columns:
                    frame["status"] = fallback_status
                else:
                    missing_status = frame["status"].isna() | frame["status"].astype(str).str.strip().eq("")
                    frame.loc[missing_status, "status"] = fallback_status
            frames.append(frame)
    if frames:
        return _with_paper_identity(pd.concat(frames, ignore_index=True, sort=False))
    return pd.DataFrame(columns=["model_name", "paper_model_id", "source_run", "source_experiment", "source_path", "source_file", "seed"])


def _collect_daily_returns(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        manifest = _manifest(run_dir)
        experiment_result = _experiment_result(run_dir)
        metrics_dir = run_dir / "metrics"
        selected_path = next((metrics_dir / name for name in DAILY_RETURN_FILES if (metrics_dir / name).exists()), None)
        if selected_path is None:
            continue
        frame = pd.read_csv(selected_path)
        if frame.empty:
            continue
        if "date" not in frame.columns and "next_valuation_date" in frame.columns:
            frame["date"] = frame["next_valuation_date"]
        if "model_name" not in frame.columns:
            frame["model_name"] = (
                _clean_value(manifest.get("model_name"))
                or _clean_value(experiment_result.get("model_name"))
                or manifest.get("run_name")
                or _source_run(run_dir, manifest)
            )
        frame["source_run"] = _source_run(run_dir, manifest)
        frame["source_experiment"] = manifest.get("experiment_type") or "unknown"
        frame["source_path"] = str(run_dir)
        frame["source_file"] = selected_path.name
        frame["_source_file_priority"] = _daily_return_file_priority(selected_path.name)
        if "seed" not in frame.columns:
            frame["seed"] = manifest.get("seed")
        frames.append(frame)
    if frames:
        return _with_paper_identity(pd.concat(frames, ignore_index=True, sort=False))
    return pd.DataFrame(columns=["date", "model_name", "paper_model_id", "net_return", "source_run", "source_experiment", "source_path", "seed"])


def _paper_main_comparison(comparison: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        result = _comparison_from_daily_returns(daily_returns)
    else:
        result = _with_paper_identity(comparison)
        if "rankable_in_unified_table" not in result.columns:
            result["rankable_in_unified_table"] = True
        else:
            result["rankable_in_unified_table"] = result["rankable_in_unified_table"].fillna(True)
    result = result.loc[_paper_main_rankable_mask(result)].copy()
    if "paper_included" not in result.columns:
        result["paper_included"] = result["rankable_in_unified_table"].map(_truthy)
    else:
        result["paper_included"] = result["paper_included"].fillna(result["rankable_in_unified_table"]).map(_truthy)
    result = _dedupe_comparison_rows(result)
    result = result.drop(columns=["_paper_model_id_explicit"], errors="ignore")
    return result.sort_values(["source_run", "paper_model_id"], kind="mergesort").reset_index(drop=True)


def _comparison_from_daily_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame(
            columns=["model_name", "paper_model_id", "source_run", "seed", "rankable_in_unified_table", "paper_included"]
        )
    rows: list[dict[str, Any]] = []
    source = _with_paper_identity(daily_returns)
    for (source_run, paper_model_id), group in source.groupby(["source_run", "paper_model_id"], dropna=False, sort=False):
        returns = pd.to_numeric(group.get("net_return"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rankable_value = _first_value(group, "rankable_in_unified_table")
        rankable = True if rankable_value is None else _truthy(rankable_value)
        row = {
            "model_name": _first_value(group, "model_name") or paper_model_id,
            "paper_model_id": paper_model_id,
            "source_run": source_run,
            "source_experiment": _first_value(group, "source_experiment") or "unknown",
            "paper_group_id": _first_value(group, "paper_group_id"),
            "source_path": _first_value(group, "source_path"),
            "source_file": _first_value(group, "source_file"),
            "_source_file_priority": _first_value(group, "_source_file_priority"),
            "seed": _first_value(group, "seed"),
            "fold_id": _first_value(group, "fold_id"),
            "status": "completed" if not returns.empty else "not_applicable",
            "rankable_in_unified_table": rankable,
            "paper_included": rankable,
            "n_steps": int(len(returns)),
            "cumulative_return": float(np.prod(1.0 + returns.to_numpy()) - 1.0) if not returns.empty else np.nan,
        }
        diagnostic_status = _clean_value(_first_value(group, "diagnostic_status"))
        if diagnostic_status:
            row["diagnostic_status"] = diagnostic_status
        reason = _clean_value(_first_value(group, "reason"))
        if reason:
            row["reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _paper_diagnostic_comparison(comparison: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    sources: list[pd.DataFrame] = []
    if not comparison.empty:
        sources.append(_with_paper_identity(comparison))
    daily_source = _comparison_from_daily_returns(daily_returns)
    if not daily_source.empty:
        sources.append(daily_source)
    if not sources:
        return pd.DataFrame(columns=PAPER_DIAGNOSTIC_COMPARISON_COLUMNS)
    source = pd.concat(sources, ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        reason = _paper_diagnostic_reason(row)
        if not reason:
            continue
        record = row.to_dict()
        record["rankable_in_unified_table"] = False
        if not _clean_value(record.get("reason")):
            record["reason"] = reason
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=PAPER_DIAGNOSTIC_COMPARISON_COLUMNS)
    result = pd.DataFrame(rows)
    if "_source_file_priority" not in result.columns:
        result["_source_file_priority"] = result.get("source_file", pd.Series("", index=result.index)).map(_comparison_file_priority)
    keys = [column for column in ("source_run", "paper_model_id", "seed", "fold_id") if column in result.columns]
    if len(keys) >= 2:
        result = result.sort_values([*keys, "_source_file_priority"], kind="mergesort")
        result = result.drop_duplicates(subset=keys, keep="first")
    result = result.drop(columns=["_paper_model_id_explicit", "_source_file_priority"], errors="ignore")
    for column in PAPER_DIAGNOSTIC_COMPARISON_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    ordered = [*PAPER_DIAGNOSTIC_COMPARISON_COLUMNS, *[column for column in result.columns if column not in PAPER_DIAGNOSTIC_COMPARISON_COLUMNS]]
    sort_cols = [column for column in ("source_run", "paper_model_id", "seed", "fold_id") if column in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols, kind="mergesort")
    return result.loc[:, ordered].reset_index(drop=True)


def _paper_paired_statistics(
    daily_returns: pd.DataFrame,
    *,
    rankable_models: set[str],
    benchmark_models: Sequence[str],
    config: Mapping[str, Any] | None,
) -> pd.DataFrame:
    if daily_returns.empty:
        return _not_applicable_stats("no_daily_returns")
    daily_returns = _with_paper_identity(daily_returns)
    daily_returns = daily_returns.loc[~_paper_diagnostic_reason_series(daily_returns).map(bool)].copy()
    if daily_returns.empty:
        return _not_applicable_stats("benchmark_or_rankable_model_missing")
    daily_returns = _dedupe_daily_return_rows(daily_returns)
    rows: list[pd.DataFrame] = []
    group_cols = ["paper_group_id" if "paper_group_id" in daily_returns.columns else "source_run"]
    group_cols.extend(column for column in ("seed", "fold_id") if column in daily_returns.columns)
    grouped = daily_returns.groupby(group_cols, dropna=False, sort=False)
    for group_key, group in grouped:
        for benchmark_model in benchmark_models:
            if str(benchmark_model) not in rankable_models:
                continue
            benchmark = group.loc[group["paper_model_id"].astype(str).eq(str(benchmark_model))].copy()
            if benchmark.empty:
                continue
            model_returns: dict[str, pd.DataFrame] = {}
            for model_name, model_frame in group.groupby("paper_model_id", sort=False):
                name = str(model_name)
                if name == str(benchmark_model) or name not in rankable_models:
                    continue
                model_returns[name] = model_frame.copy()
            if not model_returns:
                continue
            stats = run_statistical_tests(model_returns, {str(benchmark_model): benchmark}, config=config)
            stats["paper_group"] = _group_label(group_key)
            rows.append(stats)
    if not rows:
        return _not_applicable_stats("benchmark_or_rankable_model_missing")
    result = pd.concat(rows, ignore_index=True, sort=False)
    for column in STATISTICS_SUMMARY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    extra = [column for column in result.columns if column not in STATISTICS_SUMMARY_COLUMNS]
    return result.loc[:, [*STATISTICS_SUMMARY_COLUMNS, *extra]]


def _paper_seed_summary(
    paper_main: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if paper_main.empty:
        return pd.DataFrame(columns=PAPER_SEED_SUMMARY_COLUMNS)
    paper_main = _with_paper_identity(paper_main)
    metric_names = _seed_metric_columns(paper_main, metric_columns)
    rows: list[dict[str, Any]] = []
    included_mask = (
        paper_main["paper_included"].map(_truthy)
        if "paper_included" in paper_main.columns
        else pd.Series(True, index=paper_main.index)
    )
    included = paper_main.loc[included_mask].copy()
    for (source_experiment, paper_model_id), group in included.groupby(["source_experiment", "paper_model_id"], dropna=False, sort=False):
        for metric_name in metric_names:
            values = _seed_metric_values(group, metric_name)
            if values.empty:
                continue
            rows.append(
                {
                    "source_experiment": source_experiment,
                    "paper_model_id": paper_model_id,
                    "model_name": _first_value(group, "model_name") or paper_model_id,
                    "metric_name": metric_name,
                    "n_seeds": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                }
            )
    return pd.DataFrame(rows, columns=PAPER_SEED_SUMMARY_COLUMNS)


def _closest_hybrid_figure_source(paper_main: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    if paper_main.empty:
        return pd.DataFrame(columns=CLOSEST_HYBRID_FIGURE_SOURCE_COLUMNS)
    source = _with_paper_identity(paper_main)
    missing_required = [column for column in PAPER_TRAINABLE_REQUIRED_METADATA if column not in source.columns]
    if missing_required:
        return pd.DataFrame(columns=(*CLOSEST_HYBRID_FIGURE_SOURCE_COLUMNS, *[column for column in source.columns if column not in CLOSEST_HYBRID_FIGURE_SOURCE_COLUMNS]))
    mask = source["paper_model_id"].map(_clean_value).isin(PAPER_TRAINABLE_MODEL_IDS)
    mask &= source["rankable_in_unified_table"].map(_truthy)
    mask &= ~_paper_diagnostic_reason_series(source).map(bool)
    for column in PAPER_TRAINABLE_REQUIRED_METADATA:
        mask &= source[column].map(_clean_value).ne("")
    seed_summary_source = _filter_seed_summary_for_closest(seed_summary, source, mask)
    result = source.loc[mask].copy()
    if result.empty:
        return pd.DataFrame(columns=CLOSEST_HYBRID_FIGURE_SOURCE_COLUMNS)
    if "_source_file_priority" not in result.columns:
        result["_source_file_priority"] = result.get("source_file", pd.Series("", index=result.index)).map(_comparison_file_priority)
    keys = [column for column in ("paper_group_id", "source_experiment", "paper_model_id") if column in result.columns]
    sort_cols = [*keys, "_source_file_priority"]
    sort_cols.extend(column for column in ("source_run", "seed", "fold_id") if column in result.columns)
    if keys:
        result = result.sort_values(sort_cols, kind="mergesort").drop_duplicates(subset=keys, keep="first")
    seed_wide = _seed_summary_wide(seed_summary_source)
    if not seed_wide.empty:
        merge_keys = [column for column in ("source_experiment", "paper_model_id") if column in result.columns and column in seed_wide.columns]
        if merge_keys:
            result = result.merge(seed_wide, on=merge_keys, how="left")
    result = result.drop(columns=["_paper_model_id_explicit", "_source_file_priority"], errors="ignore")
    ordered = [column for column in CLOSEST_HYBRID_FIGURE_SOURCE_COLUMNS if column in result.columns]
    ordered.extend(column for column in result.columns if column not in ordered)
    return result.loc[:, ordered].reset_index(drop=True)


def _filter_seed_summary_for_closest(seed_summary: pd.DataFrame, source: pd.DataFrame, eligible_mask: pd.Series) -> pd.DataFrame:
    key_cols = [column for column in ("source_experiment", "paper_model_id") if column in seed_summary.columns and column in source.columns]
    if seed_summary.empty or len(key_cols) < 2:
        return seed_summary
    key_frame = source.loc[:, key_cols].copy()
    key_frame["_closest_eligible"] = eligible_mask.to_numpy()
    safe_keys = key_frame.groupby(key_cols, dropna=False, sort=False)["_closest_eligible"].all().reset_index()
    safe_keys = safe_keys.loc[safe_keys["_closest_eligible"]].drop(columns=["_closest_eligible"])
    if safe_keys.empty:
        return seed_summary.iloc[0:0].copy()
    return seed_summary.merge(safe_keys, on=key_cols, how="inner")


def _seed_summary_wide(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty or "paper_model_id" not in seed_summary.columns or "metric_name" not in seed_summary.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [column for column in ("source_experiment", "paper_model_id") if column in seed_summary.columns]
    if not group_cols:
        return pd.DataFrame()
    for group_key, group in seed_summary.groupby(group_cols, dropna=False, sort=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_cols, key_values, strict=False))
        for _, metric_row in group.iterrows():
            metric_name = _column_suffix(_clean_value(metric_row.get("metric_name")))
            if not metric_name:
                continue
            for stat_name in ("n_seeds", "mean", "std", "min", "max", "median"):
                if stat_name in metric_row:
                    row[f"seed_summary_{stat_name}_{metric_name}"] = metric_row.get(stat_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _benchmark_model_values(
    benchmark_model: str | None,
    benchmark_models: Sequence[str] | None,
) -> tuple[str, ...]:
    if benchmark_models is not None:
        return _dedupe_strings(benchmark_models)
    if benchmark_model is not None:
        return _dedupe_strings([benchmark_model])
    return DEFAULT_PAPER_BENCHMARK_MODELS


def _resolve_benchmark_models(
    paper_main: pd.DataFrame,
    *,
    benchmark_models: Sequence[str],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for benchmark in benchmark_models:
        key = str(benchmark)
        if key == "best_traditional":
            resolved.extend(_best_traditional_benchmarks(paper_main))
        else:
            resolved.append(key)
    return _dedupe_strings(resolved)


def _best_traditional_benchmarks(paper_main: pd.DataFrame) -> list[str]:
    if paper_main.empty or "model_name" not in paper_main.columns:
        return []
    if "baseline_family" not in paper_main.columns:
        return []
    rankable_mask = (
        paper_main["rankable_in_unified_table"].map(_truthy)
        if "rankable_in_unified_table" in paper_main.columns
        else pd.Series(True, index=paper_main.index)
    )
    frame = paper_main.loc[
        paper_main["baseline_family"].astype(str).eq(TRADITIONAL_BASELINE_FAMILY) & rankable_mask
    ].copy()
    if frame.empty:
        return []
    metric_name = _first_existing_column(frame, ("report_sharpe", "sharpe", "calmar", "cumulative_return"))
    if metric_name is None:
        return []
    frame["_benchmark_score"] = pd.to_numeric(frame[metric_name], errors="coerce")
    frame = frame.dropna(subset=["_benchmark_score"])
    if frame.empty:
        return []
    if "source_run" in frame.columns:
        idx = frame.groupby("source_run")["_benchmark_score"].idxmax()
        return [str(item) for item in frame.loc[idx, "paper_model_id"].dropna().tolist()]
    return [str(frame.sort_values("_benchmark_score", ascending=False).iloc[0]["paper_model_id"])]


def _seed_metric_columns(paper_main: pd.DataFrame, metric_columns: Sequence[str] | None) -> tuple[str, ...]:
    if metric_columns:
        requested = _dedupe_strings(metric_columns)
        if requested == ("all",):
            return tuple(
                column
                for column in paper_main.columns
                if pd.api.types.is_numeric_dtype(pd.to_numeric(paper_main[column], errors="coerce"))
            )
        return tuple(column for column in requested if column in paper_main.columns)
    return tuple(column for column in DEFAULT_PAPER_SEED_METRICS if column in paper_main.columns)


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in result:
            result.append(item)
    return tuple(result)


def _rankable_models(paper_main: pd.DataFrame) -> set[str]:
    paper_main = _with_paper_identity(paper_main)
    if paper_main.empty or "paper_model_id" not in paper_main.columns:
        return set()
    if "rankable_in_unified_table" not in paper_main.columns:
        return {str(item) for item in paper_main["paper_model_id"].dropna().unique()}
    frame = paper_main.loc[paper_main["rankable_in_unified_table"].map(_truthy)]
    return {str(item) for item in frame["paper_model_id"].dropna().unique()}


def _with_paper_identity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and len(frame.columns) == 0:
        return frame.copy()
    result = frame.copy()
    if "_paper_model_id_explicit" not in result.columns:
        if "paper_model_id" in result.columns:
            result["_paper_model_id_explicit"] = result["paper_model_id"].map(lambda value: bool(_clean_value(value)))
        else:
            result["_paper_model_id_explicit"] = False
    else:
        result["_paper_model_id_explicit"] = result["_paper_model_id_explicit"].map(_truthy)
    if "source_experiment" not in result.columns:
        result["source_experiment"] = "unknown"
    if "paper_group_id" not in result.columns:
        result["paper_group_id"] = result["source_run"] if "source_run" in result.columns else "default"
    if "model_name" not in result.columns:
        result["model_name"] = pd.NA
    result["paper_model_id"] = [_paper_model_id(row) for _, row in result.iterrows()]
    missing_model = result["model_name"].isna() | result["model_name"].astype(str).str.strip().eq("")
    result.loc[missing_model, "model_name"] = result.loc[missing_model, "paper_model_id"]
    return result


def _paper_model_id(row: pd.Series) -> str:
    explicit = _clean_value(row.get("paper_model_id"))
    explicit_is_source = "_paper_model_id_explicit" not in row or _truthy(row.get("_paper_model_id_explicit"))
    if explicit and explicit_is_source:
        return explicit
    if _is_p3_main_full_model(row):
        return "full_model"
    variant_id = _clean_value(row.get("variant_id"))
    if variant_id:
        if variant_id == "current":
            return _current_variant_label(row)
        if variant_id == "base":
            return "full_model"
        return variant_id
    hpo_model = _clean_value(row.get("hpo_model_name"))
    if hpo_model:
        return hpo_model
    model_name = _clean_value(row.get("model_name"))
    if model_name:
        return model_name
    return "unknown_model"


def _paper_main_rankable_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    paper_model_id = (
        frame["paper_model_id"].map(_clean_value)
        if "paper_model_id" in frame.columns
        else pd.Series("", index=frame.index)
    )
    if "model_name" in frame.columns:
        model_name = frame["model_name"].map(_clean_value)
        mask &= ~(model_name.eq(HYBRID_DQN_OPTIMIZER_ALIAS) & ~paper_model_id.isin(HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES))
    if "paper_model_id" in frame.columns:
        mask &= ~paper_model_id.eq(HYBRID_DQN_OPTIMIZER_ALIAS)
    if "rankable_in_unified_table" in frame.columns:
        mask &= frame["rankable_in_unified_table"].map(_truthy)
    if "diagnostic_status" in frame.columns:
        mask &= ~frame["diagnostic_status"].map(_clean_value).isin(PAPER_MAIN_EXCLUDED_DIAGNOSTIC_STATUSES)
    mask &= ~_paper_diagnostic_reason_series(frame).map(bool)
    return mask


def _paper_diagnostic_reason_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series("", index=frame.index)
    return pd.Series([_paper_diagnostic_reason(row) for _, row in frame.iterrows()], index=frame.index)


def _paper_diagnostic_reason(row: pd.Series) -> str:
    explicit_reason = _clean_value(row.get("reason")) or _clean_value(row.get("fail_reason")) or _clean_value(row.get("skip_reason"))
    diagnostic_status = _clean_value(row.get("diagnostic_status"))
    if diagnostic_status in PAPER_MAIN_EXCLUDED_DIAGNOSTIC_STATUSES:
        return explicit_reason or diagnostic_status
    paper_model_id = _clean_value(row.get("paper_model_id"))
    model_name = _clean_value(row.get("model_name"))
    if paper_model_id == HYBRID_DQN_OPTIMIZER_ALIAS:
        return explicit_reason or "hybrid_dqn_optimizer_alias"
    if model_name == HYBRID_DQN_OPTIMIZER_ALIAS and paper_model_id not in HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES:
        return explicit_reason or "hybrid_dqn_optimizer_alias"
    source_reason = _source_diagnostic_reason(row)
    if source_reason:
        return explicit_reason or source_reason
    seed_grid_reason = _seed_grid_diagnostic_reason(row)
    if seed_grid_reason:
        return explicit_reason or seed_grid_reason
    if _truthy(row.get("proxy_training", False)):
        return explicit_reason or "proxy_training"
    if _truthy(row.get("external_original_implementation", False)):
        return explicit_reason or "external_original_implementation"
    if "rankable_in_unified_table" in row and not _truthy(row.get("rankable_in_unified_table")):
        return explicit_reason or diagnostic_status or "non_rankable"
    status = _clean_value(row.get("status"))
    if status.startswith("failed") or status.startswith("skipped") or status in {"deferred_variant", "needs_paper_confirmation"}:
        return explicit_reason or status
    return ""


def _source_diagnostic_reason(row: pd.Series) -> str:
    source_path = _clean_value(row.get("source_path"))
    source_path_name = Path(source_path).name if source_path else ""
    source = " ".join(
        _clean_value(row.get(name)).lower()
        for name in ("source_experiment", "source_run", "config_path", "run_mode")
    )
    source = f"{source} {source_path_name.lower()}"
    if "smoke" in source:
        return "smoke_run"
    if "pilot" in source:
        return "pilot_run"
    if "diagnostic" in source:
        return "diagnostic_run"
    return ""


def _seed_grid_diagnostic_reason(row: pd.Series) -> str:
    for column in ("seed_grid_complete", "formal_seed_grid_complete", "complete_seed_grid", "full_seed_grid_complete"):
        if column not in row or not _clean_value(row.get(column)):
            continue
        if not _truthy(row.get(column)):
            return "incomplete_seed_grid"
    status = _clean_value(row.get("seed_grid_status")).lower()
    if status and status not in {"complete", "completed", "full", "formal"}:
        return status
    seed_count = _numeric_value(row.get("seed_count"))
    required_seed_count = _numeric_value(row.get("required_seed_count"))
    if seed_count is not None and required_seed_count is not None and seed_count < required_seed_count:
        return "incomplete_seed_grid"
    return ""


def _is_p3_main_full_model(row: pd.Series) -> bool:
    return (
        _clean_value(row.get("paper_group_id")) == "p3_components"
        and not _clean_value(row.get("variant_id"))
        and _clean_value(row.get("model_name")) == "full_dqn_gated_multitask_cnn_ppo"
    )


def _current_variant_label(row: pd.Series) -> str:
    source = " ".join(
        _clean_value(row.get(name)) or ""
        for name in ("source_run", "source_path", "changed_key_path")
    ).lower()
    if "without_dqn_gate" in source or "dqn.enabled" in source:
        return "without_dqn_gate"
    if "without_auxiliary" in source or "auxiliary.enabled" in source:
        return "without_auxiliary"
    if "mlp_encoder" in source or "model.default_encoder" in source or "model.encoder.type" in source:
        return "mlp_encoder"
    if "attention_enabled" in source or "cross_asset_attention" in source:
        return "attention_enabled"
    return "current"


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _column_suffix(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _numeric_value(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _seed_metric_values(group: pd.DataFrame, metric_name: str) -> pd.Series:
    values = pd.to_numeric(group[metric_name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = group.copy()
    frame["_metric_value"] = values
    frame = frame.dropna(subset=["_metric_value"])
    if frame.empty:
        return pd.Series(dtype=float)
    if "seed" in frame.columns and frame["seed"].notna().any():
        fallback = frame.get("source_run", pd.Series(index=frame.index, dtype=object)).astype(str)
        seed_key = fallback + "|seed=" + frame["seed"].astype(str)
        seed_key = seed_key.where(frame["seed"].notna(), fallback)
    elif "source_run" in frame.columns:
        seed_key = frame["source_run"]
    else:
        seed_key = pd.Series(range(len(frame)), index=frame.index)
    frame["_seed_unit"] = seed_key.astype(str)
    return frame.groupby("_seed_unit", sort=False)["_metric_value"].mean().dropna()


def _comparison_file_priority(filename: str) -> int:
    return int(COMPARISON_FILE_PRIORITY.get(str(filename), 100))


def _daily_return_file_priority(filename: str) -> int:
    return int(DAILY_RETURN_FILE_PRIORITY.get(str(filename), 100))


def _dedupe_daily_return_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns or "paper_model_id" not in frame.columns:
        return frame.drop(columns=["_source_file_priority"], errors="ignore")
    result = frame.copy()
    if "_source_file_priority" not in result.columns:
        result["_source_file_priority"] = result.get("source_file", pd.Series("", index=result.index)).map(
            _daily_return_file_priority
        )
    result["_date_key"] = pd.to_datetime(result["date"], errors="coerce")
    scope_col = "paper_group_id" if "paper_group_id" in result.columns else "source_run"
    keys = [scope_col, "paper_model_id", "_date_key"]
    keys.extend(column for column in ("seed", "fold_id") if column in result.columns)
    sort_cols = [*keys, "_source_file_priority"]
    if "source_run" in result.columns:
        sort_cols.append("source_run")
    result = result.sort_values(sort_cols, kind="mergesort")
    result = result.drop_duplicates(subset=keys, keep="first")
    return result.drop(columns=["_source_file_priority", "_date_key"], errors="ignore")


def _dedupe_comparison_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.drop(columns=["_source_file_priority"], errors="ignore")
    result = frame.copy()
    if "_source_file_priority" not in result.columns:
        result["_source_file_priority"] = result.get("source_file", pd.Series("", index=result.index)).map(_comparison_file_priority)
    scope_col = "paper_group_id" if "paper_group_id" in result.columns else "source_run"
    keys = [column for column in (scope_col, "paper_model_id", "seed", "fold_id") if column in result.columns]
    if len(keys) < 2:
        return result.drop(columns=["_source_file_priority"], errors="ignore")
    result = result.sort_values([*keys, "_source_file_priority"], kind="mergesort")
    result = result.drop_duplicates(subset=keys, keep="first")
    return result.drop(columns=["_source_file_priority"], errors="ignore")


def _not_applicable_stats(reason: str) -> pd.DataFrame:
    row = {column: pd.NA for column in STATISTICS_SUMMARY_COLUMNS}
    row["test_name"] = "all"
    row["status"] = "not_applicable"
    row["skip_reason"] = reason
    return pd.DataFrame([row], columns=STATISTICS_SUMMARY_COLUMNS)


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs" / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _experiment_result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logs" / "experiment_result.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _source_run(run_dir: Path, manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("run_name") or manifest.get("run_id") or run_dir.name)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "nan", "none"}
    if pd.isna(value):
        return False
    return bool(value)


def _first_value(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return None if values.empty else values.iloc[0]


def _group_label(group_key: Any) -> str:
    if isinstance(group_key, tuple):
        return "|".join(str(item) for item in group_key)
    return str(group_key)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paper-level experiment tables.")
    parser.add_argument("--run-dir", action="append", required=True, help="Experiment run directory. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Directory for paper_*.csv outputs.")
    parser.add_argument(
        "--benchmark-model",
        action="append",
        dest="benchmark_models",
        help="Benchmark model for paired statistics. Repeatable. Defaults to paper protocol benchmarks.",
    )
    parser.add_argument(
        "--seed-metric",
        action="append",
        dest="seed_metric_columns",
        help="Metric column to include in paper_seed_summary.csv. Repeatable; use 'all' for all numeric columns.",
    )
    parser.add_argument(
        "--paper-group-id",
        help="Optional shared group id for paired statistics across multiple run directories.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    return aggregate_paper_results(
        args.run_dir,
        args.output_dir,
        benchmark_models=args.benchmark_models,
        seed_metric_columns=args.seed_metric_columns,
        paper_group_id=args.paper_group_id,
    )


if __name__ == "__main__":
    main()
