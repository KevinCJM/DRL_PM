from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config import ConfigLoader, PROJECT_ROOT
from src.experiments.paper_aggregate import aggregate_paper_results
from src.utils.logger import save_json_atomic


PAPER_PILOT_CONFIG_SEQUENCE: tuple[str, ...] = (
    "configs/paper/p0_native_baseline_smoke.yaml",
    "configs/paper/hpo_equal_budget_native_pilot.yaml",
    "configs/paper/baseline_comparison_native.yaml",
)
PAPER_FORMAL_CONFIG_SEQUENCE: tuple[str, ...] = (
    "configs/paper/main_model.yaml",
    "configs/paper/baseline_comparison_native.yaml",
    "configs/paper/hpo_equal_budget_main_native.yaml",
    "configs/paper/input_matrix_ablation.yaml",
    "configs/paper/pca_ablation.yaml",
    "configs/paper/ablation_without_dqn_gate.yaml",
    "configs/paper/ablation_without_auxiliary.yaml",
    "configs/paper/ablation_mlp_encoder.yaml",
    "configs/paper/ablation_attention_enabled.yaml",
    "configs/paper/reward_ablation.yaml",
    "configs/paper/transaction_cost_sensitivity.yaml",
    "configs/paper/rebalance_frequency_analysis.yaml",
    "configs/paper/seed_stability.yaml",
    "configs/paper/asset_universe_sensitivity.yaml",
    "configs/paper/market_regime.yaml",
    "configs/paper/walk_forward.yaml",
    "configs/paper/preference_conditioned_analysis.yaml",
    "configs/paper/uncertainty_analysis.yaml",
    "configs/paper/distributional_cvar_analysis.yaml",
    "configs/paper/partial_rebalance_analysis.yaml",
)
PAPER_FULL_PROFILES = {
    "pilot": PAPER_PILOT_CONFIG_SEQUENCE,
    "formal": PAPER_FORMAL_CONFIG_SEQUENCE,
}
PAPER_AGGREGATE_SCOPES: dict[str, dict[str, tuple[str, ...]]] = {
    "pilot": {
        "pilot_hpo": ("hpo_equal_budget_native_pilot",),
        "pilot_fixed": ("p0_native_baseline_smoke", "baseline_comparison_native"),
    },
    "formal": {
        "main_hpo": ("hpo_equal_budget_main_native", "baseline_comparison_native"),
        "main_fixed": ("main_model", "baseline_comparison_native"),
        "p2_input_pca": ("input_matrix_ablation", "pca_ablation"),
        "p3_components": (
            "main_model",
            "ablation_without_dqn_gate",
            "ablation_without_auxiliary",
            "ablation_mlp_encoder",
            "ablation_attention_enabled",
        ),
        "p4_reward": ("reward_ablation",),
        "p5_cost_rebalance": ("transaction_cost_sensitivity", "rebalance_frequency_analysis"),
        "p6_robustness": ("seed_stability", "asset_universe_sensitivity", "market_regime", "walk_forward"),
        "p8_modules": (
            "preference_conditioned_analysis",
            "uncertainty_analysis",
            "distributional_cvar_analysis",
            "partial_rebalance_analysis",
        ),
    },
}


def run_paper_full(
    *,
    profile: str = "formal",
    configs: Sequence[str | Path] | None = None,
    output_root: str | Path | None = None,
    run_prefix: str = "PAPER_FULL",
    aggregate_output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_configs = _resolve_config_sequence(profile=profile, configs=configs)
    resolved_output_root = _output_root(output_root, selected_configs)
    run_specs = [_run_spec(path, run_prefix, resolved_output_root) for path in selected_configs]

    for spec in run_specs:
        if dry_run:
            continue
        subprocess.run(spec["command"], check=True)

    aggregate_outputs = None
    if not dry_run and aggregate_output_dir is not None:
        aggregate_outputs = _aggregate_scoped_outputs(profile, run_specs, aggregate_output_dir)

    payload = {
        "status": "dry_run" if dry_run else "completed",
        "profile": profile,
        "run_prefix": run_prefix,
        "output_root": str(resolved_output_root),
        "run_count": len(run_specs),
        "runs": [
            {
                "config_path": str(spec["config_path"]),
                "run_name": spec["run_name"],
                "run_dir": str(spec["run_dir"]),
                "command": list(spec["command"]),
            }
            for spec in run_specs
        ],
        "aggregate_output_dir": None if aggregate_output_dir is None else str(Path(aggregate_output_dir)),
        "aggregate_outputs": aggregate_outputs,
    }
    manifest_dir = Path(aggregate_output_dir) if aggregate_output_dir is not None else resolved_output_root
    save_json_atomic(payload, manifest_dir / f"{run_prefix}_paper_full_manifest.json")
    return payload


def _resolve_config_sequence(
    *,
    profile: str,
    configs: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    selected = tuple(str(path) for path in configs) if configs else PAPER_FULL_PROFILES.get(str(profile))
    if selected is None:
        raise ValueError(f"ERR_PAPER_FULL_PROFILE_UNKNOWN: {profile}")
    paths = tuple(_resolve_config_path(path) for path in selected)
    if not paths:
        raise ValueError("ERR_PAPER_FULL_EMPTY_CONFIG_SEQUENCE")
    return paths


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"ERR_PAPER_FULL_CONFIG_OUT_OF_SCOPE: {resolved}") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"ERR_PAPER_FULL_CONFIG_NOT_FOUND: {resolved}")
    return resolved


def _output_root(output_root: str | Path | None, configs: Sequence[Path]) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    config = ConfigLoader.load(configs[0])
    return Path(config["output"]["root"]).expanduser().resolve()


def _run_spec(config_path: Path, run_prefix: str, output_root: Path) -> dict[str, Any]:
    ConfigLoader.load(config_path)
    run_name = _run_name(run_prefix, config_path)
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_experiment",
        "--config",
        str(config_path),
        "--output",
        str(output_root),
        "--run-name",
        run_name,
    ]
    return {
        "config_path": config_path,
        "config_stem": config_path.stem,
        "run_name": run_name,
        "run_dir": output_root / run_name,
        "command": command,
    }


def _run_name(run_prefix: str, config_path: Path) -> str:
    stem = "".join(char if char.isalnum() else "_" for char in config_path.stem).strip("_")
    prefix = "".join(char if char.isalnum() or char in {"_", ".", "-"} else "_" for char in str(run_prefix)).strip("._")
    return f"{prefix}_{stem}" if prefix else stem


def _aggregate_scoped_outputs(
    profile: str,
    run_specs: Sequence[Mapping[str, Any]],
    aggregate_output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    target = Path(aggregate_output_dir).expanduser().resolve()
    scopes = PAPER_AGGREGATE_SCOPES.get(str(profile))
    if scopes is None:
        scopes = {"custom": tuple(str(spec.get("config_stem", "")) for spec in run_specs)}
    outputs: dict[str, dict[str, str]] = {}
    for scope_name, stems in scopes.items():
        selected = [spec for spec in run_specs if str(spec.get("config_stem")) in set(stems)]
        if not selected:
            continue
        paths = aggregate_paper_results(
            [spec["run_dir"] for spec in selected],
            target / scope_name,
            paper_group_id=scope_name,
        )
        outputs[scope_name] = {name: str(path) for name, path in paths.items()}
    return outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen paper experiment config sequence.")
    parser.add_argument("--profile", choices=sorted(PAPER_FULL_PROFILES), default="formal")
    parser.add_argument("--config", action="append", dest="configs", help="Override config sequence. Repeatable.")
    parser.add_argument("--output")
    parser.add_argument("--run-prefix", default="PAPER_FULL")
    parser.add_argument("--aggregate-output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_paper_full(
        profile=args.profile,
        configs=args.configs,
        output_root=args.output,
        run_prefix=args.run_prefix,
        aggregate_output_dir=args.aggregate_output_dir,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "PAPER_FORMAL_CONFIG_SEQUENCE",
    "PAPER_AGGREGATE_SCOPES",
    "PAPER_FULL_PROFILES",
    "PAPER_PILOT_CONFIG_SEQUENCE",
    "run_paper_full",
]
