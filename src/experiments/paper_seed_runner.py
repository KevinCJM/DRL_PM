from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from src.config import ConfigLoader
from src.experiments.paper_aggregate import aggregate_paper_results


def run_seed_grid(
    config_path: str | Path,
    *,
    seeds: Sequence[int],
    run_name_prefix: str,
    output_root: str | Path | None = None,
    aggregate_output_dir: str | Path | None = None,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("ERR_SEED_GRID_EMPTY")
    run_dirs: list[Path] = []
    base_output = Path(output_root) if output_root is not None else Path(ConfigLoader.load(config_path)["output"]["root"])
    for seed in seeds:
        run_name = f"{run_name_prefix}_s{int(seed)}"
        command = [
            sys.executable,
            "-m",
            "src.experiments.run_experiment",
            "--config",
            str(config_path),
            "--seed",
            str(int(seed)),
            "--run-name",
            run_name,
        ]
        if output_root is not None:
            command.extend(["--output", str(output_root)])
        subprocess.run(command, check=True)
        run_dirs.append(base_output / run_name)
    outputs = None
    if aggregate_output_dir is not None:
        outputs = aggregate_paper_results(run_dirs, aggregate_output_dir)
    return {"run_dirs": run_dirs, "aggregate_outputs": outputs}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper experiment config over an explicit seed grid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 42,123,2024,3407,9999")
    parser.add_argument("--run-name-prefix", required=True)
    parser.add_argument("--output")
    parser.add_argument("--aggregate-output-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    return run_seed_grid(
        args.config,
        seeds=seeds,
        run_name_prefix=args.run_name_prefix,
        output_root=args.output,
        aggregate_output_dir=args.aggregate_output_dir,
    )


if __name__ == "__main__":
    main()
