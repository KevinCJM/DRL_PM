from __future__ import annotations

from typing import Any


PGPORTFOLIO_EXTERNAL_MODEL_NAME = "pgportfolio_original_external"
PGPORTFOLIO_EXTERNAL_ALGORITHM = "pgportfolio_original"


def external_pgportfolio_summary(status: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": PGPORTFOLIO_EXTERNAL_MODEL_NAME,
        "baseline_family": "external_original",
        "status": str(status),
        "training_algorithm": PGPORTFOLIO_EXTERNAL_ALGORITHM,
        "rl_training": True,
        "platform_native_rl_training": False,
        "proxy_training": False,
        "external_original_implementation": True,
        "external_repo": "https://github.com/ZhengyaoJiang/PGPortfolio",
        "external_license": "GPL-3.0",
        "external_dependency_stack": "tensorflow1/tflearn",
        "rankable_in_unified_table": False,
        "source_code_vendored": False,
        "license": "GPL-3.0",
        "data_protocol": "external_export_import",
        "execution_protocol": "pgportfolio_original_external",
        "evaluation_protocol": "pgportfolio_original_external",
        "cost_model_shared": False,
        "cost_availability": "not_available",
        "constraint_protocol_shared": False,
    }
    row.update(overrides)
    return row


__all__ = [
    "PGPORTFOLIO_EXTERNAL_ALGORITHM",
    "PGPORTFOLIO_EXTERNAL_MODEL_NAME",
    "external_pgportfolio_summary",
]
