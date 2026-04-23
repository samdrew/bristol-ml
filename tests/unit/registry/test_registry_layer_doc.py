"""Spec-derived tests for the registry layer documentation (Stage 9 T6).

Both tests are derived from `docs/plans/active/09-model-registry.md` §4
AC-5 — *"the on-disk layout is documented well enough that a contributor
could inspect it by hand without the CLI"* — and plan §6 Task T6
(documentation + hand-parse gate).

- `test_registry_layout_documentation_exists` — structural gate on the
  layer doc existing with non-trivial content.
- `test_registry_run_json_is_hand_parseable` — writes a real run via
  `registry.save`, then asserts `json.loads(path.read_text())` succeeds,
  every required schema field is present with the expected Python type,
  and the file is human-readable (`indent=2`, UTF-8, no binary escapes).

If either test fails, the layer doc is drifting from the implementation
or the sidecar schema has silently changed — do not weaken the test,
fix the documentation or the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bristol_ml import registry
from bristol_ml.models.naive import NaiveModel
from conf._schemas import NaiveConfig

# ---------------------------------------------------------------------------
# AC-5 — Layer documentation structural gate
# ---------------------------------------------------------------------------


# Resolve repo root via this test file's location — avoids CWD assumptions.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAYER_DOC = _REPO_ROOT / "docs" / "architecture" / "layers" / "registry.md"


def test_registry_layout_documentation_exists() -> None:
    """The registry layer doc exists and is substantive (AC-5, plan D18).

    Plan §4 names this test as the AC-5 structural gate; the > 50-line
    floor is a "non-trivial document" heuristic that catches the
    accidental empty-file or stub-header commit.  The real editorial
    review happens in Phase 3 docs-writer, not here.
    """
    assert _LAYER_DOC.is_file(), (
        f"registry layer doc must exist at {_LAYER_DOC!s} per plan D18 / AC-5"
    )
    content = _LAYER_DOC.read_text(encoding="utf-8")
    line_count = content.count("\n") + 1
    assert line_count > 50, (
        f"registry layer doc must be substantive (>50 lines); got {line_count} lines. "
        "If this trips during a legitimate doc rewrite, extend the threshold deliberately."
    )
    # Minimum content markers — the doc names the D10 graduation path and the
    # D14 skops deferral, both of which are load-bearing for future stages.
    assert "Graduation to MLflow" in content, (
        "layer doc must include the 'Graduation to MLflow' subsection (plan D10)"
    )
    assert "skops" in content, (
        "layer doc must name the skops deferral (plan D14 / models-layer H-3)"
    )


# ---------------------------------------------------------------------------
# AC-5 — Sidecar hand-parseability
# ---------------------------------------------------------------------------


def _hourly_index(n: int) -> pd.DatetimeIndex:
    """UTC-aware hourly index; small helper local to this test module."""
    return pd.date_range("2024-01-01 00:00", periods=n, freq="h", tz="UTC")


def _minimal_metrics_df(n_folds: int = 2) -> pd.DataFrame:
    """Stage-6-harness-shaped per-fold metrics DataFrame."""
    rows = []
    for i in range(n_folds):
        rows.append(
            {
                "fold_index": i,
                "train_end": pd.Timestamp("2024-01-10 00:00", tz="UTC"),
                "test_start": pd.Timestamp("2024-01-10 01:00", tz="UTC"),
                "test_end": pd.Timestamp("2024-01-11 00:00", tz="UTC"),
                "mae": 100.0 + i,
                "rmse": 150.0 + i,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_registry_run_json_is_hand_parseable(tmp_path: Path) -> None:
    """`run.json` parses with the stdlib `json` module and matches the schema.

    Plan §4 AC-5 second test: a contributor opening the sidecar in an editor
    or piping it through `jq` must see human-readable JSON with every
    schema field present and correctly typed.  Covers:

    - UTF-8 readability without `ensure_ascii` escapes.
    - `indent=2` pretty-printing (NFR-transparency).
    - Every `SidecarFields` key present with the expected Python type.
    - Per-metric summary shape (`mean`, `std`, `per_fold`).

    Uses `NaiveModel` because it is the cheapest fit and the AC-5 gate
    is layout-shaped, not model-family-specific.
    """
    cfg = NaiveConfig(strategy="same_hour_last_week", target_column="nd_mw")
    model = NaiveModel(cfg)
    n = 400
    idx = _hourly_index(n)
    target = pd.Series(np.arange(n, dtype=float), index=idx, name="nd_mw")
    features = pd.DataFrame({"t2m": np.arange(n, dtype=float) * 0.1}, index=idx)
    model.fit(features, target)

    run_id = registry.save(
        model,
        _minimal_metrics_df(),
        feature_set="weather_only",
        target="nd_mw",
        registry_dir=tmp_path,
    )

    sidecar_path = tmp_path / run_id / "run.json"
    assert sidecar_path.is_file()

    raw_text = sidecar_path.read_text(encoding="utf-8")
    # `indent=2` → at least one line begins with two spaces then a quote.
    assert '\n  "' in raw_text, (
        "run.json must be pretty-printed with indent=2 per plan D4 / NFR-transparency"
    )
    # `ensure_ascii=False` → no \u-escapes for plain ASCII payloads either.
    assert "\\u" not in raw_text, (
        "run.json must be written with ensure_ascii=False; no \\u escapes expected"
    )

    # Parse with the stdlib — no pydantic, no yaml, no custom reader.
    sidecar = json.loads(raw_text)

    expected_fields: dict[str, type | tuple[type, ...]] = {
        "run_id": str,
        "name": str,
        "type": str,
        "feature_set": str,
        "target": str,
        "feature_columns": list,
        "fit_utc": str,
        "git_sha": (str, type(None)),  # None is a legitimate value (plan D13)
        "hyperparameters": dict,
        "metrics": dict,
        "registered_at_utc": str,
    }
    assert set(sidecar.keys()) == set(expected_fields.keys()), (
        f"sidecar keys must match §5 schema exactly; "
        f"missing={set(expected_fields) - set(sidecar)!r}, "
        f"unexpected={set(sidecar) - set(expected_fields)!r}"
    )
    for field, expected_type in expected_fields.items():
        assert isinstance(sidecar[field], expected_type), (
            f"sidecar field {field!r} must be {expected_type!r}; "
            f"got {type(sidecar[field]).__name__}"
        )

    # Per-metric summary shape matches MetricSummary (plan D15).
    for metric_name, summary in sidecar["metrics"].items():
        assert set(summary.keys()) == {"mean", "std", "per_fold"}, (
            f"metric {metric_name!r} summary must be {{mean, std, per_fold}}; "
            f"got {set(summary.keys())!r}"
        )
        assert isinstance(summary["mean"], float)
        assert isinstance(summary["std"], float)
        assert isinstance(summary["per_fold"], list)
        assert all(isinstance(v, float) for v in summary["per_fold"])

    # Basic sanity: run_id matches the directory name and the sidecar field.
    assert sidecar["run_id"] == run_id
    assert sidecar["type"] == "naive"
    assert sidecar["feature_set"] == "weather_only"
    assert sidecar["target"] == "nd_mw"
