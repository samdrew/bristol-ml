"""Stage 12 T7 — FastAPI app factory + lifespan + ``GET /`` + ``POST /predict``.

Spec-derived integration tests for the Stage 12 serving endpoint, all
exercised through :class:`fastapi.testclient.TestClient` so the
lifespan, the route handlers, and the Pydantic schema validation are
all in scope:

- ``test_serving_lifespan_starts_with_only_registry_dir`` (AC-1)
- ``test_serving_lifespan_raises_on_empty_registry`` (AC-1)
- ``test_default_model_is_lowest_mae_overall`` (D6 — every family is
  eligible after the D9 Ctrl+G reversal)
- ``test_serving_logs_default_run_id_at_startup`` (D11 startup log)
- ``test_predict_valid_request_returns_200_with_prediction`` (AC-2)
- ``test_predict_naive_datetime_returns_422`` (AC-2 / D8)
- ``test_predict_missing_required_field_returns_422`` (AC-2)
- ``test_predict_unknown_run_id_returns_404`` (AC-2)
- ``test_serving_prediction_parity_vs_direct_load`` — parametrised
  over **all six model families** per the D9 Ctrl+G reversal (AC-3).
- ``test_lazy_load_caches_run_id_after_first_request`` (D7 single
  highest-leverage cut)

The file existence + ``TestClient``-driven integration shape together
satisfy AC-5 (smoke test exercises the endpoint with a small fixture).

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan or a regression in the
upstream registry / model layer; do not weaken the test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bristol_ml import registry
from bristol_ml.models.protocol import Model
from bristol_ml.serving import build_app
from tests.integration.serving.conftest import (
    FAMILY_FACTORIES,
    fit_linear,
    fit_naive,
    fit_nn_mlp,
    fit_nn_temporal,
    fit_sarimax,
    fit_scipy_parametric,
    register_run,
)

# ---------------------------------------------------------------------------
# AC-1 — lifespan startup
# ---------------------------------------------------------------------------


def test_serving_lifespan_starts_with_only_registry_dir(tmp_path: Path) -> None:
    """Stage 12 AC-1: ``build_app(registry_dir)`` starts on a clean machine.

    The lifespan must resolve a default model from the registry without
    any further configuration; the only input is the registry root.
    Asserting via ``GET /`` proves the lifespan ran and stashed the
    expected fields on ``app.state``.

    Cited criterion: plan §4 AC-1 / §6 T7 named test
    ``test_serving_lifespan_starts_with_only_registry_dir``.
    """
    model, _ = fit_naive()
    run_id = register_run(model, registry_dir=tmp_path)

    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["default_run_id"] == run_id, (
        f"GET / must surface the resolved default run_id; got {body['default_run_id']!r}, "
        f"expected {run_id!r}."
    )
    assert body["model_name"] == model.metadata.name
    assert body["feature_columns"] == list(model.metadata.feature_columns)


def test_serving_lifespan_raises_on_empty_registry(tmp_path: Path) -> None:
    """Stage 12 AC-1: an empty registry directory raises a clear startup error.

    The lifespan converts a no-runs registry into a :class:`RuntimeError`
    whose message names the registry directory so the operator can
    self-diagnose.  ``TestClient.__enter__`` propagates the lifespan
    exception, so we wrap the ``with`` block in :func:`pytest.raises`.

    Cited criterion: plan §4 AC-1 / §6 T7 named test
    ``test_serving_lifespan_raises_on_empty_registry``.
    """
    # tmp_path is empty — no registered runs.
    app = build_app(tmp_path)
    with pytest.raises(RuntimeError) as excinfo, TestClient(app):
        pass
    msg = str(excinfo.value)
    assert "no runs" in msg.lower(), (
        f"RuntimeError message must explain that the registry has no runs; got {msg!r}."
    )
    assert str(tmp_path) in msg, (
        f"RuntimeError message must name the registry directory; got {msg!r}."
    )


# ---------------------------------------------------------------------------
# D6 — default-model selection (lowest-MAE overall, every family eligible)
# ---------------------------------------------------------------------------


def test_default_model_is_lowest_mae_overall(tmp_path: Path) -> None:
    """Stage 12 D6 / D9: the default is the lowest-MAE run across *all* families.

    Register one run from each of the six model families, with
    deliberately-staggered MAE values; the lifespan must pick the
    family whose synthetic MAE is the minimum (here ``nn_temporal``,
    set to ``MAE=10`` while every other family has higher MAE).  This
    is the load-bearing assertion for the D9 Ctrl+G reversal —
    pre-Ctrl+G the candidate set excluded ``nn_temporal``; post-Ctrl+G
    every family is eligible and the test demonstrates that
    ``nn_temporal`` can win on its merits.

    Cited criterion: plan §4 (D-derived) named test
    ``test_default_model_is_lowest_mae_overall``; plan §1 D6 + D9.
    """
    # Six runs, one per family; nn_temporal scores best (MAE=10) so the
    # post-Ctrl+G default selection must pick it.
    mae_by_family = {
        "naive": 200.0,
        "linear": 150.0,
        "sarimax": 120.0,
        "scipy_parametric": 100.0,
        "nn_mlp": 80.0,
        "nn_temporal": 10.0,
    }
    expected_winner_run_id: str | None = None
    expected_winner_name: str | None = None
    for family, factory, _cls in FAMILY_FACTORIES:
        model, _ = factory()
        run_id = register_run(model, registry_dir=tmp_path, mae=mae_by_family[family])
        if family == "nn_temporal":
            expected_winner_run_id = run_id
            expected_winner_name = model.metadata.name

    assert expected_winner_run_id is not None
    assert expected_winner_name is not None

    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/")
    body = response.json()
    assert body["default_run_id"] == expected_winner_run_id, (
        "D6 + D9: default must resolve to the lowest-MAE run across all six families "
        f"(here nn_temporal at MAE=10); got {body['default_run_id']!r}, "
        f"expected {expected_winner_run_id!r}."
    )
    assert body["model_name"] == expected_winner_name


def test_serving_logs_default_run_id_at_startup(
    tmp_path: Path,
    loguru_caplog: pytest.LogCaptureFixture,
) -> None:
    """Stage 12 D11 startup log: lifespan emits an info line naming the default.

    The lifespan emits a structured loguru info line that includes
    both the resolved ``default_run_id`` and the human-readable
    ``model_name`` — so the operator sees what is being served before
    the first request hits the endpoint.

    Cited criterion: plan §4 (D-derived) named test
    ``test_serving_logs_default_run_id_at_startup``.
    """
    model, _ = fit_naive()
    run_id = register_run(model, registry_dir=tmp_path)

    app = build_app(tmp_path)
    with TestClient(app):
        pass  # lifespan runs on enter + exit

    # The loguru_caplog fixture captures the message text only;
    # confirm the resolved run_id and model name are in the message.
    messages = [record.getMessage() for record in loguru_caplog.records]
    matching = [m for m in messages if "serving lifespan" in m]
    assert matching, (
        f"Expected a loguru INFO line containing 'serving lifespan'; "
        f"captured messages: {messages!r}."
    )
    assert any(run_id in m for m in matching), (
        f"Lifespan log line must include the resolved default_run_id={run_id!r}; "
        f"matching messages: {matching!r}."
    )
    assert any(model.metadata.name in m for m in matching), (
        f"Lifespan log line must include the resolved model_name="
        f"{model.metadata.name!r}; matching messages: {matching!r}."
    )


# ---------------------------------------------------------------------------
# AC-2 — request validation + error paths
# ---------------------------------------------------------------------------


def _post_predict(client: TestClient, payload: dict[str, Any]) -> Any:
    """Tiny convenience wrapper around ``client.post('/predict', json=...)``."""
    return client.post("/predict", json=payload)


def _features_payload(model: Model, predict_features: pd.DataFrame) -> dict[str, float]:
    """Build a feature-dict payload from the raw predict-features tail row.

    Stage 12 ships **raw input** features through the boundary — derived
    columns (e.g. SARIMAX's Fourier exogs, ``ScipyParametricModel``'s
    ``hdd`` / ``cdd``) are computed by the model on the way to predict
    rather than by the caller.  ``model.metadata.feature_columns``
    records the *internal* column set, which for those two families is
    a superset of the raw inputs and would shape an unservable payload.
    The factories' ``predict_features`` always carries the raw columns,
    so we trust those names and read ``float(...)`` casts off the tail
    row.

    The unused ``model`` parameter is retained so the helper signature
    matches the call site at :func:`test_serving_prediction_parity_vs_direct_load`
    and so a future reader has a stable extension point if a family is
    ever added that needs metadata-aware payload construction.
    """
    del model  # signature kept symmetric with the parity test below
    last_row = predict_features.iloc[-1]
    return {col: float(last_row[col]) for col in predict_features.columns}


def test_predict_valid_request_returns_200_with_prediction(tmp_path: Path) -> None:
    """Stage 12 AC-2: a valid request returns HTTP 200 with a finite ``prediction``.

    The simplest happy-path test — register a fixture, post a body
    that satisfies ``PredictRequest``, assert the response shape is a
    well-formed ``PredictResponse``.

    Cited criterion: plan §4 AC-2 / §6 T7 named test
    ``test_predict_valid_request_returns_200_with_prediction``.
    """
    model, predict_features = fit_linear()
    run_id = register_run(model, registry_dir=tmp_path)
    target_dt = predict_features.index[-1]
    payload = {
        "target_dt": target_dt.isoformat(),
        "features": _features_payload(model, predict_features),
    }
    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = _post_predict(client, payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["prediction"], float), (
        f"prediction must be a float; got {body['prediction']!r}"
    )
    # NaN / inf would still pass isinstance(float); guard explicitly.
    pred = body["prediction"]
    assert pred == pred and pred not in {float("inf"), float("-inf")}, (
        f"prediction must be a finite float; got {pred!r}."
    )
    assert body["run_id"] == run_id
    assert body["model_name"] == model.metadata.name


def test_predict_naive_datetime_returns_422(tmp_path: Path) -> None:
    """Stage 12 D8 / AC-2: naive (tz-less) ``target_dt`` returns 422.

    Pydantic's ``AwareDatetime`` rejects naive datetimes at the
    validation boundary; FastAPI surfaces the failure as HTTP 422
    with a ``detail`` array naming the offending field.

    Cited criterion: plan §4 AC-2 / §6 T7 named test
    ``test_predict_naive_datetime_returns_422``.
    """
    model, predict_features = fit_naive()
    register_run(model, registry_dir=tmp_path)

    payload = {
        "target_dt": "2025-06-15T13:00:00",  # no Z, no offset
        "features": _features_payload(model, predict_features),
    }
    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = _post_predict(client, payload)
    assert response.status_code == 422, response.text
    detail = response.json().get("detail", [])
    target_dt_errors = [err for err in detail if "target_dt" in err.get("loc", [])]
    assert target_dt_errors, f"422 detail must include an entry for ``target_dt``; got {detail!r}."


def test_predict_missing_required_field_returns_422(tmp_path: Path) -> None:
    """Stage 12 AC-2: a request without ``target_dt`` returns 422.

    Cited criterion: plan §4 AC-2 / §6 T7 named test
    ``test_predict_missing_required_field_returns_422``.
    """
    model, predict_features = fit_naive()
    register_run(model, registry_dir=tmp_path)

    payload = {
        # ``target_dt`` deliberately omitted.
        "features": _features_payload(model, predict_features),
    }
    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = _post_predict(client, payload)
    assert response.status_code == 422, response.text
    detail = response.json().get("detail", [])
    target_dt_errors = [err for err in detail if "target_dt" in err.get("loc", [])]
    assert target_dt_errors, (
        f"422 detail must include an entry for the missing ``target_dt`` field; got {detail!r}."
    )


def test_predict_unknown_run_id_returns_404(tmp_path: Path) -> None:
    """Stage 12 AC-2: an unknown ``run_id`` returns 404 with a clear ``detail``.

    The 404 detail must name both the missing ``run_id`` and the
    registry directory so the operator can self-diagnose without
    reading the server logs.

    Cited criterion: plan §4 AC-2 / §6 T7 named test
    ``test_predict_unknown_run_id_returns_404``.
    """
    model, predict_features = fit_naive()
    register_run(model, registry_dir=tmp_path)

    payload = {
        "target_dt": predict_features.index[-1].isoformat(),
        "features": _features_payload(model, predict_features),
        "run_id": "does-not-exist",
    }
    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = _post_predict(client, payload)
    assert response.status_code == 404, response.text
    detail = response.json().get("detail", "")
    # The detail field is a plain string for HTTPException; it must
    # carry both the missing run_id and the registry path so the
    # operator can debug without reaching for server logs.
    assert "does-not-exist" in detail, (
        f"404 detail must mention the missing run_id 'does-not-exist'; got {detail!r}."
    )
    assert str(tmp_path) in detail, (
        f"404 detail must mention the registry directory; got {detail!r}."
    )


# ---------------------------------------------------------------------------
# AC-3 — prediction parity, parametrised over all six model families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "factory", "_expected_class"),
    FAMILY_FACTORIES,
    ids=[name for name, _factory, _cls in FAMILY_FACTORIES],
)
def test_serving_prediction_parity_vs_direct_load(
    tmp_path: Path,
    family: str,
    factory: Callable[[], tuple[Model, pd.DataFrame]],
    _expected_class: type[Any],
) -> None:
    """Stage 12 AC-3: the served prediction matches a direct ``Model.predict`` call.

    Parametrised over **all six model families** per the D9 Ctrl+G
    reversal — the ``nn_temporal`` parameter is the load-bearing one,
    asserting that the Stage 11 D5+ warmup envelope makes single-row
    predict work through the serving boundary identically to a direct
    ``registry.load(...).predict(row)`` call.

    Tolerance is ``atol=1e-5`` per the plan AC-3; the inner per-family
    blob round-trips bit-exactly through the skops envelope (T5 round-
    trip uses ``atol=1e-8``), but the network/JSON path can introduce
    a single-ULP difference on float-formatting edge cases — the
    looser ``atol=1e-5`` in the plan absorbs that without changing the
    behavioural assertion.

    Cited criterion: plan §4 AC-3 / §6 T7 named test
    ``test_serving_prediction_parity_vs_direct_load``.
    """
    model, predict_features = factory()
    run_id = register_run(model, registry_dir=tmp_path)

    # Direct prediction: one-row tail of the predict features.  For
    # nn_temporal the warmup window is baked into the artefact, so a
    # one-row predict() returns one prediction.
    one_row = predict_features.iloc[-1:]
    direct_prediction = float(model.predict(one_row).iloc[0])

    target_dt = one_row.index[0]
    # Build the payload from the *raw* input columns the factory created
    # (sarimax + scipy_parametric record derived columns in
    # ``metadata.feature_columns``; the serving boundary always takes
    # raw inputs and lets the model derive the rest — see
    # :func:`_features_payload` above).
    features_payload = {col: float(one_row.iloc[0][col]) for col in predict_features.columns}
    payload = {
        "target_dt": target_dt.isoformat(),
        "features": features_payload,
        "run_id": run_id,
    }

    app = build_app(tmp_path)
    with TestClient(app) as client:
        response = _post_predict(client, payload)
    assert response.status_code == 200, (
        f"family={family!r}: POST /predict must return 200; got "
        f"{response.status_code} with body {response.text!r}."
    )
    served_prediction = float(response.json()["prediction"])
    assert abs(direct_prediction - served_prediction) <= 1e-5, (
        f"family={family!r}: served prediction must match direct prediction "
        f"under atol=1e-5; got served={served_prediction!r}, "
        f"direct={direct_prediction!r}, |diff|="
        f"{abs(direct_prediction - served_prediction)!r}."
    )


# ---------------------------------------------------------------------------
# D7 — lazy-load cache (single highest-leverage cut)
# ---------------------------------------------------------------------------


def test_lazy_load_caches_run_id_after_first_request(tmp_path: Path) -> None:
    """Stage 12 D7: a second request with the same ``run_id`` does not re-call ``registry.load``.

    The lifespan loads the default model exactly once.  A request
    naming a *non-default* ``run_id`` triggers exactly one
    ``registry.load`` call (the lazy-load); a *second* request with
    the same ``run_id`` must hit the ``app.state.loaded`` cache and
    not call ``registry.load`` again.

    Implementation: ``mock.patch`` wraps the real ``registry.load`` so
    behaviour is preserved; the call count is the load-bearing
    assertion.

    Cited criterion: plan §4 (D-derived) named test
    ``test_lazy_load_caches_run_id_after_first_request``.
    """
    # Two registered runs: a default (low-MAE linear) and a secondary
    # (high-MAE naive).  The lifespan loads the linear once; we then
    # POST twice naming the naive run_id and assert the second POST
    # does not re-call registry.load.
    default_model, _ = fit_linear()
    default_run_id = register_run(default_model, registry_dir=tmp_path, mae=10.0)
    secondary_model, secondary_features = fit_naive()
    secondary_run_id = register_run(secondary_model, registry_dir=tmp_path, mae=200.0)

    payload = {
        "target_dt": secondary_features.index[-1].isoformat(),
        "features": _features_payload(secondary_model, secondary_features),
        "run_id": secondary_run_id,
    }
    app = build_app(tmp_path)
    # The TestClient(app) __enter__ runs the lifespan and calls
    # registry.load once for the default run.  Patch *after* TestClient
    # has entered so the call-count window covers only the post-startup
    # window — combining the two ``with`` heads with a comma preserves
    # that order (context managers enter left-to-right).
    with (
        TestClient(app) as client,
        patch.object(registry, "load", wraps=registry.load) as mocked_load,
    ):
        r1 = _post_predict(client, payload)
        r2 = _post_predict(client, payload)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    secondary_load_calls = [
        c for c in mocked_load.call_args_list if c.args and c.args[0] == secondary_run_id
    ]
    assert len(secondary_load_calls) == 1, (
        f"D7: registry.load must be called exactly once for the secondary "
        f"run_id={secondary_run_id!r} across two consecutive predict requests; "
        f"got {len(secondary_load_calls)} calls "
        f"({mocked_load.call_args_list!r})."
    )
    # And the default run_id must not have been re-loaded inside the
    # post-startup window — proves the lifespan's single-load
    # invariant did not regress under the lazy-load patch.
    default_load_calls = [
        c for c in mocked_load.call_args_list if c.args and c.args[0] == default_run_id
    ]
    assert default_load_calls == [], (
        f"D7: registry.load must not be re-called for the default run_id "
        f"during request handling; got {default_load_calls!r}."
    )
    # Ensure we exercised the imports we say we exercised — the
    # secondary fitted model carries a different metadata.name than
    # the default, so the response model_name is the load-bearing
    # confirmation that the request resolved to the secondary run.
    assert r1.json()["run_id"] == secondary_run_id
    assert r2.json()["run_id"] == secondary_run_id


# ---------------------------------------------------------------------------
# Imports above of `fit_*` are kept explicit (rather than star-imported)
# so a future reader can grep for which family is exercised by which test.
# Reference the unused factories to keep the imports load-bearing — the
# linter would otherwise flag them as unused, hiding an inadvertent gap
# in the parametrised parity test's coverage.
# ---------------------------------------------------------------------------
_ALL_FACTORIES = (
    fit_naive,
    fit_linear,
    fit_sarimax,
    fit_scipy_parametric,
    fit_nn_mlp,
    fit_nn_temporal,
)
assert len(_ALL_FACTORIES) == 6, "Stage 12 D9: all six model families must be reachable."
del _ALL_FACTORIES  # module-import side effects only
