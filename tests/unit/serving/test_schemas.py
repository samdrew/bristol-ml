"""Spec-derived tests for the Stage 12 T6 PredictRequest / PredictResponse schemas.

Every test here is derived from:

- ``docs/plans/active/12-serving.md`` §6 T6 named tests:
  ``test_predict_request_rejects_naive_datetime`` and
  ``test_predict_request_round_trips_features_dict``.
- ``docs/plans/active/12-serving.md`` §5 schema sketch (the
  ``PredictRequest`` / ``PredictResponse`` shapes).
- Plan D8: ``AwareDatetime`` for ``target_dt``; naive datetimes
  rejected at the validation boundary.
- Plan D9 (Ctrl+G reversal): every model family is eligible as a
  ``run_id`` value — no special-casing of ``nn_temporal``.

No production code is modified here.  If any test below fails, the
failure points at a deviation from the plan — do not weaken the test;
surface the failure to the implementer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from bristol_ml.serving.schemas import PredictRequest, PredictResponse


def test_predict_request_rejects_naive_datetime() -> None:
    """Guards Stage 12 D8: naive ``target_dt`` raises a Pydantic ValidationError.

    The schema declares ``target_dt: AwareDatetime`` so a naive
    (tz-less) datetime must fail validation at the model boundary —
    which is what FastAPI surfaces to the HTTP caller as 422.  This
    test exercises the Pydantic side directly so the rule is asserted
    independently of the HTTP framework.

    Cited criterion: plan §6 T6 named test
    ``test_predict_request_rejects_naive_datetime`` (also AC-2c via T7).
    """
    naive_dt = datetime(2025, 6, 15, 13, 0, 0)  # no tzinfo
    assert naive_dt.tzinfo is None  # sanity — the test premise

    with pytest.raises(ValidationError) as excinfo:
        PredictRequest(
            target_dt=naive_dt,  # type: ignore[arg-type]  # testing the validator
            features={"hour_sin": 0.5, "hour_cos": 0.5},
            run_id=None,
        )

    # The ValidationError must point at ``target_dt`` so the operator
    # sees which field failed; the exact message text is owned by
    # Pydantic and pinned only loosely.
    err_text = str(excinfo.value)
    assert "target_dt" in err_text, (
        f"ValidationError must name the offending field ``target_dt``; got: {err_text!r}"
    )


def test_predict_request_round_trips_features_dict() -> None:
    """Guards Stage 12 D4 / T6: the ``features`` field round-trips a feature dict.

    A typical Stage-3-feature row is a small ``dict[str, float]`` of
    assembled feature values keyed by ``feature_columns`` entries.
    ``PredictRequest`` must accept the dict, expose it unchanged on the
    instance, and survive a JSON round-trip via ``model_dump`` /
    ``model_validate`` so the FastAPI handler can rely on
    feature_dict-in-feature_dict-out semantics.

    The test also asserts:
    - ``run_id=None`` is accepted (D7 default-resolution path).
    - ``run_id="some-explicit-id"`` is accepted as a plain string (D9
      reversal: every family — including ``nn_temporal`` — is a valid
      value, no special-casing).
    - An aware datetime is preserved exactly through the round-trip
      (D8 — the handler normalises to UTC at its own boundary).

    Cited criterion: plan §6 T6 named test
    ``test_predict_request_round_trips_features_dict``.
    """
    features = {
        "hour_sin": 0.5,
        "hour_cos": -0.5,
        "temperature_c_mean": 12.34,
        "is_holiday": 0.0,
    }
    target_dt = datetime(2025, 6, 15, 13, 0, 0, tzinfo=UTC)

    # Default resolution path: run_id omitted entirely.
    req_default = PredictRequest(target_dt=target_dt, features=features)
    assert req_default.target_dt == target_dt
    assert req_default.features == features, (
        f"features dict must round-trip unchanged; got {req_default.features!r}, "
        f"expected {features!r}."
    )
    assert req_default.run_id is None, (
        f"D7: run_id must default to None when omitted; got {req_default.run_id!r}."
    )

    # Explicit run_id path: every family is eligible after the D9
    # reversal — nothing about the schema special-cases nn_temporal.
    req_explicit = PredictRequest(
        target_dt=target_dt,
        features=features,
        run_id="nn-temporal-2026-04-25-abc123",
    )
    assert req_explicit.run_id == "nn-temporal-2026-04-25-abc123"

    # JSON round-trip: a serialised request must validate back to an
    # equal instance.  This is the contract FastAPI relies on at the
    # body-parsing boundary.
    payload = req_explicit.model_dump(mode="json")
    assert payload["features"] == features, (
        f"features dict must survive model_dump(mode='json'); got {payload['features']!r}."
    )
    rebuilt = PredictRequest.model_validate(payload)
    assert rebuilt == req_explicit, (
        f"JSON round-trip must produce an equal PredictRequest; "
        f"got {rebuilt!r}, expected {req_explicit!r}."
    )

    # Non-UTC aware datetimes are also accepted (the handler will
    # normalise to UTC via ``.astimezone(UTC)``); we assert the schema
    # itself does not reject a non-UTC offset, because rejecting them
    # at the schema would surprise callers in BST-like timezones.
    bst = timezone(timedelta(hours=1))
    bst_dt = datetime(2025, 6, 15, 14, 0, 0, tzinfo=bst)
    PredictRequest(target_dt=bst_dt, features=features)


def test_predict_response_round_trips_through_json() -> None:
    """Guards Stage 12 §5 PredictResponse shape: prediction + run_id + model_name + target_dt.

    A round-trip through ``model_dump(mode='json')`` and
    ``model_validate`` must preserve the four fields exactly.  This is
    the contract FastAPI exposes to the caller via OpenAPI (NFR-1 /
    AC-4); a regression here would surface as a schema mismatch in
    the auto-emitted JSON Schema.
    """
    response = PredictResponse(
        prediction=42.5,
        run_id="naive-2026-04-25-deadbeef",
        model_name="naive-same-hour-last-week",
        target_dt=datetime(2025, 6, 15, 13, 0, 0, tzinfo=UTC),
    )
    payload = response.model_dump(mode="json")
    assert payload["prediction"] == 42.5
    assert payload["run_id"] == "naive-2026-04-25-deadbeef"
    assert payload["model_name"] == "naive-same-hour-last-week"

    rebuilt = PredictResponse.model_validate(payload)
    assert rebuilt == response


def test_predict_request_rejects_extra_fields() -> None:
    """Guards ``ConfigDict(extra='forbid')`` on PredictRequest (Stage 12 §5).

    Typos in the request body must fail validation rather than be
    silently dropped.  This guards the AC-2b contract (clear error on
    invalid input) at the Pydantic layer; FastAPI surfaces the failure
    as HTTP 422.
    """
    target_dt = datetime(2025, 6, 15, 13, 0, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        PredictRequest(
            target_dt=target_dt,
            features={"hour_sin": 0.5},
            bogus_field="should_fail",  # type: ignore[call-arg]  # testing extra="forbid"
        )
