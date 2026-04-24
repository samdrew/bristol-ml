"""Spec-derived tests for :func:`bristol_ml.models.nn.temporal._SequenceDataset` — Stage 11 T2.

Every test is derived from:

- ``docs/plans/active/11-complex-nn.md`` §Task T2 (lines 464-471): the three
  T2 dataset named tests (len, getitem, no-eager-materialisation).
- ``src/bristol_ml/models/nn/temporal.py`` ``_SequenceDataset`` docstring
  (lazy-window pattern D7, float32 tensors, ``(seq_len, n_features)`` window
  shape, 0-d scalar target tensor, ``len = N - seq_len``).
- ``docs/plans/active/11-complex-nn.md`` §1 design decision D7 (lazy
  windowing — eager materialisation at the default feature-set shape would
  cost ~1.4 GB, outside the "runs on a laptop" NFR-2 envelope).

No production code is modified here.  If any test below fails the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the orchestrator.

Conventions
-----------
- British English in docstrings and comments.
- Each test docstring cites the plan clause or AC it guards.
- No ``xfail``, no ``skip``, no ``pytest.mark.gpu``.
- Fixtures use deterministic integer values so window slice comparisons are
  byte-exact verifiable without floating-point tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from bristol_ml.models.nn.temporal import _SequenceDataset

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_fixture(
    n_rows: int,
    n_features: int = 3,
    *,
    integer_values: bool = False,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic ``(features, target)`` pair.

    When ``integer_values=True`` every feature cell is an integer cast to
    ``float64`` so that the expected NumPy slice equals ``ds[i][0]`` under
    byte-exact float32 comparison — no accumulation error is possible.
    """
    index = pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC")
    if integer_values:
        rng = np.arange(n_rows * n_features, dtype=np.float64).reshape(n_rows, n_features)
        y = np.arange(n_rows, dtype=np.float64)
    else:
        rng_state = np.random.default_rng(seed)
        rng = rng_state.standard_normal(size=(n_rows, n_features))
        y = rng_state.standard_normal(size=n_rows)
    features = pd.DataFrame(rng, columns=[f"f{i}" for i in range(n_features)], index=index)
    target = pd.Series(y, index=index, name="nd_mw")
    return features, target


# ===========================================================================
# 1. test_sequence_dataset_len_is_rows_minus_seq_len
#    Plan §Task T2 named test — guards D7 ``__len__`` formula.
# ===========================================================================


def test_sequence_dataset_len_is_rows_minus_seq_len() -> None:
    """Guards plan T2 ``__len__``: ``len(ds) == len(features) - seq_len``.

    Two sub-scenarios at different fixture sizes and ``seq_len`` values
    confirm the formula is not accidentally returning the raw row count or
    the window count off-by-one.

    Plan clause: §Task T2 named test ``test_sequence_dataset_len_is_rows_minus_seq_len``.
    Implementation formula: ``_SequenceDatasetImpl.__len__`` →
    ``return len(self._features) - self._seq_len``.
    """
    # Sub-scenario A: 300 rows, seq_len=24 → expect 276.
    features_a, target_a = _make_fixture(300)
    ds_a = _SequenceDataset(features=features_a, target=target_a, seq_len=24)
    assert len(ds_a) == 276, (
        f"_SequenceDataset with 300 rows and seq_len=24 must have len=276; "
        f"got {len(ds_a)}.  Formula: len(features) - seq_len = 300 - 24 = 276 "
        "(plan §Task T2 named test)."
    )

    # Sub-scenario B: 400 rows, seq_len=168 → expect 232.
    features_b, target_b = _make_fixture(400)
    ds_b = _SequenceDataset(features=features_b, target=target_b, seq_len=168)
    assert len(ds_b) == 232, (
        f"_SequenceDataset with 400 rows and seq_len=168 must have len=232; "
        f"got {len(ds_b)}.  Formula: len(features) - seq_len = 400 - 168 = 232 "
        "(plan §Task T2 named test)."
    )


# ===========================================================================
# 2. test_sequence_dataset_lazy_window_getitem_matches_pandas_slice
#    Plan §Task T2 named test — guards D7 ``__getitem__`` window semantics
#    and dtype contract.
# ===========================================================================


@pytest.mark.parametrize("index_i", [0, 5, 42])
def test_sequence_dataset_lazy_window_getitem_matches_pandas_slice(index_i: int) -> None:
    """Guards plan T2 ``__getitem__``: window and target match the pandas slice.

    Uses integer-valued features (row*col as integer → float64 cast to float32)
    so the comparison is byte-exact without any floating-point tolerance.

    For each tested index ``i``:

    - ``ds[i][0]`` must equal ``torch.from_numpy(features.iloc[i:i+seq_len]
      .to_numpy(dtype=np.float32))``.
    - ``ds[i][1]`` must equal ``torch.tensor(float(target.iloc[i+seq_len]),
      dtype=torch.float32)``.
    - Both tensors must carry dtype ``torch.float32``.

    The boundary index (``len(ds) - 1``) is covered by a separate parametrise
    row below so that the off-by-one scenario at the tail is always exercised.

    Plan clause: §Task T2 named test
    ``test_sequence_dataset_lazy_window_getitem_matches_pandas_slice``.
    """
    seq_len = 24
    features, target = _make_fixture(200, n_features=4, integer_values=True)
    ds = _SequenceDataset(features=features, target=target, seq_len=seq_len)

    window_tensor, target_tensor = ds[index_i]

    # --- dtype contract -------------------------------------------------------
    assert window_tensor.dtype == torch.float32, (
        f"ds[{index_i}][0].dtype must be torch.float32; got {window_tensor.dtype} "
        "(plan §Task T2 / D7 lazy-window dtype)."
    )
    assert target_tensor.dtype == torch.float32, (
        f"ds[{index_i}][1].dtype must be torch.float32; got {target_tensor.dtype} "
        "(plan §Task T2 / D7 aligned-target dtype)."
    )

    # --- window shape ---------------------------------------------------------
    n_features = features.shape[1]
    assert window_tensor.shape == (seq_len, n_features), (
        f"ds[{index_i}][0].shape must be ({seq_len}, {n_features}); "
        f"got {window_tensor.shape} (plan §Task T2 / D7 window shape)."
    )

    # --- byte-exact feature window match -------------------------------------
    expected_window = torch.from_numpy(
        features.iloc[index_i : index_i + seq_len].to_numpy(dtype=np.float32)
    )
    assert torch.equal(window_tensor, expected_window), (
        f"ds[{index_i}][0] must equal features.iloc[{index_i}:{index_i + seq_len}] "
        "as float32 (plan §Task T2 lazy-window getitem semantics)."
    )

    # --- byte-exact target value match ----------------------------------------
    expected_target = torch.tensor(float(target.iloc[index_i + seq_len]), dtype=torch.float32)
    assert torch.equal(target_tensor, expected_target), (
        f"ds[{index_i}][1] must equal target.iloc[{index_i + seq_len}] as a "
        "scalar float32 tensor (plan §Task T2 aligned-target semantics)."
    )
    # The target must be a 0-d scalar tensor, not a 1-element vector.
    assert target_tensor.ndim == 0, (
        f"ds[{index_i}][1] must be a 0-d scalar tensor; got ndim={target_tensor.ndim} "
        "(nn.MSELoss contract — plan §Task T2 / D7)."
    )


def test_sequence_dataset_lazy_window_getitem_boundary_index() -> None:
    """Guards the tail-boundary index ``len(ds) - 1`` in the getitem contract.

    This is a separate test from the parametrised rows above so that the
    off-by-one boundary scenario registers as its own test item in the
    suite output — a boundary failure masked by passing middle indices
    would otherwise be silent.

    Plan clause: §Task T2 named test
    ``test_sequence_dataset_lazy_window_getitem_matches_pandas_slice``
    (boundary row).
    """
    seq_len = 24
    features, target = _make_fixture(200, n_features=4, integer_values=True)
    ds = _SequenceDataset(features=features, target=target, seq_len=seq_len)
    last_i = len(ds) - 1  # = 200 - 24 - 1 = 175

    window_tensor, target_tensor = ds[last_i]

    expected_window = torch.from_numpy(
        features.iloc[last_i : last_i + seq_len].to_numpy(dtype=np.float32)
    )
    assert torch.equal(window_tensor, expected_window), (
        f"ds[{last_i}][0] (last valid index) must equal "
        f"features.iloc[{last_i}:{last_i + seq_len}] as float32 "
        "(plan §Task T2 tail-boundary semantics)."
    )

    expected_target = torch.tensor(float(target.iloc[last_i + seq_len]), dtype=torch.float32)
    assert torch.equal(target_tensor, expected_target), (
        f"ds[{last_i}][1] (last valid index) must equal target.iloc[{last_i + seq_len}] "
        "as a scalar float32 tensor (plan §Task T2 tail-boundary semantics)."
    )


# ===========================================================================
# 3. test_sequence_dataset_does_not_eagerly_materialise_full_tensor
#    Plan §Task T2 named test — structural guard against D7 regression.
# ===========================================================================


def test_sequence_dataset_does_not_eagerly_materialise_full_tensor() -> None:
    """Guards plan D7: the dataset stores flat arrays, not a pre-windowed tensor.

    Build a 5 000-row x 50-column fixture with ``seq_len=168``.  After
    construction, introspect the stored ``_features`` and ``_target``
    numpy arrays — each must be no larger than the flat input budget::

        features: 5 000 * 50 * 4 bytes  =   1 000 000 bytes (~1 MB)
        target:   5 000      * 4 bytes  =      20 000 bytes (~20 KB)

    The 1 KB tolerance covers the ``np.ascontiguousarray`` copy overhead.

    The eagerly-windowed alternative ``(4 832, 168, 50)`` float32 tensor
    would cost ``4 832 * 168 * 50 * 4 ≈ 163 MB`` — clearly outside the
    budget.  This test fails loudly if a future refactor regresses to that
    pattern.

    Plan clause: §Task T2 named test
    ``test_sequence_dataset_does_not_eagerly_materialise_full_tensor``
    / plan §1 D7 (lazy windowing).
    """
    n_rows = 5_000
    n_features = 50
    seq_len = 168

    features, target = _make_fixture(n_rows, n_features=n_features)
    ds = _SequenceDataset(features=features, target=target, seq_len=seq_len)

    # The flat feature array must be at most n_rows * n_features * 4 bytes
    # plus a small 1 KB overhead for contiguous-copy bookkeeping.
    flat_features_budget = n_rows * n_features * 4 + 1024
    assert ds._features.nbytes <= flat_features_budget, (
        f"_SequenceDataset._features.nbytes={ds._features.nbytes} exceeds the "
        f"flat-array budget of {flat_features_budget} bytes.  An eagerly-windowed "
        f"({n_rows - seq_len}, {seq_len}, {n_features}) float32 tensor would cost "
        f"{(n_rows - seq_len) * seq_len * n_features * 4} bytes.  "
        "Regression: the dataset must store flat arrays, not pre-windowed tensors "
        "(plan §Task T2 / D7 lazy-window invariant)."
    )

    # The flat target array must be at most n_rows * 4 bytes plus 1 KB.
    flat_target_budget = n_rows * 4 + 1024
    assert ds._target.nbytes <= flat_target_budget, (
        f"_SequenceDataset._target.nbytes={ds._target.nbytes} exceeds the "
        f"flat-array budget of {flat_target_budget} bytes "
        "(plan §Task T2 / D7 lazy-window invariant — target side)."
    )
