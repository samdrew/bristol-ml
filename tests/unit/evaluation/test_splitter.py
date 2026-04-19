"""Spec-derived tests for ``bristol_ml.evaluation.splitter``.

Every test is derived from:

- ``docs/plans/completed/03-feature-assembler.md`` §4 (Acceptance Criteria) and
  §6 Task T2 (named tests and fold-count formula).
- ``src/bristol_ml/evaluation/CLAUDE.md`` "Invariants" section (the load-bearing
  list that every downstream modelling stage relies on).
- ``docs/plans/completed/03-feature-assembler.md`` §1 D4 (window-type decision:
  expanding default, ``fixed_window`` knob for sliding).

No production code is modified here.  If any test below fails, the failure
points at a deviation from the spec — do not weaken the test; surface the
failure to the implementer.

Conventions
-----------
- British English in docstrings.
- Each test docstring cites the plan AC or invariant it guards.
- Numpy is used directly; pandas is not imported (splitter is data-structure-
  agnostic per AC-4).
- ``pytest.raises(ValueError, match=...)`` is used wherever the implementation
  documents an error message.
- ``pytest.mark.parametrize`` is used wherever rolling-origin parameters vary
  in orthogonal ways.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

splitter_mod = pytest.importorskip("bristol_ml.evaluation.splitter")

rolling_origin_split = splitter_mod.rolling_origin_split
rolling_origin_split_from_config = splitter_mod.rolling_origin_split_from_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(gen) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise a generator of (train_idx, test_idx) pairs into a list."""
    return list(gen)


# ---------------------------------------------------------------------------
# AC-3: no train/test overlap and chronological order
# ---------------------------------------------------------------------------


class TestNoOverlap:
    """Guards AC-3: no train/test leakage within a fold.

    CLAUDE.md invariant: ``max(train_idx) < min(test_idx)``; train and
    test sets share no indices.
    """

    @pytest.mark.parametrize(
        "n_rows, min_train, test_len, step, gap",
        [
            (100, 20, 10, 5, 0),
            (500, 50, 24, 24, 0),
            (200, 30, 10, 1, 3),
            (1000, 100, 10, 1, 0),
        ],
    )
    def test_splitter_no_overlap(
        self,
        n_rows: int,
        min_train: int,
        test_len: int,
        step: int,
        gap: int,
    ) -> None:
        """Guards AC-3: for every fold, ``max(train_idx) < min(test_idx)``
        and the intersection of train and test indices is empty.

        Parametrised over representative configs to confirm the invariant
        holds regardless of step, gap, or window size.
        """
        folds = _collect(
            rolling_origin_split(n_rows, min_train=min_train, test_len=test_len, step=step, gap=gap)
        )
        assert folds, "Expected at least one fold from a feasible configuration."
        for i, (train_idx, test_idx) in enumerate(folds):
            assert train_idx.max() < test_idx.min(), (
                f"Fold {i}: max(train_idx)={train_idx.max()} must be strictly less "
                f"than min(test_idx)={test_idx.min()} (AC-3 no-overlap invariant)."
            )
            overlap = set(train_idx.tolist()) & set(test_idx.tolist())
            assert overlap == set(), (
                f"Fold {i}: train and test index sets must be disjoint; "
                f"overlap={overlap!r} (AC-3 no-overlap invariant)."
            )


class TestChronologicalOrder:
    """Guards AC-3: train_idx and test_idx are each monotonically ascending.

    CLAUDE.md invariant: both index arrays are monotonically ascending.
    """

    @pytest.mark.parametrize(
        "n_rows, min_train, test_len, step, gap",
        [
            (200, 30, 15, 5, 0),
            (300, 50, 10, 10, 2),
            (100, 20, 10, 1, 0),
        ],
    )
    def test_splitter_chronological_within_fold(
        self,
        n_rows: int,
        min_train: int,
        test_len: int,
        step: int,
        gap: int,
    ) -> None:
        """Guards AC-3: both train_idx and test_idx are monotonically ascending.

        Sorted ascending order is required so downstream ``df.iloc[idx]``
        produces rows in time order without an extra sort step.
        """
        folds = _collect(
            rolling_origin_split(n_rows, min_train=min_train, test_len=test_len, step=step, gap=gap)
        )
        assert folds, "Expected at least one fold."
        for i, (train_idx, test_idx) in enumerate(folds):
            assert np.all(np.diff(train_idx) > 0), (
                f"Fold {i}: train_idx must be strictly monotonically ascending; "
                f"got diffs={np.diff(train_idx).tolist()!r} (AC-3 chronological invariant)."
            )
            assert np.all(np.diff(test_idx) > 0), (
                f"Fold {i}: test_idx must be strictly monotonically ascending; "
                f"got diffs={np.diff(test_idx).tolist()!r} (AC-3 chronological invariant)."
            )


# ---------------------------------------------------------------------------
# AC-4: index arrays are numpy integer arrays compatible with DataFrame.iloc
# ---------------------------------------------------------------------------


class TestIndexArrayType:
    """Guards AC-4: the splitter returns index arrays.

    CLAUDE.md invariant: Arrays are ``numpy.int64`` and compatible with
    ``DataFrame.iloc``.  This is the contract that lets downstream code
    slice cheaply without materialising copies.
    """

    def test_splitter_returns_integer_arrays(self) -> None:
        """Guards AC-4: each yielded train_idx and test_idx is ``np.ndarray``
        with ``dtype.kind == 'i'`` and is valid as an argument to
        ``DataFrame.iloc``.

        Verifies both the type and the dtype kind; does not require pandas to
        be imported — we assert ``dtype.kind == 'i'`` directly.
        """
        folds = _collect(rolling_origin_split(100, min_train=20, test_len=10, step=5))
        assert folds, "Expected at least one fold."
        for i, (train_idx, test_idx) in enumerate(folds):
            assert isinstance(train_idx, np.ndarray), (
                f"Fold {i}: train_idx must be np.ndarray; got {type(train_idx).__name__!r} "
                f"(AC-4 index-array invariant)."
            )
            assert isinstance(test_idx, np.ndarray), (
                f"Fold {i}: test_idx must be np.ndarray; got {type(test_idx).__name__!r} "
                f"(AC-4 index-array invariant)."
            )
            assert train_idx.dtype.kind == "i", (
                f"Fold {i}: train_idx.dtype must be an integer kind; "
                f"got {train_idx.dtype!r} (AC-4 integer-array invariant)."
            )
            assert test_idx.dtype.kind == "i", (
                f"Fold {i}: test_idx.dtype must be an integer kind; "
                f"got {test_idx.dtype!r} (AC-4 integer-array invariant)."
            )


# ---------------------------------------------------------------------------
# Plan T2: fold count formula
# ---------------------------------------------------------------------------


class TestFoldCount:
    """Guards Plan T2 fold-count formula:
    ``floor((n_rows - test_len - min_train - gap) / step) + 1``.
    """

    def test_splitter_fold_count_matches_step(self) -> None:
        """Guards Plan T2: with n=1000, min_train=100, test_len=10, step=1,
        exactly 891 folds are produced.

        Formula: ``floor((1000 - 10 - 100 - 0) / 1) + 1 = 890 + 1 = 891``.
        """
        folds = _collect(rolling_origin_split(1000, min_train=100, test_len=10, step=1, gap=0))
        assert len(folds) == 891, (
            f"Expected 891 folds with n=1000, min_train=100, test_len=10, step=1; "
            f"got {len(folds)} (Plan T2 fold-count formula)."
        )

    @pytest.mark.parametrize(
        "n_rows, min_train, test_len, step, gap, expected_folds",
        [
            # Basic: step == test_len (daily non-overlapping)
            (100, 20, 10, 10, 0, 8),  # floor((100-10-20-0)/10) + 1 = 7+1 = 8
            # Step == 1 (dense sliding)
            (50, 10, 5, 1, 0, 36),  # floor((50-5-10-0)/1) + 1 = 35+1 = 36
            # Gap > 0
            (100, 20, 10, 5, 5, 14),  # floor((100-10-20-5)/5) + 1 = floor(65/5)+1 = 14
            # Exactly one fold (n_rows = min_train + gap + test_len)
            (30, 20, 5, 1, 5, 1),  # floor((30-5-20-5)/1) + 1 = 0+1 = 1
        ],
    )
    def test_splitter_fold_count_parametrised(
        self,
        n_rows: int,
        min_train: int,
        test_len: int,
        step: int,
        gap: int,
        expected_folds: int,
    ) -> None:
        """Guards Plan T2 fold-count formula across representative configs.

        Formula: ``floor((n_rows - test_len - min_train - gap) / step) + 1``.
        """
        folds = _collect(
            rolling_origin_split(n_rows, min_train=min_train, test_len=test_len, step=step, gap=gap)
        )
        assert len(folds) == expected_folds, (
            f"n={n_rows}, min_train={min_train}, test_len={test_len}, step={step}, "
            f"gap={gap}: expected {expected_folds} folds, got {len(folds)} "
            f"(Plan T2 fold-count formula)."
        )


# ---------------------------------------------------------------------------
# Plan T2: gap invariant
# ---------------------------------------------------------------------------


class TestGapInvariant:
    """Guards Plan T2 and CLAUDE.md invariant:
    if ``gap > 0``, then ``max(train_idx) + gap < min(test_idx)``.
    """

    @pytest.mark.parametrize("gap", [1, 5, 10, 24])
    def test_splitter_gap_respects_gap_hours(self, gap: int) -> None:
        """Guards Plan T2: with gap=N, for every fold
        ``max(train_idx) + N < min(test_idx)``.

        The embargo of ``gap`` rows between end-of-training and start-of-test
        must be honoured in every fold, not just the first.
        """
        folds = _collect(rolling_origin_split(200, min_train=30, test_len=10, step=5, gap=gap))
        assert folds, f"Expected at least one fold with gap={gap}."
        for i, (train_idx, test_idx) in enumerate(folds):
            assert train_idx.max() + gap < test_idx.min(), (
                f"Fold {i} with gap={gap}: max(train_idx) + gap "
                f"({train_idx.max()} + {gap} = {train_idx.max() + gap}) must be strictly "
                f"less than min(test_idx)={test_idx.min()} (CLAUDE.md gap embargo invariant)."
            )


# ---------------------------------------------------------------------------
# Plan T2 / D4: fixed_window keeps train size constant
# ---------------------------------------------------------------------------


class TestFixedWindow:
    """Guards Plan T2 and D4: under ``fixed_window=True``, every fold has
    ``len(train_idx) == min_train``.
    """

    @pytest.mark.parametrize("min_train", [10, 50, 100])
    def test_splitter_fixed_window_keeps_train_size(self, min_train: int) -> None:
        """Guards Plan T2 / D4: under ``fixed_window=True``, every fold's
        training window has exactly ``min_train`` rows (sliding window).

        The fixed-window variant should produce a constant-sized training window
        independent of fold index.
        """
        folds = _collect(
            rolling_origin_split(300, min_train=min_train, test_len=10, step=5, fixed_window=True)
        )
        assert folds, f"Expected at least one fold with min_train={min_train}."
        for i, (train_idx, _test_idx) in enumerate(folds):
            assert len(train_idx) == min_train, (
                f"Fold {i}: under fixed_window=True, len(train_idx) must equal "
                f"min_train={min_train}; got {len(train_idx)} "
                f"(Plan T2 / D4 fixed-window invariant)."
            )


# ---------------------------------------------------------------------------
# Invariant: expanding window starts at 0 and grows
# ---------------------------------------------------------------------------


class TestExpandingWindow:
    """Guards CLAUDE.md invariant: under ``fixed_window=False`` (default),
    ``train_idx`` always starts at 0 and grows with each fold.
    """

    def test_splitter_expanding_window_train_grows_with_origin(self) -> None:
        """Guards CLAUDE.md invariant: expanding window.

        Under ``fixed_window=False`` (default):
        - ``train_idx[0] == 0`` for every fold (training starts from the
          beginning of the series).
        - Each successive fold has strictly more training rows than the
          previous one (the window expands).
        - For fold at origin O: ``len(train_idx) == O - gap`` where O is the
          index of the first test row and gap is the embargo.
        """
        gap = 3
        step = 5
        min_train = 20
        test_len = 10
        n_rows = 200

        folds = _collect(
            rolling_origin_split(
                n_rows,
                min_train=min_train,
                test_len=test_len,
                step=step,
                gap=gap,
                fixed_window=False,
            )
        )
        assert folds, "Expected at least one fold."

        prev_train_len = None
        for i, (train_idx, test_idx) in enumerate(folds):
            # Training window always starts at 0.
            assert train_idx[0] == 0, (
                f"Fold {i}: expanding window train_idx must start at 0; "
                f"got train_idx[0]={train_idx[0]} (CLAUDE.md expanding-window invariant)."
            )
            # Each fold trains up to (origin - gap); origin == test_idx[0].
            origin = int(test_idx[0])
            expected_train_len = origin - gap
            assert len(train_idx) == expected_train_len, (
                f"Fold {i}: expanding window len(train_idx) must equal "
                f"origin({origin}) - gap({gap}) = {expected_train_len}; "
                f"got {len(train_idx)} (CLAUDE.md expanding-window invariant)."
            )
            # Later folds have strictly more training rows.
            if prev_train_len is not None:
                assert len(train_idx) > prev_train_len, (
                    f"Fold {i}: expanding window must grow — "
                    f"len(train_idx)={len(train_idx)} must exceed previous "
                    f"fold's {prev_train_len} (CLAUDE.md expanding-window invariant)."
                )
            prev_train_len = len(train_idx)


# ---------------------------------------------------------------------------
# Invariant: test window always has fixed width
# ---------------------------------------------------------------------------


class TestTestWindowWidth:
    """Guards CLAUDE.md invariant: ``len(test_idx) == test_len`` for every fold."""

    @pytest.mark.parametrize(
        "n_rows, min_train, test_len, step, gap, fixed_window",
        [
            (200, 30, 10, 5, 0, False),
            (200, 30, 24, 24, 0, False),
            (200, 30, 10, 5, 3, True),
            (500, 100, 48, 24, 12, False),
        ],
    )
    def test_splitter_test_window_always_has_fixed_width(
        self,
        n_rows: int,
        min_train: int,
        test_len: int,
        step: int,
        gap: int,
        fixed_window: bool,
    ) -> None:
        """Guards CLAUDE.md invariant: ``len(test_idx) == test_len`` for every fold.

        The forecast horizon is fixed per evaluation run (set by ``test_len``),
        regardless of window type or fold index.  A varying test window would
        make fold-level metrics incomparable.
        """
        folds = _collect(
            rolling_origin_split(
                n_rows,
                min_train=min_train,
                test_len=test_len,
                step=step,
                gap=gap,
                fixed_window=fixed_window,
            )
        )
        assert folds, "Expected at least one fold."
        for i, (_train_idx, test_idx) in enumerate(folds):
            assert len(test_idx) == test_len, (
                f"Fold {i}: len(test_idx) must equal test_len={test_len}; "
                f"got {len(test_idx)} (CLAUDE.md fixed-test-width invariant)."
            )


# ---------------------------------------------------------------------------
# Invariant: non-overlapping test windows across folds
# ---------------------------------------------------------------------------


class TestNoLeakageAcrossFolds:
    """Guards cross-fold ordering: for step >= test_len, test windows do not
    overlap across adjacent folds.
    """

    @pytest.mark.parametrize(
        "step_multiplier",
        [
            1,  # step == test_len: non-overlapping, exactly touching
            2,  # step == 2 * test_len: non-overlapping with a gap between test windows
        ],
    )
    def test_splitter_no_leakage_across_folds(self, step_multiplier: int) -> None:
        """Guards cross-fold ordering: for any earlier fold A and later fold B,
        ``min(B.test_idx) > max(A.test_idx)`` when ``step >= test_len``.

        With ``step == test_len`` (non-overlapping daily step), successive test
        windows tile the evaluation period without repetition.  With
        ``step == 2 * test_len`` there is a gap between consecutive test windows
        but the ordering invariant still holds.

        This is parametrised over ``step == test_len`` and
        ``step == 2 * test_len`` to make the coverage matrix explicit.
        """
        test_len = 10
        step = test_len * step_multiplier
        folds = _collect(rolling_origin_split(500, min_train=50, test_len=test_len, step=step))
        assert len(folds) >= 2, (
            f"Need at least two folds to test cross-fold ordering; got {len(folds)}."
        )
        for i in range(len(folds) - 1):
            _, test_a = folds[i]
            _, test_b = folds[i + 1]
            assert test_b.min() > test_a.max(), (
                f"Folds {i} and {i + 1}: min(test_b)={test_b.min()} must be strictly "
                f"greater than max(test_a)={test_a.max()} when step >= test_len "
                f"(cross-fold ordering invariant, step_multiplier={step_multiplier})."
            )


# ---------------------------------------------------------------------------
# Error cases: non-positive n_rows
# ---------------------------------------------------------------------------


class TestRaisesNonPositiveNRows:
    """Guards ValueError on non-positive ``n_rows`` (CLAUDE.md + docstring)."""

    @pytest.mark.parametrize("n_rows", [0, -1, -100])
    def test_splitter_raises_on_non_positive_n_rows(self, n_rows: int) -> None:
        """Guards ValueError: ``rolling_origin_split(0, …)`` raises ``ValueError``.

        A non-positive row count is physically meaningless; the splitter must
        reject it immediately rather than silently yielding zero folds.
        """
        with pytest.raises(ValueError, match="n_rows"):
            _collect(rolling_origin_split(n_rows, min_train=10, test_len=5, step=1))


# ---------------------------------------------------------------------------
# Error cases: non-positive min_train, test_len, step
# ---------------------------------------------------------------------------


class TestRaisesNonPositiveParameters:
    """Guards ValueError on non-positive ``min_train``, ``test_len``, ``step``."""

    @pytest.mark.parametrize("min_train", [0, -1, -50])
    def test_splitter_raises_on_non_positive_min_train(self, min_train: int) -> None:
        """Guards ValueError: ``min_train`` must be at least 1.

        A zero-length training window is undefined for a supervised model;
        the splitter must reject it with a ``ValueError`` naming the parameter.
        """
        with pytest.raises(ValueError, match="min_train"):
            _collect(rolling_origin_split(100, min_train=min_train, test_len=5, step=1))

    @pytest.mark.parametrize("test_len", [0, -1, -5])
    def test_splitter_raises_on_non_positive_test_len(self, test_len: int) -> None:
        """Guards ValueError: ``test_len`` must be at least 1.

        A zero-length test window produces no evaluation data; the splitter
        must reject it with a ``ValueError`` naming the parameter.
        """
        with pytest.raises(ValueError, match="test_len"):
            _collect(rolling_origin_split(100, min_train=20, test_len=test_len, step=1))

    @pytest.mark.parametrize("step", [0, -1, -10])
    def test_splitter_raises_on_non_positive_step(self, step: int) -> None:
        """Guards ValueError: ``step`` must be at least 1.

        A step of zero would loop infinitely over the same fold; negative step
        is nonsensical.  The splitter must reject both with a ``ValueError``
        naming the parameter.
        """
        with pytest.raises(ValueError, match="step"):
            _collect(rolling_origin_split(100, min_train=20, test_len=5, step=step))


# ---------------------------------------------------------------------------
# Error case: negative gap
# ---------------------------------------------------------------------------


class TestRaisesNegativeGap:
    """Guards ValueError on negative ``gap`` (CLAUDE.md + docstring)."""

    @pytest.mark.parametrize("gap", [-1, -5, -100])
    def test_splitter_raises_on_negative_gap(self, gap: int) -> None:
        """Guards ValueError: ``gap`` must be non-negative.

        A negative embargo is physically meaningless (training data cannot
        postdate the test window).  The splitter must raise with a
        ``ValueError`` naming the parameter.
        """
        with pytest.raises(ValueError, match="gap"):
            _collect(rolling_origin_split(100, min_train=20, test_len=5, step=1, gap=gap))


# ---------------------------------------------------------------------------
# Error case: infeasible configuration
# ---------------------------------------------------------------------------


class TestRaisesInfeasibleConfig:
    """Guards ValueError when min_train + gap + test_len > n_rows."""

    def test_splitter_raises_on_infeasible_config(self) -> None:
        """Guards ValueError: ``min_train + gap + test_len > n_rows`` raises.

        With n_rows=50, min_train=40, gap=5, test_len=10:
        ``40 + 5 + 10 = 55 > 50``, which is infeasible.

        The error message must name the offending numbers so the user can
        diagnose the conflict without reading the source.
        """
        with pytest.raises(ValueError, match=r"55") as exc_info:
            _collect(rolling_origin_split(50, min_train=40, gap=5, test_len=10, step=1))
        msg = str(exc_info.value)
        # The message must name the key numbers from the violation.
        assert "40" in msg or "min_train" in msg.lower(), (
            f"Infeasible-config error must mention min_train or 40; got: {msg!r}"
        )
        assert "50" in msg or "n_rows" in msg.lower(), (
            f"Infeasible-config error must mention n_rows or 50; got: {msg!r}"
        )


# ---------------------------------------------------------------------------
# Edge case: boundary condition produces exactly one fold
# ---------------------------------------------------------------------------


class TestBoundaryOneFold:
    """Guards boundary condition: n_rows == min_train + gap + test_len => 1 fold."""

    @pytest.mark.parametrize(
        "min_train, gap, test_len",
        [
            (20, 0, 10),  # n_rows = 30
            (50, 5, 20),  # n_rows = 75
            (100, 0, 24),  # n_rows = 124
        ],
    )
    def test_splitter_empty_yield_when_config_saturates_exactly(
        self, min_train: int, gap: int, test_len: int
    ) -> None:
        """Guards boundary condition: ``n_rows = min_train + gap + test_len``
        produces exactly one fold.

        Formula: ``floor((n_rows - test_len - min_train - gap) / step) + 1``
        = ``floor(0 / step) + 1 = 1``.

        Verifies the off-by-one boundary without accidentally triggering an
        infeasible-config ``ValueError``.
        """
        n_rows = min_train + gap + test_len
        folds = _collect(
            rolling_origin_split(n_rows, min_train=min_train, test_len=test_len, step=1, gap=gap)
        )
        assert len(folds) == 1, (
            f"n_rows={n_rows} (=min_train+gap+test_len) must yield exactly 1 fold; "
            f"got {len(folds)} (boundary-condition invariant)."
        )
        train_idx, test_idx = folds[0]
        # Training window in the boundary fold.
        if gap == 0:
            assert len(train_idx) == min_train, (
                f"Boundary fold train length must be min_train={min_train}; got {len(train_idx)}."
            )
        # Test window must fill the tail exactly.
        assert len(test_idx) == test_len, (
            f"Boundary fold test length must be test_len={test_len}; got {len(test_idx)}."
        )
        assert test_idx[-1] == n_rows - 1, (
            f"Boundary fold must exhaust the full series; "
            f"last test index must be {n_rows - 1}, got {test_idx[-1]}."
        )


# ---------------------------------------------------------------------------
# rolling_origin_split_from_config equivalence
# ---------------------------------------------------------------------------


class TestFromConfigEquivalence:
    """Guards the wrapper function contract: ``rolling_origin_split_from_config``
    must yield the same sequence as the kernel with matching kwargs.
    """

    def test_splitter_from_config_equivalence(self) -> None:
        """Guards public API: ``rolling_origin_split_from_config(n, SplitterConfig(...))``
        yields the same sequence as ``rolling_origin_split(n, ...)`` with matching
        kwargs.

        Uses a directly constructed ``SplitterConfig`` (no Hydra round-trip)
        to keep the test isolated from the config layer.  This guards the
        wrapper's responsibility to unpack ``config.min_train_periods`` →
        ``min_train`` correctly.
        """
        from conf._schemas import SplitterConfig  # type: ignore[import-not-found]

        n = 500
        cfg = SplitterConfig(
            min_train_periods=80,
            test_len=20,
            step=10,
            gap=3,
            fixed_window=True,
        )

        folds_kernel = _collect(
            rolling_origin_split(
                n,
                min_train=cfg.min_train_periods,
                test_len=cfg.test_len,
                step=cfg.step,
                gap=cfg.gap,
                fixed_window=cfg.fixed_window,
            )
        )
        folds_wrapper = _collect(rolling_origin_split_from_config(n, cfg))

        assert len(folds_kernel) == len(folds_wrapper), (
            f"rolling_origin_split_from_config must produce the same fold count as "
            f"the kernel: expected {len(folds_kernel)}, got {len(folds_wrapper)}."
        )
        for i, ((train_k, test_k), (train_w, test_w)) in enumerate(
            zip(folds_kernel, folds_wrapper, strict=True)
        ):
            np.testing.assert_array_equal(
                train_k,
                train_w,
                err_msg=(
                    f"Fold {i}: train_idx from wrapper differs from kernel result "
                    f"(rolling_origin_split_from_config equivalence)."
                ),
            )
            np.testing.assert_array_equal(
                test_k,
                test_w,
                err_msg=(
                    f"Fold {i}: test_idx from wrapper differs from kernel result "
                    f"(rolling_origin_split_from_config equivalence)."
                ),
            )

    @pytest.mark.parametrize("fixed_window", [False, True])
    def test_splitter_from_config_equivalence_both_window_types(self, fixed_window: bool) -> None:
        """Guards wrapper equivalence for both expanding and fixed window variants.

        Parametrised over both ``fixed_window`` values to confirm the wrapper
        forwards the flag correctly in each case (D4 decision).
        """
        from conf._schemas import SplitterConfig  # type: ignore[import-not-found]

        n = 300
        cfg = SplitterConfig(
            min_train_periods=50,
            test_len=10,
            step=7,
            gap=2,
            fixed_window=fixed_window,
        )

        folds_kernel = _collect(
            rolling_origin_split(
                n,
                min_train=cfg.min_train_periods,
                test_len=cfg.test_len,
                step=cfg.step,
                gap=cfg.gap,
                fixed_window=cfg.fixed_window,
            )
        )
        folds_wrapper = _collect(rolling_origin_split_from_config(n, cfg))

        assert len(folds_kernel) == len(folds_wrapper), (
            f"fixed_window={fixed_window}: wrapper fold count {len(folds_wrapper)} "
            f"must equal kernel fold count {len(folds_kernel)}."
        )
        for i, ((train_k, test_k), (train_w, test_w)) in enumerate(
            zip(folds_kernel, folds_wrapper, strict=True)
        ):
            np.testing.assert_array_equal(
                train_k,
                train_w,
                err_msg=f"Fold {i} (fixed_window={fixed_window}): train mismatch.",
            )
            np.testing.assert_array_equal(
                test_k,
                test_w,
                err_msg=f"Fold {i} (fixed_window={fixed_window}): test mismatch.",
            )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_splitter_cli_help_exits_zero() -> None:
    """Guards §2.1.1: every module runs standalone.

    ``python -m bristol_ml.evaluation.splitter --help`` must exit 0.
    """
    result = subprocess.run(
        [sys.executable, "-m", "bristol_ml.evaluation.splitter", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI --help must exit 0; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
