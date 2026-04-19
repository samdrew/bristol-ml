"""Rolling-origin train/test splitter for time-indexed data.

Stage 3 (§DESIGN 5.1, §DESIGN 5.3). The splitter yields integer-array
``(train_idx, test_idx)`` pairs that every downstream modelling stage
consumes. The fold origin advances by ``step`` rows at a time; the test
window is fixed at ``test_len`` rows; the training window is either
expanding (``fixed_window=False``, default) or sliding
(``fixed_window=True``) with length ``min_train_periods``.

Terminology follows Tashman (2000); the "walk-forward validation" name in
ML practitioner literature refers to the same construct. The shape of the
return value — pairs of ``numpy`` integer arrays compatible with
``pandas.DataFrame.iloc`` — is chosen so that downstream code slices
cheaply without materialising copies (Intent AC-4,
``docs/intent/03-feature-assembler.md``).

Run standalone::

    python -m bristol_ml.evaluation.splitter [--help]

The CLI loads ``conf/config.yaml`` via Hydra, applies the resolved
``evaluation.rolling_origin`` config to a synthetic index of length
``--n-rows`` (default: 8760 * 2 hours, i.e. two years), and prints the
fold count plus the first fold's train/test index heads. Useful for a
live-demo sanity check.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — typing-only import
    from conf._schemas import SplitterConfig

__all__ = ["rolling_origin_split", "rolling_origin_split_from_config"]


def rolling_origin_split(
    n_rows: int,
    *,
    min_train: int,
    test_len: int,
    step: int,
    gap: int = 0,
    fixed_window: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` integer-array pairs in chronological order.

    Parameters
    ----------
    n_rows
        Total number of rows in the time-indexed series. Folds are enumerated
        over ``range(n_rows)``; downstream code slices its own DataFrame by
        these indices (the splitter itself is data-structure-agnostic).
    min_train
        Minimum number of rows that must precede the first test origin. Under
        the default expanding window, this is also the training window of the
        first fold; subsequent folds' training windows grow. Under
        ``fixed_window=True``, this is the (fixed) length of every fold's
        training window.
    test_len
        Number of rows in each fold's test window — the forecast horizon.
    step
        Number of rows by which the origin advances between successive folds.
        A ``step`` equal to ``test_len`` produces non-overlapping test windows
        (the convention for day-ahead daily-step evaluation: ``step = test_len = 24``).
    gap
        Optional embargo between the end of training and the start of testing,
        in rows. Zero is the default for historical training; non-zero encodes
        a gate-closure-style discipline where the last ``gap`` observations
        before the test window are not available to the model.
    fixed_window
        ``False`` (default) produces an expanding training window: every fold
        trains from row 0 up to ``origin - gap``. ``True`` produces a sliding
        window of exactly ``min_train`` rows per fold.

    Yields
    ------
    (train_idx, test_idx) : tuple[np.ndarray, np.ndarray]
        Integer index arrays (``dtype int64``) suitable for
        ``DataFrame.iloc[train_idx]`` / ``DataFrame.iloc[test_idx]``.

    Raises
    ------
    ValueError
        If any of ``n_rows``, ``min_train``, ``test_len``, or ``step`` is
        non-positive, or if ``gap`` is negative, or if the configuration is
        infeasible (``min_train + gap + test_len > n_rows``).

    Notes
    -----
    The first fold's test window starts at row ``min_train + gap``. Folds
    continue while ``origin + test_len <= n_rows``. The total number of
    folds is therefore
    ``floor((n_rows - test_len - min_train - gap) / step) + 1`` when the
    configuration is feasible, else zero (and nothing is yielded).
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive; got {n_rows}.")
    if min_train < 1:
        raise ValueError(f"min_train must be at least 1; got {min_train}.")
    if test_len < 1:
        raise ValueError(f"test_len must be at least 1; got {test_len}.")
    if step < 1:
        raise ValueError(f"step must be at least 1; got {step}.")
    if gap < 0:
        raise ValueError(f"gap must be non-negative; got {gap}.")
    if min_train + gap + test_len > n_rows:
        raise ValueError(
            f"Infeasible split: min_train ({min_train}) + gap ({gap}) + test_len "
            f"({test_len}) = {min_train + gap + test_len} exceeds n_rows ({n_rows})."
        )

    first_origin = min_train + gap
    # origin indexes the first row of the test window; loop while a full test
    # window fits (origin + test_len <= n_rows).
    for origin in range(first_origin, n_rows - test_len + 1, step):
        train_end = origin - gap  # exclusive
        train_start = train_end - min_train if fixed_window else 0
        train_idx = np.arange(train_start, train_end, dtype=np.int64)
        test_idx = np.arange(origin, origin + test_len, dtype=np.int64)
        yield train_idx, test_idx


def rolling_origin_split_from_config(
    n_rows: int,
    config: SplitterConfig,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Convenience wrapper: unpack ``SplitterConfig`` fields into ``rolling_origin_split``.

    This keeps the kernel function (``rolling_origin_split``) free of any
    Pydantic dependency so the splitter is usable outside the Hydra-config
    lifecycle (e.g. in a synthetic-data unit test).
    """
    return rolling_origin_split(
        n_rows,
        min_train=config.min_train_periods,
        test_len=config.test_len,
        step=config.step,
        gap=config.gap,
        fixed_window=config.fixed_window,
    )


# ---------------------------------------------------------------------------
# CLI — `python -m bristol_ml.evaluation.splitter`
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bristol_ml.evaluation.splitter",
        description=(
            "Enumerate rolling-origin folds against a synthetic index of length "
            "`--n-rows`, using the resolved `evaluation.rolling_origin` config. "
            "Prints the fold count and the first fold's train/test index heads."
        ),
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=8760 * 2,
        help="Synthetic index length to enumerate folds over (default: 17520 = 2 years hourly).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. evaluation.rolling_origin.step=48",
    )
    return parser


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Local import so `--help` does not pull Hydra into the import chain.
    from bristol_ml.config import load_config

    cfg = load_config(overrides=list(args.overrides))
    splitter_cfg = cfg.evaluation.rolling_origin
    if splitter_cfg is None:
        print(
            "No rolling-origin splitter config resolved. Ensure "
            "`evaluation/rolling_origin@evaluation.rolling_origin` is in "
            "`conf/config.yaml` defaults.",
            file=sys.stderr,
        )
        return 2

    folds = list(rolling_origin_split_from_config(args.n_rows, splitter_cfg))
    logger.info(
        "Rolling-origin splitter: n_rows={} folds={} "
        "(min_train={} test_len={} step={} gap={} fixed_window={})",
        args.n_rows,
        len(folds),
        splitter_cfg.min_train_periods,
        splitter_cfg.test_len,
        splitter_cfg.step,
        splitter_cfg.gap,
        splitter_cfg.fixed_window,
    )
    print(f"fold_count={len(folds)}")
    if folds:
        first_train, first_test = folds[0]
        last_train, last_test = folds[-1]
        print(
            f"first_fold: train[0..3]={first_train[:3].tolist()} "
            f"test[0..3]={first_test[:3].tolist()} "
            f"(train_len={len(first_train)}, test_len={len(first_test)})"
        )
        print(
            f"last_fold:  train[-3:]={last_train[-3:].tolist()} "
            f"test[-3:]={last_test[-3:].tolist()} "
            f"(train_len={len(last_train)}, test_len={len(last_test)})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli_main())
