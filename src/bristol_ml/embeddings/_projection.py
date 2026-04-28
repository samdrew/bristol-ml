"""Stage 15 — UMAP wrapper for the notebook's 2D projection cell.

Plan §1 D6 / A3 (Ctrl+G 2026-04-27 — flipped from PCA): UMAP preserves
local neighbourhood structure in 768-dim embedding space better than
PCA's two principal components, which the user judged inadequate for
the demo moment ("the 2D projection shows the corpus falling into
visible clusters by event type and fuel").

Determinism (R-3): UMAP with ``random_state=42`` is coordinate-exact-
reproducible *only* when single-threaded — per the umap-learn docs,
multi-threaded runs introduce floating-point non-determinism even
with a fixed seed. The wrapper sets ``n_jobs=1`` to guarantee the
projection-coordinate determinism that
``tests/integration/test_notebook_15.py`` relies on.

Lazy import: ``umap-learn`` brings ~200 MB of compiled deps (numba +
llvmlite + pynndescent), so the import is deferred to the wrapper
function. A stub-only or factory-only caller never pays the cost.

Cross-references:

- Layer contract — ``docs/architecture/layers/embeddings.md``.
- Stage 15 plan — ``docs/plans/completed/15-embedding-index.md`` §1 A3, §6 T9.
- R-6 — UMAP install footprint mitigation.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "DEFAULT_RANDOM_STATE",
    "project_to_2d",
]


# Plan R-3: a single-source random_state used across the project for
# UMAP-driven artefacts. Re-exposed as a module constant so a notebook
# can import it (rather than hard-coding ``42`` in two places).
DEFAULT_RANDOM_STATE: Final[int] = 42


def project_to_2d(
    vectors: np.ndarray,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project an ``(n, dim)`` matrix to 2D via UMAP.

    Parameters
    ----------
    vectors
        Pre-normalised float32 corpus matrix (one row per event).
    random_state
        Seed forwarded to UMAP. Defaults to
        :data:`DEFAULT_RANDOM_STATE`.
    n_neighbors
        Local-neighbourhood size; UMAP's default is 15. Smaller
        values emphasise local structure.
    min_dist
        Minimum separation between projected points; UMAP's default
        is 0.1.

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` float32. Coordinate-exact-reproducible
        across processes when ``random_state`` and ``n_jobs=1`` hold
        (this wrapper pins both).
    """
    if vectors.ndim != 2:
        raise ValueError(
            f"vectors must be 2-D (n, dim); got ndim={vectors.ndim} shape={vectors.shape}."
        )
    if vectors.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Lazy import — see module docstring.
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=1,  # R-3: required for coordinate-exact reproducibility.
        metric="cosine",
    )
    coords = reducer.fit_transform(vectors)
    return np.asarray(coords, dtype=np.float32)
