"""Repo-wide pytest fixtures.

Shared test plumbing that more than one test file needs.  Kept deliberately
small — fixtures that are only used by a single test file live next to that
file, not here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from loguru import logger

# Stage 15 D12 / NFR-1: set ``HF_HUB_OFFLINE=1`` at module-import time so
# any sentence-transformers / huggingface_hub import in the test process
# refuses to fetch over the network. This is belt-and-braces with the
# project's stub-first triple gate (``BRISTOL_ML_EMBEDDING_STUB``); a
# regression that re-introduces a network call surfaces here as an
# OSError rather than a silent download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture()
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Route loguru records into pytest's ``caplog`` fixture at INFO and above.

    The repo's production code uses loguru (house style; see
    ``src/bristol_ml/ingestion/neso.py`` and friends).  Pytest's stdlib
    ``caplog`` does not capture loguru records without this adapter because
    loguru does not dispatch through ``logging.Logger``.

    Install this fixture once at repo scope so any test that asserts on a
    structured INFO line can depend on the same adapter.  Stage 3's feature
    assembler is the first caller (``tests/unit/features/test_assembler.py``);
    Stage 4's evaluation harness is the planned second caller.
    """
    handler_id = logger.add(caplog.handler, format="{message}", level="INFO")
    try:
        yield caplog
    finally:
        logger.remove(handler_id)
