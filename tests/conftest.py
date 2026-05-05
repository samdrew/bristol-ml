"""Repo-wide pytest fixtures.

Shared test plumbing that more than one test file needs.  Kept
deliberately small — fixtures that are only used by a single test
file live next to that file, not here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger


@pytest.fixture()
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Route loguru records into pytest's ``caplog`` fixture at INFO and above.

    Production code uses loguru; pytest's stdlib ``caplog`` does not
    capture loguru records without this adapter because loguru does
    not dispatch through ``logging.Logger``.  Install once at repo
    scope so any test that asserts on a structured INFO line can
    depend on the same adapter.
    """
    handler_id = logger.add(caplog.handler, format="{message}", level="INFO")
    try:
        yield caplog
    finally:
        logger.remove(handler_id)
