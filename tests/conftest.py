from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    """Provide an event loop for environments that do not auto-create one."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
def enable_event_loop_debug() -> None:
    return None


@pytest.fixture(autouse=True)
def verify_cleanup() -> None:
    return None
