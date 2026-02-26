from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from custom_components.mindml import (
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.mindml.const import DOMAIN


def test_async_setup_initializes_domain_data() -> None:
    hass = MagicMock()
    hass.data = {}

    result = asyncio.run(async_setup(hass, {}))

    assert result is True
    assert DOMAIN in hass.data


def test_async_setup_entry_forwards_platform_and_stores_entry_data() -> None:
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "entry-1"

    result = asyncio.run(async_setup_entry(hass, entry))

    assert result is True
    hass.config_entries.async_forward_entry_setups.assert_awaited_once()
    assert entry.entry_id in hass.data[DOMAIN]


def test_async_unload_entry_unloads_platforms_and_cleans_up() -> None:
    hass = MagicMock()
    hass.data = {DOMAIN: {"entry-1": {"sensor": "value"}}}
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

    entry = MagicMock()
    entry.entry_id = "entry-1"

    result = asyncio.run(async_unload_entry(hass, entry))

    assert result is True
    hass.config_entries.async_unload_platforms.assert_awaited_once()
    assert "entry-1" not in hass.data[DOMAIN]
