"""Startup cost resolution must not let a bad sibling taxonomy abort LIVE.

The external taxonomy only feeds the effective-costs log line and the v2 fee
gate. With every v2 flag off, an invalid (or older, unknown-field) sibling file
must degrade to the built-in defaults instead of failing the whole startup.
When a v2 flag is enabled the taxonomy becomes load-bearing and the error is
still fatal.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import start
from src.config.regime_taxonomy import (
    CostsSettings,
    TaxonomyConfigError,
    resolve_costs_config,
)
from tests.conftest import make_config


def _config(**flags):
    config = make_config(**flags)
    config.costs_overrides = lambda: {}
    return config


def _raise_invalid_taxonomy(*_args, **_kwargs):
    raise TaxonomyConfigError("setups.trend_pullback has unknown keys: tp_mode")


def test_invalid_taxonomy_falls_back_to_defaults_when_all_v2_flags_are_off(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(start, "load_taxonomy_config", _raise_invalid_taxonomy)

    costs = start._resolve_startup_costs(_config(), logger)

    assert costs == resolve_costs_config(None, {})
    assert costs == CostsSettings()
    logger.warning.assert_called_once()
    assert "Invalid external taxonomy config ignored" in logger.warning.call_args[0][0]


@pytest.mark.parametrize(
    "flag",
    ["LLM_FOCUS_V2", "SHORTLIST_V2", "EVENT_POSITIONS_V2"],
)
def test_invalid_taxonomy_stays_fatal_when_a_v2_flag_is_enabled(monkeypatch, flag):
    logger = MagicMock()
    monkeypatch.setattr(start, "load_taxonomy_config", _raise_invalid_taxonomy)

    with pytest.raises(TaxonomyConfigError, match="unknown keys"):
        start._resolve_startup_costs(_config(**{flag: True}), logger)

    logger.warning.assert_not_called()
