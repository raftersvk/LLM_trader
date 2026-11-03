"""Feature flags for the LLM-focus v2 rollout.

Every flag defaults to ``False``. The helpers accept the repository ``Config``
object, a ``SimpleNamespace`` style test double, or a plain mapping so tests and
offline tooling do not need the full configuration loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LLM_FOCUS_V2 = "llm_focus_v2"
SHORTLIST_V2 = "shortlist_v2"
EVENT_POSITIONS_V2 = "event_positions_v2"

CONFIG_SECTION = "llm_focus"

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled", ""}


def coerce_flag_value(value: Any) -> bool | None:
    """Convert a configuration value to a boolean when it is unambiguous.

    Returns ``None`` for values that are not recognisable booleans (for example
    a ``MagicMock`` returned by an unconfigured test double) so callers can fall
    back to another lookup strategy instead of treating arbitrary objects as
    truthy.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    return None


def _from_mapping(config: Mapping[str, Any], flag_name: str) -> bool | None:
    for key in (flag_name, flag_name.upper(), f"{CONFIG_SECTION}.{flag_name}"):
        if key in config:
            parsed = coerce_flag_value(config[key])
            if parsed is not None:
                return parsed
    section = config.get(CONFIG_SECTION)
    if isinstance(section, Mapping) and flag_name in section:
        return coerce_flag_value(section[flag_name])
    return None


def flag_enabled(config: Any, flag_name: str, default: bool = False) -> bool:
    """Return whether ``flag_name`` is enabled, defaulting to off.

    Lookup order:
    1. Mapping keys (``llm_focus_v2``, ``LLM_FOCUS_V2`` or
       ``llm_focus.llm_focus_v2``) and a nested ``llm_focus`` section.
    2. A ``LLM_FOCUS_V2`` style attribute on the config object.
    3. ``config.get_config("llm_focus", flag_name)`` when available.
    """
    if config is None:
        return bool(default)

    if isinstance(config, Mapping):
        parsed = _from_mapping(config, flag_name)
        return bool(default) if parsed is None else parsed

    for attribute in (flag_name.upper(), flag_name):
        if hasattr(config, attribute):
            parsed = coerce_flag_value(getattr(config, attribute))
            if parsed is not None:
                return parsed

    get_config = getattr(config, "get_config", None)
    if callable(get_config):
        sentinel = object()
        try:
            value = get_config(CONFIG_SECTION, flag_name, sentinel)
        except TypeError:
            value = sentinel
        if value is not sentinel:
            parsed = coerce_flag_value(value)
            if parsed is not None:
                return parsed

    return bool(default)


def llm_focus_v2_enabled(config: Any) -> bool:
    """Return whether the v2 prompt/schema flag is enabled."""
    return flag_enabled(config, LLM_FOCUS_V2)


def shortlist_v2_enabled(config: Any) -> bool:
    """Return whether the recall-oriented shortlist flag is enabled."""
    return flag_enabled(config, SHORTLIST_V2)


def event_positions_v2_enabled(config: Any) -> bool:
    """Return whether event-driven position management is enabled."""
    return flag_enabled(config, EVENT_POSITIONS_V2)
