"""Fallback target-geometry model for LLM-focus v2 (policy C).

Policy C makes the LLM's absolute stop-loss/take-profit levels authoritative.
This resolver is kept only as an explicitly documented fallback for signals that
carry no levels at all (expected to be none in practice); it must not be used
when the LLM supplied levels.

The canonical setup taxonomy lives in the external trading-stack repository
(PR #15) and is not modified here. ``src.config.regime_taxonomy.SetupDefinition``
is the single validated source of ``tp_mode``/``tp_submode``; this module reads
those parsed fields and falls back to ``atr_multiple`` only when the field is
absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from src.config.regime_taxonomy import (
    STRUCTURE_MEASURED_MOVE,
    STRUCTURE_MID_RANGE,
    STRUCTURE_VWAP,
    TP_MODE_ATR_MULTIPLE,
    TP_MODE_STRUCTURE,
    VALID_TP_MODES,
    VALID_TP_SUBMODES,
)

TP_MODE_LLM_LEVEL = "llm_level"

VALID_STRUCTURE_MODES = VALID_TP_SUBMODES

#: Measured move stays an explicit, unvalidated option; only the two range
#: setups get a structure default, matching their taxonomy descriptions.
DEFAULT_STRUCTURE_MODES: dict[str, tuple[str, ...]] = {
    "RANGE_MEAN_REVERSION": (STRUCTURE_MID_RANGE, STRUCTURE_VWAP),
    "RANGE_FADE_EXTREME": (STRUCTURE_MID_RANGE, STRUCTURE_VWAP),
}


@dataclass(frozen=True)
class TargetGeometry:
    """Resolved target geometry for one setup."""

    setup_name: str
    direction: str
    requested_mode: str
    effective_mode: str
    structure_mode: str | None
    target_price: float | None
    fallback_used: bool
    fallback_reason: str | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetValidation:
    """Result of validating LLM/strategy target levels."""

    valid: bool
    issues: tuple[str, ...]
    rr_ratio: float | None
    sl_distance_pct: float | None
    tp_distance_pct: float | None


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _normalize_direction(direction: Any) -> str:
    normalized = str(direction or "").strip().upper()
    if normalized not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    return normalized


def _target_on_correct_side(direction: str, entry_price: float, target: float) -> bool:
    if direction == "LONG":
        return target > entry_price
    return target < entry_price


def read_tp_mode(setup: Any) -> tuple[str, str | None, str | None]:
    """Read ``tp_mode``/``tp_submode`` from a parsed setup definition.

    ``SetupDefinition`` validates both fields, so this reader treats them as the
    single source of truth and falls back to ``atr_multiple`` only when the
    field is absent. A raw mapping is still accepted for callers that have not
    adopted the taxonomy parser.

    Returns ``(requested_mode, structure_mode, note)``.
    """
    raw_mode = getattr(setup, "tp_mode", None)
    if raw_mode is None and isinstance(setup, Mapping):
        raw_mode = setup.get("tp_mode")
    if raw_mode in (None, ""):
        return TP_MODE_ATR_MULTIPLE, None, "tp_mode absent: default atr_multiple"
    mode = str(raw_mode).strip().lower()
    if mode == "atr":
        mode = TP_MODE_ATR_MULTIPLE
    if mode not in VALID_TP_MODES:
        return TP_MODE_ATR_MULTIPLE, None, f"invalid tp_mode '{raw_mode}': fallback atr_multiple"

    raw_structure = getattr(setup, "tp_submode", None)
    if raw_structure is None and isinstance(setup, Mapping):
        raw_structure = setup.get("tp_submode")
    structure_mode = None
    if raw_structure not in (None, ""):
        candidate = str(raw_structure).strip().lower()
        if candidate in VALID_TP_SUBMODES:
            structure_mode = candidate
        else:
            return TP_MODE_ATR_MULTIPLE, None, f"invalid structure mode '{raw_structure}': fallback atr_multiple"

    if mode == TP_MODE_STRUCTURE and structure_mode is None:
        return mode, None, None
    return mode, structure_mode, None


def _structure_target(
    mode: str,
    direction: str,
    entry_price: float,
    range_high: float | None,
    range_low: float | None,
    vwap: float | None,
    measured_move_target: float | None,
) -> float | None:
    if mode == STRUCTURE_MID_RANGE:
        if range_high is None or range_low is None or range_high <= range_low:
            return None
        target = (range_high + range_low) / 2
    elif mode == STRUCTURE_VWAP:
        target = vwap
    elif mode == STRUCTURE_MEASURED_MOVE:
        target = measured_move_target
    else:
        return None
    if target is None or not math.isfinite(float(target)) or target <= 0:
        return None
    return float(target)


def resolve_target_geometry(
    setup: Any,
    *,
    direction: str,
    entry_price: float,
    atr: float,
    range_high: float | None = None,
    range_low: float | None = None,
    vwap: float | None = None,
    measured_move_target: float | None = None,
) -> TargetGeometry:
    """Resolve an ATR/structure fallback target (policy C fallback only).

    Never call this when the LLM supplied ``stop_loss``/``take_profit``: policy
    C makes those levels authoritative. This resolver exists only for signals
    carrying no levels at all.

    ``range_mean_reversion`` and ``range_fade_extreme`` try mid-range then
    VWAP when no explicit submode is supplied. If the structural measurement is
    unavailable or on the wrong side of entry, the resolver falls back to the
    setup's ``tp_atr`` multiple. Setups without ``tp_atr`` return ``None`` so
    the caller can keep its own target or reject the signal.
    """
    normalized_direction = _normalize_direction(direction)
    if entry_price <= 0 or atr <= 0:
        raise ValueError("entry_price and atr must be positive")

    setup_name = _normalize_name(_field(setup, "name") or _field(setup, "setup_name"))
    tp_atr = _field(setup, "tp_atr")
    requested_mode, structure_mode, mode_note = read_tp_mode(setup)
    notes: list[str] = [mode_note] if mode_note else []

    def atr_fallback(reason: str) -> TargetGeometry:
        if tp_atr is None:
            return TargetGeometry(
                setup_name=setup_name,
                direction=normalized_direction,
                requested_mode=requested_mode,
                effective_mode=TP_MODE_LLM_LEVEL,
                structure_mode=structure_mode,
                target_price=None,
                fallback_used=True,
                fallback_reason=reason + "; tp_atr absent: keep LLM target",
                notes=tuple(notes),
            )
        distance = float(tp_atr) * atr
        target = entry_price + distance if normalized_direction == "LONG" else entry_price - distance
        return TargetGeometry(
            setup_name=setup_name,
            direction=normalized_direction,
            requested_mode=requested_mode,
            effective_mode=TP_MODE_ATR_MULTIPLE,
            structure_mode=structure_mode,
            target_price=target,
            fallback_used=True,
            fallback_reason=reason,
            notes=tuple(notes),
        )

    if requested_mode == TP_MODE_ATR_MULTIPLE:
        if tp_atr is None:
            return atr_fallback("atr_multiple requested")
        distance = float(tp_atr) * atr
        target = entry_price + distance if normalized_direction == "LONG" else entry_price - distance
        return TargetGeometry(
            setup_name=setup_name,
            direction=normalized_direction,
            requested_mode=requested_mode,
            effective_mode=TP_MODE_ATR_MULTIPLE,
            structure_mode=None,
            target_price=target,
            fallback_used=False,
            fallback_reason=None,
            notes=tuple(notes),
        )

    candidate_modes: tuple[str, ...]
    if structure_mode is not None:
        candidate_modes = (structure_mode,)
    else:
        candidate_modes = DEFAULT_STRUCTURE_MODES.get(setup_name, ())

    failures: list[str] = []
    for mode in candidate_modes:
        target = _structure_target(
            mode,
            normalized_direction,
            entry_price,
            range_high,
            range_low,
            vwap,
            measured_move_target,
        )
        if target is None:
            failures.append(f"{mode} measurement unavailable")
            continue
        if not _target_on_correct_side(normalized_direction, entry_price, target):
            failures.append(f"{mode} target {target:g} is on the wrong side of entry")
            continue
        if mode == STRUCTURE_MEASURED_MOVE:
            notes.append("measured_move is an unvalidated target path")
        return TargetGeometry(
            setup_name=setup_name,
            direction=normalized_direction,
            requested_mode=requested_mode,
            effective_mode=TP_MODE_STRUCTURE,
            structure_mode=mode,
            target_price=target,
            fallback_used=False,
            fallback_reason=None,
            notes=tuple(notes),
        )

    if not candidate_modes:
        failures.append("no structure mode configured")
    return atr_fallback("; ".join(failures) or "structure measurement unavailable")


def validate_target(
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    min_rr: float = 1.0,
    min_distance_pct: float = 0.5,
    max_distance_pct: float = 10.0,
) -> TargetValidation:
    """Validate side, R/R and optional distance bounds.

    The default 0.5-10% bounds are retained for the ATR fallback path. Policy-C
    LLM levels call this with ``min_distance_pct=0`` and an infinite max so no
    legacy absolute cap is applied.
    """
    issues: list[str] = []
    try:
        normalized_direction = _normalize_direction(direction)
    except ValueError as exc:
        return TargetValidation(False, (str(exc),), None, None, None)

    values = (entry_price, stop_loss, take_profit)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
        return TargetValidation(False, ("entry/SL/TP must be finite positive numbers",), None, None, None)

    if normalized_direction == "LONG":
        if not stop_loss < entry_price:
            issues.append("LONG stop_loss must be below entry_price")
        if not take_profit > entry_price:
            issues.append("LONG take_profit must be above entry_price")
    else:
        if not stop_loss > entry_price:
            issues.append("SHORT stop_loss must be above entry_price")
        if not take_profit < entry_price:
            issues.append("SHORT take_profit must be below entry_price")

    sl_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
    tp_distance_pct = abs(take_profit - entry_price) / entry_price * 100
    for label, distance in (("stop_loss", sl_distance_pct), ("take_profit", tp_distance_pct)):
        if distance < min_distance_pct or distance > max_distance_pct:
            issues.append(
                f"{label} distance {distance:.3f}% outside {min_distance_pct:g}-{max_distance_pct:g}%"
            )

    rr_ratio = None
    if sl_distance_pct > 0:
        rr_ratio = tp_distance_pct / sl_distance_pct
        if rr_ratio < min_rr:
            issues.append(f"R/R {rr_ratio:.3f} below minimum {min_rr:g}")

    return TargetValidation(
        valid=not issues,
        issues=tuple(issues),
        rr_ratio=rr_ratio,
        sl_distance_pct=sl_distance_pct,
        tp_distance_pct=tp_distance_pct,
    )
