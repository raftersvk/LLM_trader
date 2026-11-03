"""Pure recall-oriented shortlist for LLM-focus v2.

``build_shortlist`` filters the full candidate universe before any paid LLM
call. P0-A regime/allowed setups, ATR/volatility and open positions are hard
filters. JEV bias/tradeability is advisory: it is attached to the candidate but
does not silently drop a potentially valid signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.llm_focus.deterministic_risk import (
    UNKNOWN,
    assess_regime,
    normalize_setups,
)
from src.llm_focus.flags import coerce_flag_value, shortlist_v2_enabled

REASON_REGIME_UNKNOWN = "regime_unknown"
REASON_NO_ALLOWED_SETUPS = "no_allowed_setups"
REASON_SETUP_NOT_ALLOWED = "setup_not_allowed"
REASON_OPEN_POSITION = "open_position"
REASON_ATR_UNAVAILABLE = "atr_unavailable"
REASON_ATR_TOO_LOW = "atr_too_low"
REASON_ATR_TOO_HIGH = "atr_too_high"

ADVISORY_JEV_NOT_TRADEABLE = "jev_not_tradeable"
ADVISORY_JEV_BIAS_MISMATCH = "jev_bias_mismatch"

DEFAULT_MIN_ATR_PCT = 0.2
DEFAULT_MAX_ATR_PCT = 8.0


@dataclass(frozen=True)
class ShortlistCandidate:
    """Input snapshot for one symbol at one analysis cycle."""

    symbol: str
    regime: Any = None
    allowed_setups: tuple[str, ...] = ()
    setup: str | None = None
    atr_pct: float | None = None
    has_open_position: bool = False
    direction: str | None = None
    jev_bias: str | None = None
    jev_tradeable: bool | None = None


@dataclass(frozen=True)
class ShortlistEntry:
    """Accepted candidate with normalized fields and advisory notes."""

    symbol: str
    setup: str
    regime: str
    allowed_setups: tuple[str, ...]
    atr_pct: float
    volatility: str
    direction: str | None = None
    jev_bias: str | None = None
    jev_tradeable: bool | None = None
    advisories: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShortlistRejection:
    """Rejected symbol with explicit, machine-readable reasons."""

    symbol: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShortlistResult:
    """Shortlist output plus the full rejection audit trail."""

    candidates: tuple[ShortlistEntry, ...]
    rejections: tuple[ShortlistRejection, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(candidate.symbol for candidate in self.candidates)


@dataclass(frozen=True)
class RecallReport:
    """Recall and LLM-call accounting for one universe/cycle."""

    total_signals: int
    captured: int
    missed: tuple[str, ...]
    full_llm_calls: int
    shortlist_llm_calls: int

    @property
    def recall(self) -> float:
        if self.total_signals == 0:
            return 1.0
        return self.captured / self.total_signals

    @property
    def llm_calls_saved(self) -> int:
        return self.full_llm_calls - self.shortlist_llm_calls


def _normalize_direction(direction: Any) -> str | None:
    if direction is None:
        return None
    normalized = str(direction).strip().upper()
    return normalized or None


def _bias_conflicts(direction: str | None, jev_bias: str | None) -> bool:
    if not direction or not jev_bias:
        return False
    bias = jev_bias.strip().upper()
    if bias in ("NEUTRAL", "NONE"):
        return False
    if direction == "LONG":
        return bias in ("SHORT", "SELL", "BEARISH")
    if direction == "SHORT":
        return bias in ("LONG", "BUY", "BULLISH")
    return False


def _reject_reasons(
    candidate: ShortlistCandidate,
    min_atr_pct: float,
    max_atr_pct: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    regime = assess_regime(
        candidate.regime,
        candidate.allowed_setups,
        candidate.atr_pct,
    )

    if regime.regime == UNKNOWN:
        reasons.append(REASON_REGIME_UNKNOWN)
    if not regime.allowed_setups:
        reasons.append(REASON_NO_ALLOWED_SETUPS)

    setup = normalize_setups([candidate.setup]) if candidate.setup else ()
    if regime.allowed_setups and (not setup or setup[0] not in regime.allowed_setups):
        reasons.append(REASON_SETUP_NOT_ALLOWED)

    if candidate.has_open_position:
        reasons.append(REASON_OPEN_POSITION)

    if candidate.atr_pct is None:
        reasons.append(REASON_ATR_UNAVAILABLE)
    else:
        atr_pct = float(candidate.atr_pct)
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            reasons.append(REASON_ATR_UNAVAILABLE)
        elif atr_pct < min_atr_pct:
            reasons.append(REASON_ATR_TOO_LOW)
        elif atr_pct > max_atr_pct:
            reasons.append(REASON_ATR_TOO_HIGH)

    # Keep reason order stable and duplicate-free.
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return tuple(deduped)


def build_shortlist(
    candidates: Iterable[ShortlistCandidate],
    min_atr_pct: float = DEFAULT_MIN_ATR_PCT,
    max_atr_pct: float = DEFAULT_MAX_ATR_PCT,
) -> ShortlistResult:
    """Filter candidates without side effects or live calls.

    The output order follows input order, so identical inputs always produce
    identical outputs.
    """
    if min_atr_pct < 0 or max_atr_pct <= 0 or min_atr_pct >= max_atr_pct:
        raise ValueError("ATR bounds must satisfy 0 <= min_atr_pct < max_atr_pct")

    accepted: list[ShortlistEntry] = []
    rejected: list[ShortlistRejection] = []

    for candidate in candidates:
        reasons = _reject_reasons(candidate, min_atr_pct, max_atr_pct)
        if reasons:
            rejected.append(ShortlistRejection(symbol=candidate.symbol, reasons=reasons))
            continue

        assessment = assess_regime(
            candidate.regime,
            candidate.allowed_setups,
            candidate.atr_pct,
        )
        setup = normalize_setups([candidate.setup])[0]
        advisories: list[str] = []
        if candidate.jev_tradeable is False:
            advisories.append(ADVISORY_JEV_NOT_TRADEABLE)
        if _bias_conflicts(_normalize_direction(candidate.direction), candidate.jev_bias):
            advisories.append(ADVISORY_JEV_BIAS_MISMATCH)

        accepted.append(
            ShortlistEntry(
                symbol=candidate.symbol,
                setup=setup,
                regime=assessment.regime,
                allowed_setups=assessment.allowed_setups,
                atr_pct=float(candidate.atr_pct),
                volatility=assessment.volatility,
                direction=_normalize_direction(candidate.direction),
                jev_bias=candidate.jev_bias,
                jev_tradeable=candidate.jev_tradeable,
                advisories=tuple(advisories),
            )
        )

    return ShortlistResult(candidates=tuple(accepted), rejections=tuple(rejected))


def maybe_build_shortlist(
    config: Any,
    candidates: Iterable[ShortlistCandidate],
    min_atr_pct: float = DEFAULT_MIN_ATR_PCT,
    max_atr_pct: float = DEFAULT_MAX_ATR_PCT,
) -> ShortlistResult | None:
    """Run the shortlist only when ``shortlist_v2`` is enabled.

    Returning ``None`` when disabled is the explicit isolation boundary: the
    legacy full-universe path stays untouched.
    """
    if not shortlist_v2_enabled(config):
        return None
    return build_shortlist(candidates, min_atr_pct=min_atr_pct, max_atr_pct=max_atr_pct)


def evaluate_recall(
    reference_signals: Iterable[str],
    shortlisted_symbols: Iterable[str],
    full_llm_calls: int,
    shortlist_llm_calls: int | None = None,
) -> RecallReport:
    """Compare shortlist coverage against archived BUY/SELL reference signals."""
    references = list(reference_signals)
    shortlisted = set(shortlisted_symbols)
    captured_symbols = [symbol for symbol in references if symbol in shortlisted]
    missed = tuple(symbol for symbol in references if symbol not in shortlisted)
    shortlist_calls = len(shortlisted) if shortlist_llm_calls is None else shortlist_llm_calls
    return RecallReport(
        total_signals=len(references),
        captured=len(captured_symbols),
        missed=missed,
        full_llm_calls=full_llm_calls,
        shortlist_llm_calls=shortlist_calls,
    )


def merge_recall_reports(reports: Iterable[RecallReport]) -> RecallReport:
    """Aggregate per-cycle recall reports."""
    report_list = list(reports)
    total = sum(report.total_signals for report in report_list)
    captured = sum(report.captured for report in report_list)
    missed = tuple(symbol for report in report_list for symbol in report.missed)
    return RecallReport(
        total_signals=total,
        captured=captured,
        missed=missed,
        full_llm_calls=sum(report.full_llm_calls for report in report_list),
        shortlist_llm_calls=sum(report.shortlist_llm_calls for report in report_list),
    )


def meets_recall_guardrail(report: RecallReport, min_recall: float = 0.95) -> bool:
    """Return whether the shortlist preserves enough reference signals."""
    return report.recall >= min_recall


def candidate_from_mapping(record: Mapping[str, Any]) -> ShortlistCandidate:
    """Build a shortlist candidate from an archived/backtest record.

    Accepts both flat records and records with nested ``p0a`` / ``jev``
    sections so the A/B script does not force a storage migration.
    """
    p0a = record.get("p0a") if isinstance(record.get("p0a"), Mapping) else record
    jev = record.get("jev") if isinstance(record.get("jev"), Mapping) else {}

    allowed_setups = p0a.get("allowed_setups", record.get("allowed_setups", ()))
    if isinstance(allowed_setups, str):
        allowed_setups = tuple(part.strip() for part in allowed_setups.split(",") if part.strip())

    atr_pct = p0a.get("atr_pct", p0a.get("atr_percentage", record.get("atr_pct")))
    # Archived records may carry JSON strings ("false", "0") that are truthy
    # when passed to bool(); coerce them so a string "false" is not an open
    # position.
    has_open_position = coerce_flag_value(
        record.get("has_open_position", record.get("open_position", False))
    ) is True

    return ShortlistCandidate(
        symbol=str(record.get("symbol", record.get("pair", ""))),
        regime=p0a.get("regime", record.get("regime")),
        allowed_setups=tuple(allowed_setups),
        setup=record.get("setup", p0a.get("setup")),
        atr_pct=float(atr_pct) if atr_pct is not None else None,
        has_open_position=has_open_position,
        direction=record.get("direction"),
        jev_bias=jev.get("bias", record.get("jev_bias")),
        jev_tradeable=jev.get("tradeable", record.get("jev_tradeable")),
    )
