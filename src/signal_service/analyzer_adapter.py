"""Dedicated adapter: AnalysisEngine -> schema-valid Signal.

Per-pair AnalysisEngine instances are used stateless: no previous response,
no position context, no brain service. Extraction goes through
PositionExtractor. `TradingStrategy`, `PersistenceManager` and
`CryptoTradingBot` are never reused for position semantics.

The raw LLM response is hashed and never stored or logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.signal_service.position_policy import (
    PositionView,
    direction_for_side,
    evaluate_update,
)
from src.signal_service.contract import (
    ALLOWED_ACTIONS,
    Signal,
    SignalDraft,
    SignalValidationError,
    build_decision_id,
    confidence_to_float,
    hash_raw_response,
    iso_z,
    parse_iso,
    timeframe_seconds,
)
from src.signal_service.jev_gate import GateResult, JevGate, build_state
from src.signal_service.storage import SignalStorage, write_heartbeat

logger = logging.getLogger(__name__)

DEFAULT_MIN_RR = 1.5
DEFAULT_MIN_CONFIDENCE = 0.6
def _normalize_position_pct(raw_size) -> float:
    """Normalize a position size to a fraction in [0, 1].

    The upstream parser already emits fractions, while legacy AI replies and the
    fork's text path give percentages > 1; divide those once here.
    """
    if raw_size is None:
        return DEFAULT_POSITION_SIZE_PCT
    try:
        size = float(raw_size)
    except (TypeError, ValueError):
        return DEFAULT_POSITION_SIZE_PCT
    if size > 1:
        size = size / 100.0
    return round(max(0.0, min(1.0, size)), 4)


DEFAULT_POSITION_SIZE_PCT = 0.05
PROMPT_VERSION = "llm-trader-signal-v1"


@dataclass
class UpdateContext:
    """The open position an analysis should produce an UPDATE for."""

    position_id: str
    position_side: str  # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    revision: int
    based_on_entry: str
    current_price: Optional[float] = None
    entry_sl: Optional[float] = None
    entry_tp: Optional[float] = None

    def position_view(self) -> PositionView:
        """Build the duck-typed view consumed by the tightening policy."""
        return PositionView(
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            direction=direction_for_side(self.position_side),
        )

    def effective_entry_sl(self) -> Optional[float]:
        """Original entry-signal SL, the widening base (falls back to current)."""
        if self.entry_sl is not None:
            return self.entry_sl
        return self.stop_loss

    def effective_entry_tp(self) -> Optional[float]:
        """Original entry-signal TP, used when the current TP is unknown."""
        if self.take_profit is not None:
            return self.take_profit
        return self.entry_tp
_UNKNOWN_LABELS = frozenset({"", "unknown", "unspecified", "none", "null"})
_NEWS_ORIGINAL_ATTR = "_signal_service_original_retrieve_context"

#: Signal-only output contract appended as the LAST block of the system prompt
#: via ``system_prompt_suffix``. The shared trading template is also used by the
#: live bot (its JSON branch triggers for the DeepSeek provider) and still
#: allows CLOSE/UPDATE/narrative, so the strict v1 signal schema is imposed as a
#: trailing system instruction instead of changing global template behavior.
#: The suffix is unset for the live bot, whose prompt is unchanged.
SIGNAL_OUTPUT_CONTRACT = (
    "SIGNAL OUTPUT CONTRACT (authoritative; overrides any conflicting style guidance):\n"
    "- Respond with exactly ONE JSON object and nothing else: no prose, no markdown, "
    "no code fences and no comments before or after it.\n"
    '- Required shape: {"analysis": {"signal": "BUY|SELL|HOLD", '
    '"confidence": <integer 0-100>, "entry_price": <number|null>, '
    '"stop_loss": <number|null>, "take_profit": <number|null>, '
    '"position_size": <0.0-1.0>, "reasoning": "<why>", "symbol": "<pair>"}}.\n'
    "- signal MUST be exactly one of BUY, SELL or HOLD. Never emit CLOSE, UPDATE or "
    "any other verb.\n"
    "- confidence MUST be an integer from 0 to 100, or one of the labels HIGH, MEDIUM, "
    "LOW. Never emit MODERATE, \"moderate\", \"neutral\" or any other word.\n"
    "- For BUY: stop_loss < entry_price < take_profit. For SELL: "
    "take_profit < entry_price < stop_loss. HOLD uses null entry_price/stop_loss/take_profit.\n"
    "- NEVER answer with an object keyed by \"educational_analysis_only\", "
    "\"analysis_only\" or any analysis-only envelope, and never replace the decision "
    "object with a narrative report or a \"final_synthesis\". The decision object above "
    "is mandatory on every cycle.\n"
)

#: One-shot repair instruction. Kept short so a repair reuses the original
#: answer instead of paying for a second full market-analysis prompt.
JSON_REPAIR_INSTRUCTION = (
    "Return ONLY one JSON object of the form "
    '{"analysis": {"signal": "BUY|SELL|HOLD", "confidence": <integer 0-100>, '
    '"entry_price": <number|null>, "stop_loss": <number|null>, '
    '"take_profit": <number|null>, "position_size": <0.0-1.0>, '
    '"reasoning": "<why>", "symbol": "<pair>"}} and no prose. '
    "confidence must be an integer 0-100 or one of HIGH/MEDIUM/LOW, never MODERATE. "
    "For BUY: stop_loss < entry_price < take_profit. For SELL: "
    "take_profit < entry_price < stop_loss. symbol is required. Never return an "
    "educational_analysis_only or analysis-only object."
)
JSON_REPAIR_SYSTEM = (
    "You convert a trading analysis answer into the required signal JSON object. "
    "Do not add commentary. The object must contain analysis.signal (BUY/SELL/HOLD), "
    "analysis.confidence (integer 0-100 or HIGH/MEDIUM/LOW, never MODERATE), "
    "analysis.reasoning, analysis.symbol, and entry_price/stop_loss/take_profit/"
    "position_size. For BUY: stop_loss < entry_price < take_profit; for SELL: "
    "take_profit < entry_price < stop_loss. Never answer analysis-only."
)
#: Cap on the previous answer embedded in the repair prompt. The decision and
#: narrative normally sit at the end, so keep the tail when truncating.
MAX_REPAIR_SOURCE_CHARS = 24000
REPAIR_FAILED_REASON = "invalid LLM output: JSON parse error after repair"
ECHO_RERUN_FAILED_REASON = "invalid LLM output: response_format echo after one retry without JSON mode"
#: Keys that make a repaired object a decision rather than a parser/provider error payload.
_DECISION_KEYS = frozenset({"action", "recommendation", "signal", "decision"})
_DECISION_CONTAINERS = ("analysis", "trading_decision", "decision")
#: Keys a provider echoes back from the requested response_format instead of answering.
_FORMAT_ECHO_KEYS = frozenset({"type", "response_format", "format"})
#: Alternate flat schemas the model sometimes emits instead of the canonical
#: ``{"analysis": {"signal", "confidence", "reasoning"}}`` contract. The decision
#: and its execution note are still a real answer, so map the known aliases
#: instead of discarding the candle as "empty reasoning".
_CANONICAL_ACTION_KEYS = ("signal", "action")
_CANONICAL_REASONING_KEYS = ("reasoning", "rationale")
#: Sibling keys that carry the narrative when no canonical reasoning exists.
#: ``reason`` lives inside the ``decision`` object seen in production on the
#: 2026-10-29 candle (candle open 16:00Z, logged 18:07 local); ``final_synthesis``
#: is the long-form fallback.
_ALT_REASONING_KEYS = ("reason", "execution_note", "summary", "final_synthesis")
#: Keys read from a nested ``decision`` object instead of the flat payload.
_DECISION_REASONING_KEYS = ("reason", "rationale", "reasoning", "summary")


def _normalize_extraction_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill canonical extraction keys from known alternate LLM schemas.

    ``PositionExtractor`` only reads ``signal``/``action``, ``confidence`` and
    ``reasoning``/``rationale``. Some replies instead use ``decision``,
    ``confidence_score``, ``execution_note`` or an equity-fraction size key, so
    the decision was silently downgraded to HOLD with an empty reasoning. This
    maps those aliases onto the canonical keys. Existing canonical fields always
    win; aliases only fill a gap, so a genuinely empty answer stays empty.
    """
    payload = dict(data)

    if not any(payload.get(key) for key in _CANONICAL_ACTION_KEYS):
        decision = payload.get("decision")
        if isinstance(decision, str) and decision.strip():
            payload["action"] = decision
        elif isinstance(decision, dict):
            # Some providers nest the decision instead of flattening it:
            # {"decision": {"action": "HOLD", "reason": "..."}}.
            nested_action = decision.get("signal") or decision.get("action")
            if isinstance(nested_action, str) and nested_action.strip():
                payload["action"] = nested_action

    if payload.get("confidence") in (None, ""):
        score = payload.get("confidence_score")
        if score is not None:
            payload["confidence"] = score
        else:
            # A JSON null would stringify to "NONE" and raise in
            # confidence_to_float; let the extractor's MEDIUM default apply.
            payload.pop("confidence", None)

    existing_reasoning = next(
        (
            payload.get(key)
            for key in _CANONICAL_REASONING_KEYS
            if isinstance(payload.get(key), str) and payload.get(key).strip()
        ),
        "",
    )
    if not str(existing_reasoning).strip():
        candidates = []
        decision = payload.get("decision")
        if isinstance(decision, dict):
            candidates.extend(
                decision.get(key) for key in _DECISION_REASONING_KEYS
            )
        candidates.extend(payload.get(key) for key in _ALT_REASONING_KEYS)
        for value in candidates:
            if isinstance(value, str) and value.strip():
                payload["reasoning"] = value
                break

    if payload.get("stop_loss") is None and payload.get("sl") is not None:
        payload["stop_loss"] = payload["sl"]
    if payload.get("take_profit") is None and payload.get("tp") is not None:
        payload["take_profit"] = payload["tp"]

    # A base-currency quantity (e.g. 0.216 ETH) must never be read as an equity
    # fraction; the explicit fraction key wins when both are present.
    fraction = payload.get("position_size_equity_fraction")
    if fraction is not None:
        payload["position_size"] = fraction
    elif payload.get("position_size") is None:
        nested = payload.get("position_sizing")
        if isinstance(nested, dict) and nested.get("final_capital_pct") is not None:
            payload["position_size"] = nested["final_capital_pct"]

    return payload


def _as_utc(moment: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def truncate_context_to_as_of(
    context: Any,
    candles: Any,
    timestamps: Sequence[datetime],
    timeframe: str,
    as_of: datetime,
) -> bool:
    """Keep only candles closed at or before `as_of` in the analysis context.

    A candle opened at T closes at T + timeframe; it is only usable when that
    close is at or before `as_of`. This is the single no-look-ahead rule used
    both by live analysis (hide the forming candle) and by replay.

    Args:
        context: AnalysisContext-like object carrying ohlcv_candles/timestamps.
        candles: OHLCV rows aligned with `timestamps`.
        timestamps: Candle open times.
        timeframe: Candle timeframe label.
        as_of: Latest allowed candle close time.

    Returns:
        True when at least one closed candle remains, False otherwise.
    """
    period = timedelta(seconds=timeframe_seconds(timeframe))
    ordered = [(_as_utc(ts), index) for index, ts in enumerate(timestamps)]
    keep = [(ts, index) for ts, index in ordered if ts + period <= as_of]
    if candles is None or len(candles) == 0 or not keep:
        context.ohlcv_candles = None
        context.timestamps = []
        context.current_price = None
        return False
    count = min(len(keep), len(candles))
    kept_candles = candles[:count]
    context.ohlcv_candles = kept_candles
    context.timestamps = [ts for ts, _index in keep[:count]]
    try:
        context.current_price = float(kept_candles[-1][4])
    except (IndexError, TypeError, ValueError):
        context.current_price = None
    return True


class ClosedCandleWindow:
    """Hide the still-forming candle from a single analysis.

    Installed around one `PairAnalyzer.analyze(candle_open_time=...)` call:
    after the engine collects market data, the context is truncated to candles
    closed at or before the analysed candle's close, and `current_price` is
    recomputed from the last closed candle.
    """

    def __init__(self, engine: Any, timeframe: str, candle_open_dt: datetime) -> None:
        """Bind the window to an engine and the analysed candle.

        Args:
            engine: AnalysisEngine whose collector must be wrapped.
            timeframe: Candle timeframe label.
            candle_open_dt: Open time of the candle being analysed.
        """
        self.engine = engine
        self.timeframe = timeframe
        self.as_of = candle_open_dt + timedelta(seconds=timeframe_seconds(timeframe))
        self.state: Dict[str, bool] = {"empty": False}
        self._original_collect = None

    async def _collect_and_truncate(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._original_collect(*args, **kwargs)
        context = self.engine.context
        timestamps = getattr(context, "timestamps", None)
        kept = truncate_context_to_as_of(
            context,
            getattr(context, "ohlcv_candles", None),
            list(timestamps) if timestamps is not None else [],
            self.timeframe,
            self.as_of,
        )
        self.state["empty"] = not kept
        return result

    def __enter__(self) -> Dict[str, bool]:
        collector = getattr(self.engine, "data_collector", None)
        if collector is None:
            # Test doubles and injected engines may not expose a collector; the
            # production AnalysisEngine always does. Never fail the analysis.
            logger.warning(
                "Engine (%s) exposes no data_collector: closed-candle window not enforced",
                self.timeframe,
            )
            return self.state
        self._original_collect = collector.collect_data
        collector.collect_data = self._collect_and_truncate
        return self.state

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self._original_collect is not None:
            self.engine.data_collector.collect_data = self._original_collect



class PairAnalyzer:
    """Turns one AnalysisEngine analysis into a contract-valid Signal."""

    def __init__(
        self,
        engine: Any,
        symbol: str,
        timeframe: str = "1h",
        min_rr: float = DEFAULT_MIN_RR,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        use_news: bool = True,
        use_chart: bool = True,
        jev_gate: Optional[JevGate] = None,
        json_response_format: Optional[bool] = None,
    ) -> None:
        """Bind a dedicated engine for a single pair.

        Args:
            engine: AnalysisEngine instance (one per pair).
            symbol: Trading pair, e.g. "BTC/USDT".
            timeframe: Candle timeframe label.
            min_rr: Minimum risk/reward ratio to approve risk.
            min_confidence: Minimum confidence to approve risk.
            use_news: Enable news context (ablation flag).
            use_chart: Enable chart image analysis (ablation flag).
            jev_gate: Optional JEV gate; modes pre/post/pre_post. When omitted
                or off, no JEV call is made and behaviour is unchanged.
            json_response_format: Opt into response_format={"type":"json_object"}
                for this pair's signal analysis only. ``None`` reads
                [signal_service] json_response_format from the engine config.
        """
        self.engine = engine
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_rr = min_rr
        self.min_confidence = min_confidence
        self.use_news = use_news
        self.use_chart = use_chart
        self.jev_gate = jev_gate
        self.json_response_format = json_response_format
        self._flags_applied = False

    # ------------------------------------------------------------ ablation flags
    def _apply_flags(self) -> None:
        """Bind this pair's use_news / use_chart ablations to its engine.

        `model_manager` and `rag_engine` are process-wide singletons shared by
        every per-pair engine, so this must run before *each* analysis: another
        pair's flags may be in effect, and a one-shot application would let the
        last started pair silently override the earlier ones.
        """
        self._flags_applied = True
        try:
            self.engine.model_manager.supports_image_analysis = lambda *_args, **_kwargs: self.use_chart  # type: ignore[assignment]
        except AttributeError:
            pass
        self._bind_news_ablation(self.use_news)

    def _bind_news_ablation(self, enabled: bool) -> None:
        """Enable or disable news retrieval on the (possibly shared) RAG engine.

        The original retrieval is saved on the RAG engine itself, so an earlier
        pair's neutralization can be undone for a pair that keeps news enabled.
        """
        rag_engine = getattr(self.engine, "rag_engine", None)
        if rag_engine is None:
            return
        original = getattr(rag_engine, _NEWS_ORIGINAL_ATTR, None)
        if enabled:
            if original is not None:
                rag_engine.retrieve_context = original  # type: ignore[assignment]
            return
        if original is None:
            try:
                setattr(rag_engine, _NEWS_ORIGINAL_ATTR, rag_engine.retrieve_context)
            except AttributeError:
                return
        async def _empty_context(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            return ("", {})

        rag_engine.retrieve_context = _empty_context  # type: ignore[assignment]

    # ---------------------------------------------------------------- analysis
    async def analyze(
        self,
        candle_open_time: Optional[str] = None,
        position_context: str = "",
        update_context: Optional[UpdateContext] = None,
    ) -> Signal:
        """Run an analysis and return a validated Signal.

        The default call is stateless (``position_context=""``) and produces an
        entry signal, exactly as before. Passing ``update_context`` switches to
        the position-aware path: the analysis receives ``position_context`` and
        may return an UPDATE (v1.1) instead of an entry. A refused/unusable
        update is returned as HOLD so callers simply ignore it.

        Args:
            candle_open_time: Optional open time of the closed candle to
                analyse (ISO-8601). When provided, the still-forming candle is
                hidden from the analysis: only candles closed at or before this
                candle's close are visible. Defaults to the engine's current
                (live) behaviour.

        Returns:
            A validated Signal (HOLD fallback on any failure).
        """
        # Re-apply on every analysis: engine model managers and RAG engines are
        # shared process-wide, so another pair's flags may currently be bound.
        self._apply_flags()

        # JEV gate: `pre` only decides whether the LLM is called; `post` vetoes a
        # non-HOLD decision. The phase-1 risk layer keeps flagging an explicit
        # JEV bias conflict whenever a JEV payload is attached.
        as_of = self._analysis_as_of(candle_open_time)
        # The prompt for a closed candle must describe that candle's close, not
        # the wall-clock time when the analysis happens to run. This holds for
        # live (candle_open_time from the scheduler) and replay alike.
        self._set_engine_analysis_as_of(as_of)
        pre_result: Optional[GateResult] = None
        state: Optional[Dict[str, Any]] = None
        # Position updates are exempt from the entry gate: JEV, TTL and
        # risk_approved never block managing or exiting an open position.
        if update_context is None and self.jev_gate is not None and self.jev_gate.uses_pre():
            # Live-mode context may still hold the candles truncated during the
            # previous cycle: refresh and truncate before gating so the
            # pre-gate decides on the candle that just closed.
            await self._refresh_context(as_of)
            state = self._jev_state(as_of)
            if state is None:
                logger.warning("JEV pre-gate skipped for %s: no closed-candle state", self.symbol)
            else:
                pre_result = await self.jev_gate.check_pre(state)
                if pre_result.consulted and not pre_result.allowed:
                    logger.info("JEV pre-gate blocked %s: %s", self.symbol, pre_result.reason)
                    return self._gate_blocked_signal(pre_result, candle_open_time)

        # Hide the still-forming candle from the analysis window.
        candle_open_dt = parse_iso(candle_open_time) if candle_open_time is not None else None
        result: Dict[str, Any]
        window_state: Dict[str, bool] = {"empty": False}
        try:
            if candle_open_dt is None:
                result = await self._analyze_market(position_context=position_context)
            else:
                with ClosedCandleWindow(self.engine, self.timeframe, candle_open_dt) as window_state:
                    result = await self._analyze_market(position_context=position_context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Analysis failed for %s: %s", self.symbol, exc)
            return self._hold_fallback(reasoning=f"analysis error: {exc}", candle_open_dt=candle_open_dt)

        if window_state["empty"]:
            return self._hold_fallback(
                reasoning="no closed candle available at candle close",
                candle_open_dt=candle_open_dt,
            )

        raw_response = ""
        if isinstance(result, dict):
            raw_response = getattr(self.engine, "last_llm_response", None) or result.get("raw_response") or ""

        # Existing terminal guards run before any repair: a truncated, filtered
        # or empty completion must keep its real diagnostic and never be turned
        # into a valid-looking decision by the repair call.
        invalid_reason = self._invalid_response_reason(result, raw_response)
        if invalid_reason:
            logger.warning("Invalid LLM response for %s: %s", self.symbol, invalid_reason)
            model, provider = self._resolve_model_provider(result if isinstance(result, dict) else {})
            return self._hold_fallback(
                reasoning=invalid_reason,
                candle_open_dt=candle_open_dt,
                model=model,
                provider=provider,
                raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
            )

        # A format echo ({"type": "json_object"}) carries no analysis: converting
        # it would force the model to invent a decision. Re-run the FULL analysis
        # once with JSON mode off so the original prompt produces the analysis
        # that was never emitted. Only applies while JSON mode is actually on.
        format_echo = self._json_format_enabled() and self._is_response_format_echo(raw_response)
        if format_echo:
            original_raw = raw_response
            finish_reason = self._last_finish_reason(result)
            logger.warning(
                "response_format echo for %s: re-running full analysis once without JSON mode "
                "(finish_reason=%s, content_chars=%d)",
                self.symbol, finish_reason or "unknown", len(original_raw),
            )
            rerun = await self._rerun_without_json_mode(candle_open_dt, position_context)
            if rerun is None:
                logger.warning("Invalid LLM response for %s: %s", self.symbol, ECHO_RERUN_FAILED_REASON)
                model, provider = self._resolve_model_provider(result if isinstance(result, dict) else {})
                return self._hold_fallback(
                    reasoning=ECHO_RERUN_FAILED_REASON,
                    candle_open_dt=candle_open_dt,
                    model=model,
                    provider=provider,
                    raw_response_hash=hash_raw_response(original_raw) if original_raw else None,
                )
            result, raw_response = rerun
            rerun_invalid = self._invalid_response_reason(result, raw_response)
            if rerun_invalid:
                logger.warning("Invalid LLM response for %s after retry: %s", self.symbol, rerun_invalid)
                model, provider = self._resolve_model_provider(result)
                return self._hold_fallback(
                    reasoning=rerun_invalid,
                    candle_open_dt=candle_open_dt,
                    model=model,
                    provider=provider,
                    raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
                )
            if self._is_json_parse_error(result) or self._is_response_format_echo(raw_response):
                logger.warning("Invalid LLM response for %s: %s", self.symbol, ECHO_RERUN_FAILED_REASON)
                model, provider = self._resolve_model_provider(result)
                return self._hold_fallback(
                    reasoning=ECHO_RERUN_FAILED_REASON,
                    candle_open_dt=candle_open_dt,
                    model=model,
                    provider=provider,
                    raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
                )
        elif self._is_json_parse_error(result):
            original_raw = raw_response
            finish_reason = self._last_finish_reason(result)
            logger.warning(
                "JSON parse error for %s: requesting one repair (finish_reason=%s, content_chars=%d)",
                self.symbol, finish_reason or "unknown", len(original_raw),
            )
            repaired = await self._repair_json_response(result, original_raw)
            if repaired is None:
                logger.warning(
                    "Invalid LLM response for %s: %s (finish_reason=%s, content_chars=%d)",
                    self.symbol, REPAIR_FAILED_REASON, finish_reason or "unknown", len(original_raw),
                )
                model, provider = self._resolve_model_provider(result if isinstance(result, dict) else {})
                return self._hold_fallback(
                    reasoning=REPAIR_FAILED_REASON,
                    candle_open_dt=candle_open_dt,
                    model=model,
                    provider=provider,
                    raw_response_hash=hash_raw_response(original_raw) if original_raw else None,
                )
            result, raw_response = repaired
            # Re-check the repaired answer: a repair that itself comes back
            # truncated, filtered or empty must not bypass the same guards.
            repaired_invalid = self._invalid_response_reason(result, raw_response)
            if repaired_invalid:
                logger.warning("Invalid LLM response for %s after repair: %s", self.symbol, repaired_invalid)
                model, provider = self._resolve_model_provider(result)
                return self._hold_fallback(
                    reasoning=repaired_invalid,
                    candle_open_dt=candle_open_dt,
                    model=model,
                    provider=provider,
                    raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
                )

        veto_reason: Optional[str] = None
        try:
            draft = self._extract_draft(result, raw_response)

            # Part of the invalid-response guard above: a completion that only
            # yields an empty/whitespace reasoning is unusable. Accepting it
            # would persist a maximum-confidence HOLD backed by no analysis.
            if not draft.reasoning.strip():
                # Distinguish an analysis-only reply (full write-up, no usable
                # decision) from a genuinely empty answer. Both stay HOLD and
                # are never mapped to BUY/SELL.
                motif = (
                    "analysis-only response"
                    if self._is_analysis_only_response(raw_response)
                    else "empty reasoning"
                )
                logger.warning(
                    "Invalid LLM response for %s: %s (finish_reason=%s, content_chars=%d)",
                    self.symbol, motif, self._last_finish_reason(result) or "unknown", len(raw_response),
                )
                return self._hold_fallback(
                    reasoning=f"invalid LLM output: {motif}",
                    candle_open_dt=candle_open_dt,
                    model=draft.model,
                    provider=draft.provider,
                    raw_response_hash=draft.raw_response_hash,
                )

            post_result: Optional[GateResult] = None
            if self.jev_gate is not None and self.jev_gate.uses_post() and draft.action in ("BUY", "SELL"):
                # Rebuild the state from the just-refreshed context: after
                # _analyze_market (and the closed-candle window truncation) the
                # context includes the candle analysed this cycle, so the post
                # gate never decides on a one-candle-stale series.
                state = self._jev_state(as_of)
                if state is None:
                    logger.warning("JEV post-gate skipped for %s: no closed-candle state", self.symbol)
                else:
                    post_result = await self.jev_gate.check_post(state, draft.action)
                    if not post_result.allowed:
                        veto_reason = post_result.reason
                        logger.info("JEV post-gate vetoed %s %s: %s", self.symbol, draft.action, veto_reason)

            jev_result = post_result if post_result is not None and post_result.signal else pre_result
            if jev_result is not None and jev_result.signal is not None:
                draft.jev = jev_result.signal.to_payload()

            if update_context is not None:
                return self._build_update_signal(
                    draft, raw_response, candle_open_time, update_context,
                )
            return self._build_signal(draft, raw_response, candle_open_time, veto_reason=veto_reason)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Extraction and contract validation failures must still produce a
            # signal: one explicable-but-invalid LLM answer must not lose the
            # whole candle (no row and no heartbeat) for the pair.
            logger.warning("Signal build failed for %s: %s", self.symbol, exc)
            model, provider = self._resolve_model_provider(result)
            return self._hold_fallback(
                reasoning=f"invalid LLM output: {exc}",
                candle_open_dt=candle_open_dt,
                model=model,
                provider=provider,
                raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
            )

    # --------------------------------------------------------------- jev state
    async def _refresh_context(self, as_of: datetime) -> None:
        """Collect fresh candles and truncate the context to closed-at-or-before as_of.

        The pre-gate runs before the analysis, but the engine context may still
        hold the candles truncated during the previous cycle. Refreshing here
        lets the pre-gate decide on the candle that just closed instead of the
        previous one. Best-effort: any collection error leaves the context
        untouched and the gate fails open downstream.
        """
        collector = getattr(self.engine, "data_collector", None)
        if collector is None:
            return
        try:
            await collector.collect_data()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("JEV pre-gate data refresh failed for %s: %s", self.symbol, exc)
            return
        context = self.engine.context
        if context is None:
            return
        timestamps = getattr(context, "timestamps", None)
        truncate_context_to_as_of(
            context,
            getattr(context, "ohlcv_candles", None),
            list(timestamps) if timestamps is not None else [],
            self.timeframe,
            as_of,
        )

    def _analysis_as_of(self, candle_open_time: Optional[str]) -> datetime:
        """Reference time of the analysis: the analysed candle close or now."""
        if candle_open_time is not None:
            return parse_iso(candle_open_time) + timedelta(seconds=timeframe_seconds(self.timeframe))
        return datetime.now(timezone.utc)

    def _set_engine_analysis_as_of(self, as_of: datetime) -> None:
        """Push the prompt clock to the engine (or its prompt builder).

        Real engines expose `set_analysis_as_of`; lightweight test doubles may
        only accept the attribute, so both shapes are supported.
        """
        setter = getattr(self.engine, "set_analysis_as_of", None)
        if callable(setter):
            setter(as_of)
            return
        try:
            self.engine.analysis_as_of = as_of
        except AttributeError:
            logger.warning("Engine for %s cannot accept an analysis clock", self.symbol)

    def _jev_state(self, as_of: datetime) -> Optional[Dict[str, Any]]:
        """Build the no-look-ahead JEV state from the engine's closed candles."""
        context = getattr(self.engine, "context", None)
        candles = getattr(context, "ohlcv_candles", None) if context is not None else None
        # OHLCV data is a NumPy array in the real engine: never use truthiness.
        if candles is None or len(candles) == 0:
            return None
        timestamps = getattr(context, "timestamps", None)
        try:
            if timestamps is not None and len(timestamps) == len(candles):
                return build_state(
                    self.symbol,
                    self.timeframe,
                    candles,
                    timestamps=[self._as_datetime(ts) for ts in timestamps],
                    as_of=as_of,
                )
            # No timestamps: be conservative and drop the last (possibly forming) candle.
            if len(candles) < 3:
                return None
            return build_state(self.symbol, self.timeframe, list(candles)[:-1])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # A state-building failure must never abort the analysis: the
            # documented JEV fail-open contract covers every failure mode.
            logger.warning("Cannot build JEV state for %s: %s", self.symbol, exc)
            return None

    @staticmethod
    def _as_datetime(value: Any) -> datetime:
        """Coerce datetime / epoch milliseconds / ISO string to aware UTC."""
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            seconds = float(value) / 1000.0 if float(value) > 1e11 else float(value)
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        return parse_iso(str(value))

    def _resolve_model_provider(self, result: Dict[str, Any]) -> tuple[str, str]:
        """Resolve the real model/provider, never returning 'unspecified'."""
        engine_config = getattr(self.engine, "config", None)
        model = str(result.get("model") or "").strip()
        provider = str(result.get("provider") or "").strip()
        fallback_model = str(getattr(engine_config, "LITELLM_MODEL", "") or "").strip()
        fallback_provider = str(getattr(engine_config, "PROVIDER", "") or "").strip()
        if not model or model.lower() in _UNKNOWN_LABELS:
            model = fallback_model or "unknown"
        if not provider or provider.lower() in _UNKNOWN_LABELS:
            provider = fallback_provider or "unknown"
        return model, provider

    def _json_format_enabled(self) -> bool:
        """Resolve the opt-in structured-output flag for this analyzer."""
        if self.json_response_format is not None:
            return bool(self.json_response_format)
        config = getattr(self.engine, "config", None)
        if config is None:
            return False
        value = getattr(config, "SIGNAL_JSON_RESPONSE_FORMAT", None)
        if value is None:
            getter = getattr(config, "get_config", None)
            if callable(getter):
                value = getter("signal_service", "json_response_format", False)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _json_response_format(self) -> Optional[Dict[str, str]]:
        """Return the response_format payload for the signal call, if enabled."""
        return {"type": "json_object"} if self._json_format_enabled() else None

    def _is_response_format_echo(self, raw_response: str) -> bool:
        """True when the reply only repeats the requested ``response_format`` object.

        Live JSON-mode providers can answer with the literal ``{"type":
        "json_object"}`` (seen on OpenCode Go with provider=litellm). It parses as
        valid JSON, so the parser reports no error; without this check the empty
        decision would fall through to an unexplained HOLD. Treat it as a repair
        trigger, never as an answer.
        """
        text = (raw_response or "").strip()
        if not text:
            return False
        parsed: Any = None
        parser = self._unified_parser()
        parse_json_object = getattr(parser, "parse_json_object", None)
        if callable(parse_json_object):
            parsed = parse_json_object(text)
        if parsed is None:
            try:
                parsed = json.loads(text)
            except ValueError:
                return False
        return (
            isinstance(parsed, dict)
            and bool(parsed)
            and set(parsed) <= _FORMAT_ECHO_KEYS
        )

    @staticmethod
    def _is_json_parse_error(result: Any) -> bool:
        """Detect the parser's explicit json_parse_error signal."""
        if not isinstance(result, dict):
            return False
        if result.get("parse_error"):
            return True
        validation = result.get("response_validation")
        if not isinstance(validation, dict):
            return False
        return any(
            isinstance(error, dict) and error.get("type") == "json_parse_error"
            for error in validation.get("errors", [])
        )

    def _last_finish_reason(self, result: Any) -> Optional[str]:
        """Best-effort provider finish_reason for diagnostics (never the body)."""
        if isinstance(result, dict) and result.get("finish_reason"):
            return str(result["finish_reason"])
        model_manager = getattr(self.engine, "model_manager", None)
        reason = getattr(model_manager, "last_finish_reason", None)
        return str(reason) if reason else None

    @classmethod
    def _has_decision_payload(cls, parsed: Any) -> bool:
        """Return True when a parsed object actually carries a trading decision.

        A bare JSON object is not enough: ``format_error_response`` is also a
        fenced JSON object (``{"analysis": {"summary": ...}}``), so accepting it
        would turn a failed provider call into a fake "successful" repair.
        """
        if not isinstance(parsed, dict):
            return False
        if any(key in parsed for key in _DECISION_KEYS):
            return True
        for container_key in _DECISION_CONTAINERS:
            nested = parsed.get(container_key)
            if isinstance(nested, dict) and any(key in nested for key in _DECISION_KEYS):
                return True
        return False

    async def _repair_json_response(
        self, result: Any, raw_response: str,
    ) -> Optional[tuple[Dict[str, Any], str]]:
        """Relaunch exactly once with a short JSON-only repair instruction.

        Reuses the original answer instead of re-running the full market
        analysis. Returns ``(result, repaired_text)`` only when the repaired
        answer parses as a JSON object; otherwise None (the caller falls back
        to HOLD). Raises nothing: a repair failure is never fatal.
        """
        if not raw_response.strip():
            return None
        model_manager = getattr(self.engine, "model_manager", None)
        send_prompt = getattr(model_manager, "send_prompt", None)
        if not callable(send_prompt):
            logger.warning("Cannot repair JSON for %s: model manager has no send_prompt", self.symbol)
            return None
        source = raw_response.strip()
        if len(source) > MAX_REPAIR_SOURCE_CHARS:
            source = source[-MAX_REPAIR_SOURCE_CHARS:]
        prompt = f"{JSON_REPAIR_INSTRUCTION}\n\nPrevious analysis response:\n{source}"
        try:
            # Deliberately no response_format here: when the first call echoed
            # the requested JSON object, re-sending the same forced format would
            # reproduce the echo. The prompt itself asks for the JSON object.
            repaired_text = await send_prompt(
                prompt=prompt,
                system_message=JSON_REPAIR_SYSTEM,
            )
        except TypeError:
            # Test doubles / older model managers may not accept response_format.
            logger.warning("Cannot repair JSON for %s: send_prompt lacks response_format", self.symbol)
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("JSON repair request failed for %s: %s", self.symbol, exc)
            return None
        if not isinstance(repaired_text, str) or not repaired_text.strip():
            return None
        parser = self._unified_parser()
        if parser is None:
            return None
        parsed = parser.parse_json_object(repaired_text)
        if parsed is None or not self._has_decision_payload(parsed):
            logger.warning(
                "JSON repair for %s returned no decision payload; treating it as failed",
                self.symbol,
            )
            return None
        repaired_result = dict(result) if isinstance(result, dict) else {}
        repaired_result.pop("parse_error", None)
        repaired_result.pop("error", None)
        repaired_result["raw_response"] = repaired_text
        repaired_result["response_validation"] = {
            "schema": "1.0", "status": "valid", "valid": True, "errors": [],
        }
        try:
            self.engine.last_llm_response = repaired_text
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        logger.info("JSON repair succeeded for %s", self.symbol)
        return repaired_result, repaired_text

    def _invalid_response_reason(self, result: Any, raw_response: str) -> Optional[str]:
        """Return why an LLM response is unusable, or None when it is usable.

        Detects the silent-HOLD trap: an empty/truncated completion must never
        become a clean, risk-approved HOLD. The matching check on the extracted
        reasoning (empty/whitespace-only) runs in ``analyze`` after extraction.
        """
        if not isinstance(result, dict):
            return "invalid LLM response: no result"
        if result.get("error"):
            return f"invalid LLM response: {result['error']}"
        finish_reason = (
            result.get("finish_reason")
            or getattr(getattr(self.engine, "model_manager", None), "last_finish_reason", None)
        )
        empty = not raw_response.strip()
        if finish_reason == "length":
            label = "empty content" if empty else "truncated content"
            return f"invalid LLM response: {label} (finish_reason=length)"
        if finish_reason == "content_filter":
            return "invalid LLM response: content filtered (finish_reason=content_filter)"
        if empty:
            suffix = f" (finish_reason={finish_reason})" if finish_reason else ""
            return f"invalid LLM response: empty content{suffix}"
        return None

    async def _analyze_market(
        self, json_mode: Optional[bool] = None, position_context: str = ""
    ) -> Dict[str, Any]:
        """Call the engine statelessly (one analysis, no continuity state).

        ``json_mode`` defaults to the signal opt-in. The echo-recovery path
        passes ``False`` so the same full prompt is re-run as prose + JSON,
        which gives the model the market context the echo never answered.

        ``position_context`` is empty in the default (stateless) mode. The
        position-aware mode passes a compact block so the model can propose a
        position UPDATE; ``previous_response``/``brain_service`` stay None.
        """
        if json_mode is None:
            json_mode = self._json_format_enabled()
        return await self.engine.analyze_market(
            previous_response=None,
            previous_indicators=None,
            position_context=position_context,
            performance_context="",
            brain_service=None,
            dynamic_thresholds=None,
            system_prompt_suffix=SIGNAL_OUTPUT_CONTRACT,
            response_format={"type": "json_object"} if json_mode else None,
            json_object_output=json_mode,
        )

    async def _rerun_without_json_mode(
        self, candle_open_dt: Optional[datetime], position_context: str = "",
    ) -> Optional[tuple[Dict[str, Any], str]]:
        """Re-run the full analysis once without response_format after an echo.

        The retry must see the SAME closed-candle window as the original call:
        ``AnalysisEngine.analyze_market`` re-fetches OHLCV, so without a fresh
        window it would analyse the still-forming candle (no-look-ahead breach).
        """
        window_state: Dict[str, bool] = {"empty": False}
        try:
            if candle_open_dt is None:
                result = await self._analyze_market(json_mode=False, position_context=position_context)
            else:
                with ClosedCandleWindow(self.engine, self.timeframe, candle_open_dt) as window_state:
                    result = await self._analyze_market(
                        json_mode=False, position_context=position_context
                    )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Full-analysis retry without JSON mode failed for %s: %s", self.symbol, exc)
            return None
        if window_state["empty"]:
            logger.warning(
                "Full-analysis retry without JSON mode for %s found no closed candle",
                self.symbol,
            )
            return None
        if not isinstance(result, dict):
            return None
        raw_response = (
            result.get("raw_response")
            or getattr(self.engine, "last_llm_response", None)
            or ""
        )
        return result, raw_response


    # ----------------------------------------------------------------- extraction
    def _unified_parser(self) -> Any:
        """Return the engine's UnifiedParser, matching the live bot's extraction.

        The live bot builds `PositionExtractor(logger, parser)` from the
        AnalysisResultProcessor's parser; without it the extractor never sees
        JSON decisions (numeric confidence, `action` keys) and silently degrades
        to the weaker regex fallback.
        """
        result_processor = getattr(self.engine, "result_processor", None)
        return getattr(result_processor, "unified_parser", None)

    def _alternate_payload(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Return the parsed answer normalized from a known alternate schema.

        ``None`` when no parser is available or the text holds no JSON object.
        Only used as a recovery path after canonical extraction produced no
        reasoning, so normal answers keep the existing extraction untouched.
        """
        parser = self._unified_parser()
        parse_json_object = getattr(parser, "parse_json_object", None)
        if not callable(parse_json_object):
            return None
        parsed = parse_json_object(raw_response)
        if not isinstance(parsed, dict):
            return None
        for container in ("analysis", "trading_decision", "decision"):
            nested = parsed.get(container)
            if isinstance(nested, dict):
                # Merge rather than replace: the narrative keys
                # (execution_note/final_synthesis) often sit next to the nested
                # decision and must survive the unwrap.
                parsed = {**parsed, **nested}
                break
        return _normalize_extraction_payload(parsed)

    def _is_analysis_only_response(self, raw_response: str) -> bool:
        """Return True when the reply is explicitly flagged analysis-only.

        Such a reply carries a full technical write-up but is not an executable
        signal; it must never become an order and must reach a distinct
        diagnostic instead of the generic "empty reasoning" motif. The flag may
        sit at the top level or inside a known decision container.
        """
        parser = self._unified_parser()
        parse_json_object = getattr(parser, "parse_json_object", None)
        if not callable(parse_json_object):
            return False
        parsed = parse_json_object(raw_response)
        if not isinstance(parsed, dict):
            return False
        candidates = [parsed]
        for key in _DECISION_CONTAINERS:
            nested = parsed.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        for candidate in candidates:
            for key in ("educational_analysis_only", "analysis_only"):
                value = candidate.get(key)
                if value is True:
                    return True
                if isinstance(value, str) and value.strip().lower() == "true":
                    return True
        return False

    def _extract_draft(self, result: Dict[str, Any], raw_response: str) -> SignalDraft:
        """Extract the decision from the raw LLM response via PositionExtractor."""
        from src.trading.position_extractor import PositionExtractor  # local import: optional dep

        model, provider = self._resolve_model_provider(result)
        raw_hash = hash_raw_response(raw_response) if raw_response else None

        if raw_response:
            parser = self._unified_parser()
            if parser is None:
                logger.warning(
                    "Engine for %s exposes no UnifiedParser: falling back to regex extraction",
                    self.symbol,
                )
            extractor = PositionExtractor(logger=None, unified_parser=parser)
            (signal, confidence, stop_loss, take_profit, position_size, reasoning,
             _symbol, numeric_confidence) = extractor.extract_trading_info_with_confidence(raw_response)
            # Keep the raw 0-100 score so contract.confidence_to_float can apply
            # its 5% rounding; fall back to the label when none was provided.
            if numeric_confidence is not None:
                confidence = numeric_confidence
            if not str(reasoning).strip():
                # The model sometimes answers a known alternate schema
                # (decision/execution_note/...) that the canonical extractor
                # ignores. Map the aliases and retry before declaring the answer
                # unusable; a truly empty answer still yields "".
                alternate = self._alternate_payload(raw_response)
                if alternate is not None:
                    alt = extractor.extract_trading_info_with_confidence(alternate)
                    alt_signal, alt_confidence, alt_sl, alt_tp, alt_size, alt_reasoning = alt[:6]
                    alt_numeric = alt[6] if len(alt) > 6 else None
                    if str(alt_reasoning).strip():
                        signal = alt_signal
                        confidence = alt_numeric if alt_numeric is not None else alt_confidence
                        stop_loss = alt_sl
                        take_profit = alt_tp
                        position_size = alt_size
                        reasoning = alt_reasoning
        else:
            signal = str(result.get("recommendation", "HOLD")).upper()
            confidence = result.get("confidence", "LOW")
            stop_loss = result.get("stop_loss")
            take_profit = result.get("take_profit")
            position_size = result.get("position_size")
            reasoning = str(result.get("reasoning", ""))

        if raw_response and self._is_analysis_only_response(raw_response):
            # Fail closed: an analysis-only envelope must never become an order,
            # even when it embeds a BUY/SELL decision with coherent SL/TP. Drop
            # the executable fields and force HOLD; keep a marked narrative for
            # audit. confidence is zeroed so a forced HOLD never looks approved.
            logger.warning(
                "Analysis-only response for %s: forcing HOLD (fail-closed, was %s)",
                self.symbol,
                signal,
            )
            signal = "HOLD"
            confidence = 0.0
            stop_loss = None
            take_profit = None
            position_size = 0.0
            if str(reasoning).strip():
                reasoning = f"analysis-only response: {reasoning}"

        if signal not in ALLOWED_ACTIONS:
            # Unknown labels fall back to HOLD. UPDATE/CLOSE are preserved here
            # for the position-aware path; the entry builder downgrades them to
            # HOLD so a position action can never open a new trade.
            signal = "HOLD"

        return SignalDraft(
            action=signal,
            confidence_raw=confidence,
            sl=stop_loss,
            tp=take_profit,
            position_size=position_size,
            reasoning=reasoning[:8000],
            model=model,
            provider=provider,
            prompt_version=PROMPT_VERSION,
            latency_ms=None,
            raw_response_hash=raw_hash,
        )

    # ------------------------------------------------------------------ timing
    def current_candle_bounds(self, now: Optional[datetime] = None) -> datetime:
        """Return the open time of the currently running candle aligned to `now`.

        Args:
            now: Reference time; defaults to current UTC time.
        """
        ref = now or datetime.now(timezone.utc)
        period = timeframe_seconds(self.timeframe)
        epoch = ref.replace(tzinfo=timezone.utc).timestamp()
        return datetime.fromtimestamp(epoch // period * period, tz=timezone.utc)

    def _candle_times(self, candle_open_dt: datetime) -> tuple[str, str, str]:
        period = timedelta(seconds=timeframe_seconds(self.timeframe))
        open_at = iso_z(candle_open_dt)
        close_at = iso_z(candle_open_dt + period)
        expires_at = iso_z(candle_open_dt + 2 * period)
        return open_at, close_at, expires_at

    # ------------------------------------------------------------ risk checks
    def _risk_approved(
        self,
        draft: SignalDraft,
        confidence_float: float,
        entry_price: Optional[float],
    ) -> tuple[bool, str]:
        """Evaluate risk rules and return (approved, reason).

        risk_approved=false when the SL/TP is missing or incoherent, the RR is
        under threshold, the confidence is under threshold or the signal is
        expired. JEV veto and spread/liquidity bounds are the caller's gate in
        v1 (see PLAN.md section 4); they are honoured when provided via jev.

        A HOLD never triggers an entry (Freqtrade requires action BUY/SELL and
        risk_approved=true), so it is never approved: keeping it false avoids a
        misleading "approved" state in monitoring.
        """
        if draft.action == "HOLD":
            return False, "HOLD does not trigger an entry"
        if draft.action in ("BUY", "SELL"):
            if draft.sl is None or draft.tp is None:
                return False, "missing SL/TP"
            if entry_price is None:
                return False, "entry price unavailable"
            if draft.action == "BUY" and not (draft.sl < entry_price < draft.tp):
                return False, "incoherent SL/TP for BUY"
            if draft.action == "SELL" and not (draft.tp < entry_price < draft.sl):
                return False, "incoherent SL/TP for SELL"
            risk = abs(entry_price - draft.sl)
            reward = abs(draft.tp - entry_price)
            if risk <= 0 or reward / risk < self.min_rr:
                return False, f"RR below threshold ({self.min_rr})"
        if confidence_float < self.min_confidence:
            return False, f"confidence below threshold ({self.min_confidence})"
        jev = draft.jev or {}
        if jev.get("tradeability") is False:
            return False, "JEV veto"
        if draft.action == "BUY" and jev.get("bias") == "short":
            return False, "JEV bias conflict"
        if draft.action == "SELL" and jev.get("bias") == "long":
            return False, "JEV bias conflict"
        return True, "ok"

    # ------------------------------------------------------------- signal build
    def _build_update_signal(
        self,
        draft: SignalDraft,
        raw_response: str,
        candle_open_time: Optional[str],
        ctx: UpdateContext,
    ) -> Signal:
        """Build a v1.1 UPDATE/CLOSE record for an open position.

        Entry gates (TTL, JEV, min RR) never apply here: this is a position
        management action. Only UPDATE/CLOSE are honoured; a BUY/SELL/HOLD
        answer becomes a HOLD (ignored) so an entry schema can never be
        reinterpreted as a position update. A non-compliant proposal returns a
        HOLD signal with the refusal reason in the reasoning.
        """
        candle_open_dt = (
            parse_iso(candle_open_time)
            if candle_open_time is not None
            else self.current_candle_bounds()
        )
        open_at, close_at, expires_at = self._candle_times(candle_open_dt)
        created_at = iso_z(datetime.now(timezone.utc))
        decision_id = f"{ctx.based_on_entry}-u{ctx.revision}"

        def _base(action: str, **kwargs: Any) -> Signal:
            payload = dict(
                schema_version="1.1",
                decision_id=decision_id,
                source="llm_trader",
                symbol=self.symbol,
                timeframe=self.timeframe,
                candle_open_time=open_at,
                candle_close_time=close_at,
                created_at=created_at,
                expires_at=expires_at,
                action=action,
                confidence=0.0,
                risk_approved=False,
                sl=None,
                tp=None,
                position_size_pct=0.0,
                reasoning=draft.reasoning[:8000],
                model=draft.model,
                provider=draft.provider,
                prompt_version=draft.prompt_version,
                raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
                position_id=ctx.position_id,
                position_side=ctx.position_side,
                revision=ctx.revision,
                based_on_entry=ctx.based_on_entry,
                source_price=ctx.current_price,
            )
            payload.update(kwargs)
            return Signal(**payload)

        if draft.action not in ("UPDATE", "CLOSE"):
            return self._hold_fallback(
                reasoning=(
                    f"position mode requires UPDATE or CLOSE, got {draft.action}"
                ),
                candle_open_dt=candle_open_dt,
                model=draft.model,
                provider=draft.provider,
                raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
            )
        if draft.action == "CLOSE":
            signal = _base("CLOSE", close_reason="llm_close")
            signal.validate()
            return signal

        current_price = ctx.current_price or self._entry_price() or ctx.entry_price
        # v1.1 deliberately does not synthesise a dynamic TP: no ATR reaches the
        # position path. When the TP is unknown, only a safe SL tightening can
        # be emitted (SL-only), never an invented take-profit.
        decision = evaluate_update(
            ctx.position_view(),
            draft.sl,
            draft.tp,
            current_price,
            max(timeframe_seconds(self.timeframe) // 60, 1),
            original_sl=ctx.effective_entry_sl(),
        )
        if not decision.allowed:
            logger.info(
                "Position update refused for %s %s: %s",
                self.symbol, ctx.position_id, decision.reason,
            )
            return self._hold_fallback(
                reasoning=f"update refused: {decision.reason}",
                candle_open_dt=candle_open_dt,
                model=draft.model,
                provider=draft.provider,
                raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
            )

        final_sl = decision.stop_loss if decision.stop_loss is not None else ctx.stop_loss
        final_tp = decision.take_profit if decision.take_profit is not None else ctx.take_profit
        signal = _base(
            "UPDATE",
            sl=final_sl,
            tp=final_tp,
            risk_approved=True,
            reasoning=f"{draft.reasoning} [{decision.reason}]"[:8000],
        )
        signal.validate()
        return signal

    def _build_signal(
        self,
        draft: SignalDraft,
        raw_response: str,
        candle_open_time: Optional[str],
        veto_reason: Optional[str] = None,
    ) -> Signal:
        candle_open_dt = (
            parse_iso(candle_open_time)
            if candle_open_time is not None
            else self.current_candle_bounds()
        )
        if draft.action in ("UPDATE", "CLOSE"):
            # A position action must never open a trade. In stateless/entry mode
            # it is downgraded to HOLD.
            return self._hold_fallback(
                reasoning=f"position action {draft.action} is not an entry",
                candle_open_dt=candle_open_dt,
                model=draft.model,
                provider=draft.provider,
                raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
            )
        open_at, close_at, expires_at = self._candle_times(candle_open_dt)
        confidence_float = confidence_to_float(draft.confidence_raw)
        entry_price = self._entry_price()
        approved, reason = self._risk_approved(draft, confidence_float, entry_price)
        if veto_reason is not None:
            approved = False
            reason = veto_reason
        if not approved:
            logger.info(
                "risk_approved=false for %s %s: %s", self.symbol, draft.action, reason
            )
        reasoning = draft.reasoning
        if veto_reason is not None:
            reasoning = f"{reasoning} [JEV veto: {veto_reason}]"[:8000]
        signal = Signal(
            schema_version="1.0",
            decision_id=build_decision_id(self.symbol, self.timeframe, open_at),
            source="llm_trader",
            symbol=self.symbol,
            timeframe=self.timeframe,
            candle_open_time=open_at,
            candle_close_time=close_at,
            created_at=iso_z(datetime.now(timezone.utc)),
            expires_at=expires_at,
            action=draft.action,
            confidence=round(confidence_float, 4),
            risk_approved=approved,
            sl=draft.sl,
            tp=draft.tp,
            position_size_pct=_normalize_position_pct(draft.position_size),
            reasoning=reasoning,
            model=draft.model,
            provider=draft.provider,
            prompt_version=draft.prompt_version,
            latency_ms=None,
            raw_response_hash=hash_raw_response(raw_response) if raw_response else None,
            jev=draft.jev,
        )
        signal.validate()
        return signal

    def _entry_price(self) -> Optional[float]:
        """Best-effort entry price from the engine's collected context."""
        context = getattr(self.engine, "context", None)
        if context is None:
            return None
        current_price = getattr(context, "current_price", None)
        if isinstance(current_price, (int, float)) and current_price > 0:
            return float(current_price)
        candles = getattr(context, "ohlcv_candles", None)
        if candles is not None and len(candles) > 0:
            try:
                return float(candles[-1][4])  # last close
            except (IndexError, TypeError, ValueError):
                return None
        return None

    # --------------------------------------------------------------- fallback
    # --------------------------------------------------------------- fallback
    def _gate_blocked_signal(self, result: GateResult, candle_open_time: Optional[str]) -> Signal:
        """Build the HOLD signal produced when the JEV pre-gate blocks the LLM.

        The LLM is never called in this path. The signal stays contract-valid:
        HOLD, risk_approved=false, `source="jev"` and the JEV payload.
        """
        candle_open_dt = (
            parse_iso(candle_open_time)
            if candle_open_time is not None
            else self.current_candle_bounds()
        )
        open_at, close_at, expires_at = self._candle_times(candle_open_dt)
        jev_signal = result.signal
        signal = Signal(
            schema_version="1.0",
            decision_id=build_decision_id(self.symbol, self.timeframe, open_at),
            source="jev",
            symbol=self.symbol,
            timeframe=self.timeframe,
            candle_open_time=open_at,
            candle_close_time=close_at,
            created_at=iso_z(datetime.now(timezone.utc)),
            expires_at=expires_at,
            action="HOLD",
            confidence=0.0,
            risk_approved=False,
            sl=None,
            tp=None,
            position_size_pct=DEFAULT_POSITION_SIZE_PCT,
            reasoning=result.reason[:8000],
            model=jev_signal.model if jev_signal is not None else "unknown",
            provider="jev",
            prompt_version=PROMPT_VERSION,
            jev=jev_signal.to_payload() if jev_signal is not None else None,
        )
        signal.validate()
        return signal

    def _hold_fallback(
        self,
        reasoning: str,
        candle_open_dt: Optional[datetime] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        raw_response_hash: Optional[str] = None,
    ) -> Signal:
        """Build the deterministic HOLD fallback signal.

        Args:
            reasoning: Human-readable cause, stored in the signal.
            candle_open_dt: Open time of the analysed candle; defaults to the
                current candle when no closed candle was supplied.
            model: Real model name; falls back to config.LITELLM_MODEL.
            provider: Real provider name; falls back to config.PROVIDER.
            raw_response_hash: sha256 fingerprint of the raw response, if any.
        """
        candle_open_dt = candle_open_dt or self.current_candle_bounds()
        resolved_model, resolved_provider = self._resolve_model_provider(
            {"model": model, "provider": provider}
        )
        open_at, close_at, expires_at = self._candle_times(candle_open_dt)
        signal = Signal(
            schema_version="1.0",
            decision_id=build_decision_id(self.symbol, self.timeframe, open_at),
            source="llm_trader",
            symbol=self.symbol,
            timeframe=self.timeframe,
            candle_open_time=open_at,
            candle_close_time=close_at,
            created_at=iso_z(datetime.now(timezone.utc)),
            expires_at=expires_at,
            action="HOLD",
            confidence=0.0,
            risk_approved=False,
            sl=None,
            tp=None,
            position_size_pct=DEFAULT_POSITION_SIZE_PCT,
            reasoning=reasoning[:8000],
            model=resolved_model,
            provider=resolved_provider,
            prompt_version=PROMPT_VERSION,
            raw_response_hash=raw_response_hash,
        )
        signal.validate()
        return signal


class ReplayDataInjector:
    """Feed historical candles into one engine with strict no-look-ahead.

    Replay replaces the live data collection with a candle slice that is
    strictly older than or equal to the replayed candle close, neutralizes
    live enrichment and news so no future data can reach the prompt.
    """

    def __init__(
        self,
        analyzer: PairAnalyzer,
        candles: Sequence[Sequence[float]],
        timestamps: List[datetime],
    ) -> None:
        """Store the full historical series.

        Args:
            analyzer: PairAnalyzer bound to the replayed engine.
            candles: OHLCV rows (timestamp_ms or index, open, high, low, close, volume).
            timestamps: Candle open times parallel to `candles`.
        """
        self.analyzer = analyzer
        self.engine = analyzer.engine
        # Keep the CCXT 6-column NumPy layout the live collector produces:
        # consumers slice columns (ohlcv[:, 1:6]) and read column 4 as the close.
        self.candles: np.ndarray = np.array(candles, dtype=np.float64)
        self.timestamps = list(timestamps)
        self._original_collect = None
        self._original_enrich = None
        self._original_article_snapshot = None
        self._original_collector_article_urls = None
        self._collector_had_article_urls = False
        self._original_engine_article_urls: Any = {}

    def install(self) -> None:
        """Bypass live collection and live enrichment for this engine."""
        engine = self.engine
        context = engine.context

        async def _replay_collect(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            return {"success": True, "errors": [], "market_context": "", "article_urls": {}}

        async def _replay_enrich(*_args: Any, **_kwargs: Any) -> None:
            return None

        self._original_collect = engine.data_collector.collect_data
        self._original_enrich = getattr(engine, "_enrich_market_context", None)
        engine.data_collector.collect_data = _replay_collect  # type: ignore[assignment]
        engine._enrich_market_context = _replay_enrich  # type: ignore[assignment]

        # Replay must never attach current news metadata: the engine's cached
        # article URLs describe today, not the replayed candle. Neutralize both
        # the snapshot source and the collector's cached URLs, then restore on
        # exit so live behaviour is unchanged.
        collector = engine.data_collector
        self._collector_had_article_urls = hasattr(collector, "article_urls")
        if self._collector_had_article_urls:
            self._original_collector_article_urls = collector.article_urls
        collector.article_urls = {}
        rag_engine = getattr(engine, "rag_engine", None)
        snapshot = getattr(rag_engine, "get_latest_article_urls_snapshot", None)
        if callable(snapshot):
            self._original_article_snapshot = snapshot
            rag_engine.get_latest_article_urls_snapshot = lambda: {}  # type: ignore[assignment]
        self._original_engine_article_urls = getattr(engine, "article_urls", {})
        engine.article_urls = {}

        # Bind the same use_news/use_chart ablation flags as the run command;
        # forcing news off and pre-setting _flags_applied made --use-chart and
        # --no-news no-ops during replay.
        self.analyzer._apply_flags()  # pylint: disable=protected-access
        context.ohlcv_candles = self.candles.copy()
        context.timestamps = self.timestamps.copy()

    def restore(self) -> None:
        """Restore the original live data collection hooks."""
        engine = self.engine
        if self._original_collect is not None:
            engine.data_collector.collect_data = self._original_collect
        if self._original_enrich is not None:
            engine._enrich_market_context = self._original_enrich
        if self._collector_had_article_urls:
            engine.data_collector.article_urls = self._original_collector_article_urls
        else:
            try:
                del engine.data_collector.article_urls
            except AttributeError:
                pass
        engine.article_urls = self._original_engine_article_urls
        if self._original_article_snapshot is not None:
            engine.rag_engine.get_latest_article_urls_snapshot = self._original_article_snapshot

    def set_as_of(self, as_of: datetime) -> None:
        """Slice the stored series strictly to candles closed at or before `as_of`."""
        context = self.engine.context
        kept = truncate_context_to_as_of(
            context, self.candles, self.timestamps, self.analyzer.timeframe, as_of,
        )
        if not kept:
            raise SignalValidationError("no candles available at or before as_of")


async def analyze_with_timeout(
    analyzer: PairAnalyzer,
    timeout_seconds: float,
    candle_open_time: Optional[str] = None,
) -> Signal:
    """Run one analysis with a hard timeout and HOLD fallback.

    Args:
        analyzer: PairAnalyzer to run.
        timeout_seconds: Maximum allowed wall time for the LLM call.
        candle_open_time: Open time of the closed candle being analysed.

    Returns:
        The produced Signal, or a deterministic HOLD fallback on timeout.
    """
    try:
        return await asyncio.wait_for(
            analyzer.analyze(candle_open_time=candle_open_time), timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Analysis timed out after %.0fs: HOLD fallback", timeout_seconds)
        candle_open_dt = parse_iso(candle_open_time) if candle_open_time is not None else None
        return analyzer._hold_fallback(  # pylint: disable=protected-access
            reasoning="analysis timeout", candle_open_dt=candle_open_dt,
        )


def _refresh_heartbeat(
    analyzer: PairAnalyzer,
    storage: SignalStorage,
    candle_open_time: Optional[str],
    signal: Optional[Signal] = None,
) -> None:
    """Best-effort heartbeat refresh; never masks the caller's own failure."""
    try:
        if signal is not None:
            heartbeat_candle = signal.candle_open_time
        elif candle_open_time is not None:
            heartbeat_candle = iso_z(parse_iso(candle_open_time))
        else:
            heartbeat_candle = iso_z(analyzer.current_candle_bounds())
        write_heartbeat(
            signals_dir=storage.path.rsplit("/", 1)[0],
            symbol=analyzer.symbol,
            timeframe=analyzer.timeframe,
            candle_open_time=heartbeat_candle,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Liveness must not turn a storage/analysis error into a different one.
        logger.warning("Heartbeat refresh failed for %s: %s", analyzer.symbol, exc)


async def analyze_and_store(
    analyzer: PairAnalyzer,
    storage: SignalStorage,
    timeout_seconds: float = 300.0,
    candle_open_time: Optional[str] = None,
    dedupe: bool = True,
) -> Signal:
    """Analyze one closed candle for a pair and append the signal to its JSONL file.

    Heartbeat is refreshed regardless of the outcome, including analysis,
    validation and append failures.

    Args:
        dedupe: Passed to storage. Set False only for debug output directories.
    """
    signal: Optional[Signal] = None
    appended = False
    try:
        signal = await analyze_with_timeout(
            analyzer, timeout_seconds, candle_open_time=candle_open_time,
        )
        appended = storage.append(signal, dedupe=dedupe)
    finally:
        _refresh_heartbeat(analyzer, storage, candle_open_time, signal)
    if not appended:
        logger.info("Signal for %s already stored: skipped", signal.decision_id)
    return signal
