"""Dedicated adapter: AnalysisEngine -> schema-valid Signal.

Per-pair AnalysisEngine instances are used stateless: no previous response,
no position context, no brain service. Extraction goes through
PositionExtractor. `TradingStrategy`, `PersistenceManager` and
`CryptoTradingBot` are never reused for position semantics.

The raw LLM response is hashed and never stored or logged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

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
from src.signal_service.storage import SignalStorage, write_heartbeat

logger = logging.getLogger(__name__)

DEFAULT_MIN_RR = 1.5
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_POSITION_SIZE_PCT = 0.05
PROMPT_VERSION = "llm-trader-signal-v1"


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
        """
        self.engine = engine
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_rr = min_rr
        self.min_confidence = min_confidence
        self.use_news = use_news
        self.use_chart = use_chart
        self._flags_applied = False

    # ------------------------------------------------------------ ablation flags
    def _apply_flags(self) -> None:
        """Apply use_news / use_chart ablation flags to the engine."""
        self._flags_applied = True
        try:
            self.engine.model_manager.supports_image_analysis = lambda *_args, **_kwargs: self.use_chart  # type: ignore[assignment]
        except AttributeError:
            pass
        if not self.use_news:
            self._neutralize_news()

    def _neutralize_news(self) -> None:
        """Make the RAG/news layer a no-op so no article can reach the prompt."""
        rag_engine = getattr(self.engine, "rag_engine", None)
        if rag_engine is None:
            return

        async def _empty_context(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            return ("", {})

        rag_engine.retrieve_context = _empty_context  # type: ignore[assignment]

    # ---------------------------------------------------------------- analysis
    async def analyze(self, candle_open_time: Optional[str] = None) -> Signal:
        """Run a stateless analysis and return a validated Signal.

        Args:
            candle_open_time: Optional override of the analysed candle open
                time (ISO-8601). Defaults to the engine's latest candle.

        Returns:
            A validated Signal (HOLD fallback on any failure).
        """
        if not self._flags_applied:
            self._apply_flags()

        result: Dict[str, Any]
        raw_response: str = ""
        try:
            result = await self.engine.analyze_market(
                previous_response=None,
                previous_indicators=None,
                position_context="",
                performance_context="",
                brain_service=None,
                dynamic_thresholds=None,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Analysis failed for %s: %s", self.symbol, exc)
            return self._hold_fallback(reasoning=f"analysis error: {exc}")

        if not isinstance(result, dict) or result.get("error"):
            error = str(result.get("error")) if isinstance(result, dict) else "unknown"
            return self._hold_fallback(reasoning=f"analysis error: {error}")

        raw_response = getattr(self.engine, "last_llm_response", None) or result.get("raw_response") or ""
        draft = self._extract_draft(result, raw_response)
        return self._build_signal(draft, raw_response, candle_open_time)

    # ----------------------------------------------------------------- extraction
    def _extract_draft(self, result: Dict[str, Any], raw_response: str) -> SignalDraft:
        """Extract the decision from the raw LLM response via PositionExtractor."""
        from src.trading.position_extractor import PositionExtractor  # local import: optional dep

        model = str(result.get("model") or "unknown")
        provider = str(result.get("provider") or "unknown")
        raw_hash = hash_raw_response(raw_response) if raw_response else None

        if raw_response:
            extractor = PositionExtractor(logger=None)
            signal, confidence, stop_loss, take_profit, position_size, reasoning, _symbol = \
                extractor.extract_trading_info(raw_response)
        else:
            signal = str(result.get("recommendation", "HOLD")).upper()
            confidence = result.get("confidence", "LOW")
            stop_loss = result.get("stop_loss")
            take_profit = result.get("take_profit")
            position_size = result.get("position_size")
            reasoning = str(result.get("reasoning", ""))

        if signal not in ALLOWED_ACTIONS:
            # CLOSE/UPDATE and unknown labels fall back to HOLD in v1.
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
        """
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
    def _build_signal(
        self,
        draft: SignalDraft,
        raw_response: str,
        candle_open_time: Optional[str],
    ) -> Signal:
        candle_open_dt = (
            parse_iso(candle_open_time)
            if candle_open_time is not None
            else self.current_candle_bounds()
        )
        open_at, close_at, expires_at = self._candle_times(candle_open_dt)
        confidence_float = confidence_to_float(draft.confidence_raw)
        entry_price = self._entry_price()
        approved, reason = self._risk_approved(draft, confidence_float, entry_price)
        if not approved:
            logger.info(
                "risk_approved=false for %s %s: %s", self.symbol, draft.action, reason
            )
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
            position_size_pct=(
                round(float(draft.position_size), 4)
                if draft.position_size is not None
                else DEFAULT_POSITION_SIZE_PCT
            ),
            reasoning=draft.reasoning,
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
    def _hold_fallback(self, reasoning: str) -> Signal:
        """Build the deterministic HOLD fallback signal."""
        candle_open_dt = self.current_candle_bounds()
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
            model="unknown",
            provider="unknown",
            prompt_version=PROMPT_VERSION,
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
        self.candles: List[Sequence[float]] = list(candles)
        self.timestamps = list(timestamps)
        self._original_collect = None
        self._original_enrich = None

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
        self.analyzer._neutralize_news()  # pylint: disable=protected-access
        context.ohlcv_candles = self.candles.copy()
        context.timestamps = self.timestamps.copy()
        self.analyzer._flags_applied = True  # pylint: disable=protected-access

    def restore(self) -> None:
        """Restore the original live data collection hooks."""
        engine = self.engine
        if self._original_collect is not None:
            engine.data_collector.collect_data = self._original_collect
        if self._original_enrich is not None:
            engine._enrich_market_context = self._original_enrich

    def set_as_of(self, as_of: datetime) -> None:
        """Slice the stored series strictly to candles closed at or before `as_of`."""
        context = self.engine.context
        keep = [ts for ts in self.timestamps
                if ts + timedelta(seconds=timeframe_seconds(self.analyzer.timeframe)) <= as_of]
        if not keep:
            raise SignalValidationError("no candles available at or before as_of")
        count = len(keep)
        context.ohlcv_candles = self.candles[:count]  # type: ignore[assignment]
        context.timestamps = keep  # type: ignore[assignment]
        context.current_price = float(context.ohlcv_candles[-1][4])


async def analyze_with_timeout(analyzer: PairAnalyzer, timeout_seconds: float) -> Signal:
    """Run one analysis with a hard timeout and HOLD fallback.

    Args:
        analyzer: PairAnalyzer to run.
        timeout_seconds: Maximum allowed wall time for the LLM call.

    Returns:
        The produced Signal, or a deterministic HOLD fallback on timeout.
    """
    try:
        return await asyncio.wait_for(analyzer.analyze(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("Analysis timed out after %.0fs: HOLD fallback", timeout_seconds)
        return analyzer._hold_fallback(reasoning="analysis timeout")  # pylint: disable=protected-access


async def analyze_and_store(
    analyzer: PairAnalyzer,
    storage: SignalStorage,
    timeout_seconds: float = 300.0,
) -> Signal:
    """Analyze one candle for a pair and append the signal to its JSONL file.

    Heartbeat is refreshed regardless of the outcome.
    """
    signal = await analyze_with_timeout(analyzer, timeout_seconds)
    appended = storage.append(signal)
    write_heartbeat(
        signals_dir=storage.path.rsplit("/", 1)[0],
        symbol=analyzer.symbol,
        timeframe=analyzer.timeframe,
        candle_open_time=signal.candle_open_time,
    )
    if not appended:
        logger.info("Signal for %s already stored: skipped", signal.decision_id)
    return signal