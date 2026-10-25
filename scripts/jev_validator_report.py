"""Read-only measurement of the JEV post gate as validator of ``risk_approved``.

In ``post`` mode the signal adapter calls JEV only for BUY/SELL decisions. When
JEV vetoes, the signal keeps its action but ``risk_approved`` becomes false and
the reasoning gets a ``[JEV veto: <reason>]`` suffix. This tool answers one
question: did those vetoes save money or cost money?

For every BUY/SELL signal it computes the counterfactual trade (enter at the
close of the analysed candle, in the signal's direction, exit ``h`` hours
later, net of a fixed round-trip cost) from hourly Binance candles and then
reports, per symbol and horizon:

* how many signals were approved, vetoed (tradeability or bias conflict) or
  rejected by a non-JEV risk rule;
* the net P&L of the vetoed trades (negative = the veto saved money);
* the false-veto cost: the vetoed trades that would have been profitable.

The tool is strictly read-only with respect to the signals directory: it only
reads ``*.jsonl`` files from it, and it refuses to write the report or the
candle cache inside that directory. The candle source is injectable so unit
tests never touch the network.

Usage::

    python scripts/jev_validator_report.py --out /tmp/jev-validator.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_service.contract import parse_iso, timeframe_seconds  # noqa: E402
from src.signal_service.storage import default_signals_dir  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_COST_PCT = 0.10
DEFAULT_HORIZONS = (1, 2, 4)
DEFAULT_TARGET = 200
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "runtime", "jev_bench", "jev_validator_report.md")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "runtime", "jev_bench", "candles_cache")

#: Minimum number of vetoed, candle-matched signals before a horizon counts.
MIN_VETO_SAMPLE = 20
#: A false-veto cost above this share of the avoided losses is not "small".
MAX_FALSE_VETO_SHARE = 0.25
#: Number of rows in the worst-false-veto table.
MAX_FALSE_VETO_ROWS = 8

VETO_MARKER = "[JEV veto:"
ACTION_DIRECTION = {"BUY": 1.0, "SELL": -1.0}
TRADE_ACTIONS = frozenset(ACTION_DIRECTION)


# --------------------------------------------------------------------- data
@dataclass(frozen=True)
class Candle:
    """One closed OHLCV candle."""

    open_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


#: A candle source maps (symbol, timeframe, start, end) to candle rows.
CandleSource = Callable[[str, str, datetime, datetime], Sequence[Candle]]


class CandleSourceError(RuntimeError):
    """Raised when candles cannot be fetched and no usable cache exists."""


@dataclass(frozen=True)
class SignalRecord:
    """The subset of one stored signal this report needs."""

    decision_id: str
    symbol: str
    timeframe: str
    candle_open_time: datetime
    action: str
    risk_approved: bool
    reasoning: str
    jev: Optional[Dict[str, object]]


@dataclass
class LoadStats:
    """Counters produced while reading the signals directory."""

    files_read: int = 0
    lines_read: int = 0
    holds_ignored: int = 0
    malformed: int = 0
    duplicates: int = 0
    empty_files: int = 0


@dataclass(frozen=True)
class VetoInfo:
    """Parsed ``[JEV veto: ...]`` suffix."""

    present: bool
    kind: Optional[str] = None
    noul: Optional[float] = None
    detail: str = ""


@dataclass(frozen=True)
class Counterfactual:
    """Counterfactual result of one signal at one horizon."""

    horizon: int
    status: str
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    net_pct: Optional[float] = None


@dataclass(frozen=True)
class EvaluatedSignal:
    """One BUY/SELL signal with its classification and P&L results."""

    record: SignalRecord
    status: str  # approved | vetoed | other_rejected
    veto_kind: Optional[str]
    noul: Optional[float]
    results: Dict[int, Counterfactual]


@dataclass
class SignalCounts:
    """Per-symbol signal counts."""

    total: int = 0
    approved: int = 0
    vetoed: int = 0
    tradeability: int = 0
    bias: int = 0
    other_veto: int = 0
    other_rejected: int = 0

    def add_veto(self, kind: Optional[str]) -> None:
        """Record one veto in the matching reason bucket."""
        self.vetoed += 1
        if kind == "tradeability":
            self.tradeability += 1
        elif kind == "bias":
            self.bias += 1
        else:
            self.other_veto += 1


@dataclass
class HorizonStats:
    """Aggregated counterfactual P&L at one horizon."""

    horizon: int
    approved_total: int = 0
    approved_evaluable: int = 0
    approved_net_pct: float = 0.0
    vetoed_total: int = 0
    vetoed_evaluable: int = 0
    vetoed_net_pct: float = 0.0
    false_veto_count: int = 0
    false_veto_cost_pct: float = 0.0
    loss_avoided_pct: float = 0.0


@dataclass
class SymbolSummary:
    """Counts and horizon statistics for one symbol."""

    symbol: str
    counts: SignalCounts = field(default_factory=SignalCounts)
    horizons: Dict[int, HorizonStats] = field(default_factory=dict)


@dataclass
class ReportData:
    """Everything the renderer needs, independent of input paths."""

    horizons: Tuple[int, ...]
    cost_pct: float
    target: int
    stats: LoadStats
    symbols: Dict[str, SymbolSummary]
    overall: Dict[int, HorizonStats]
    evaluations: List[EvaluatedSignal]
    notes: List[str]

    @property
    def total_signals(self) -> int:
        """Number of BUY/SELL signals collected."""
        return sum(summary.counts.total for summary in self.symbols.values())

    @property
    def sample_is_sufficient(self) -> bool:
        """True when the collected sample reaches the target."""
        return self.total_signals >= self.target


# ------------------------------------------------------------------ loading
def parse_veto_info(reasoning: object) -> VetoInfo:
    """Parse the ``[JEV veto: <reason>]`` suffix from a signal reasoning.

    The adapter wraps the gate reason, and the gate reason already starts with
    ``JEV veto:``, so live rows read ``[JEV veto: JEV veto: not tradeable
    (noul=0.273 < 0.5)]``. Repeated prefixes are stripped. The ``noul``
    probability is only returned when the text actually contains it; it is
    never inferred from the ``tradeability`` boolean.

    Args:
        reasoning: Stored reasoning text (may be None or non-string).

    Returns:
        Parsed veto information; ``present`` is False without the marker.
    """
    text = reasoning if isinstance(reasoning, str) else ""
    index = text.rfind(VETO_MARKER)
    if index < 0:
        return VetoInfo(present=False)
    detail = text[index + len(VETO_MARKER):].strip()
    if detail.endswith("]"):
        detail = detail[:-1].strip()
    while detail.lower().startswith("jev veto:"):
        detail = detail[len("jev veto:"):].strip()
    lowered = detail.lower()
    noul: Optional[float] = None
    match = re.search(r"noul\s*=\s*([0-9]+(?:\.[0-9]+)?)", lowered)
    if match:
        try:
            noul = float(match.group(1))
        except ValueError:
            noul = None
    if "not tradeable" in lowered or noul is not None:
        kind = "tradeability"
    elif "bias conflict" in lowered or "conflicts with" in lowered:
        kind = "bias"
    elif "confidence" in lowered:
        kind = "confidence"
    else:
        kind = "other"
    return VetoInfo(present=True, kind=kind, noul=noul, detail=detail)


def classify_signal(record: SignalRecord) -> Tuple[str, Optional[str], Optional[float]]:
    """Classify a BUY/SELL signal as approved, vetoed or otherwise rejected.

    The suffix is authoritative. When a row predates the suffix but carries a
    JEV payload and ``risk_approved=false``, the payload's tradeability/bias
    fields are used as a fallback. No ``noul`` value is invented in that case.

    Args:
        record: One stored BUY/SELL signal.

    Returns:
        ``(status, veto_kind, noul)``.
    """
    veto = parse_veto_info(record.reasoning)
    if veto.present:
        return "vetoed", veto.kind, veto.noul
    if record.risk_approved:
        return "approved", None, None
    jev = record.jev or {}
    if jev.get("tradeability") is False:
        return "vetoed", "tradeability", None
    bias = jev.get("bias")
    if (record.action == "BUY" and bias == "short") or (
        record.action == "SELL" and bias == "long"
    ):
        return "vetoed", "bias", None
    return "other_rejected", None, None


def _decision_key(data: Dict[str, object]) -> Optional[str]:
    """Return the deduplication key of one raw row, or None when unusable."""
    symbol = str(data.get("symbol") or "").strip()
    timeframe = str(data.get("timeframe") or "1h").strip()
    open_at = str(data.get("candle_open_time") or "").strip()
    if not symbol or not open_at:
        return None
    decision_id = str(data.get("decision_id") or "").strip()
    if decision_id:
        return decision_id
    return f"{symbol}|{timeframe}|{open_at}"


def load_signals(signals_dir: str) -> Tuple[List[SignalRecord], LoadStats]:
    """Read every ``*.jsonl`` file in a directory, tolerating bad rows.

    The directory is only read, never written. Files are visited in sorted
    order and the first occurrence of a decision id wins, which makes the
    output deterministic for a fixed input.

    Args:
        signals_dir: Directory containing the per-pair JSONL files.

    Returns:
        BUY/SELL records sorted by symbol, candle open time and decision id,
        plus loading counters.

    Raises:
        SystemExit: when the directory does not exist.
    """
    stats = LoadStats()
    if not os.path.isdir(signals_dir):
        raise SystemExit(f"signals directory not found: {signals_dir}")
    records: List[SignalRecord] = []
    seen: set[str] = set()
    for name in sorted(os.listdir(signals_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(signals_dir, name)
        if not os.path.isfile(path):
            continue
        stats.files_read += 1
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                stats.lines_read += 1
                try:
                    data = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    stats.malformed += 1
                    continue
                if not isinstance(data, dict):
                    stats.malformed += 1
                    continue
                key = _decision_key(data)
                if key is None:
                    stats.malformed += 1
                    continue
                try:
                    action = str(data.get("action") or "").strip().upper()
                    open_at = parse_iso(str(data["candle_open_time"]))
                    timeframe = str(data.get("timeframe") or "1h").strip()
                    timeframe_seconds(timeframe)
                except (KeyError, TypeError, ValueError):
                    stats.malformed += 1
                    continue
                if action not in ("BUY", "SELL", "HOLD"):
                    stats.malformed += 1
                    continue
                if action == "HOLD":
                    stats.holds_ignored += 1
                    continue
                if key in seen:
                    stats.duplicates += 1
                    continue
                seen.add(key)
                jev = data.get("jev")
                records.append(
                    SignalRecord(
                        decision_id=key,
                        symbol=str(data.get("symbol") or "").strip(),
                        timeframe=timeframe,
                        candle_open_time=open_at,
                        action=action,
                        # Only a real boolean true approves; strings never do.
                        risk_approved=data.get("risk_approved") is True,
                        reasoning=str(data.get("reasoning") or ""),
                        jev=jev if isinstance(jev, dict) else None,
                    )
                )
    records.sort(key=lambda item: (item.symbol, item.candle_open_time, item.decision_id))
    return records, stats


# -------------------------------------------------------------------- prices
def read_candles_csv(path: str) -> List[Candle]:
    """Read ``time,open,high,low,close,volume`` candles from a CSV file.

    Args:
        path: CSV path.

    Returns:
        Candles sorted by open time.

    Raises:
        FileNotFoundError: when the file does not exist.
    """
    candles: List[Candle] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return candles
        for row in reader:
            if not row or len(row) < 6:
                continue
            try:
                candles.append(
                    Candle(
                        open_at=parse_iso(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                )
            except (TypeError, ValueError):
                continue
    candles.sort(key=lambda candle: candle.open_at)
    return candles


def write_candles_csv(path: str, candles: Sequence[Candle]) -> None:
    """Write candles atomically in the calibration CSV format."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write("time,open,high,low,close,volume\n")
        for candle in candles:
            handle.write(
                f"{candle.open_at.strftime('%Y-%m-%dT%H:%M:%SZ')},"
                f"{candle.open:.8f},{candle.high:.8f},{candle.low:.8f},"
                f"{candle.close:.8f},{candle.volume:.8f}\n"
            )
    os.replace(temporary, path)


def _merge_candles(*series: Sequence[Candle]) -> List[Candle]:
    """Merge candle series by open time; later series win."""
    merged: Dict[datetime, Candle] = {}
    for candles in series:
        for candle in candles:
            merged[candle.open_at] = candle
    return [merged[key] for key in sorted(merged)]


def _filter_window(
    candles: Sequence[Candle], start: datetime, end: datetime
) -> List[Candle]:
    """Keep candles whose open time is inside ``[start, end]``."""
    return [candle for candle in candles if start <= candle.open_at <= end]


def _covers(candles: Sequence[Candle], start: datetime, end: datetime) -> bool:
    """True when the series spans the whole requested window."""
    if not candles:
        return False
    return candles[0].open_at <= start and candles[-1].open_at >= end


def closed_candles(
    candles: Sequence[Candle], timeframe: str, now: Optional[datetime] = None
) -> List[Candle]:
    """Drop the still-forming candle so only closed candles are usable."""
    reference = now or datetime.now(timezone.utc)
    period = timedelta(seconds=timeframe_seconds(timeframe))
    return [candle for candle in candles if candle.open_at + period <= reference]


class CcxtCandleSource:
    """Binance hourly candles with a local CSV cache.

    The cache directory is the only place this class writes. A cache file that
    already spans the requested window avoids the network entirely; otherwise
    the missing window is fetched with ccxt and merged into the cache.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        exchange: Optional[object] = None,
    ) -> None:
        """Bind a cache directory and an optional pre-built ccxt exchange."""
        self.cache_dir = cache_dir
        self._exchange = exchange

    @staticmethod
    def cache_path(cache_dir: str, symbol: str, timeframe: str) -> str:
        """Return the cache CSV path for one symbol/timeframe."""
        token = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        return os.path.join(cache_dir, token)

    def _exchange_instance(self) -> object:
        if self._exchange is None:
            import ccxt  # local import: tests never need the network stack

            self._exchange = ccxt.binance({"enableRateLimit": True})
        return self._exchange

    def _fetch(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> List[Candle]:
        exchange = self._exchange_instance()
        period_ms = timeframe_seconds(timeframe) * 1000
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        fetched: List[Candle] = []
        while cursor <= end_ms:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            for row in batch:
                try:
                    fetched.append(
                        Candle(
                            open_at=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=float(row[4]),
                            volume=float(row[5]),
                        )
                    )
                except (IndexError, TypeError, ValueError):
                    continue
            last_open = int(batch[-1][0])
            next_cursor = last_open + period_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
        return fetched

    def __call__(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> List[Candle]:
        """Return closed candles covering ``[start, end]``.

        Raises:
            CandleSourceError: when the cache is insufficient and the fetch
                fails, so the caller can report missing candles explicitly.
        """
        cached: List[Candle] = []
        path: Optional[str] = None
        if self.cache_dir:
            path = self.cache_path(self.cache_dir, symbol, timeframe)
            try:
                cached = closed_candles(read_candles_csv(path), timeframe)
            except (FileNotFoundError, OSError):
                cached = []
        if _covers(cached, start, end):
            return _filter_window(cached, start, end)
        try:
            fetched = self._fetch(symbol, timeframe, start, end)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise CandleSourceError(
                f"{symbol} {timeframe}: candle fetch failed ({type(exc).__name__}: {exc})"
            ) from exc
        merged = _merge_candles(cached, closed_candles(fetched, timeframe))
        if path is not None:
            write_candles_csv(path, merged)
        if not merged:
            raise CandleSourceError(
                f"{symbol} {timeframe}: candle source returned no data for the window"
            )
        # A window ending on the still-forming candle is expected to be short:
        # the missing exit is reported per signal instead of dropping the data.
        return _filter_window(merged, start, end)


def _index_candles(candles: Sequence[Candle]) -> Dict[datetime, Candle]:
    return {candle.open_at: candle for candle in candles}


def counterfactual(
    record: SignalRecord,
    candles_by_open: Dict[datetime, Candle],
    horizon: int,
    cost_pct: float,
) -> Counterfactual:
    """Compute the net P&L of executing one signal for ``horizon`` hours.

    The entry is the close of the signal's own candle (the candle it was
    analysed on). The exit is the close of the candle ``horizon`` hours later.
    Cost is subtracted once per round trip.

    Args:
        record: BUY/SELL signal.
        candles_by_open: Candle index keyed by open time.
        horizon: Holding period in hours.
        cost_pct: Round-trip cost in percentage points.

    Returns:
        A counterfactual with ``status`` ``ok``, ``missing_entry`` or
        ``missing_exit``.
    """
    entry = candles_by_open.get(record.candle_open_time)
    if entry is None or entry.close <= 0:
        return Counterfactual(horizon=horizon, status="missing_entry")
    period = timeframe_seconds(record.timeframe)
    exit_at = record.candle_open_time + timedelta(seconds=period * horizon)
    exit_candle = candles_by_open.get(exit_at)
    if exit_candle is None or exit_candle.close <= 0:
        return Counterfactual(
            horizon=horizon,
            status="missing_exit",
            entry_price=entry.close,
        )
    direction = ACTION_DIRECTION[record.action]
    gross_pct = direction * (exit_candle.close / entry.close - 1.0) * 100.0
    return Counterfactual(
        horizon=horizon,
        status="ok",
        entry_price=entry.close,
        exit_price=exit_candle.close,
        net_pct=gross_pct - cost_pct,
    )


# ----------------------------------------------------------------- aggregate
def evaluate_signals(
    records: Sequence[SignalRecord],
    candles: Dict[Tuple[str, str], Sequence[Candle]],
    horizons: Sequence[int],
    cost_pct: float,
) -> List[EvaluatedSignal]:
    """Classify every signal and compute its counterfactual P&L.

    Args:
        records: BUY/SELL signal records.
        candles: Candles keyed by ``(symbol, timeframe)``.
        horizons: Holding periods in hours.
        cost_pct: Round-trip cost in percentage points.

    Returns:
        Evaluated signals in input order.
    """
    indexes: Dict[Tuple[str, str], Dict[datetime, Candle]] = {
        key: _index_candles(series) for key, series in candles.items()
    }
    evaluated: List[EvaluatedSignal] = []
    for record in records:
        status, veto_kind, noul = classify_signal(record)
        index = indexes.get((record.symbol, record.timeframe), {})
        results = {
            horizon: counterfactual(record, index, horizon, cost_pct)
            for horizon in horizons
        }
        evaluated.append(
            EvaluatedSignal(
                record=record,
                status=status,
                veto_kind=veto_kind,
                noul=noul,
                results=results,
            )
        )
    return evaluated


def _empty_horizons(horizons: Sequence[int]) -> Dict[int, HorizonStats]:
    return {horizon: HorizonStats(horizon=horizon) for horizon in horizons}


def summarize(
    evaluated: Sequence[EvaluatedSignal],
    horizons: Sequence[int],
    cost_pct: float,
    target: int,
    stats: LoadStats,
    notes: Optional[Sequence[str]] = None,
) -> ReportData:
    """Aggregate evaluations into per-symbol and overall statistics."""
    symbols: Dict[str, SymbolSummary] = {}
    overall = _empty_horizons(horizons)
    for item in evaluated:
        summary = symbols.setdefault(
            item.record.symbol,
            SymbolSummary(symbol=item.record.symbol, horizons=_empty_horizons(horizons)),
        )
        summary.counts.total += 1
        if item.status == "approved":
            summary.counts.approved += 1
        elif item.status == "vetoed":
            summary.counts.add_veto(item.veto_kind)
        else:
            summary.counts.other_rejected += 1
        for horizon in horizons:
            result = item.results[horizon]
            for bucket in (summary.horizons[horizon], overall[horizon]):
                _accumulate(bucket, item, result)
    return ReportData(
        horizons=tuple(horizons),
        cost_pct=cost_pct,
        target=target,
        stats=stats,
        symbols=dict(sorted(symbols.items())),
        overall=overall,
        evaluations=list(evaluated),
        notes=list(notes or []),
    )


def _accumulate(
    bucket: HorizonStats, item: EvaluatedSignal, result: Counterfactual
) -> None:
    """Add one signal's counterfactual result to a horizon bucket."""
    if item.status == "approved":
        bucket.approved_total += 1
        if result.net_pct is not None:
            bucket.approved_evaluable += 1
            bucket.approved_net_pct += result.net_pct
        return
    if item.status != "vetoed":
        return
    bucket.vetoed_total += 1
    if result.net_pct is None:
        return
    bucket.vetoed_evaluable += 1
    bucket.vetoed_net_pct += result.net_pct
    if result.net_pct > 0:
        bucket.false_veto_count += 1
        bucket.false_veto_cost_pct += result.net_pct
    else:
        bucket.loss_avoided_pct += result.net_pct


def horizon_validated(
    stats: HorizonStats, sample_sufficient: bool
) -> Tuple[bool, str]:
    """Apply the documented decision rule to one horizon.

    Args:
        stats: Aggregated statistics for the horizon.
        sample_sufficient: True when the total collected sample reaches target.

    Returns:
        ``(validated, reason)`` where reason explains a failure.
    """
    if not sample_sufficient:
        return False, "sample below target"
    if stats.vetoed_evaluable < MIN_VETO_SAMPLE:
        return False, f"fewer than {MIN_VETO_SAMPLE} vetoed signals with candles"
    if stats.vetoed_net_pct >= 0:
        return False, "vetoed trades would have made money on net"
    if stats.false_veto_cost_pct > MAX_FALSE_VETO_SHARE * abs(stats.loss_avoided_pct):
        return False, "false-veto cost is not small relative to avoided losses"
    return True, "net avoided is negative and false-veto cost is small"


# ------------------------------------------------------------------ rendering
def _signed_pct(value: Optional[float]) -> str:
    """Format an optional signed percentage, normalizing negative zero."""
    if value is None:
        return "n/a"
    cleaned = 0.0 if abs(value) < 0.005 else value
    return f"{cleaned:+.2f}%"


def _plain_pct(value: Optional[float]) -> str:
    """Format an optional unsigned percentage, normalizing negative zero."""
    if value is None:
        return "n/a"
    cleaned = 0.0 if abs(value) < 0.005 else value
    return f"{cleaned:.2f}%"


def _noul_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_sample(data: ReportData, source_label: str) -> List[str]:
    horizons = ", ".join(f"{horizon}h" for horizon in data.horizons)
    return [
        "## Sample",
        "",
        f"- Signals directory: `{source_label}` (read-only)",
        f"- Cost assumption: {data.cost_pct:.2f}% per round trip",
        f"- Horizons: {horizons}",
        f"- Target sample: {data.target} BUY/SELL signals",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| JSONL files read | {data.stats.files_read} |",
        f"| Lines read | {data.stats.lines_read} |",
        f"| BUY/SELL signals collected | {data.total_signals} |",
        f"| HOLD signals ignored | {data.stats.holds_ignored} |",
        f"| Malformed lines skipped | {data.stats.malformed} |",
        f"| Duplicate decisions skipped | {data.stats.duplicates} |",
        "",
    ]


def _render_counts(data: ReportData) -> List[str]:
    lines = [
        "## Signal counts per symbol",
        "",
        "| Symbol | BUY/SELL | Approved | Vetoed | Tradeability veto | Bias veto | Other veto | Other rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in data.symbols.values():
        counts = summary.counts
        lines.append(
            f"| {summary.symbol} | {counts.total} | {counts.approved} | {counts.vetoed} "
            f"| {counts.tradeability} | {counts.bias} | {counts.other_veto} "
            f"| {counts.other_rejected} |"
        )
    lines.append("")
    return lines


def _render_overall_pnl(data: ReportData) -> List[str]:
    lines = [
        "## Counterfactual P&L, all symbols (net of cost)",
        "",
        "Avoided net is the P&L the vetoed trades would have produced: negative means the veto saved money. "
        "False-veto cost is the sum of the vetoed trades that would have been profitable.",
        "",
        "| Horizon | Approved evaluated | Approved net | Vetoed evaluated | Avoided net | False vetoes "
        "| False-veto cost | Loss avoided |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in data.horizons:
        stats = data.overall[horizon]
        approved_net = stats.approved_net_pct if stats.approved_evaluable else None
        avoided_net = stats.vetoed_net_pct if stats.vetoed_evaluable else None
        lines.append(
            f"| {horizon}h | {stats.approved_evaluable} | {_signed_pct(approved_net)} "
            f"| {stats.vetoed_evaluable} | {_signed_pct(avoided_net)} "
            f"| {stats.false_veto_count} | {_signed_pct(stats.false_veto_cost_pct)} "
            f"| {_signed_pct(stats.loss_avoided_pct)} |"
        )
    lines.append("")
    return lines


def _render_symbol_pnl(data: ReportData) -> List[str]:
    lines = [
        "## P&L by symbol and horizon",
        "",
        "| Symbol | Horizon | Approved net | Vetoed evaluated | Avoided net | False vetoes | False-veto cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in data.symbols.values():
        for horizon in data.horizons:
            stats = summary.horizons[horizon]
            approved_net = stats.approved_net_pct if stats.approved_evaluable else None
            avoided_net = stats.vetoed_net_pct if stats.vetoed_evaluable else None
            lines.append(
                f"| {summary.symbol} | {horizon}h | {_signed_pct(approved_net)} "
                f"| {stats.vetoed_evaluable} | {_signed_pct(avoided_net)} "
                f"| {stats.false_veto_count} | {_signed_pct(stats.false_veto_cost_pct)} |"
            )
    lines.append("")
    return lines


def worst_false_vetoes(
    data: ReportData, limit: int = MAX_FALSE_VETO_ROWS
) -> List[Tuple[float, int, EvaluatedSignal]]:
    """Return the most costly missed gains, one row per vetoed signal.

    Each signal is represented by its best horizon so the table shows distinct
    vetoes instead of repeating the same trade once per horizon.
    """
    rows: List[Tuple[float, int, EvaluatedSignal]] = []
    for item in data.evaluations:
        if item.status != "vetoed":
            continue
        best: Optional[Tuple[float, int]] = None
        for horizon, result in item.results.items():
            if result.net_pct is not None and result.net_pct > 0:
                if best is None or result.net_pct > best[0]:
                    best = (result.net_pct, horizon)
        if best is not None:
            rows.append((best[0], best[1], item))
    rows.sort(
        key=lambda row: (
            -row[0],
            row[2].record.symbol,
            row[2].record.candle_open_time,
            row[1],
        )
    )
    return rows[:limit]


def _render_worst_false(data: ReportData) -> List[str]:
    rows = worst_false_vetoes(data)
    lines = [
        "## Worst false vetoes",
        "",
        "Vetoed signals that would have been profitable, largest net P&L first.",
        "",
    ]
    if not rows:
        lines.extend(["None: no vetoed signal would have been profitable.", ""])
        return lines
    lines.extend(
        [
            "| Symbol | Action | Candle open (UTC) | Horizon | Net P&L | Veto kind | noul |",
            "| --- | --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for net_pct, horizon, item in rows:
        lines.append(
            f"| {item.record.symbol} | {item.record.action} "
            f"| {item.record.candle_open_time.strftime('%Y-%m-%d %H:%M')} "
            f"| {horizon}h | {_signed_pct(net_pct)} | {item.veto_kind or 'other'} "
            f"| {_noul_text(item.noul)} |"
        )
    lines.append("")
    return lines


def _render_verdict(data: ReportData) -> List[str]:
    total = data.total_signals
    suffix = "is" if data.sample_is_sufficient else "is NOT"
    lines = [
        "## Verdict on the sample",
        "",
        f"Collected **{total}** BUY/SELL signals against a target of **{data.target}**. "
        f"The sample {suffix} sufficient to judge the validator.",
    ]
    if not data.sample_is_sufficient:
        lines.append(
            f"A verdict needs at least {data.target} BUY/SELL signals and at least "
            f"{MIN_VETO_SAMPLE} vetoed signals with candle data per horizon. "
            "Treat this report as a first reading, not as evidence."
        )
    lines.extend(["", "Decision rule per horizon:", ""])
    for horizon in data.horizons:
        stats = data.overall[horizon]
        validated, reason = horizon_validated(stats, data.sample_is_sufficient)
        state = "VALIDATED" if validated else "NOT VALIDATED"
        lines.append(
            f"- {horizon}h: **{state}** ({reason}); avoided {_signed_pct(stats.vetoed_net_pct)}, "
            f"false-veto cost {_signed_pct(stats.false_veto_cost_pct)}."
        )
    lines.extend(
        [
            "",
            "A veto role is validated only if (a) net avoided P&L is clearly negative over a "
            "sufficient sample and (b) false-veto cost stays small; otherwise the validator is not "
            "demonstrated.",
            "",
        ]
    )
    return lines


def _render_conclusions(data: ReportData) -> List[str]:
    total = data.total_signals
    actions = sorted({item.record.action for item in data.evaluations})
    h1 = data.overall.get(1)
    lines = ["## Conclusions", ""]
    if not data.sample_is_sufficient:
        lines.append(
            f"- **Too early to judge.** {total} BUY/SELL signals is far from the {data.target} "
            "needed; one or two unusual candles can flip the result."
        )
    else:
        lines.append(
            f"- **Sample is at target.** {total} BUY/SELL signals were collected, enough to apply "
            "the decision rule, though more data always helps."
        )
    if h1 is None:
        lines.append("- No 1h horizon was requested, so the headline number is unavailable.")
    elif h1.vetoed_evaluable == 0:
        lines.append(
            "- **No vetoed trade could be priced.** The signals lack matching candles, so the "
            "validator cannot be judged from this run."
        )
    else:
        if h1.vetoed_net_pct < 0:
            effect = "would have lost money, so avoiding them was correct"
            verb = "saved"
        else:
            effect = "would have made money, so the veto gave up that gain"
            verb = "cost"
        lines.append(
            f"- **At 1h the veto {verb} money.** Across {h1.vetoed_evaluable} vetoed trades the "
            f"counterfactual net was {_signed_pct(h1.vetoed_net_pct)} (negative means the vetoed "
            f"trades {effect})."
        )
        lines.append(
            f"- **False vetoes.** {h1.false_veto_count} of {h1.vetoed_evaluable} vetoed trades would "
            f"have been profitable, an opportunity cost of {_signed_pct(h1.false_veto_cost_pct)}."
        )
    if len(actions) == 1:
        lines.append(
            f"- **One direction only.** Every collected signal is {actions[0]}, so this measures "
            "the veto in a single market phase, not across cycles."
        )
    else:
        lines.append(
            f"- **Direction mix.** The sample contains {', '.join(actions)}; results may differ per "
            "direction and should be split once the sample is larger."
        )
    lines.append(
        f"- **Costs and limits.** P&L subtracts a fixed {data.cost_pct:.2f}% round trip and ignores "
        "slippage and funding; rerun weekly until the sample is sufficient and both decision-rule "
        "conditions can be checked."
    )
    lines.append("")
    return lines


def render_report(data: ReportData, source_label: str) -> str:
    """Render the full Markdown report deterministically.

    Args:
        data: Aggregated report data.
        source_label: Text shown as the input location (stable for golden tests).

    Returns:
        The Markdown document, ending with a newline.
    """
    lines = [
        "# JEV post-gate veto impact report",
        "",
        "Read-only measurement of the JEV **post** gate as validator of `risk_approved`, the last "
        "stop before an approved BUY/SELL can execute.",
        "",
    ]
    if data.notes:
        lines.extend(["## Candle data notes", ""])
        lines.extend(f"- {note}" for note in data.notes)
        lines.append("")
    lines.extend(_render_sample(data, source_label))
    lines.extend(_render_counts(data))
    lines.extend(_render_overall_pnl(data))
    lines.extend(_render_symbol_pnl(data))
    lines.extend(_render_worst_false(data))
    lines.extend(_render_verdict(data))
    lines.extend(_render_conclusions(data))
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------- CLI
def parse_horizons(value: str) -> Tuple[int, ...]:
    """Parse ``--horizons`` into sorted unique positive integers."""
    horizons = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            horizon = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid horizon: {token!r}") from exc
        if horizon <= 0:
            raise argparse.ArgumentTypeError(f"horizon must be positive: {horizon}")
        horizons.append(horizon)
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return tuple(sorted(set(horizons)))


def resolve_signals_dir(value: Optional[str]) -> str:
    """Resolve the signals directory, honoring ``LLM_TRADER_SIGNALS_DIR``.

    An explicit ``--signals-dir`` always wins. Otherwise the shared
    ``default_signals_dir()`` resolver is used, so the tool behaves like every
    other signal reader instead of assuming this machine's live-stack path.

    Args:
        value: Explicit ``--signals-dir`` value, or None.

    Returns:
        The directory to read.
    """
    return value if value else default_signals_dir()


def _ensure_outside(path: str, forbidden_dir: str, label: str) -> None:
    """Refuse to write under the read-only signals directory."""
    if not forbidden_dir:
        return
    target = os.path.realpath(os.path.abspath(path))
    forbidden = os.path.realpath(os.path.abspath(forbidden_dir))
    if target == forbidden or target.startswith(forbidden + os.sep):
        raise SystemExit(f"refusing to write {label} inside the signals directory: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only report of the JEV post-gate veto impact on BUY/SELL signals."
        )
    )
    parser.add_argument(
        "--signals-dir",
        default=None,
        help=(
            "directory holding the per-pair signal JSONL files, read-only "
            "(default: $LLM_TRADER_SIGNALS_DIR or <repo>/data/signals; the live "
            "stack uses .../trading-stack/runtime/signals)"
        ),
    )
    parser.add_argument(
        "--cost-pct",
        type=float,
        default=DEFAULT_COST_PCT,
        help="round-trip cost in percentage points (default 0.10)",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=DEFAULT_HORIZONS,
        help="comma-separated holding periods in hours (default 1,2,4)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="number of BUY/SELL signals needed for a verdict (default 200)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help="Markdown output path (default runtime/jev_bench/jev_validator_report.md)",
    )
    parser.add_argument(
        "--candles-cache",
        default=DEFAULT_CACHE_DIR,
        help="local candle cache directory (default runtime/jev_bench/candles_cache)",
    )
    return parser


def _collect_candles(
    records: Sequence[SignalRecord],
    source: CandleSource,
    horizons: Sequence[int],
) -> Tuple[Dict[Tuple[str, str], List[Candle]], List[str]]:
    """Fetch the candle window each symbol needs, recording failures as notes."""
    groups: Dict[Tuple[str, str], List[datetime]] = {}
    for record in records:
        groups.setdefault((record.symbol, record.timeframe), []).append(
            record.candle_open_time
        )
    candles: Dict[Tuple[str, str], List[Candle]] = {}
    notes: List[str] = []
    for (symbol, timeframe), opens in sorted(groups.items()):
        period = timeframe_seconds(timeframe)
        start = min(opens)
        end = max(opens) + timedelta(seconds=period * max(horizons))
        try:
            candles[(symbol, timeframe)] = list(source(symbol, timeframe, start, end))
        except CandleSourceError as exc:
            candles[(symbol, timeframe)] = []
            notes.append(str(exc))
    return candles, notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read the signals, compute the report and write it to ``--out``."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.cost_pct < 0:
        raise SystemExit("--cost-pct must be non-negative")
    if args.target <= 0:
        raise SystemExit("--target must be positive")
    signals_dir = resolve_signals_dir(args.signals_dir)
    _ensure_outside(args.out, signals_dir, "the report")
    if args.candles_cache:
        _ensure_outside(args.candles_cache, signals_dir, "the candle cache")

    records, stats = load_signals(signals_dir)
    source = CcxtCandleSource(cache_dir=args.candles_cache or None)
    candles, notes = _collect_candles(records, source, args.horizons)
    evaluations = evaluate_signals(records, candles, args.horizons, args.cost_pct)
    data = summarize(
        evaluations,
        args.horizons,
        args.cost_pct,
        args.target,
        stats,
        notes,
    )
    report = render_report(data, signals_dir)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
