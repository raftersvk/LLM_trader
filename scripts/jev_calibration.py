"""JEV tradeability calibration on historical candles.

The gate turns a JEV ``noul`` probability into a boolean with a fixed
threshold (``DEFAULT_TRADEABILITY_THRESHOLD = 0.5``). This script measures
whether that threshold is calibrated: for every state it asks JEV once with
the exact production questions, records the raw answer, and compares ``noul``
with a forward-looking "worth trading" label.

The state is built by the same no-look-ahead builder the gate uses: candles
that are still forming at the reference time (the close of the analysed
candle) are excluded. The forward label only reads candles strictly after that
close.

    python scripts/jev_calibration.py \\
        --candles runtime/jev_bench/BTC_USDT_1h.csv \\
        --symbol BTC/USDT --limit 200 \\
        --output runtime/jev_bench/calibration_btc_200.json

The "worth trading" label is ``abs(forward_return) >= move_threshold_pct`` at
the primary horizon (default 1 candle). The base decision is the rule-based
momentum proxy already used by ``scripts/bench_jev.py``; it is NOT an LLM and
the base-trade precision numbers must be read with that caveat.

A 429 / quota error stops the run immediately. The partial report is still
written and marked ``aborted`` so an exhausted quota can never masquerade as a
full calibration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bench_jev import (  # noqa: E402  (path setup must run first)
    DEFAULT_HORIZONS,
    DEFAULT_MOVE_THRESHOLD_PCT,
    MomentumBaseline,
    _build_real_client,
    _rounded,
    build_samples,
    latency_summary,
    load_candles_csv,
)
from src.signal_service.jev_client import (  # noqa: E402
    JevAuthError,
    JevError,
    JevRateLimitError,
)
from src.signal_service.contract import iso_z  # noqa: E402
from src.signal_service.jev_gate import (  # noqa: E402
    build_questions,
    parse_jev_signal,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = tuple(round(0.20 + 0.05 * step, 2) for step in range(11))  # 0.20 .. 0.70
HISTOGRAM_BINS = 10
RELIABILITY_BINS = 5
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_DELAY_SECONDS = 0.5


@dataclass
class CalibrationRecord:
    """One state: JEV answer, forward labels and base decision."""

    index: int
    as_of: str
    noul: Optional[float]
    candle_open_time: str = ""
    bias: Optional[str] = None
    regime: Optional[str] = None
    atr_pct: Optional[float] = None
    atr_percentile: Optional[float] = None
    confidence: Optional[float] = None
    score_raw: Optional[float] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    cost_pct: float = 0.10
    forward_returns: Dict[int, float] = field(default_factory=dict)
    worth_trading: Dict[int, bool] = field(default_factory=dict)
    base_action: str = "HOLD"
    base_direction_correct: Dict[int, Optional[bool]] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Serialize the record without any secret or raw prompt.

        ``noul``, ``atr_pct`` and ``forward_returns`` are metric inputs and
        must keep full precision. Rounding them would make a saved report
        disagree with a later ``--aggregate-from`` rebuild: a tiny positive
        return could become ``0.0`` and move out of the positive class, or a
        value just below a threshold could land exactly on it.
        """
        return {
            "index": self.index,
            "as_of": self.as_of,
            "candle_open_time": self.candle_open_time,
            "noul": self.noul,
            "bias": self.bias,
            "regime": self.regime,
            "atr_pct": self.atr_pct,
            "atr_percentile": _rounded(self.atr_percentile),
            "confidence": _rounded(self.confidence),
            "score_raw": _rounded(self.score_raw),
            "latency_ms": self.latency_ms,
            "cost_pct": self.cost_pct,
            "error": self.error,
            "error_kind": self.error_kind,
            "forward_returns": {
                str(horizon): value
                for horizon, value in sorted(self.forward_returns.items())
            },
            "worth_trading": {
                str(horizon): value for horizon, value in sorted(self.worth_trading.items())
            },
            "base_action": self.base_action,
            "base_direction_correct": {
                str(horizon): value
                for horizon, value in sorted(self.base_direction_correct.items())
            },
        }


# ------------------------------------------------------------------- metrics
def _bin_label(low: float, high: float) -> str:
    """Return the ``low-high`` label used by the histogram and reliability bins."""
    return f"{low:.1f}-{high:.1f}"


def noul_histogram(
    values: Sequence[float],
    bins: int = HISTOGRAM_BINS,
    low: float = 0.0,
    high: float = 1.0,
) -> List[Dict[str, Any]]:
    """Count values in ``bins`` equal-width buckets from ``low`` to ``high``.

    Values outside the range are clamped, and a value exactly equal to
    ``high`` lands in the last bucket.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    counts = [0] * bins
    width = (high - low) / bins
    for value in values:
        offset = int((float(value) - low) / width) if width else 0
        index = min(bins - 1, max(0, offset))
        counts[index] += 1
    return [
        {
            "bin": _bin_label(low + index * width, low + (index + 1) * width),
            "low": round(low + index * width, 4),
            "high": round(low + (index + 1) * width, 4),
            "count": counts[index],
        }
        for index in range(bins)
    ]


def noul_distribution(values: Sequence[float]) -> Dict[str, Any]:
    """Return the noul histogram and its summary statistics."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "histogram": noul_histogram([]),
        }
    return {
        "count": len(values),
        "min": _rounded(min(values)),
        "median": _rounded(statistics.median(values)),
        "mean": _rounded(statistics.fmean(values)),
        "max": _rounded(max(values)),
        "histogram": noul_histogram(values),
    }


def calibration_brier(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> Optional[float]:
    """Brier score for the worth-trading label (lower is better)."""
    if not probabilities:
        return None
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    return sum(
        (float(probability) - (1.0 if label else 0.0)) ** 2
        for probability, label in zip(probabilities, labels)
    ) / len(probabilities)


def calibration_reliability(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    bins: int = RELIABILITY_BINS,
) -> List[Dict[str, Any]]:
    """Bucket predicted noul and report the observed worth-trading rate."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(bins)]
    width = 1.0 / bins
    for probability, label in zip(probabilities, labels):
        index = min(bins - 1, max(0, int(float(probability) / width)))
        buckets[index].append((float(probability), bool(label)))
    result: List[Dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        entry: Dict[str, Any] = {
            "bin": _bin_label(index * width, (index + 1) * width),
            "count": len(bucket),
        }
        if bucket:
            entry["mean_predicted"] = _rounded(
                statistics.fmean(probability for probability, _label in bucket)
            )
            entry["observed_worth_rate"] = _rounded(
                sum(1 for _probability, label in bucket if label) / len(bucket)
            )
        result.append(entry)
    return result


def bias_direction_auc(records: Sequence[CalibrationRecord], horizon: int) -> Optional[float]:
    """AUC of JEV ``bias`` predicting the sign of the forward return.

    Only records with a directional bias and a forward label are used. The
    score is 1.0 for ``long`` and 0.0 for ``short``; the label is
    ``forward_return > 0``. Returns None when one class is missing.
    """
    scored = [
        (1.0 if record.bias == "long" else 0.0, record.forward_returns[horizon] > 0)
        for record in records
        if record.bias in ("long", "short") and horizon in record.forward_returns
    ]
    positives = [score for score, label in scored if label]
    negatives = [score for score, label in scored if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _is_blocked(record: CalibrationRecord, threshold: float) -> bool:
    """True when the pre-gate holds a base trade at this threshold.

    Mirrors the production pre-gate: a state is tradeable when ``noul`` is at
    or above the threshold and the bias is not ``none``; a bias that directly
    contradicts the base action also blocks the trade. A missing or
    unparseable bias (``None``) is allowed by ``JevGate.check_pre`` and must
    not be treated as a hold here.
    """
    if record.base_action not in ("BUY", "SELL"):
        return False
    if record.noul is None or record.noul < threshold:
        return True
    if record.bias == "none":
        return True
    if record.base_action == "BUY" and record.bias == "short":
        return True
    if record.base_action == "SELL" and record.bias == "long":
        return True
    return False


def threshold_curve(
    records: Sequence[CalibrationRecord],
    thresholds: Sequence[float],
    horizon: int,
    move_threshold_pct: float = DEFAULT_MOVE_THRESHOLD_PCT,
) -> List[Dict[str, Any]]:
    """Compute coverage, trades, precision, false holds and AUC per threshold.

    Definitions:

    * ``coverage``: share of scored states whose ``noul`` clears the threshold
      (tradeability alone, without the bias rule);
    * ``trade_count``: base BUY/SELL decisions that survive the full pre-gate
      rule at this threshold;
    * ``base_trade_precision``: share of those surviving trades whose direction
      matched the forward return;
    * ``false_hold_rate``: share of blocked base trades that were actually
      worth trading (``abs(forward_return) >= move_threshold_pct``).
    """
    scored = [record for record in records if record.noul is not None]
    result: List[Dict[str, Any]] = []
    for threshold in thresholds:
        tradable = [record for record in scored if record.noul >= threshold]
        taken = [record for record in scored if not _is_blocked(record, threshold)
                 and record.base_action in ("BUY", "SELL")]
        blocked = [record for record in scored if _is_blocked(record, threshold)]
        correct = [
            record for record in taken
            if record.base_direction_correct.get(horizon) is True
        ]
        false_holds = [
            record for record in blocked
            if record.worth_trading.get(horizon) is True
        ]
        directional = [
            record for record in tradable if record.bias in ("long", "short")
        ]
        result.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": _rounded(len(tradable) / len(scored) if scored else None),
                "trade_count": len(taken),
                "base_trade_precision": _rounded(
                    len(correct) / len(taken) if taken else None
                ),
                "hold_count": len(blocked),
                "false_hold_count": len(false_holds),
                "false_hold_rate": _rounded(
                    len(false_holds) / len(blocked) if blocked else None
                ),
                "directional_auc": _rounded(bias_direction_auc(directional, horizon)),
            }
        )
    return result


def _worth_rate(records: Sequence[CalibrationRecord], horizon: int) -> Optional[float]:
    """Share of scored states labelled worth trading at one horizon."""
    labels = [
        record.worth_trading[horizon]
        for record in records
        if record.noul is not None and horizon in record.worth_trading
    ]
    return _rounded(sum(1 for label in labels if label) / len(labels) if labels else None)


def assign_atr_percentiles(records: Sequence[CalibrationRecord]) -> None:
    """Fill ``atr_percentile`` within the report (0 = lowest ATR)."""
    scored = [record for record in records if record.atr_pct is not None]
    ordered = sorted(scored, key=lambda record: record.atr_pct)  # type: ignore[arg-type,return-value]
    if len(ordered) < 2:
        for record in scored:
            record.atr_percentile = 0.0
        return
    last = len(ordered) - 1
    for rank, record in enumerate(ordered):
        record.atr_percentile = round(100.0 * rank / last, 2)


def observation_line(record: CalibrationRecord, pair: str) -> Dict[str, Any]:
    """Return one JSONL observation for a scored state."""
    line: Dict[str, Any] = {
        "candle_open_time": record.candle_open_time or record.as_of,
        "pair": pair,
        "noul": record.noul,
        "bias": record.bias,
        "regime": record.regime,
        "atr_pct": record.atr_pct,
        "atr_percentile": _rounded(record.atr_percentile),
        "base_action": record.base_action,
        "cost_pct": record.cost_pct,
    }
    for horizon, value in sorted(record.forward_returns.items()):
        line[f"forward_{horizon}h"] = value
    return line


def write_observations(report: Dict[str, Any], path: str) -> int:
    """Write one JSONL line per scored state; return the line count."""
    pair = str(report.get("dataset", {}).get("symbol") or "")
    payloads = [
        payload for payload in report.get("states") or []
        if payload.get("noul") is not None
    ]
    records = records_from_payload(payloads)
    assign_atr_percentiles(records)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(observation_line(record, pair)) + "\n")
    return len(records)


def records_from_payload(payloads: Sequence[Dict[str, Any]]) -> List[CalibrationRecord]:
    """Rebuild records from a saved report so metrics regenerate offline."""
    records: List[CalibrationRecord] = []
    for payload in payloads:
        records.append(
            CalibrationRecord(
                index=int(payload["index"]),
                as_of=str(payload.get("as_of") or ""),
                noul=payload.get("noul"),
                candle_open_time=str(payload.get("candle_open_time") or ""),
                bias=payload.get("bias"),
                regime=payload.get("regime"),
                atr_pct=payload.get("atr_pct"),
                atr_percentile=payload.get("atr_percentile"),
                confidence=payload.get("confidence"),
                score_raw=payload.get("score_raw"),
                latency_ms=payload.get("latency_ms"),
                cost_pct=float(payload.get("cost_pct") or 0.10),
                error=payload.get("error"),
                error_kind=payload.get("error_kind"),
                forward_returns={
                    int(key): float(value)
                    for key, value in (payload.get("forward_returns") or {}).items()
                },
                worth_trading={
                    int(key): bool(value)
                    for key, value in (payload.get("worth_trading") or {}).items()
                },
                base_action=str(payload.get("base_action") or "HOLD"),
                base_direction_correct={
                    int(key): value
                    for key, value in (payload.get("base_direction_correct") or {}).items()
                },
            )
        )
    return records


def validate_calibration_report(report: Any) -> List[str]:
    """Return schema problems for a saved calibration report (empty = valid)."""
    problems: List[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    for key in ("dataset", "states"):
        if key not in report:
            problems.append(f"missing {key}")
    if problems:
        return problems
    dataset = report["dataset"]
    if not isinstance(dataset, dict):
        problems.append("dataset must be an object")
    else:
        for key in ("symbol", "timeframe", "horizons", "move_threshold_pct"):
            if key not in dataset:
                problems.append(f"dataset missing {key}")
        horizons = dataset.get("horizons")
        if horizons is not None and not isinstance(horizons, (list, tuple)):
            problems.append("dataset.horizons must be a list")
        if dataset.get("candles") is not None and not isinstance(dataset["candles"], int):
            problems.append("dataset.candles must be an int")
    states = report["states"]
    if not isinstance(states, list) or not states:
        problems.append("states must be a non-empty list")
        return problems
    required_state = ("index", "noul", "forward_returns", "worth_trading", "base_action")
    for position, state in enumerate(states):
        if not isinstance(state, dict):
            problems.append(f"state {position} must be an object")
            continue
        missing = [key for key in required_state if key not in state]
        if missing:
            problems.append(f"state {position} missing {', '.join(missing)}")
        noul = state.get("noul")
        if noul is not None and (not isinstance(noul, (int, float)) or not 0.0 <= float(noul) <= 1.0):
            problems.append(f"state {position} noul must be in [0, 1]")
        if not isinstance(state.get("forward_returns"), dict):
            problems.append(f"state {position} forward_returns must be an object")
    return problems


def _parse_horizons(value: Any, default: Sequence[int] = DEFAULT_HORIZONS) -> Tuple[int, ...]:
    """Parse a CLI horizons value or fall back to the report's horizons."""
    if value is None:
        return tuple(int(horizon) for horizon in default)
    if isinstance(value, (list, tuple)):
        return tuple(int(horizon) for horizon in value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise SystemExit("--horizons must contain at least one horizon")
    return tuple(int(part) for part in parts)


def _carry_aggregate_dataset(
    args: argparse.Namespace, dataset: Dict[str, Any]
) -> None:
    """Fill aggregate-from fields from the report unless the CLI overrode them.

    ``None`` means the option was not given (all provenance options default to
    ``None``), so a rebuild without flags preserves the original dataset.
    """
    if args.symbol is None:
        args.symbol = dataset.get("symbol") or DEFAULT_SYMBOL
    if args.timeframe is None:
        args.timeframe = dataset.get("timeframe") or DEFAULT_TIMEFRAME
    if args.move_threshold_pct is None:
        args.move_threshold_pct = dataset.get("move_threshold_pct", DEFAULT_MOVE_THRESHOLD_PCT)
    if args.delay_seconds is None:
        args.delay_seconds = dataset.get("delay_seconds", DEFAULT_DELAY_SECONDS)
    if args.candles is None:
        source = dataset.get("source")
        args.candles = source if isinstance(source, str) and source else None


def _apply_run_defaults(args: argparse.Namespace) -> None:
    """Resolve CLI defaults for a fresh calibration run."""
    if args.symbol is None:
        args.symbol = DEFAULT_SYMBOL
    if args.timeframe is None:
        args.timeframe = DEFAULT_TIMEFRAME
    if args.move_threshold_pct is None:
        args.move_threshold_pct = DEFAULT_MOVE_THRESHOLD_PCT
    if args.delay_seconds is None:
        args.delay_seconds = DEFAULT_DELAY_SECONDS


def build_report(
    records: Sequence[CalibrationRecord],
    args: argparse.Namespace,
    horizons: Sequence[int],
    started_at: str,
    duration_seconds: float,
    aborted: bool,
    abort_reason: Optional[str],
    model: str,
    base_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the machine-readable calibration report."""
    horizon = args.horizon
    scored = [
        record for record in records
        if record.noul is not None and horizon in record.worth_trading
    ]
    probabilities = [record.noul for record in scored]  # type: ignore[misc]
    labels = [record.worth_trading[horizon] for record in scored]
    latencies = [record.latency_ms for record in records if record.latency_ms is not None]
    base_trades = [
        record for record in records
        if record.base_action in ("BUY", "SELL") and horizon in record.base_direction_correct
    ]
    base_correct = [
        record for record in base_trades
        if record.base_direction_correct[horizon] is True
    ]
    rate_limit_errors = sum(1 for record in records if record.error_kind == "rate_limit")
    auth_errors = sum(1 for record in records if record.error_kind == "auth")
    other_errors = sum(1 for record in records if record.error_kind == "other")
    assign_atr_percentiles(records)
    by_horizon: Dict[str, Any] = {}
    for other in horizons:
        other_scored = [
            record for record in records
            if record.noul is not None and other in record.worth_trading
        ]
        by_horizon[f"h{other}"] = {
            "worth_rate": _worth_rate(records, other),
            "brier": _rounded(
                calibration_brier(
                    [record.noul for record in other_scored],  # type: ignore[misc]
                    [record.worth_trading[other] for record in other_scored],
                )
            ),
            "reliability": calibration_reliability(
                [record.noul for record in other_scored],  # type: ignore[misc]
                [record.worth_trading[other] for record in other_scored],
            ),
            "threshold_curve": threshold_curve(
                records, args.thresholds, other, args.move_threshold_pct
            ),
        }

    return {
        "generated_at": started_at,
        "kind": "jev-tradeability-calibration",
        "real_benchmark": True,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "dataset": {
            "source": args.candles,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "candles": args.candle_count,
            "states": len(records),
            "horizons": list(horizons),
            "horizon": horizon,
            "move_threshold_pct": args.move_threshold_pct,
            "base": "momentum-proxy (rule-based, NOT an LLM)",
            "delay_seconds": args.delay_seconds,
            "model": model,
        },
        "scoring": {
            "calls": len(records),
            "scored": len(scored),
            "errors": len(records) - len(scored),
            "rate_limit_errors": rate_limit_errors,
            "auth_errors": auth_errors,
            "other_errors": other_errors,
            "latency": latency_summary(latencies),
            "duration_seconds": round(duration_seconds, 2),
            "base_duration_seconds": base_seconds,
        },
        "noul": noul_distribution(probabilities),
        "brier": _rounded(calibration_brier(probabilities, labels)),
        "reliability": calibration_reliability(probabilities, labels),
        "worth_rate": {
            f"h{other}": _worth_rate(records, other) for other in horizons
        },
        "base": {
            "trade_count": len(base_trades),
            "trade_precision": _rounded(
                len(base_correct) / len(base_trades) if base_trades else None
            ),
        },
        "threshold_curve": threshold_curve(
            records, args.thresholds, horizon, args.move_threshold_pct
        ),
        "by_horizon": by_horizon,
        "states": [record.to_payload() for record in records],
    }


# --------------------------------------------------------------------- runner
async def run_calibration(args: argparse.Namespace) -> Dict[str, Any]:
    """Call JEV once per state and build the calibration report."""
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.monotonic()

    if args.aggregate_from:
        with open(args.aggregate_from, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        problems = validate_calibration_report(existing)
        if problems:
            raise SystemExit(
                f"invalid calibration report {args.aggregate_from}: {'; '.join(problems)}"
            )
        dataset = existing["dataset"]
        _carry_aggregate_dataset(args, dataset)
        horizons = _parse_horizons(
            args.horizons, dataset.get("horizons") or DEFAULT_HORIZONS
        )
        if args.horizon is None:
            args.horizon = int(dataset.get("horizon") or horizons[0])
        if args.horizon not in horizons:
            raise SystemExit(f"--horizon {args.horizon} must be one of {horizons}")

        records = records_from_payload(existing["states"])
        args.candle_count = dataset.get("candles")
        if args.candles and os.path.exists(args.candles):
            # Older reports do not carry candle time or ATR: rebuild the same
            # no-look-ahead samples from the source candles and enrich by index.
            try:
                candles = load_candles_csv(args.candles)
            except (OSError, ValueError) as exc:
                logger.warning("cannot enrich from %s: %s", args.candles, exc)
                candles = []
            if candles:
                args.candle_count = len(candles)
                samples = build_samples(candles, args.symbol, args.timeframe, horizons=horizons)
                enrichment = {
                    index: (iso_z(candles[index].open_at), state.get("atr_pct"))
                    for index, state, _future in samples
                }
                for record in records:
                    if record.index in enrichment:
                        record.candle_open_time, record.atr_pct = enrichment[record.index]
        for record in records:
            record.cost_pct = args.cost_pct
        scoring = existing.get("scoring") or {}
        return build_report(
            records,
            args,
            horizons,
            existing.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            scoring.get("duration_seconds") or 0.0,
            bool(existing.get("aborted")),
            existing.get("abort_reason"),
            dataset.get("model") or "unknown",
            scoring.get("base_duration_seconds"),
        )

    _apply_run_defaults(args)
    horizons = _parse_horizons(args.horizons)
    if args.horizon is None:
        args.horizon = horizons[0]
    if args.horizon not in horizons:
        raise SystemExit(f"--horizon {args.horizon} must be one of {horizons}")
    if not args.candles:
        raise SystemExit("--candles is required unless --aggregate-from is given")
    candles = load_candles_csv(args.candles)
    if len(candles) < 60:
        raise SystemExit(f"not enough candles: {len(candles)}")
    args.candle_count = len(candles)

    samples = build_samples(candles, args.symbol, args.timeframe, horizons=horizons)
    if not samples:
        raise SystemExit("no samples could be built from the dataset")
    if args.limit > 0 and len(samples) > args.limit:
        samples = samples[-args.limit:]

    client = _build_real_client()
    model = client.model
    base = MomentumBaseline()
    records: List[CalibrationRecord] = []
    aborted = False
    abort_reason: Optional[str] = None
    base_started = time.monotonic()

    try:
        for position, (index, state, future) in enumerate(samples):
            decision = await base.decide(candles, index, state)
            record = CalibrationRecord(
                index=index,
                as_of=state.get("as_of") or "",
                noul=None,
                candle_open_time=iso_z(candles[index].open_at),
                atr_pct=state.get("atr_pct"),
                cost_pct=args.cost_pct,
                base_action=decision.action,
                forward_returns={horizon: future[horizon] for horizon in horizons},
            )
            record.worth_trading = {
                horizon: abs(future[horizon]) * 100.0 >= args.move_threshold_pct
                for horizon in horizons
            }
            record.base_direction_correct = {
                horizon: (
                    (decision.action == "BUY" and future[horizon] > 0)
                    or (decision.action == "SELL" and future[horizon] < 0)
                )
                if decision.action in ("BUY", "SELL")
                else None
                for horizon in horizons
            }

            if args.delay_seconds > 0:
                await asyncio.sleep(args.delay_seconds)
            try:
                response = await client.ask(state, build_questions())
                signal = parse_jev_signal(response)
            except JevRateLimitError as exc:
                record.error = f"rate limit: {exc}"
                record.error_kind = "rate_limit"
                records.append(record)
                aborted = True
                abort_reason = str(exc)
                logger.error("JEV rate limit at state %d; stopping", position)
                break
            except JevAuthError as exc:
                record.error = f"auth: {exc}"
                record.error_kind = "auth"
                records.append(record)
                aborted = True
                abort_reason = str(exc)
                logger.error("JEV authentication error at state %d; stopping", position)
                break
            except JevError as exc:
                record.error = str(exc)
                record.error_kind = "other"
                records.append(record)
                logger.warning("JEV error at state %d: %s", position, exc)
                continue
            except Exception as exc:  # pylint: disable=broad-exception-caught
                record.error = f"{type(exc).__name__}: {exc}"
                record.error_kind = "other"
                records.append(record)
                logger.warning("unexpected JEV failure at state %d: %s", position, exc)
                continue

            record.noul = signal.tradeability_probability
            record.bias = signal.bias
            record.regime = signal.regime
            record.confidence = signal.confidence
            record.score_raw = signal.score_raw
            record.latency_ms = signal.latency_ms
            records.append(record)
            if (position + 1) % 25 == 0:
                logger.info("%d/%d states scored", position + 1, len(samples))
    finally:
        await client.close()

    base_seconds = round(time.monotonic() - base_started, 2)
    return build_report(
        records,
        args,
        horizons,
        started_at,
        time.monotonic() - started,
        aborted,
        abort_reason,
        model,
        base_seconds,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JEV tradeability threshold calibration")
    parser.add_argument("--candles", default=None, help="OHLCV csv (time,open,high,low,close,volume)")
    parser.add_argument(
        "--aggregate-from", default=None,
        help="Rebuild metrics from a saved report's states (no JEV calls)",
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Trading pair (default BTC/USDT, or the report's symbol with --aggregate-from)",
    )
    parser.add_argument(
        "--timeframe", default=None,
        help="Candle timeframe (default 1h, or the report's timeframe with --aggregate-from)",
    )
    parser.add_argument("--limit", type=int, default=200, help="Most recent states to evaluate (0 = all)")
    parser.add_argument(
        "--horizons", default=None,
        help="Comma-separated horizons (default 1,2,3,4, or the report's horizons)",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Primary horizon for labels and precision (default 1, or the report's horizon)",
    )
    parser.add_argument(
        "--move-threshold-pct", type=float, default=None,
        help="Absolute forward move that makes a state worth trading (percent; default 1.0)",
    )
    parser.add_argument(
        "--delay-seconds", type=float, default=None,
        help="Pause before each real JEV call (default 0.5; carried from the report on rebuild)",
    )
    parser.add_argument(
        "--cost-pct", type=float, default=0.10,
        help="Round-trip trading cost in percent, stored in every observation",
    )
    parser.add_argument(
        "--observations", default=None,
        help="Optional JSONL path; one line per scored state is written there",
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.20,
        help="First tradeability threshold in the curve",
    )
    parser.add_argument(
        "--max-threshold", type=float, default=0.70,
        help="Last tradeability threshold in the curve",
    )
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--output", default=os.path.join(_REPO_ROOT, "runtime", "jev_bench", "calibration.json"),
    )
    return parser


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if args.step <= 0:
        raise SystemExit("--step must be > 0")
    steps = int(round((args.max_threshold - args.min_threshold) / args.step))
    args.thresholds = [round(args.min_threshold + index * args.step, 2)
                       for index in range(steps + 1)]
    if not args.thresholds:
        raise SystemExit("empty threshold curve")
    return args


def _print_summary(report: Dict[str, Any]) -> None:
    print(f"states={report['dataset']['states']} scored={report['scoring']['scored']} "
          f"aborted={report['aborted']}")
    noul = report["noul"]
    print(f"noul: min={noul['min']} median={noul['median']} mean={noul['mean']} max={noul['max']}")
    print(f"brier={report['brier']}")
    print("threshold  coverage  trades  base_precision  false_hold  auc")
    for row in report["threshold_curve"]:
        print(f"{row['threshold']:<10} {row['coverage']!s:<9} {row['trade_count']:<7} "
              f"{row['base_trade_precision']!s:<15} {row['false_hold_rate']!s:<11} "
              f"{row['directional_auc']}")
    if report["aborted"]:
        print(f"ABORTED: {report['abort_reason']}")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: run the calibration and write the JSON report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    report = asyncio.run(run_calibration(args))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if args.observations:
        written = write_observations(report, args.observations)
        print(f"observations: {os.path.abspath(args.observations)} ({written} lines)")
    _print_summary(report)
    print(f"report: {output}")
    return 2 if report["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main())
