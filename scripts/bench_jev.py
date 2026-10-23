"""JEV benchmark scaffold (PLAN.md 7.5).

Only a real run over real historical candles produces a benchmark result:

    python scripts/bench_jev.py --candles data/BTC_USDT_1h.csv --symbol BTC/USDT

The tool evaluates four decision pipelines on the same states:

* ``llm_alone``   : the base LLM decision, untouched;
* ``jev_pre``     : JEV may skip the LLM decision entirely;
* ``jev_post``    : JEV may veto a non-HOLD LLM decision;
* ``jev_pre_post``: both gates (two JEV calls per state).

Labels are the raw forward return over 1 to 4 candles, computed only from
candles strictly after the state's last closed candle. Evaluation is
walk-forward: states are split into consecutive folds and metrics are reported
per fold and pooled. Nothing is fitted, so every fold is out-of-sample.

``--offline`` is a plumbing dry-run: it generates seeded synthetic candles and
uses a deterministic local stand-in for JEV. It exists to verify the harness,
never to measure JEV. Its report is explicitly marked
``"real_benchmark": false`` and must not be quoted as a performance result.

The base decision source is either a rule-based momentum proxy
(``--base momentum``) or the real LLM service (``--base llm``). Without an LLM
the report states that the LLM-alone numbers are unavailable rather than
inventing them.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_service.contract import iso_z, parse_iso, timeframe_seconds
from src.signal_service.jev_client import JevClient, JevResponse
from src.signal_service.jev_gate import JevGate, build_state

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/home/djelinux/TradingBots/trading-stack/runtime/jev_bench/bench_jev.json"
DEFAULT_HORIZONS = (1, 2, 3, 4)
DEFAULT_MOVE_THRESHOLD_PCT = 1.0
MIN_STATES = 200
MAX_STATES = 500


# ------------------------------------------------------------------ dataset
@dataclass
class Candle:
    """One closed OHLCV candle."""

    open_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_candles_csv(path: str) -> List[Candle]:
    """Load an OHLCV csv with columns time,open,high,low,close,volume."""
    import csv

    candles: List[Candle] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty csv: {path}")
        first = header[0].strip().lower()
        if first not in ("time", "date", "timestamp"):
            raise ValueError(f"unsupported csv header: {header}")
        for row in reader:
            if not row or len(row) < 6:
                continue
            open_at = parse_iso(row[0].replace("Z", "+00:00"))
            candles.append(
                Candle(
                    open_at=open_at,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
    candles.sort(key=lambda candle: candle.open_at)
    return candles


def synthetic_candles(count: int, seed: int, timeframe: str = "1h") -> List[Candle]:
    """Deterministic random-walk candles for the offline plumbing dry-run."""
    rng = random.Random(seed)
    candles: List[Candle] = []
    price = 60_000.0
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    period = timedelta(seconds=timeframe_seconds(timeframe))
    for index in range(count):
        drift = 0.0002 * math.sin(index / 37.0)
        shock = rng.gauss(0.0, 0.006)
        open_price = price
        close = max(1.0, open_price * (1.0 + drift + shock))
        high = max(open_price, close) * (1.0 + abs(rng.gauss(0.0, 0.002)))
        low = min(open_price, close) * (1.0 - abs(rng.gauss(0.0, 0.002)))
        volume = 100.0 * (1.0 + abs(rng.gauss(0.0, 0.4)))
        candles.append(
            Candle(
                open_at=start + index * period,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close
    return candles


def build_samples(
    candles: List[Candle],
    symbol: str,
    timeframe: str,
    warmup: int = 30,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> List[Tuple[int, Dict[str, Any], Dict[int, float]]]:
    """Build (index, state, forward returns) samples with strict no-look-ahead."""
    period = timeframe_seconds(timeframe)
    rows = [[c.open, c.high, c.low, c.close, c.volume] for c in candles]
    timestamps = [c.open_at for c in candles]
    samples: List[Tuple[int, Dict[str, Any], Dict[int, float]]] = []
    last_index = len(candles) - max(horizons) - 1
    for index in range(warmup, max(warmup, last_index + 1)):
        as_of = candles[index].open_at + timedelta(seconds=period)
        history_rows = rows[: index + 1]
        history_times = timestamps[: index + 1]
        try:
            state = build_state(
                symbol=symbol,
                timeframe=timeframe,
                candles=history_rows,
                timestamps=history_times,
                as_of=as_of,
            )
        except ValueError:
            continue
        future = {
            horizon: candles[index + horizon].close / candles[index].close - 1.0
            for horizon in horizons
        }
        samples.append((index, state, future))
    return samples


def split_folds(count: int, folds: int) -> List[Tuple[int, int]]:
    """Return contiguous [start, end) walk-forward folds."""
    if folds < 1:
        raise ValueError("folds must be >= 1")
    bounds: List[Tuple[int, int]] = []
    for fold in range(folds):
        start = count * fold // folds
        end = count * (fold + 1) // folds
        bounds.append((start, end))
    return bounds


# ---------------------------------------------------------------- decisions
@dataclass
class BaseDecision:
    """Decision produced by the base pipeline (LLM or momentum proxy)."""

    action: str
    confidence: float
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class MomentumBaseline:
    """Rule-based 4-candle momentum proxy used when no LLM is available."""

    name = "momentum-proxy"

    def decide(self, state: Dict[str, Any]) -> BaseDecision:
        """BUY above +1%, SELL below -1%, HOLD otherwise."""
        returns = state.get("returns") or {}
        momentum = returns.get("4")
        if momentum is None:
            return BaseDecision("HOLD", 0.0)
        if momentum > 0.01:
            return BaseDecision("BUY", min(1.0, abs(momentum) * 20))
        if momentum < -0.01:
            return BaseDecision("SELL", min(1.0, abs(momentum) * 20))
        return BaseDecision("HOLD", 0.0)


class LLMBackend:
    """Real LLM decisions through the signal service composition root.

    Construction is lazy and best-effort: without a working config the backend
    reports `available = False` instead of producing invented decisions.
    """

    name = "llm"

    def __init__(self, symbol: str, timeframe: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.available = False
        self.reason: Optional[str] = None
        self._analyzer = None
        self._injector = None

    async def initialize(self) -> None:
        """Build the real analyzer; record a clear reason when impossible."""
        try:
            from start import CompositionRoot  # local import: heavy optional dependency

            root = CompositionRoot()
            deps = await root.build_dependencies()
            engine = deps["market_analyzer_factory"](self.symbol)
            from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector

            self._analyzer = PairAnalyzer(
                engine=engine, symbol=self.symbol, timeframe=self.timeframe
            )
            self._injector = ReplayDataInjector
            self.available = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.reason = f"LLM backend unavailable: {exc}"

    async def decide(self, candles: List[Candle], index: int, state: Dict[str, Any]) -> BaseDecision:
        """Run the real analyzer on the closed-candle slice."""
        if not self.available or self._analyzer is None or self._injector is None:
            return BaseDecision("HOLD", 0.0, error=self.reason or "LLM backend unavailable")
        injector = self._injector(
            self._analyzer,
            [[c.open, c.high, c.low, c.close, c.volume] for c in candles[: index + 1]],
            [c.open_at for c in candles[: index + 1]],
        )
        injector.install()
        try:
            signal = await self._analyzer.analyze(candle_open_time=iso_z(candles[index].open_at))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return BaseDecision("HOLD", 0.0, error=str(exc))
        finally:
            injector.restore()
        return BaseDecision(signal.action, signal.confidence)


class OfflineStandInClient:
    """Deterministic local stand-in for JEV used only by `--offline`.

    It is a pure heuristic over the state and is NOT JEV. Results produced
    with it must never be presented as a JEV benchmark.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def ask(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> JevResponse:
        """Return a synthetic answer set derived from the state's momentum."""
        self.calls += 1
        momentum = (state.get("returns") or {}).get("4") or 0.0
        atr = state.get("atr_pct") or 0.0
        probability = max(0.0, min(1.0, 0.5 + momentum * 25.0))
        if momentum > 0.01:
            bias = "long"
        elif momentum < -0.01:
            bias = "short"
        else:
            bias = "none"
        if atr > 0.04:
            regime = "high_vol"
        elif atr < 0.008:
            regime = "low_vol"
        elif abs(momentum) > 0.01:
            regime = "trend"
        else:
            regime = "range"
        return JevResponse(
            model="offline-stand-in (NOT JEV)",
            answers={
                "tradeability": {"type": "noul", "noul": probability},
                "bias": {"type": "choice", "choice": bias, "confidence": 0.5},
                "regime": {"type": "choice", "choice": regime, "confidence": 0.5},
                "confidence": {
                    "type": "score",
                    "score": 2.0,
                    "legend": {"0": "none", "1": "weak", "2": "fair", "3": "strong", "4": "very strong"},
                },
            },
            latency_ms=0,
        )


@dataclass
class Prediction:
    """One decision after the configured pipeline, with its label."""

    index: int
    action: str
    probability_up: float
    fold: int
    latency_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ModeRun:
    """Aggregated predictions and side metrics for one pipeline."""

    mode: str
    predictions: List[Prediction] = field(default_factory=list)
    jev_calls: int = 0
    jev_errors: int = 0
    latencies_ms: List[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def action_probability_up(action: str) -> float:
    """Map a decision to P(up): BUY=1, SELL=0, HOLD=0.5."""
    if action == "BUY":
        return 1.0
    if action == "SELL":
        return 0.0
    return 0.5


# ------------------------------------------------------------------ metrics
def direction_auc(probabilities: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-based AUC with tie handling; None when one class is absent."""
    positives = [p for p, label in zip(probabilities, labels) if label]
    negatives = [p for p, label in zip(probabilities, labels) if not label]
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


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Brier score (lower is better); None for an empty sample."""
    if not probabilities:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    bins: int = 5,
) -> List[Dict[str, Any]]:
    """Bucket probabilities and report observed frequency per bin."""
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        index = min(bins - 1, max(0, int(probability * bins)))
        buckets[index].append((probability, label))
    result: List[Dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            result.append({"bin": f"{index / bins:.1f}-{(index + 1) / bins:.1f}", "count": 0})
            continue
        mean_probability = statistics.fmean(p for p, _ in bucket)
        observed = sum(1 for _p, label in bucket if label) / len(bucket)
        result.append(
            {
                "bin": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
                "count": len(bucket),
                "mean_probability": round(mean_probability, 4),
                "observed_up_rate": round(observed, 4),
            }
        )
    return result


def evaluate_predictions(
    predictions: Sequence[Prediction],
    samples: Sequence[Tuple[int, Dict[str, Any], Dict[int, float]]],
    horizon: int,
    move_threshold_pct: float,
) -> Dict[str, Any]:
    """Compute the PLAN metric set for one horizon."""
    returns = {index: future[horizon] for index, _state, future in samples}
    probabilities: List[float] = []
    labels: List[bool] = []
    trades = 0
    correct_trades = 0
    holds = 0
    false_holds = 0
    threshold = move_threshold_pct / 100.0
    for prediction in predictions:
        result = returns.get(prediction.index)
        if result is None:
            continue
        labels.append(result > 0)
        probabilities.append(prediction.probability_up)
        if prediction.action in ("BUY", "SELL"):
            trades += 1
            if (prediction.action == "BUY" and result > 0) or (prediction.action == "SELL" and result < 0):
                correct_trades += 1
        else:
            holds += 1
            if abs(result) >= threshold:
                false_holds += 1
    return {
        "samples": len(probabilities),
        "direction_auc": _rounded(direction_auc(probabilities, labels)),
        "trade_count": trades,
        "trade_precision": _rounded(correct_trades / trades if trades else None),
        "hold_count": holds,
        "false_hold_rate": _rounded(false_holds / holds if holds else None),
        "brier": _rounded(brier_score(probabilities, labels)),
        "reliability": reliability_bins(probabilities, labels),
        "coverage": _rounded(len(probabilities) / len(predictions) if predictions else None),
    }


def _rounded(value: Optional[float]) -> Optional[float]:
    """Round optional floats to 4 decimals."""
    return None if value is None else round(float(value), 4)


def latency_summary(latencies: Sequence[int]) -> Dict[str, Optional[float]]:
    """Return p50/p95/max latency in milliseconds."""
    if not latencies:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }


# ----------------------------------------------------------------- pipeline
async def run_mode(
    mode: str,
    gate: JevGate,
    base_backend: Any,
    candles: List[Candle],
    samples: List[Tuple[int, Dict[str, Any], Dict[int, float]]],
    folds: int,
    is_llm: bool,
) -> ModeRun:
    """Evaluate one pipeline over all states, walk-forward fold by fold."""
    run = ModeRun(mode=mode)
    bounds = split_folds(len(samples), folds)
    fold_of: List[int] = [0] * len(samples)
    for fold, (start, end) in enumerate(bounds):
        for position in range(start, end):
            fold_of[position] = fold

    for position, (index, state, _future) in enumerate(samples):
        base_latency: Optional[int] = None
        if is_llm:
            decision = await base_backend.decide(candles, index, state)
            base_latency = decision.latency_ms
        else:
            decision = base_backend.decide(state)

        action = decision.action
        error = decision.error
        pre_signal = None
        if mode in ("jev_pre", "jev_pre_post"):
            pre = await gate.check_pre(state)
            run.jev_calls += 1 if pre.consulted else 0
            if pre.error is not None:
                run.jev_errors += 1
            if not pre.allowed:
                action = "HOLD"
            else:
                pre_signal = pre.signal
                # Mirror the signal service risk layer: an explicit bias
                # conflict marks the decision as non-tradeable.
                if pre_signal is not None and (
                    (action == "BUY" and pre_signal.bias == "short")
                    or (action == "SELL" and pre_signal.bias == "long")
                ):
                    action = "HOLD"
        if action in ("BUY", "SELL") and mode in ("jev_post", "jev_pre_post"):
            post = await gate.check_post(state, action)
            run.jev_calls += 1 if post.consulted else 0
            if post.error is not None:
                run.jev_errors += 1
            if not post.allowed:
                action = "HOLD"
            signal = post.signal or pre_signal
            if signal is not None and signal.latency_ms is not None:
                run.latencies_ms.append(signal.latency_ms)
        elif pre_signal is not None and pre_signal.latency_ms is not None:
            run.latencies_ms.append(pre_signal.latency_ms)

        run.predictions.append(
            Prediction(
                index=index,
                action=action,
                probability_up=action_probability_up(action),
                fold=fold_of[position],
                latency_ms=base_latency,
                error=error,
            )
        )
    return run


def summarize_run(
    run: ModeRun,
    samples: Sequence[Tuple[int, Dict[str, Any], Dict[int, float]]],
    horizons: Sequence[int],
    move_threshold_pct: float,
    folds: int,
) -> Dict[str, Any]:
    """Aggregate one mode into per-fold and pooled metrics."""
    pooled = {
        f"h{horizon}": evaluate_predictions(run.predictions, samples, horizon, move_threshold_pct)
        for horizon in horizons
    }
    per_fold: List[Dict[str, Any]] = []
    bounds = split_folds(len(samples), folds)
    fold_of = {}
    for fold, (start, end) in enumerate(bounds):
        for position in range(start, end):
            fold_of[position] = fold
    index_order = [index for index, _state, _future in samples]
    index_to_position = {index: position for position, index in enumerate(index_order)}
    for fold in range(folds):
        fold_predictions = [
            prediction for prediction in run.predictions
            if fold_of.get(index_to_position.get(prediction.index, -1)) == fold
        ]
        per_fold.append(
            {
                "fold": fold,
                "samples": len(fold_predictions),
                f"h{horizons[0]}": evaluate_predictions(
                    fold_predictions, samples, horizons[0], move_threshold_pct
                ),
            }
        )
    errors = sum(1 for prediction in run.predictions if prediction.error)
    return {
        "mode": run.mode,
        "states": len(run.predictions),
        "errors": errors,
        "jev_calls": run.jev_calls,
        "jev_errors": run.jev_errors,
        "jev_latency": latency_summary(run.latencies_ms),
        "cost": {
            "usd": None,
            "note": "jev-1.13-free has no monetary cost; token usage is reported when available",
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
        },
        "metrics": pooled,
        "per_fold": per_fold,
    }


# ---------------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JEV gate benchmark scaffold (PLAN.md 7.5)")
    parser.add_argument("--candles", default=None, help="OHLCV csv (time,open,high,low,close,volume)")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=300, help="States to evaluate (200-500 recommended)")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--horizons", default="1,2,3,4")
    parser.add_argument("--move-threshold-pct", type=float, default=DEFAULT_MOVE_THRESHOLD_PCT)
    parser.add_argument("--tradeability-threshold", type=float, default=0.5)
    parser.add_argument("--fold-confidence-floor", type=float, default=None)
    parser.add_argument("--base", choices=["momentum", "llm"], default="momentum")
    parser.add_argument("--offline", action="store_true", help="Synthetic candles + local JEV stand-in (NOT a benchmark)")
    parser.add_argument("--synthetic-candles", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Pause between real JEV calls")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def _gate_for(mode: str, client: Any, threshold: float, floor: Optional[float]) -> JevGate:
    return JevGate(
        client=client,
        mode=mode,
        tradeability_threshold=threshold,
        min_confidence=floor,
    )


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run every pipeline and return the report payload."""
    horizons = tuple(int(value) for value in args.horizons.split(","))
    folds = max(1, args.folds)
    limit = max(1, min(MAX_STATES, args.limit))

    if args.offline:
        candles = synthetic_candles(args.synthetic_candles, args.seed, args.timeframe)
        data_source = f"synthetic (seed={args.seed})"
        client: Any = OfflineStandInClient()
        real_benchmark = False
    else:
        if not args.candles:
            raise SystemExit("--candles is required for a real benchmark (or pass --offline for a dry-run)")
        candles = load_candles_csv(args.candles)
        data_source = args.candles
        client = JevClient()
        real_benchmark = True
    if len(candles) < 60:
        raise SystemExit(f"not enough candles for a benchmark: {len(candles)}")

    samples = build_samples(candles, args.symbol, args.timeframe, horizons=horizons)
    if not samples:
        raise SystemExit("no samples could be built from the dataset")
    if len(samples) > limit:
        samples = samples[-limit:]
    if real_benchmark and len(samples) < MIN_STATES:
        logger.warning(
            "Only %d states available (PLAN recommends %d-%d); results are indicative",
            len(samples), MIN_STATES, MAX_STATES,
        )

    if args.base == "llm" and not args.offline:
        llm_backend = LLMBackend(args.symbol, args.timeframe)
        await llm_backend.initialize()
        base_backend: Any = llm_backend
        is_llm = llm_backend.available
        base_name = "llm" if llm_backend.available else f"unavailable ({llm_backend.reason})"
    else:
        base_backend = MomentumBaseline()
        is_llm = False
        base_name = (
            f"{base_backend.name} (synthetic dry-run proxy)" if args.offline else base_backend.name
        )

    modes = ["llm_alone", "jev_pre", "jev_post", "jev_pre_post"]
    results: List[Dict[str, Any]] = []
    for mode in modes:
        if mode != "llm_alone" and real_benchmark and args.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(args.delay_seconds)
        gate_mode = "off" if mode == "llm_alone" else mode.replace("jev_", "")
        gate = _gate_for(gate_mode, client, args.tradeability_threshold, args.fold_confidence_floor)
        run = await run_mode(
            mode=mode,
            gate=gate,
            base_backend=base_backend,
            candles=candles,
            samples=samples,
            folds=folds,
            is_llm=is_llm,
        )
        results.append(
            summarize_run(run, samples, horizons, args.move_threshold_pct, folds)
        )

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "real_benchmark": real_benchmark,
        "kind": "synthetic-offline-dry-run" if not real_benchmark else "historical-walk-forward",
        "disclaimer": (
            "SYNTHETIC OFFLINE DRY-RUN: synthetic candles and a local stand-in for JEV. "
            "These numbers do not measure JEV and must not be reported as a benchmark result."
            if not real_benchmark
            else "Walk-forward evaluation on historical candles; no parameter fitting."
        ),
        "dataset": {
            "source": data_source,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "candles": len(candles),
            "states": len(samples),
            "folds": folds,
            "horizons": list(horizons),
            "move_threshold_pct": args.move_threshold_pct,
            "base": base_name,
        },
        "modes": results,
        "notes": [
            "direction_auc: probability that a random up-state scores above a random down-state.",
            "trade_precision: share of non-HOLD decisions whose direction matched the forward return.",
            "false_hold_rate: HOLD decisions followed by a move >= move_threshold_pct.",
            "coverage: share of states that produced a decision.",
            "llm_alone with --base llm is the only LLM baseline; the momentum proxy is not an LLM.",
        ],
    }
    if not real_benchmark:
        report["real_benchmark_result"] = None
    if args.base == "llm" and not is_llm:
        report["llm_alone_available"] = False
        report["llm_unavailable_reason"] = base_backend.reason if isinstance(base_backend, LLMBackend) else None
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    print(f"kind: {report['kind']}")
    print(f"real_benchmark: {report['real_benchmark']}")
    if not report["real_benchmark"]:
        print("*** NO REAL BENCHMARK RESULT: synthetic data + local JEV stand-in ***")
    dataset = report["dataset"]
    print(f"dataset: {dataset['source']} | states={dataset['states']} | folds={dataset['folds']}")
    primary = list(report["modes"][0]["metrics"])[0] if report["modes"] else "h1"
    for mode in report["modes"]:
        metrics = mode["metrics"].get(primary, {})
        print(
            f"{mode['mode']:<14} auc={metrics.get('direction_auc')} "
            f"trade_precision={metrics.get('trade_precision')} "
            f"false_hold={metrics.get('false_hold_rate')} "
            f"coverage={metrics.get('coverage')} jev_calls={mode['jev_calls']}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: run the benchmark and write the JSON report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)
    import asyncio

    started = time.monotonic()
    report = asyncio.run(run_benchmark(args))
    report["duration_seconds"] = round(time.monotonic() - started, 2)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    _print_summary(report)
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
