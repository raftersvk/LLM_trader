"""Aggregate a multi-symbol JEV spot and futures calibration campaign.

Reads the per-symbol ``calibration_<base>_200.json`` and
``calibration_<base>_200_futures.json`` reports produced by
``scripts/jev_calibration.py`` and answers four questions per symbol-market at
the h1 horizon (``abs(forward_return) >= 1 %`` means "worth trading"):

1. how often the event happens (prevalence);
2. whether a high ``noul`` identifies it (top vs bottom group event rate and
   ranking AUC, both with 95 % bootstrap intervals);
3. whether ``noul`` says anything about direction (AUC for ``forward > 0``);
4. what a rank-based rejection rule would do (reject the lowest 30/40/50 %).

All intervals are deterministic bootstrap intervals with the same seed as the
calibration tool. Symbols are never pooled: pooling hides one symbol's fluke
behind another's. When futures reports are supplied, each symbol also gets a
spot-minus-futures comparison with an independent-sample bootstrap interval.

    python scripts/jev_campaign_analysis.py \\
        --reports runtime/jev_bench/calibration_*_200.json \\
        --futures-reports runtime/jev_bench/calibration_*_200_futures.json \\
        --output runtime/jev_bench/campaign_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jev_calibration import (  # noqa: E402  (path setup must run first)
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    _percentile,
    bootstrap_auc_ci,
    ranking_auc,
)

TOP_FRACTION = 0.30
REJECT_FRACTIONS = (0.30, 0.40, 0.50)
HORIZON = 1
Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> Optional[Dict[str, float]]:
    """Wilson score 95 % interval for a binomial proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"low": round(max(0.0, centre - half), 4), "high": round(min(1.0, centre + half), 4)}


def _scored_states(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return states with a usable noul and an h1 label."""
    states = []
    for state in report.get("states") or []:
        if state.get("noul") is None:
            continue
        worth = (state.get("worth_trading") or {}).get(str(HORIZON))
        forward = (state.get("forward_returns") or {}).get(str(HORIZON))
        if worth is None or forward is None:
            continue
        states.append(state)
    return states


def _rates(states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Event count, rate and Wilson interval for one group of states."""
    total = len(states)
    events = sum(1 for state in states if state["worth_trading"][str(HORIZON)])
    return {
        "states": total,
        "events": events,
        "rate": round(events / total, 4) if total else None,
        "ci95": wilson_interval(events, total),
    }


def _partition_by_score(
    records: Sequence[Tuple[float, bool]], fraction: float
) -> Tuple[List[Tuple[float, bool]], List[Tuple[float, bool]]]:
    """Split (score, event) records into (rejected-lowest, kept) by score."""
    ordered = sorted(records, key=lambda item: item[0])
    cut = int(round(fraction * len(ordered)))
    cut = min(max(cut, 0), len(ordered) - 1) if ordered else 0
    return ordered[:cut], ordered[cut:]


def _bootstrap_difference(
    records: Sequence[Tuple[float, bool]],
    select: Any,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[Dict[str, Any]]:
    """Bootstrap CI for the event-rate difference returned by ``select``.

    ``select`` maps a sample of (score, event) records to
    ``(better_group, worse_group)``. The full sample is resampled and the groups
    are re-derived on every resample, so the interval includes the uncertainty
    of where the rank cut lands.
    """
    better, worse = select(records)
    if not better or not worse:
        return None
    observed = (
        sum(1 for _score, event in better if event) / len(better)
        - sum(1 for _score, event in worse if event) / len(worse)
    )
    generator = random.Random(seed)
    count = len(records)
    values: List[float] = []
    for _ in range(resamples):
        sample = [records[generator.randrange(count)] for _ in range(count)]
        better_sample, worse_sample = select(sample)
        if not better_sample or not worse_sample:
            continue
        better_rate = sum(1 for _score, event in better_sample if event) / len(better_sample)
        worse_rate = sum(1 for _score, event in worse_sample if event) / len(worse_sample)
        values.append(better_rate - worse_rate)
    if not values:
        return None
    values.sort()
    share_not_better = sum(1 for value in values if value <= 0.0) / len(values)
    return {
        "observed": round(observed, 4),
        "ci95": {
            "low": round(_percentile(values, 2.5), 4),
            "high": round(_percentile(values, 97.5), 4),
        },
        "bootstrap_share_diff_le_zero": round(share_not_better, 4),
        "resamples": len(values),
    }


def _reject_select(fraction: float) -> Any:
    """Return a selector comparing the kept top group against the rejected tail."""
    def select(records: Sequence[Tuple[float, bool]]) -> Tuple[List[Tuple[float, bool]], List[Tuple[float, bool]]]:
        rejected, kept = _partition_by_score(records, fraction)
        return kept, rejected
    return select


def _top_bottom_select(fraction: float) -> Any:
    """Return a selector comparing the top ``fraction`` against the bottom one."""
    def select(records: Sequence[Tuple[float, bool]]) -> Tuple[List[Tuple[float, bool]], List[Tuple[float, bool]]]:
        ordered = sorted(records, key=lambda item: item[0])
        size = int(round(fraction * len(ordered)))
        size = min(max(size, 0), len(ordered) // 2) if ordered else 0
        if size == 0:
            return [], []
        return ordered[-size:], ordered[:size]
    return select


def _direction_report(states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """AUC of noul ranking the sign of the forward return (up vs down)."""
    scores = [float(state["noul"]) for state in states]
    labels = [
        float(state["forward_returns"][str(HORIZON)]) > 0.0
        for state in states
    ]
    auc = ranking_auc(scores, labels)
    ci = bootstrap_auc_ci(scores, labels)
    proven = bool(
        ci and auc is not None and (ci["low"] is not None and ci["high"] is not None)
        and (ci["low"] > 0.5 or ci["high"] < 0.5)
    )
    return {
        "states": len(states),
        "up_states": sum(1 for label in labels if label),
        "auc": round(auc, 4) if auc is not None else None,
        "ci95": ci,
        "direction_proven_95": proven,
    }


def _event_rate_stat(states: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Share of states labelled worth trading at the primary horizon."""
    if not states:
        return None
    return sum(
        1 for state in states if state["worth_trading"][str(HORIZON)]
    ) / len(states)


def _tradeability_auc_stat(states: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Ranking AUC of noul for the worth-trading label."""
    if not states:
        return None
    return ranking_auc(
        [float(state["noul"]) for state in states],
        [bool(state["worth_trading"][str(HORIZON)]) for state in states],
    )


def _direction_auc_stat(states: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Ranking AUC of noul for the sign of the forward return."""
    if not states:
        return None
    return ranking_auc(
        [float(state["noul"]) for state in states],
        [float(state["forward_returns"][str(HORIZON)]) > 0.0 for state in states],
    )


def _rejection_diff_stat(fraction: float) -> Any:
    """Return a stat function for kept-minus-rejected event rate at one cut."""
    def stat(states: Sequence[Dict[str, Any]]) -> Optional[float]:
        records = [
            (float(state["noul"]), bool(state["worth_trading"][str(HORIZON)]))
            for state in states
        ]
        rejected, kept = _partition_by_score(records, fraction)
        if not rejected or not kept:
            return None
        kept_rate = sum(1 for _score, event in kept if event) / len(kept)
        rejected_rate = sum(1 for _score, event in rejected if event) / len(rejected)
        return kept_rate - rejected_rate
    return stat


def _bootstrap_independent_delta(
    spot_states: Sequence[Dict[str, Any]],
    futures_states: Sequence[Dict[str, Any]],
    stat: Any,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[Dict[str, Any]]:
    """Bootstrap CI for ``stat(spot) - stat(futures)`` with independent samples.

    Each market is resampled with replacement independently, which is the right
    null for two separate candle series. The observed difference is reported as
    spot minus futures: positive means spot has the larger value.
    """
    spot_value = stat(spot_states)
    futures_value = stat(futures_states)
    if spot_value is None or futures_value is None:
        return None
    observed = spot_value - futures_value
    generator = random.Random(seed)
    spot_count = len(spot_states)
    futures_count = len(futures_states)
    values: List[float] = []
    for _ in range(resamples):
        spot_sample = [spot_states[generator.randrange(spot_count)] for _ in range(spot_count)]
        futures_sample = [
            futures_states[generator.randrange(futures_count)] for _ in range(futures_count)
        ]
        spot_sample_value = stat(spot_sample)
        futures_sample_value = stat(futures_sample)
        if spot_sample_value is None or futures_sample_value is None:
            continue
        values.append(spot_sample_value - futures_sample_value)
    if not values:
        return None
    values.sort()
    ci = {
        "low": round(_percentile(values, 2.5), 4),
        "high": round(_percentile(values, 97.5), 4),
    }
    return {
        "observed": round(observed, 4),
        "ci95": ci,
        "within_noise": bool(ci["low"] <= 0.0 <= ci["high"]),
        "resamples": len(values),
    }


def _compare_markets(
    spot_states: Sequence[Dict[str, Any]],
    futures_states: Sequence[Dict[str, Any]],
    spot_summary: Dict[str, Any],
    futures_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare one symbol's spot and futures metrics.

    ``observed`` deltas are spot minus futures. A delta whose CI contains zero
    is reported as within noise; with this sample size most deltas will be.
    """
    metrics: Dict[str, Any] = {}
    definitions = {
        "event_rate": _event_rate_stat,
        "tradeability_auc": _tradeability_auc_stat,
        "direction_auc": _direction_auc_stat,
    }
    for fraction in REJECT_FRACTIONS:
        definitions[f"rejection_diff_{int(round(fraction * 100))}"] = _rejection_diff_stat(fraction)
    for name, stat in definitions.items():
        metrics[name] = _bootstrap_independent_delta(spot_states, futures_states, stat)

    significant = [
        name for name, result in metrics.items()
        if result is not None and not result["within_noise"]
    ]
    spot_reliable = spot_summary.get("verdict") != "no reliable rejection rule today"
    futures_reliable = futures_summary.get("verdict") != "no reliable rejection rule today"
    if not significant:
        conclusion = (
            "same conclusion; all spot-vs-futures metric differences are within noise"
        )
    elif not spot_reliable and not futures_reliable:
        conclusion = (
            "same conclusion (neither market has a reliable rule), but "
            f"{len(significant)} metric difference(s) exceed noise: "
            + ", ".join(significant)
        )
    elif spot_reliable != futures_reliable:
        winning = "futures" if futures_reliable else "spot"
        conclusion = (
            f"different: only {winning} shows a nominal rejection signal; "
            f"{len(significant)} metric difference(s) exceed noise"
        )
    else:
        conclusion = (
            f"both markets show nominal signals; {len(significant)} metric "
            "difference(s) exceed noise: " + ", ".join(significant)
        )
    return {
        "symbol": spot_summary.get("symbol"),
        "spot_verdict": spot_summary.get("verdict"),
        "futures_verdict": futures_summary.get("verdict"),
        "metrics": metrics,
        "significant_metrics": significant,
        "all_metric_differences_within_noise": not significant,
        "conclusion": conclusion,
    }


def _symbol_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compute every campaign metric for one symbol report."""
    states = _scored_states(report)
    records = [
        (float(state["noul"]), bool(state["worth_trading"][str(HORIZON)]))
        for state in states
    ]
    scores = [score for score, _event in records]
    labels = [event for _score, event in records]
    overall = _rates(states)
    auc = ranking_auc(scores, labels)
    auc_ci = bootstrap_auc_ci(scores, labels)
    proven = bool(
        auc_ci and auc_ci["low"] is not None and auc_ci["high"] is not None
        and (auc_ci["low"] > 0.5 or auc_ci["high"] < 0.5)
    )

    ordered = sorted(records, key=lambda item: item[0])
    group_size = int(round(TOP_FRACTION * len(ordered)))
    bottom = ordered[:group_size]
    top = ordered[-group_size:] if group_size else []
    top_bottom = None
    if top and bottom:
        top_events = sum(1 for _score, event in top if event)
        bottom_events = sum(1 for _score, event in bottom if event)
        top_bottom = {
            "fraction": TOP_FRACTION,
            "top": {
                "states": len(top),
                "events": top_events,
                "rate": round(top_events / len(top), 4),
                "ci95": wilson_interval(top_events, len(top)),
            },
            "bottom": {
                "states": len(bottom),
                "events": bottom_events,
                "rate": round(bottom_events / len(bottom), 4),
                "ci95": wilson_interval(bottom_events, len(bottom)),
            },
            "difference": _bootstrap_difference(records, _top_bottom_select(TOP_FRACTION)),
        }

    rejection = []
    for fraction in REJECT_FRACTIONS:
        rejected, kept = _partition_by_score(records, fraction)
        kept_events = sum(1 for _score, event in kept if event)
        rejected_events = sum(1 for _score, event in rejected if event)
        difference = _bootstrap_difference(records, _reject_select(fraction))
        rejection.append(
            {
                "reject_fraction": fraction,
                "skipped_states": len(rejected),
                "kept_states": len(kept),
                "kept": {
                    "states": len(kept),
                    "events": kept_events,
                    "rate": round(kept_events / len(kept), 4) if kept else None,
                    "ci95": wilson_interval(kept_events, len(kept)),
                },
                "rejected": {
                    "states": len(rejected),
                    "events": rejected_events,
                    "rate": round(rejected_events / len(rejected), 4) if rejected else None,
                    "ci95": wilson_interval(rejected_events, len(rejected)),
                },
                "difference": difference,
                "better_than_noise": bool(
                    difference
                    and difference["ci95"]["low"] is not None
                    and difference["ci95"]["low"] > 0.0
                ),
            }
        )

    scoring = report.get("scoring") or {}
    dataset = report.get("dataset") or {}
    return {
        "symbol": dataset.get("symbol"),
        "market_type": dataset.get("market_type") or "spot",
        "report": dataset.get("source"),
        "model": dataset.get("model"),
        "aborted": report.get("aborted"),
        "abort_reason": report.get("abort_reason"),
        "calls": scoring.get("calls"),
        "scored": scoring.get("scored"),
        "errors": scoring.get("errors"),
        "rate_limit_errors": scoring.get("rate_limit_errors"),
        "other_errors": scoring.get("other_errors"),
        "prevalence": overall,
        "tradeability_auc": {
            "auc": round(auc, 4) if auc is not None else None,
            "ci95": auc_ci,
            "proven_95": proven,
        },
        "top_vs_bottom": top_bottom,
        "direction": _direction_report(states),
        "rank_rejection": rejection,
    }


def _verdict(symbol: Dict[str, Any]) -> str:
    """Plain-language per-symbol verdict for a rank-based rejection rule."""
    rejection = symbol.get("rank_rejection") or []
    wins = [row for row in rejection if row.get("better_than_noise")]
    if not wins:
        return "no reliable rejection rule today"
    if len(wins) == len(rejection):
        return "rank rejection separates event rates at all three cuts"
    return "rank rejection separates event rates at some cuts only"


def _load_reports(paths: Sequence[str]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Load reports and their per-symbol summaries in path order."""
    loaded: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        summary = _symbol_summary(report)
        summary["verdict"] = _verdict(summary)
        loaded.append((report, summary))
    return loaded


def build_summary(
    paths: Sequence[str],
    futures_paths: Sequence[str] = (),
) -> Dict[str, Any]:
    """Load every report and build the per-symbol spot/futures summary."""
    spot_loaded = _load_reports(paths)
    futures_loaded = _load_reports(futures_paths)
    symbols = [summary for _report, summary in spot_loaded]
    futures_symbols = [summary for _report, summary in futures_loaded]

    spot_by_symbol = {
        summary["symbol"]: (report, summary) for report, summary in spot_loaded
    }
    futures_by_symbol = {
        summary["symbol"]: (report, summary) for report, summary in futures_loaded
    }
    comparisons = []
    for symbol in sorted(set(spot_by_symbol) & set(futures_by_symbol)):
        spot_report, spot_summary = spot_by_symbol[symbol]
        futures_report, futures_summary = futures_by_symbol[symbol]
        comparisons.append(
            _compare_markets(
                _scored_states(spot_report),
                _scored_states(futures_report),
                spot_summary,
                futures_summary,
            )
        )
    futures_skipped = [
        {
            "symbol": symbol,
            "reason": (
                "no Binance USDT-M perpetual market"
                if symbol == "NEXO/USDT"
                else "no futures report supplied"
            ),
        }
        for symbol in sorted(set(spot_by_symbol) - set(futures_by_symbol))
    ]

    spot_checks = len(symbols) * (1 + 1 + len(REJECT_FRACTIONS))
    futures_checks = len(futures_symbols) * (1 + 1 + len(REJECT_FRACTIONS))
    return {
        "kind": "jev-spot-futures-campaign-summary",
        "horizon": HORIZON,
        "move_threshold_pct": 1.0,
        "top_fraction": TOP_FRACTION,
        "reject_fractions": list(REJECT_FRACTIONS),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "multiple_testing": {
            "spot_symbols": len(symbols),
            "futures_symbols": len(futures_symbols),
            "symbol_markets": len(symbols) + len(futures_symbols),
            "metric_checks": spot_checks + futures_checks,
            "note": (
                "13 symbol-market evaluations, each with correlated AUC and "
                "rank-rejection checks; a single good-looking symbol is not evidence"
            ),
        },
        "symbols": symbols,
        "futures_symbols": futures_symbols,
        "futures_skipped": futures_skipped,
        "comparisons": comparisons,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _fmt_ci(ci: Optional[Dict[str, float]]) -> str:
    if not ci or ci.get("low") is None or ci.get("high") is None:
        return "n/a"
    return f"{100.0 * ci['low']:.2f}-{100.0 * ci['high']:.2f}%"


def _fmt_auc_ci(value: Optional[float], ci: Optional[Dict[str, float]]) -> str:
    if value is None:
        return "n/a"
    if not ci or ci.get("low") is None:
        return f"{value:.3f}"
    return f"{value:.3f} ({ci['low']:.3f}-{ci['high']:.3f})"


def _fmt_delta(result: Optional[Dict[str, Any]], percent: bool = False) -> str:
    """Format one spot-minus-futures delta with its interval."""
    if not result:
        return "n/a"
    scale = 100.0 if percent else 1.0
    unit = " pp" if percent else ""
    ci = result.get("ci95") or {}
    low = ci.get("low")
    high = ci.get("high")
    observed = result.get("observed")
    text = f"{observed * scale:+.2f}{unit}"
    if low is not None and high is not None:
        text += f" ({low * scale:+.2f} to {high * scale:+.2f}{unit})"
    if result.get("within_noise"):
        text += " [noise]"
    return text


def _print_market_tables(symbols: Sequence[Dict[str, Any]], label: str) -> None:
    """Print the two per-symbol tables for one market."""
    print(f"\n### {label}")
    print("| Symbol | Worth-trading rate | Top 30% event rate | Bottom 30% event rate | "
          "Difference (95% CI) | Tradeability AUC (95% CI) | Direction AUC (95% CI) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for symbol in symbols:
        prevalence = symbol["prevalence"]
        top_bottom = symbol.get("top_vs_bottom") or {}
        top = top_bottom.get("top") or {}
        bottom = top_bottom.get("bottom") or {}
        difference = top_bottom.get("difference") or {}
        direction = symbol["direction"]
        print(
            f"| {symbol['symbol']} | {prevalence['events']}/{prevalence['states']} "
            f"({_fmt_pct(prevalence['rate'])}) | {_fmt_pct(top.get('rate'))} | "
            f"{_fmt_pct(bottom.get('rate'))} | "
            f"{_fmt_pct(difference.get('observed'))} "
            f"({_fmt_ci(difference.get('ci95'))}) | "
            f"{_fmt_auc_ci(symbol['tradeability_auc']['auc'], symbol['tradeability_auc']['ci95'])} | "
            f"{_fmt_auc_ci(direction['auc'], direction['ci95'])} |"
        )
    print("\n| Symbol | Reject 30%: skipped | Kept rate | Rejected rate | Difference (95% CI) | "
          "Reject 40%: diff | Reject 50%: diff | Verdict |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for symbol in symbols:
        rows = {row["reject_fraction"]: row for row in symbol["rank_rejection"]}
        first = rows[0.30]
        print(
            f"| {symbol['symbol']} | {first['skipped_states']} | "
            f"{_fmt_pct(first['kept']['rate'])} | {_fmt_pct(first['rejected']['rate'])} | "
            f"{_fmt_pct(first['difference']['observed'])} "
            f"({_fmt_ci(first['difference']['ci95'])}) | "
            f"{_fmt_pct(rows[0.40]['difference']['observed'])} | "
            f"{_fmt_pct(rows[0.50]['difference']['observed'])} | {symbol['verdict']} |"
        )


def _print_comparisons(summary: Dict[str, Any]) -> None:
    """Print the per-symbol spot-vs-futures comparison table."""
    spot = {symbol["symbol"]: symbol for symbol in summary["symbols"]}
    futures = {symbol["symbol"]: symbol for symbol in summary.get("futures_symbols") or []}
    print("\n### Spot vs USDT-M perpetual futures (delta = spot minus futures)")
    print("| Symbol | Event rate spot -> futures (delta, 95% CI) | "
          "Tradeability AUC spot -> futures (delta, 95% CI) | "
          "Direction AUC spot -> futures (delta, 95% CI) | Rejection diffs 30/40/50 spot | "
          "Rejection diffs 30/40/50 futures | Conclusion |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for comparison in summary.get("comparisons") or []:
        name = comparison["symbol"]
        spot_row = spot[name]
        futures_row = futures[name]
        metrics = comparison["metrics"]
        spot_rejection = "/".join(
            _fmt_pct(row["difference"]["observed"]) for row in spot_row["rank_rejection"]
        )
        futures_rejection = "/".join(
            _fmt_pct(row["difference"]["observed"]) for row in futures_row["rank_rejection"]
        )
        print(
            f"| {name} | {_fmt_pct(spot_row['prevalence']['rate'])} -> "
            f"{_fmt_pct(futures_row['prevalence']['rate'])} "
            f"({_fmt_delta(metrics['event_rate'], percent=True)}) | "
            f"{spot_row['tradeability_auc']['auc']:.3f} -> "
            f"{futures_row['tradeability_auc']['auc']:.3f} "
            f"({_fmt_delta(metrics['tradeability_auc'])}) | "
            f"{spot_row['direction']['auc']:.3f} -> "
            f"{futures_row['direction']['auc']:.3f} "
            f"({_fmt_delta(metrics['direction_auc'])}) | {spot_rejection} | "
            f"{futures_rejection} | {comparison['conclusion']} |"
        )
    for skipped in summary.get("futures_skipped") or []:
        print(f"\nSkipped futures: {skipped['symbol']} ({skipped['reason']}).")


def print_markdown(summary: Dict[str, Any]) -> None:
    """Print compact Markdown tables ready to paste into the campaign doc."""
    _print_market_tables(summary["symbols"], "Spot")
    if summary.get("futures_symbols"):
        _print_market_tables(summary["futures_symbols"], "USDT-M perpetual futures")
    if summary.get("comparisons"):
        _print_comparisons(summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate a JEV multi-symbol campaign")
    parser.add_argument("--reports", nargs="+", default=None, help="Spot calibration reports")
    parser.add_argument(
        "--futures-reports", nargs="+", default=None,
        help="USDT-M futures calibration reports (optional)",
    )
    parser.add_argument("--output", default=os.path.join(_REPO_ROOT, "runtime", "jev_bench", "campaign_summary.json"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the aggregation and write the JSON summary."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    paths = args.reports
    if not paths:
        paths = glob.glob(os.path.join(_REPO_ROOT, "runtime", "jev_bench", "calibration_*_200.json"))
    futures_paths = args.futures_reports
    if futures_paths is None:
        futures_paths = glob.glob(
            os.path.join(_REPO_ROOT, "runtime", "jev_bench", "calibration_*_200_futures.json")
        )
    if not paths:
        raise SystemExit("no calibration reports found")
    summary = build_summary(paths, futures_paths)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print_markdown(summary)
    print(f"\nsummary: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
