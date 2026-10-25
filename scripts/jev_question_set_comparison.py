"""Compare the v1 and v2 JEV question-set campaigns symbol by symbol.

v1 is the production wording; v2 separates "will price move >= 1 % in the next
hour" (noul) from "which direction" (bias). Both campaigns score the SAME
states, so this tool pairs them by candle index and reports:

* prevalence, tradeability AUC, direction AUC of noul, direction AUC of the
  bias answer, and the rank-rejection difference at 30/40/50 %;
* a paired bootstrap 95 % interval for every v2-minus-v1 difference (the same
  states are resampled for both versions, which removes state-selection noise);
* an overall summary that counts symbols, never pools levels across symbols.

A difference whose interval contains zero is reported as noise. Seven symbols
times several metrics is a multiple-comparison problem; a single symbol whose
interval clears zero is not evidence, and the output says so explicitly.

    python scripts/jev_question_set_comparison.py \\
        --v1-reports 'runtime/jev_bench/calibration_*_200.json' \\
        --v2-reports 'runtime/jev_bench/calibration_*_200_v2.json' \\
        --output runtime/jev_bench/v2_comparison.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jev_calibration import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    _percentile,
)

HORIZON = 1
MOVE_THRESHOLD_PCT = 1.0
REJECT_FRACTIONS = (0.30, 0.40, 0.50)
DIRECTIONAL_BIAS = ("long", "short")
Z_95 = 1.959963984540054

Record = Dict[str, Any]
MetricFn = Callable[[Sequence[Record]], Optional[float]]


# ------------------------------------------------------------------- metrics
def _average_ranks(values: Sequence[float]) -> List[float]:
    """Return 1-based average ranks, ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def ranking_auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank AUC with ties at half, in O(n log n); None when a class is absent."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks([float(score) for score in scores])
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _auc_metric(score_key: str) -> MetricFn:
    """AUC of one record score against the worth-trading label."""
    def metric(records: Sequence[Record]) -> Optional[float]:
        return ranking_auc(
            [float(record[score_key]) for record in records],
            [bool(record["label"]) for record in records],
        )
    return metric


def _direction_auc_noul_metric(score_key: str) -> MetricFn:
    """AUC of one record score against the sign of the forward return."""
    def metric(records: Sequence[Record]) -> Optional[float]:
        return ranking_auc(
            [float(record[score_key]) for record in records],
            [bool(record["up"]) for record in records],
        )
    return metric


def _bias_metric(bias_key: str) -> MetricFn:
    """AUC of long=1 / short=0 against the forward sign.

    Only states with an explicit long/short bias are used; ``none`` is not a
    direction prediction and is excluded instead of being scored as 0.5.
    """
    def metric(records: Sequence[Record]) -> Optional[float]:
        selected = [
            record for record in records
            if record[bias_key] in DIRECTIONAL_BIAS
        ]
        return ranking_auc(
            [1.0 if record[bias_key] == "long" else 0.0 for record in selected],
            [bool(record["up"]) for record in selected],
        )
    return metric


def _bias_coverage_metric(bias_key: str) -> MetricFn:
    """Share of states with an explicit long/short bias."""
    def metric(records: Sequence[Record]) -> Optional[float]:
        if not records:
            return None
        return sum(
            1 for record in records if record[bias_key] in DIRECTIONAL_BIAS
        ) / len(records)
    return metric


def _rejection_difference(records: Sequence[Record], score_key: str, fraction: float) -> Optional[float]:
    """Kept-minus-rejected event rate after rejecting the lowest score tail."""
    if not records:
        return None
    ordered = sorted(records, key=lambda record: float(record[score_key]))
    cut = int(round(fraction * len(ordered)))
    cut = min(max(cut, 0), len(ordered) - 1)
    rejected, kept = ordered[:cut], ordered[cut:]
    if not rejected or not kept:
        return None
    kept_rate = sum(1 for record in kept if record["label"]) / len(kept)
    rejected_rate = sum(1 for record in rejected if record["label"]) / len(rejected)
    return kept_rate - rejected_rate


def _rejection_metric(score_key: str, fraction: float) -> MetricFn:
    """Factory for one rejection cut."""
    def metric(records: Sequence[Record]) -> Optional[float]:
        return _rejection_difference(records, score_key, fraction)
    return metric


def precision_interval(successes: int, total: int, z: float = Z_95) -> Optional[Dict[str, float]]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * (
        (proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) ** 0.5
    ) / denominator
    return {
        "low": round(max(0.0, centre - half), 4),
        "high": round(min(1.0, centre + half), 4),
    }


def bootstrap_ci(
    records: Sequence[Record],
    metric: MetricFn,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[Dict[str, Optional[float]]]:
    """Deterministic percentile bootstrap 95 % CI for one metric."""
    if metric(records) is None:
        return None
    generator = random.Random(seed)
    count = len(records)
    values: List[float] = []
    for _ in range(resamples):
        sample = [records[generator.randrange(count)] for _ in range(count)]
        value = metric(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return {
        "low": round(_percentile(values, 2.5), 4),
        "high": round(_percentile(values, 97.5), 4),
        "resamples": len(values),
    }


def paired_bootstrap_delta(
    records: Sequence[Record],
    metric_v1: MetricFn,
    metric_v2: MetricFn,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[Dict[str, Any]]:
    """Paired bootstrap CI for ``metric_v2 - metric_v1`` on the same states."""
    observed_v1 = metric_v1(records)
    observed_v2 = metric_v2(records)
    if observed_v1 is None or observed_v2 is None:
        return None
    observed = observed_v2 - observed_v1
    generator = random.Random(seed)
    count = len(records)
    values: List[float] = []
    for _ in range(resamples):
        sample = [records[generator.randrange(count)] for _ in range(count)]
        value_v1 = metric_v1(sample)
        value_v2 = metric_v2(sample)
        if value_v1 is None or value_v2 is None:
            continue
        values.append(value_v2 - value_v1)
    if not values:
        return None
    values.sort()
    low = round(_percentile(values, 2.5), 4)
    high = round(_percentile(values, 97.5), 4)
    return {
        "observed": round(observed, 4),
        "ci95": {"low": low, "high": high},
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "share_le_zero": round(
            sum(1 for value in values if value <= 0.0) / len(values), 4
        ),
        "resamples": len(values),
    }


def _interval_result(
    records: Sequence[Record],
    metric: MetricFn,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """A level and its bootstrap interval."""
    value = metric(records)
    return {
        "value": round(float(value), 4) if value is not None else None,
        "ci95": bootstrap_ci(records, metric, resamples, seed),
    }


def _difference_result(
    records: Sequence[Record],
    metric_v1: MetricFn,
    metric_v2: MetricFn,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """A v2-minus-v1 difference and its paired bootstrap interval."""
    delta = paired_bootstrap_delta(records, metric_v1, metric_v2, resamples, seed)
    if delta is None:
        return {"observed": None, "ci95": None, "excludes_zero": None}
    return delta


# -------------------------------------------------------------------- loading
def _scored_states(report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Return usable states keyed by candle index."""
    scored: Dict[int, Dict[str, Any]] = {}
    for state in report.get("states") or []:
        if state.get("noul") is None:
            continue
        worth = (state.get("worth_trading") or {}).get(str(HORIZON))
        forward = (state.get("forward_returns") or {}).get(str(HORIZON))
        if worth is None or forward is None:
            continue
        scored[int(state["index"])] = {
            "noul": float(state["noul"]),
            "bias": state.get("bias"),
            "label": bool(worth),
            "up": float(forward) > 0.0,
        }
    return scored


def _paired_records(
    v1_report: Dict[str, Any],
    v2_report: Dict[str, Any],
) -> List[Record]:
    """Pair v1 and v2 states on the same candle index."""
    v1_states = _scored_states(v1_report)
    v2_states = _scored_states(v2_report)
    records: List[Record] = []
    for index in sorted(set(v1_states) & set(v2_states)):
        first = v1_states[index]
        second = v2_states[index]
        records.append(
            {
                "index": index,
                "label": first["label"],
                "up": first["up"],
                "v1_noul": first["noul"],
                "v2_noul": second["noul"],
                "v1_bias": first["bias"],
                "v2_bias": second["bias"],
            }
        )
    return records


def _load_by_symbol(paths: Iterable[str], pattern: str = "") -> Dict[str, Dict[str, Any]]:
    """Load reports and key them by their dataset symbol."""
    loaded: Dict[str, Dict[str, Any]] = {}
    for path in sorted(paths):
        if pattern and pattern not in os.path.basename(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        symbol = (report.get("dataset") or {}).get("symbol")
        if not symbol:
            continue
        loaded[symbol] = report
    return loaded


# ------------------------------------------------------------------- analysis
def compare_symbol(
    symbol: str,
    v1_report: Dict[str, Any],
    v2_report: Dict[str, Any],
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute every paired metric for one symbol."""
    records = _paired_records(v1_report, v2_report)
    events = sum(1 for record in records if record["label"])
    prevalence = {
        "states": len(records),
        "events": events,
        "rate": round(events / len(records), 4) if records else None,
        "ci95": precision_interval(events, len(records)),
    }

    tradeability_v1 = _auc_metric("v1_noul")
    tradeability_v2 = _auc_metric("v2_noul")
    direction_noul_v1 = _direction_auc_noul_metric("v1_noul")
    direction_noul_v2 = _direction_auc_noul_metric("v2_noul")
    bias_v1 = _bias_metric("v1_bias")
    bias_v2 = _bias_metric("v2_bias")

    rank_rejection: Dict[str, Any] = {}
    for fraction in REJECT_FRACTIONS:
        key = f"{int(round(fraction * 100))}"
        metric_v1 = _rejection_metric("v1_noul", fraction)
        metric_v2 = _rejection_metric("v2_noul", fraction)
        level_v1 = _interval_result(records, metric_v1, resamples, seed)
        level_v2 = _interval_result(records, metric_v2, resamples, seed)
        rank_rejection[key] = {
            "reject_fraction": fraction,
            "v1": {
                **level_v1,
                "better_than_noise": bool(
                    level_v1["ci95"]
                    and level_v1["ci95"]["low"] is not None
                    and level_v1["ci95"]["low"] > 0.0
                ),
            },
            "v2": {
                **level_v2,
                "better_than_noise": bool(
                    level_v2["ci95"]
                    and level_v2["ci95"]["low"] is not None
                    and level_v2["ci95"]["low"] > 0.0
                ),
            },
            "delta_v2_minus_v1": _difference_result(
                records, metric_v1, metric_v2, resamples, seed
            ),
        }

    v2_bias_coverage = _interval_result(
        records, _bias_coverage_metric("v2_bias"), resamples, seed
    )
    v1_bias_coverage = _interval_result(
        records, _bias_coverage_metric("v1_bias"), resamples, seed
    )

    return {
        "symbol": symbol,
        "v1_model": (v1_report.get("dataset") or {}).get("model"),
        "v2_model": (v2_report.get("dataset") or {}).get("model"),
        "prevalence": prevalence,
        "tradeability_auc": {
            "v1": _interval_result(records, tradeability_v1, resamples, seed),
            "v2": _interval_result(records, tradeability_v2, resamples, seed),
            "delta_v2_minus_v1": _difference_result(
                records, tradeability_v1, tradeability_v2, resamples, seed
            ),
        },
        "direction_auc_noul": {
            "v1": _interval_result(records, direction_noul_v1, resamples, seed),
            "v2": _interval_result(records, direction_noul_v2, resamples, seed),
            "delta_v2_minus_v1": _difference_result(
                records, direction_noul_v1, direction_noul_v2, resamples, seed
            ),
        },
        "direction_bias": {
            "v1": {
                **_interval_result(records, bias_v1, resamples, seed),
                "directional_states": sum(
                    1 for record in records if record["v1_bias"] in DIRECTIONAL_BIAS
                ),
                "coverage": v1_bias_coverage,
            },
            "v2": {
                **_interval_result(records, bias_v2, resamples, seed),
                "directional_states": sum(
                    1 for record in records if record["v2_bias"] in DIRECTIONAL_BIAS
                ),
                "coverage": v2_bias_coverage,
            },
            "delta_v2_minus_v1": _difference_result(
                records, bias_v1, bias_v2, resamples, seed
            ),
        },
        "rank_rejection": rank_rejection,
        "v2_rejection_usable_all_cuts": all(
            rank_rejection[key]["v2"]["better_than_noise"]
            for key in rank_rejection
        ),
        "v2_rejection_cuts_better_than_noise": sum(
            1
            for key in rank_rejection
            if rank_rejection[key]["v2"]["better_than_noise"]
        ),
    }


def _summarize_deltas(rows: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Summarize v2-minus-v1 deltas across symbols without pooling levels."""
    usable = [(symbol, delta) for symbol, delta in rows if delta.get("observed") is not None]
    if not usable:
        return {
            "symbols": 0,
            "v2_better": 0,
            "v1_better": 0,
            "outside_noise_v2_better": 0,
            "outside_noise_v1_better": 0,
            "median_delta": None,
            "deltas": [],
        }
    deltas = [float(delta["observed"]) for _symbol, delta in usable]
    outside_up = [
        symbol for symbol, delta in usable
        if delta.get("excludes_zero") and float(delta["observed"]) > 0.0
    ]
    outside_down = [
        symbol for symbol, delta in usable
        if delta.get("excludes_zero") and float(delta["observed"]) < 0.0
    ]
    return {
        "symbols": len(usable),
        "v2_better": sum(1 for value in deltas if value > 0.0),
        "v1_better": sum(1 for value in deltas if value < 0.0),
        "outside_noise_v2_better": len(outside_up),
        "outside_noise_v1_better": len(outside_down),
        "outside_noise_v2_better_symbols": outside_up,
        "outside_noise_v1_better_symbols": outside_down,
        "median_delta": round(statistics.median(deltas), 4),
        "deltas": [
            {
                "symbol": symbol,
                "delta": delta["observed"],
                "ci95": delta.get("ci95"),
                "excludes_zero": delta.get("excludes_zero"),
            }
            for symbol, delta in usable
        ],
    }


def _overall(symbols: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the cross-symbol summary for every key metric."""
    def rows(path: Callable[[Dict[str, Any]], Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        return [(symbol["symbol"], path(symbol)) for symbol in symbols]

    rejection_rows: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for fraction in REJECT_FRACTIONS:
        key = f"{int(round(fraction * 100))}"
        rejection_rows[key] = rows(
            lambda symbol, key=key: symbol["rank_rejection"][key]["delta_v2_minus_v1"]
        )

    return {
        "tradeability_auc": _summarize_deltas(
            rows(lambda symbol: symbol["tradeability_auc"]["delta_v2_minus_v1"])
        ),
        "direction_auc_noul": _summarize_deltas(
            rows(lambda symbol: symbol["direction_auc_noul"]["delta_v2_minus_v1"])
        ),
        "direction_bias_auc": _summarize_deltas(
            rows(lambda symbol: symbol["direction_bias"]["delta_v2_minus_v1"])
        ),
        "rejection_difference": {
            key: _summarize_deltas(values) for key, values in rejection_rows.items()
        },
        "v2_symbols_with_usable_rejection_at_all_cuts": sum(
            1 for symbol in symbols if symbol["v2_rejection_usable_all_cuts"]
        ),
        "v2_symbols_with_at_least_one_usable_cut": sum(
            1 for symbol in symbols
            if symbol["v2_rejection_cuts_better_than_noise"] > 0
        ),
        "v1_symbols_with_any_usable_cut": sum(
            1 for symbol in symbols
            if any(
                row["v1"]["better_than_noise"]
                for row in symbol["rank_rejection"].values()
            )
        ),
        "direction_bias_ci_excludes_chance": {
            "v1": [
                symbol["symbol"] for symbol in symbols
                if symbol["direction_bias"]["v1"]["ci95"]
                and (
                    symbol["direction_bias"]["v1"]["ci95"]["low"] > 0.5
                    or symbol["direction_bias"]["v1"]["ci95"]["high"] < 0.5
                )
            ],
            "v2": [
                symbol["symbol"] for symbol in symbols
                if symbol["direction_bias"]["v2"]["ci95"]
                and (
                    symbol["direction_bias"]["v2"]["ci95"]["low"] > 0.5
                    or symbol["direction_bias"]["v2"]["ci95"]["high"] < 0.5
                )
            ],
        },
    }


def _verdict(symbols: Sequence[Dict[str, Any]], overall: Dict[str, Any]) -> Dict[str, Any]:
    """Derive plain-language verdict flags from the measured deltas."""
    magnitude = overall["tradeability_auc"]
    usable = overall["v2_symbols_with_usable_rejection_at_all_cuts"]
    total = len(symbols)
    magnitude_gain = (
        magnitude["outside_noise_v2_better"] >= 2
        and magnitude["outside_noise_v2_better"] > magnitude["outside_noise_v1_better"]
        and (magnitude["median_delta"] or 0.0) > 0.0
    )
    direction_gain = (
        len(overall["direction_bias_ci_excludes_chance"]["v2"])
        > len(overall["direction_bias_ci_excludes_chance"]["v1"])
        and len(overall["direction_bias_ci_excludes_chance"]["v2"]) >= max(3, total // 2)
    )
    rule_usable = usable >= max(4, (2 * total + 2) // 3)

    if not magnitude_gain and not direction_gain and not rule_usable:
        text = "no improvement; keep v1"
    else:
        helped: List[str] = []
        missed: List[str] = []
        if magnitude_gain:
            helped.append("big-move ranking")
        else:
            missed.append("big-move ranking")
        if direction_gain:
            helped.append("direction")
        else:
            missed.append("direction")
        if rule_usable:
            helped.append("a cross-symbol rank-rejection rule")
        else:
            missed.append("a cross-symbol rank-rejection rule")
        if helped and missed:
            text = (
                "partial improvement: v2 is better for "
                + ", ".join(helped)
                + "; not demonstrated for "
                + ", ".join(missed)
                + ". Production should keep v1 where the missing part matters."
            )
        elif helped:
            text = "v2 improves " + ", ".join(helped) + " over v1"
        else:
            text = "no improvement; keep v1"
    return {
        "magnitude_gain": magnitude_gain,
        "direction_gain": direction_gain,
        "rejection_rule_usable": rule_usable,
        "v2_rejection_usable_symbols": usable,
        "symbols": total,
        "text": text,
    }


def build_comparison(
    v1_paths: Sequence[str],
    v2_paths: Sequence[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Load both campaigns and compare every symbol present in both."""
    v1 = _load_by_symbol(v1_paths)
    v2 = _load_by_symbol(v2_paths)
    symbols: List[Dict[str, Any]] = []
    for symbol in sorted(set(v1) & set(v2)):
        symbols.append(
            compare_symbol(symbol, v1[symbol], v2[symbol], resamples, seed)
        )
    if not symbols:
        raise SystemExit("no symbol is present in both the v1 and v2 reports")
    overall = _overall(symbols)
    metric_checks = len(symbols) * (3 + len(REJECT_FRACTIONS))
    return {
        "kind": "jev-question-set-comparison",
        "horizon": HORIZON,
        "move_threshold_pct": MOVE_THRESHOLD_PCT,
        "v1_question_set": "v1",
        "v2_question_set": "v2",
        "v2_calibration_context": True,
        "bootstrap": {"resamples": resamples, "seed": seed, "paired": True},
        "multiple_testing": {
            "symbols": len(symbols),
            "metric_checks": metric_checks,
            "familywise_note": (
                "Each symbol contributes correlated tradeability, direction and "
                "rejection checks; 7 symbols times 6 checks is about 42 "
                "comparisons, so roughly 2 intervals can clear zero by chance "
                "at 95 %. A lone symbol is not evidence; look for a consistent "
                "pattern and a median delta in the same direction."
            ),
        },
        "symbols": symbols,
        "overall": overall,
        "verdict": _verdict(symbols, overall),
    }


# --------------------------------------------------------------------- output
def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_ci(interval: Optional[Dict[str, Any]]) -> str:
    if not interval:
        return "n/a"
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None:
        return "n/a"
    return f"{low:.3f}-{high:.3f}"


def _fmt_delta(delta: Optional[Dict[str, Any]]) -> str:
    if not delta or delta.get("observed") is None:
        return "n/a"
    text = f"{delta['observed']:+.3f}"
    ci = delta.get("ci95") or {}
    if ci.get("low") is not None and ci.get("high") is not None:
        text += f" ({ci['low']:+.3f} to {ci['high']:+.3f})"
    if delta.get("excludes_zero"):
        text += " *"
    return text


def print_markdown(summary: Dict[str, Any]) -> None:
    """Print compact Markdown tables for the PR body and the campaign doc."""
    print("\n### Per-symbol summary\n")
    print("| Symbol | States | Prevalence | v1 tradeability AUC (95% CI) | "
          "v2 tradeability AUC (95% CI) | v2-v1 (paired 95% CI) | "
          "v1 direction AUC (95% CI) | v2 direction AUC (95% CI) |")
    print("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
    for symbol in summary["symbols"]:
        prevalence = symbol["prevalence"]
        trade_v1 = symbol["tradeability_auc"]["v1"]
        trade_v2 = symbol["tradeability_auc"]["v2"]
        direction_v1 = symbol["direction_bias"]["v1"]
        direction_v2 = symbol["direction_bias"]["v2"]
        print(
            f"| {symbol['symbol']} | {prevalence['states']} | "
            f"{prevalence['events']}/{prevalence['states']} ({_fmt(prevalence['rate'], 3)}) | "
            f"{_fmt(trade_v1['value'])} ({_fmt_ci(trade_v1['ci95'])}) | "
            f"{_fmt(trade_v2['value'])} ({_fmt_ci(trade_v2['ci95'])}) | "
            f"{_fmt_delta(symbol['tradeability_auc']['delta_v2_minus_v1'])} | "
            f"{_fmt(direction_v1['value'])} ({_fmt_ci(direction_v1['ci95'])}) | "
            f"{_fmt(direction_v2['value'])} ({_fmt_ci(direction_v2['ci95'])}) |"
        )

    print("\n### Rank rejection (kept minus rejected event rate)\n")
    print("| Symbol | v1 reject 30% | v2 reject 30% | v2-v1 30% | "
          "v1 reject 40% | v2 reject 40% | v2-v1 40% | "
          "v1 reject 50% | v2 reject 50% | v2-v1 50% | v2 cuts usable |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |")
    for symbol in summary["symbols"]:
        cells = []
        for fraction in REJECT_FRACTIONS:
            key = f"{int(round(fraction * 100))}"
            row = symbol["rank_rejection"][key]
            cells.append(f"{_fmt(row['v1']['value'])}{' *' if row['v1']['better_than_noise'] else ''}")
            cells.append(f"{_fmt(row['v2']['value'])}{' *' if row['v2']['better_than_noise'] else ''}")
            cells.append(_fmt_delta(row["delta_v2_minus_v1"]))
        print(f"| {symbol['symbol']} | " + " | ".join(cells) + " | "
              f"{symbol['v2_rejection_cuts_better_than_noise']}/3 |")

    overall = summary["overall"]
    print("\n### Overall (paired deltas, v2 minus v1)\n")
    print("| Metric | v2 better | v1 better | v2 better outside noise | "
          "v1 better outside noise | Median delta |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    labels = {
        "tradeability_auc": "Tradeability AUC",
        "direction_auc_noul": "Direction AUC of noul",
        "direction_bias_auc": "Direction AUC of bias",
    }
    for key, label in labels.items():
        row = overall[key]
        print(
            f"| {label} | {row['v2_better']}/{row['symbols']} | "
            f"{row['v1_better']}/{row['symbols']} | "
            f"{row['outside_noise_v2_better']} | "
            f"{row['outside_noise_v1_better']} | {_fmt(row['median_delta'])} |"
        )
    for fraction in REJECT_FRACTIONS:
        key = f"{int(round(fraction * 100))}"
        row = overall["rejection_difference"][key]
        print(
            f"| Rejection difference {key}% | {row['v2_better']}/{row['symbols']} | "
            f"{row['v1_better']}/{row['symbols']} | "
            f"{row['outside_noise_v2_better']} | "
            f"{row['outside_noise_v1_better']} | {_fmt(row['median_delta'])} |"
        )
    print(
        f"\nSymbols with a usable v2 rejection rule at all three cuts: "
        f"{overall['v2_symbols_with_usable_rejection_at_all_cuts']}/"
        f"{len(summary['symbols'])}."
    )
    print(f"Bias AUC CI excludes chance: v1 "
          f"{overall['direction_bias_ci_excludes_chance']['v1']}, v2 "
          f"{overall['direction_bias_ci_excludes_chance']['v2']}.")
    print(f"\n**Verdict: {summary['verdict']['text']}**")
    print("\n`*` marks an interval that excludes zero. Intervals include zero "
          "for every other entry and are therefore noise.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare v1 and v2 JEV campaigns")
    parser.add_argument(
        "--v1-reports", nargs="+", default=None,
        help="v1 calibration reports (default: runtime/jev_bench/calibration_*_200.json)",
    )
    parser.add_argument(
        "--v2-reports", nargs="+", default=None,
        help="v2 calibration reports (default: runtime/jev_bench/calibration_*_200_v2.json)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_REPO_ROOT, "runtime", "jev_bench", "v2_comparison.json"),
    )
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the comparison and write the JSON summary."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    v1_paths = args.v1_reports or glob.glob(
        os.path.join(_REPO_ROOT, "runtime", "jev_bench", "calibration_*_200.json")
    )
    v2_paths = args.v2_reports or glob.glob(
        os.path.join(_REPO_ROOT, "runtime", "jev_bench", "calibration_*_200_v2.json")
    )
    # Exclude futures reports from the default spot globs.
    v1_paths = [path for path in v1_paths if "_futures" not in os.path.basename(path)]
    v2_paths = [path for path in v2_paths if "_futures" not in os.path.basename(path)]
    if not v1_paths or not v2_paths:
        raise SystemExit("need at least one v1 and one v2 report")
    summary = build_comparison(v1_paths, v2_paths, args.resamples, args.seed)
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
