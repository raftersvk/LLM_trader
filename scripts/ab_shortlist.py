#!/usr/bin/env python3
"""Offline A/B comparison for the recall-oriented v2 shortlist.

This script never calls an LLM and never touches the live trading service. It
reads archived signals/backtest records, applies ``build_shortlist`` and reports:

* recall of archived BUY/SELL signals,
* missed signal candidates and their rejection reasons,
* LLM calls saved versus the current full-universe path.

Input formats
-------------
JSON array, a JSON object with a ``records``/``signals``/``candidates`` key, or
JSON Lines. A record may be flat or nested::

    {
      "cycle": "2026-01-01T12:00:00Z",
      "symbol": "BTC/USDT",
      "p0a": {"regime": "TRENDING_UP", "allowed_setups": ["TREND_PULLBACK"], "atr_pct": 2.1},
      "setup": "TREND_PULLBACK",
      "open_position": false,
      "jev": {"bias": "LONG", "tradeable": true},
      "expected_signal": "BUY"
    }

Only ``expected_signal`` in BUY/SELL counts as a reference signal for recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm_focus.shortlist import (  # noqa: E402
    DEFAULT_MAX_ATR_PCT,
    DEFAULT_MIN_ATR_PCT,
    build_shortlist,
    candidate_from_mapping,
    evaluate_recall,
    meets_recall_guardrail,
    merge_recall_reports,
)

SIGNAL_ACTIONS = {"BUY", "SELL"}


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load archived records from JSON or JSON Lines without network access."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(parsed, dict):
            for key in ("records", "signals", "candidates", "data"):
                if isinstance(parsed.get(key), list):
                    records = list(parsed[key])
                    break
            else:
                records = [parsed]
        elif isinstance(parsed, list):
            records = list(parsed)
        else:
            raise ValueError("JSON root must be a list, object or JSON Lines")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every archived record must be a JSON object")
    return records


def expected_signal(record: dict[str, Any]) -> str:
    """Return the archived reference action, defaulting to HOLD."""
    value = record.get(
        "expected_signal",
        record.get("reference_signal", record.get("signal", record.get("action", "HOLD"))),
    )
    return str(value).strip().upper()


def _group_by_cycle(records: Iterable[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, record in enumerate(records):
        cycle = str(record.get("cycle", record.get("timestamp", f"record-{index}")))
        grouped.setdefault(cycle, []).append(record)
    return grouped


def run_ab(
    records: list[dict[str, Any]],
    min_atr_pct: float = DEFAULT_MIN_ATR_PCT,
    max_atr_pct: float = DEFAULT_MAX_ATR_PCT,
    full_pairs: int | None = None,
    min_recall: float = 0.95,
) -> dict[str, Any]:
    """Run the shortlist A/B over archived records and return a summary."""
    reports = []
    reason_counts: Counter[str] = Counter()
    missed_with_reasons: list[dict[str, Any]] = []
    shortlisted_symbols: list[str] = []

    for cycle_records in _group_by_cycle(records).values():
        candidates = [candidate_from_mapping(record) for record in cycle_records]
        result = build_shortlist(
            candidates,
            min_atr_pct=min_atr_pct,
            max_atr_pct=max_atr_pct,
        )
        rejection_by_symbol = {
            rejection.symbol: rejection.reasons for rejection in result.rejections
        }
        for rejection in result.rejections:
            reason_counts.update(rejection.reasons)

        references = [
            candidate.symbol
            for candidate, record in zip(candidates, cycle_records)
            if expected_signal(record) in SIGNAL_ACTIONS
        ]
        full_calls = len(cycle_records) if full_pairs is None else full_pairs
        reports.append(
            evaluate_recall(
                references,
                result.symbols,
                full_llm_calls=full_calls,
                shortlist_llm_calls=len(result.symbols),
            )
        )
        shortlisted_symbols.extend(result.symbols)

        shortlisted = set(result.symbols)
        for symbol in references:
            if symbol not in shortlisted:
                missed_with_reasons.append(
                    {
                        "symbol": symbol,
                        "reasons": list(rejection_by_symbol.get(symbol, ["not_in_shortlist"])),
                    }
                )

    merged = merge_recall_reports(reports)
    return {
        "records": len(records),
        "cycles": len(reports),
        "universe_symbols": sorted({candidate.symbol for candidate in _all_candidates(records)}),
        "shortlisted_symbols": shortlisted_symbols,
        "reference_signals": merged.total_signals,
        "captured": merged.captured,
        "recall": merged.recall,
        "missed": list(merged.missed),
        "missed_with_reasons": missed_with_reasons,
        "full_llm_calls": merged.full_llm_calls,
        "shortlist_llm_calls": merged.shortlist_llm_calls,
        "llm_calls_saved": merged.llm_calls_saved,
        "rejection_reason_counts": dict(reason_counts),
        "min_recall": min_recall,
        "meets_recall_guardrail": meets_recall_guardrail(merged, min_recall=min_recall),
        "offline": True,
    }


def _all_candidates(records: Iterable[dict[str, Any]]):
    return [candidate_from_mapping(record) for record in records]


def format_summary(summary: dict[str, Any]) -> str:
    """Render the A/B summary for a human-readable CLI report."""
    lines = [
        "LLM-focus v2 shortlist A/B (offline, no LLM calls)",
        f"Records: {summary['records']} across {summary['cycles']} cycle(s)",
        f"Reference BUY/SELL signals: {summary['reference_signals']}",
        f"Captured: {summary['captured']} | Recall: {summary['recall']:.1%}",
        (
            f"Recall guardrail >= {summary['min_recall']:.0%}: "
            f"{'PASS' if summary['meets_recall_guardrail'] else 'FAIL'}"
        ),
        (
            f"LLM calls: full={summary['full_llm_calls']} "
            f"shortlist={summary['shortlist_llm_calls']} "
            f"saved={summary['llm_calls_saved']}"
        ),
        f"Missed: {', '.join(summary['missed']) if summary['missed'] else 'none'}",
    ]
    if summary["missed_with_reasons"]:
        lines.append("Missed signal details:")
        for item in summary["missed_with_reasons"]:
            lines.append(f"  - {item['symbol']}: {', '.join(item['reasons'])}")
    if summary["rejection_reason_counts"]:
        lines.append("Rejection reasons:")
        for reason, count in sorted(summary["rejection_reason_counts"].items()):
            lines.append(f"  - {reason}: {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Archived JSON/JSONL records")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional local JSON output path")
    parser.add_argument("--min-atr-pct", type=float, default=DEFAULT_MIN_ATR_PCT)
    parser.add_argument("--max-atr-pct", type=float, default=DEFAULT_MAX_ATR_PCT)
    parser.add_argument(
        "--full-pairs",
        type=int,
        default=10,
        help="Nominal full-universe LLM calls per cycle (default: 10-pair path)",
    )
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument(
        "--fail-below-guardrail",
        action="store_true",
        help="Exit with status 2 when recall is below --min-recall",
    )
    args = parser.parse_args(argv)

    records = load_records(args.dataset)
    summary = run_ab(
        records,
        min_atr_pct=args.min_atr_pct,
        max_atr_pct=args.max_atr_pct,
        full_pairs=args.full_pairs,
        min_recall=args.min_recall,
    )
    print(format_summary(summary))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON summary written to {args.json_out}")

    if args.fail_below_guardrail and not summary["meets_recall_guardrail"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
