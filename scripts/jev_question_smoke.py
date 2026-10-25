"""One-call smoke test for the JEV question sets before a campaign.

The calibration campaign is expensive, so check first that the provider accepts
the exact payload each question set produces. For every symbol this script
sends ONE call per question set with the last evaluable state, then verifies:

* the HTTP call succeeded (a provider ``400`` raises ``JevError``);
* all four typed answers are present;
* the tradeability answer carries a numeric ``noul``/``probability``;
* the bias and regime choices are in the allowed sets;
* no answer contains an ``invalid_question`` marker.

Any rejection makes the script exit non-zero and write the failure report; do
not start a campaign while a smoke call is rejected.

    python scripts/jev_question_smoke.py \\
        --csv-dir runtime/jev_bench \\
        --output runtime/jev_bench/v2_question_smoke.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bench_jev import _build_real_client, build_samples, load_candles_csv  # noqa: E402
from src.signal_service.jev_client import JevError  # noqa: E402
from src.signal_service.jev_gate import (  # noqa: E402
    ALLOWED_BIAS,
    ALLOWED_REGIME,
    QUESTION_SETS,
    build_questions,
    parse_jev_signal,
)

DEFAULT_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "ARB/USDT",
    "AXS/USDT",
    "NEXO/USDT",
)


def _csv_path(csv_dir: str, symbol: str) -> str:
    """Return the committed spot candle path for a symbol."""
    base = symbol.replace("/", "_")
    return os.path.join(csv_dir, f"{base}_1h.csv")


def _discover_symbols(csv_dir: str) -> List[str]:
    """Derive symbols from ``*_USDT_1h.csv`` files, excluding futures."""
    symbols: List[str] = []
    for path in sorted(glob.glob(os.path.join(csv_dir, "*_USDT_1h.csv"))):
        name = os.path.basename(path)[: -len("_1h.csv")]
        if name.endswith("_futures"):
            continue
        symbols.append(name.replace("_", "/", 1))
    return symbols


def _has_invalid_question(payload: Any) -> bool:
    """Recursively look for an ``invalid_question`` marker in an answer."""
    if isinstance(payload, str):
        return "invalid_question" in payload
    if isinstance(payload, dict):
        return any(_has_invalid_question(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_has_invalid_question(value) for value in payload)
    return False


def _numeric(value: Any) -> Optional[float]:
    """Return a finite float or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check_response(response: Any) -> List[str]:
    """Return the list of acceptance problems for one response."""
    problems: List[str] = []
    expected = {"tradeability", "bias", "regime", "confidence"}
    missing = expected - set(response.answers)
    if missing:
        problems.append(f"missing answers: {sorted(missing)}")
    tradeability = response.answer("tradeability")
    probability = _numeric(tradeability.get("noul", tradeability.get("probability")))
    if probability is None or not 0.0 <= probability <= 1.0:
        problems.append("tradeability.noul is missing or outside [0, 1]")
    bias = str(response.answer("bias").get("choice") or "").strip().lower()
    if bias not in ALLOWED_BIAS:
        problems.append(f"bias.choice invalid: {bias!r}")
    regime = str(response.answer("regime").get("choice") or "").strip().lower()
    if regime not in ALLOWED_REGIME:
        problems.append(f"regime.choice invalid: {regime!r}")
    if _has_invalid_question(response.answers):
        problems.append("provider returned an invalid_question marker")
    return problems


async def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    """Send one call per symbol and question set and collect the verdicts."""
    csv_dir = args.csv_dir
    symbols = args.symbols or _discover_symbols(csv_dir)
    if not symbols:
        raise SystemExit(f"no spot candle csv found in {csv_dir}")
    question_sets = args.question_sets or list(QUESTION_SETS)

    client = _build_real_client()
    results: List[Dict[str, Any]] = []
    try:
        for symbol in symbols:
            path = _csv_path(csv_dir, symbol)
            if not os.path.exists(path):
                results.append(
                    {
                        "symbol": symbol,
                        "question_set": None,
                        "accepted": False,
                        "problems": [f"missing candle file: {path}"],
                    }
                )
                continue
            candles = load_candles_csv(path)
            for question_set in question_sets:
                context = question_set == "v2"
                samples = build_samples(
                    candles,
                    symbol,
                    "1h",
                    horizons=(1,),
                    calibration_context=context,
                )
                if not samples:
                    results.append(
                        {
                            "symbol": symbol,
                            "question_set": question_set,
                            "accepted": False,
                            "problems": ["no evaluable state"],
                        }
                    )
                    continue
                _index, state, _future = samples[-1]
                questions = build_questions(question_set)
                entry: Dict[str, Any] = {
                    "symbol": symbol,
                    "question_set": question_set,
                    "calibration_context": context,
                    "candles": len(candles),
                    "state_keys": sorted(state),
                    "question_keys": sorted(questions),
                }
                try:
                    response = await client.ask(state, questions)
                except JevError as exc:
                    entry.update(
                        accepted=False,
                        problems=[f"provider rejected the call: {exc}"],
                        error_status=getattr(exc, "status", None),
                    )
                    results.append(entry)
                    continue
                problems = _check_response(response)
                signal = parse_jev_signal(response)
                entry.update(
                    accepted=not problems,
                    problems=problems,
                    model=response.model,
                    latency_ms=response.latency_ms,
                    tradeability_probability=signal.tradeability_probability,
                    bias=signal.bias,
                    regime=signal.regime,
                    confidence=signal.confidence,
                )
                results.append(entry)
    finally:
        await client.close()

    accepted = sum(1 for entry in results if entry.get("accepted"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "jev-question-set-smoke",
        "provider_model": client.model,
        "provider_host": client.api_key_host,
        "calls": len(results),
        "accepted": accepted,
        "rejected": len(results) - accepted,
        "all_accepted": accepted == len(results) and bool(results),
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test JEV question-set payloads")
    parser.add_argument("--csv-dir", default=os.path.join(_REPO_ROOT, "runtime", "jev_bench"))
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to test (default: every *_USDT_1h.csv in --csv-dir)",
    )
    parser.add_argument(
        "--question-sets", nargs="+", default=None, choices=QUESTION_SETS,
        help="Question sets to test (default: v1 v2)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_REPO_ROOT, "runtime", "jev_bench", "question_smoke.json"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the smoke calls and write the JSON report."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = asyncio.run(run_smoke(args))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("| Symbol | Set | Accepted | noul | bias | regime | problems |")
    print("| --- | --- | --- | ---: | --- | --- | --- |")
    for entry in report["results"]:
        print(
            f"| {entry.get('symbol')} | {entry.get('question_set')} | "
            f"{'yes' if entry.get('accepted') else 'NO'} | "
            f"{entry.get('tradeability_probability')} | {entry.get('bias')} | "
            f"{entry.get('regime')} | {'; '.join(entry.get('problems') or []) or '-'} |"
        )
    print(f"accepted: {report['accepted']}/{report['calls']}")
    print(f"report: {output}")
    return 0 if report["all_accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
