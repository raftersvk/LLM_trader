"""Tests for the v1-versus-v2 JEV question-set comparison math.

No network: synthetic calibration reports are compared in memory.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "jev_question_set_comparison",
    _SCRIPTS / "jev_question_set_comparison.py",
)
assert _SPEC is not None and _SPEC.loader is not None
comparison = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = comparison
_SPEC.loader.exec_module(comparison)


def _report(
    symbol: str,
    nouls: List[float],
    biases: List[str],
    labels: List[bool],
    ups: List[bool],
) -> Dict[str, Any]:
    states = []
    for index, (noul, bias, label, up) in enumerate(zip(nouls, biases, labels, ups)):
        states.append(
            {
                "index": index,
                "noul": noul,
                "bias": bias,
                "worth_trading": {"1": label},
                "forward_returns": {"1": 0.02 if up else -0.02},
            }
        )
    return {"dataset": {"symbol": symbol, "model": "test"}, "states": states}


def test_ranking_auc_matches_reference_with_ties():
    reference_spec = importlib.util.spec_from_file_location(
        "jev_calibration_ref", _SCRIPTS / "jev_calibration.py"
    )
    assert reference_spec is not None and reference_spec.loader is not None
    reference = importlib.util.module_from_spec(reference_spec)
    sys.modules["jev_calibration_ref"] = reference
    reference_spec.loader.exec_module(reference)

    generator = random.Random(11)
    scores = [round(generator.random(), 1) for _ in range(80)]
    labels = [generator.random() > 0.5 for _ in range(80)]
    assert comparison.ranking_auc(scores, labels) == pytest.approx(
        reference.ranking_auc(scores, labels)
    )


def test_paired_delta_detects_a_real_improvement():
    generator = random.Random(3)
    labels = [generator.random() < 0.3 for _ in range(120)]
    ups = [generator.random() < 0.5 for _ in range(120)]
    v1_noul = [generator.random() for _ in range(120)]
    v2_noul = [0.9 if label else 0.1 for label in labels]
    v1_bias = ["long" if generator.random() < 0.5 else "short" for _ in range(120)]
    v2_bias = ["long" if up else "short" for up in ups]

    result = comparison.compare_symbol(
        "BTC/USDT",
        _report("BTC/USDT", v1_noul, v1_bias, labels, ups),
        _report("BTC/USDT", v2_noul, v2_bias, labels, ups),
        resamples=300,
        seed=7,
    )

    tradeability = result["tradeability_auc"]["delta_v2_minus_v1"]
    assert tradeability["observed"] > 0.4
    assert tradeability["excludes_zero"] is True

    direction = result["direction_bias"]["delta_v2_minus_v1"]
    assert direction["observed"] > 0.4
    assert direction["excludes_zero"] is True
    assert result["direction_bias"]["v2"]["directional_states"] == 120
    assert result["direction_bias"]["v1"]["directional_states"] == 120

    # A perfect magnitude score also makes every rejection cut separate.
    assert result["v2_rejection_usable_all_cuts"] is True
    assert result["v2_rejection_cuts_better_than_noise"] == 3


def test_identical_campaigns_are_within_noise():
    generator = random.Random(5)
    labels = [generator.random() < 0.3 for _ in range(80)]
    ups = [generator.random() < 0.5 for _ in range(80)]
    noul = [generator.random() for _ in range(80)]
    biases = ["long" if up else "short" for up in ups]
    report = _report("ETH/USDT", noul, biases, labels, ups)

    result = comparison.compare_symbol(
        "ETH/USDT", report, report, resamples=200, seed=7
    )

    delta = result["tradeability_auc"]["delta_v2_minus_v1"]
    assert delta["observed"] == pytest.approx(0.0)
    assert delta["excludes_zero"] is False
    assert result["v2_rejection_cuts_better_than_noise"] in (0, 1, 2, 3)


def test_comparison_pairs_only_shared_indices():
    v1 = _report("SOL/USDT", [0.2, 0.8], ["long", "short"], [False, True], [False, True])
    v2 = _report("SOL/USDT", [0.3], ["long"], [True], [True])
    v2["states"][0]["index"] = 1
    records = comparison._paired_records(v1, v2)
    assert len(records) == 1
    assert records[0]["index"] == 1
    assert records[0]["v1_noul"] == pytest.approx(0.8)
    assert records[0]["v2_noul"] == pytest.approx(0.3)


def test_overall_summary_and_verdict_are_derived_not_pooled():
    generator = random.Random(9)
    symbols = []
    for symbol_index in range(3):
        labels = [generator.random() < 0.3 for _ in range(60)]
        ups = [generator.random() < 0.5 for _ in range(60)]
        v1 = _report(
            f"S{symbol_index}/USDT",
            [generator.random() for _ in range(60)],
            ["none"] * 60,
            labels,
            ups,
        )
        v2 = _report(
            f"S{symbol_index}/USDT",
            [0.9 if label else 0.1 for label in labels],
            ["long" if up else "short" for up in ups],
            labels,
            ups,
        )
        symbols.append(
            comparison.compare_symbol(
                f"S{symbol_index}/USDT", v1, v2, resamples=200, seed=7
            )
        )
    overall = comparison._overall(symbols)
    verdict = comparison._verdict(symbols, overall)

    assert overall["tradeability_auc"]["symbols"] == 3
    assert overall["tradeability_auc"]["v2_better"] == 3
    assert verdict["magnitude_gain"] is True
    assert verdict["direction_gain"] is True
    assert "partial improvement" in verdict["text"] or "improves" in verdict["text"]


def test_load_by_symbol_skips_aborted_reports(tmp_path):
    complete = _report("BTC/USDT", [0.2], ["long"], [False], [False])
    aborted = _report("ETH/USDT", [0.2], ["long"], [False], [False])
    aborted["aborted"] = True
    aborted["abort_reason"] = "JEV rate limited (429)"
    aborted["scoring"] = {"calls": 200, "scored": 80}
    first = tmp_path / "calibration_btc_200_v1.json"
    second = tmp_path / "calibration_eth_200_v1.json"
    first.write_text(__import__("json").dumps(complete), encoding="utf-8")
    second.write_text(__import__("json").dumps(aborted), encoding="utf-8")

    loaded, aborted_rows = comparison._load_by_symbol([str(first), str(second)])

    assert set(loaded) == {"BTC/USDT"}
    assert len(aborted_rows) == 1
    assert aborted_rows[0]["symbol"] == "ETH/USDT"
    assert aborted_rows[0]["scored"] == 80
    assert "429" in aborted_rows[0]["reason"]


def test_compare_symbol_flags_different_state_sets():
    v1 = _report("SOL/USDT", [0.2, 0.8], ["long", "short"], [False, True], [False, True])
    v2 = _report("SOL/USDT", [0.2, 0.8], ["long", "short"], [False, True], [False, True])
    v2["states"][1]["index"] = 9

    result = comparison.compare_symbol(
        "SOL/USDT", v1, v2, resamples=100, seed=7
    )

    assert result["same_states"] is False
    assert result["paired_states"] == 1
