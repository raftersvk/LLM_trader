"""Unit tests for the weekly JEV policy calibration (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "jev_calibrate",
    Path(__file__).resolve().parents[1] / "scripts" / "jev_calibrate.py",
)
assert _SPEC is not None and _SPEC.loader is not None
jev_calibrate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = jev_calibrate
_SPEC.loader.exec_module(jev_calibrate)

Observation = jev_calibrate.Observation
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _obs(
    noul: float,
    base_action: str = "BUY",
    forward: float = 0.02,
    *,
    pair: str = "BTC/USDT",
    bias: str = "long",
    regime: str = "trend",
    atr_pct: float | None = 0.01,
    opened: datetime | None = None,
    cost_pct: float = 0.10,
) -> Any:
    return Observation(
        candle_open_time=opened or (NOW - timedelta(hours=1)),
        pair=pair,
        noul=noul,
        base_action=base_action,
        forward={1: forward, 2: forward, 3: forward, 4: forward},
        bias=bias,
        regime=regime,
        atr_pct=atr_pct,
        cost_pct=cost_pct,
    )


def _policy(generated_at: datetime, threshold: float = 0.6, version: int = 3) -> Dict[str, Any]:
    return {
        "version": version,
        "generated_at": generated_at.isoformat(),
        "window_days": 60,
        "min_states": 200,
        "threshold": threshold,
        "fallback": 0.5,
        "expire_at": (generated_at + timedelta(days=14)).isoformat(),
        "offset_rules": {},
        "metrics": {"sample": 400},
    }


def test_isotonic_is_monotonic_and_pools_violations():
    """PAVA merges decreasing neighbours into one monotone block."""
    model = jev_calibrate.fit_isotonic([0.1, 0.2, 0.3, 0.4], [False, True, False, True])

    rates = [rate for _x, rate in model]
    assert rates == sorted(rates)
    assert jev_calibrate.isotonic_predict(model, 0.1) == 0.0
    assert jev_calibrate.isotonic_predict(model, 0.3) == pytest.approx(0.5)
    assert jev_calibrate.isotonic_predict(model, 0.9) == 1.0
    assert jev_calibrate.fit_isotonic([], []) == []


def test_brier_score_known_value():
    """Brier is the mean squared error of the calibrated probability."""
    assert jev_calibrate.brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)
    assert jev_calibrate.brier_score([], []) is None


def test_clamp_bounds():
    """Clamp keeps values inside the band and rejects an inverted band."""
    assert jev_calibrate.clamp(0.2, 0.35, 0.65) == 0.35
    assert jev_calibrate.clamp(0.9, 0.35, 0.65) == 0.65
    assert jev_calibrate.clamp(0.5, 0.35, 0.65) == 0.5
    with pytest.raises(ValueError):
        jev_calibrate.clamp(0.5, 0.7, 0.3)


def test_smooth_threshold_uses_ewma_and_clamps():
    """Weekly smoothing is 0.7 current + 0.3 candidate, clamped to 0.35-0.65."""
    assert jev_calibrate.smooth_threshold(0.5, 0.9) == pytest.approx(0.62)
    assert jev_calibrate.smooth_threshold(0.35, 0.3) == pytest.approx(0.35)
    assert jev_calibrate.smooth_threshold(0.65, 0.7) == pytest.approx(0.65)


def test_limit_delta_caps_weekly_move():
    """A single week can never move more than 0.05."""
    assert jev_calibrate.limit_delta(0.5, 0.62) == pytest.approx(0.55)
    assert jev_calibrate.limit_delta(0.5, 0.40) == pytest.approx(0.45)
    assert jev_calibrate.limit_delta(0.5, 0.53) == pytest.approx(0.53)
    with pytest.raises(ValueError):
        jev_calibrate.limit_delta(0.5, 0.6, max_delta=-0.1)


def test_should_promote_requires_two_percent():
    """Promotion needs a 2 % relative out-of-sample improvement."""
    assert jev_calibrate.should_promote(10.0, 10.25) is True
    assert jev_calibrate.should_promote(10.0, 10.1) is False
    assert jev_calibrate.should_promote(0.0, 0.0) is False
    assert jev_calibrate.should_promote(-5.0, 1.0) is True


def test_vol_offset_high_and_low_rules():
    """High vol is OR (regime or ATR); low vol is AND (regime and ATR)."""
    rules = {
        "high_vol": {
            "offset": 0.05,
            "regimes": ["high_vol"],
            "atr_cutoff": {"BTC/USDT": 0.02},
        },
        "low_vol": {
            "offset": -0.05,
            "regimes": ["range"],
            "atr_cutoff": {"BTC/USDT": 0.005},
        },
    }
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=0.03, regime="range"), rules) == 0.05
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=0.01, regime="high_vol"), rules) == 0.05
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=0.004, regime="range"), rules) == -0.05
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=0.004, regime="trend"), rules) == 0.0
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=0.01, regime="range"), rules) == 0.0
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=None, regime="high_vol"), rules) == 0.05
    assert jev_calibrate.vol_offset(_obs(0.5, atr_pct=None, regime="range"), rules) == 0.0
    assert jev_calibrate.vol_offset(_obs(0.5), None) == 0.0


def test_build_offset_rules_uses_pair_percentiles():
    """Per-pair ATR cutoffs are the 20th/80th percentiles of the window."""
    observations = [_obs(noul, atr_pct=atr) for noul, atr in
                    [(0.1, 0.001), (0.2, 0.002), (0.3, 0.003), (0.4, 0.004), (0.5, 0.010)]]
    rules = jev_calibrate.build_offset_rules(observations)

    assert rules["high_vol"]["atr_cutoff"]["BTC/USDT"] == pytest.approx(0.0052)
    assert rules["low_vol"]["atr_cutoff"]["BTC/USDT"] == pytest.approx(0.0018)
    assert rules["high_vol"]["combine"] == "any"
    assert rules["low_vol"]["combine"] == "all"


def test_fallback_below_min_states():
    """Fewer than 200 trade-labeled states keeps 0.5 and marks the fallback."""
    policy = jev_calibrate.calibrate_policy(
        [_obs(0.5) for _ in range(199)], None, NOW,
    )
    assert policy["status"] == "fallback"
    assert policy["threshold"] == 0.5
    assert policy["metrics"]["sample"] == 199
    assert policy["metrics"]["label_states_used"] == 199
    assert policy["metrics"]["label_states_excluded"] == 0
    assert "insufficient trade-labeled states" in policy["metrics"]["reason"]
    assert policy["label"]["type"] == "trade_pnl_directional"
    assert jev_calibrate.validate_policy(policy) == []


def test_hold_states_are_excluded_from_the_trade_label():
    """HOLD states carry no trade direction and must not count toward min_states."""
    observations = [_obs(0.5, base_action="HOLD", forward=0.05) for _ in range(250)]
    policy = jev_calibrate.calibrate_policy(observations, None, NOW)

    assert policy["status"] == "fallback"
    assert policy["metrics"]["label_states_used"] == 0
    assert policy["metrics"]["label_states_excluded"] == 250
    assert "0 < 200" in policy["metrics"]["reason"]


def test_trade_label_uses_base_direction_over_each_horizon():
    """BUY uses +forward, SELL uses -forward, on every 1-4 horizon."""
    buy = Observation(
        candle_open_time=NOW - timedelta(hours=1), pair="BTC/USDT", noul=0.6,
        base_action="BUY", forward={1: -0.01, 2: 0.005, 3: 0.02, 4: 0.03},
        bias="long", regime="trend", atr_pct=0.01, cost_pct=0.1,
    )
    sell = Observation(
        candle_open_time=NOW - timedelta(hours=2), pair="BTC/USDT", noul=0.4,
        base_action="SELL", forward={1: 0.01, 2: -0.005, 3: -0.02, 4: -0.03},
        bias="short", regime="trend", atr_pct=0.01, cost_pct=0.1,
    )
    assert jev_calibrate.net_label(buy, 1) is False
    assert jev_calibrate.net_label(buy, 4) is True
    assert jev_calibrate.net_label(sell, 1) is False
    assert jev_calibrate.net_label(sell, 4) is True
    assert jev_calibrate.net_return_pct(buy, 4) == pytest.approx(2.9)
    assert jev_calibrate.net_return_pct(sell, 4) == pytest.approx(2.9)


def test_cost_pct_override_changes_the_label_enabled_by_cli():
    """A higher explicit cost can flip a marginal trade to a loss."""
    observation = _obs(0.5, "BUY", 0.01)  # 1.0 % gross against a 0.1 % cost
    assert jev_calibrate.net_label(observation, 1, cost_pct=0.1) is True
    assert jev_calibrate.net_label(observation, 1, cost_pct=1.2) is False
    assert jev_calibrate.net_return_pct(observation, 1, cost_pct=1.2) == pytest.approx(-0.2)


def test_fallback_below_min_states_preserves_previous_threshold():
    """A fresh previous policy keeps its threshold when data is insufficient."""
    previous = _policy(NOW - timedelta(days=1), threshold=0.6)
    policy = jev_calibrate.calibrate_policy([_obs(0.5) for _ in range(50)], previous, NOW)

    assert policy["status"] == "fallback"
    assert policy["threshold"] == pytest.approx(0.6)
    assert policy["version"] == 4


def test_stale_policy_resolves_to_fallback():
    """A policy older than 14 days must not be used."""
    fresh = _policy(NOW - timedelta(days=13), threshold=0.6)
    stale = _policy(NOW - timedelta(days=15), threshold=0.6)

    assert jev_calibrate.policy_is_stale(stale, NOW) is True
    assert jev_calibrate.policy_is_stale(fresh, NOW) is False
    assert jev_calibrate.resolve_threshold(stale, NOW) == 0.5
    assert jev_calibrate.resolve_threshold(fresh, NOW) == pytest.approx(0.6)
    assert jev_calibrate.resolve_threshold(None, NOW) == 0.5
    assert jev_calibrate.resolve_threshold({"generated_at": "bad"}, NOW) == 0.5


def test_stale_previous_forces_fallback_threshold_during_calibration():
    """A stale previous policy is ignored, so calibration starts from 0.5."""
    previous = _policy(NOW - timedelta(days=30), threshold=0.65)
    observations = [_obs(0.2 + (index % 5) * 0.1) for index in range(200)]
    policy = jev_calibrate.calibrate_policy(observations, previous, NOW)

    assert 0.35 <= policy["threshold"] <= 0.65
    assert abs(policy["threshold"] - 0.5) <= 0.05


def test_calibrate_policy_schema_and_weekly_cap():
    """A full window produces a valid policy with a capped weekly move."""
    observations = [
        _obs(
            0.2 + (index % 7) * 0.1,
            base_action="BUY" if index % 2 == 0 else "SELL",
            forward=0.02 if index % 3 else -0.005,
            opened=NOW - timedelta(hours=index),
        )
        for index in range(240)
    ]
    policy = jev_calibrate.calibrate_policy(observations, None, NOW)

    assert jev_calibrate.validate_policy(policy) == []
    assert policy["status"] == "active"
    assert policy["window_days"] == 60
    assert policy["min_states"] == 200
    assert policy["fallback"] == 0.5
    assert abs(policy["threshold"] - 0.5) <= 0.05
    assert policy["expire_at"]
    assert "threshold_curve" in policy["metrics"]
    assert set(policy["offset_rules"]) == {"high_vol", "low_vol"}


def test_policy_reports_trade_pnl_by_horizon_and_horizon_choice():
    """The policy records the selected horizon and all four horizon metrics."""
    observations = [
        Observation(
            candle_open_time=NOW - timedelta(hours=index),
            pair="BTC/USDT",
            noul=0.2 + (index % 6) * 0.1,
            base_action="BUY",
            forward={1: -0.005, 2: 0.005, 3: 0.015, 4: 0.025},
            bias="long",
            regime="trend",
            atr_pct=0.01,
            cost_pct=0.1,
        )
        for index in range(240)
    ]
    policy = jev_calibrate.calibrate_policy(
        observations, None, NOW, cost_pct=0.1, trade_horizon=4
    )

    assert policy["label"]["horizon"] == 4
    assert policy["label"]["horizons_evaluated"] == [1, 2, 3, 4]
    assert policy["label"]["cost_pct"] == pytest.approx(0.1)
    assert set(policy["metrics"]["trade_pnl_by_horizon"]) == {"h1", "h2", "h3", "h4"}
    assert policy["metrics"]["trade_pnl_by_horizon"]["h4"]["states_used"] == 240
    assert policy["metrics"]["label_states_used"] == 240


def test_validate_policy_reports_missing_fields_and_bad_label():
    """Schema validation lists missing keys and rejects a non-trade label."""
    problems = jev_calibrate.validate_policy({})
    assert "missing version" in problems
    assert "missing threshold" in problems
    assert "missing label" in problems

    policy = jev_calibrate.calibrate_policy([_obs(0.5) for _ in range(250)], None, NOW)
    policy = dict(policy)
    policy["label"] = {"type": "absolute_move", "horizon": 0, "cost_pct": -1}
    problems = jev_calibrate.validate_policy(policy)
    assert "label.type must be trade_pnl_directional" in problems
    assert "label.horizon must be a positive int" in problems
    assert "label.cost_pct must be >= 0" in problems


def test_load_observations_skips_malformed_lines(tmp_path):
    """One malformed line does not abort the whole log."""
    path = tmp_path / "observations.jsonl"
    valid = {
        "candle_open_time": (NOW - timedelta(hours=1)).isoformat(),
        "pair": "BTC/USDT",
        "noul": 0.4,
        "base_action": "BUY",
        "forward_1h": 0.01,
        "forward_2h": 0.02,
        "forward_3h": 0.03,
        "forward_4h": 0.04,
        "cost_pct": 0.1,
    }
    path.write_text(
        json.dumps(valid) + "\n"
        + "{not json}\n"
        + json.dumps({"noul": 0.5, "pair": "ETH/USDT"}) + "\n",
        encoding="utf-8",
    )
    observations = jev_calibrate.load_observations(str(path))

    assert len(observations) == 1
    assert observations[0].forward[4] == pytest.approx(0.04)
    assert jev_calibrate.filter_window(observations, NOW) == observations
    assert jev_calibrate.filter_window(observations, NOW + timedelta(days=61)) == []


def test_missing_bias_is_tradeable_but_explicit_none_is_not():
    """Match JevGate.check_pre: None bias is allowed, ``none`` is a hold."""
    missing = jev_calibrate.evaluate_threshold([_obs(0.9, "BUY", 0.02, bias=None)], 0.5)
    explicit = jev_calibrate.evaluate_threshold([_obs(0.9, "BUY", 0.02, bias="none")], 0.5)

    assert missing["trades"] == 1
    assert missing["net_benefit"] == pytest.approx(1.9)
    assert explicit["trades"] == 0
    assert explicit["net_benefit"] == 0.0


def test_net_labels_apply_cost_and_direction():
    """Labels and net returns use the base direction and the cost."""
    buy_win = _obs(0.5, "BUY", 0.02)
    buy_loss = _obs(0.5, "BUY", -0.02)
    sell_win = _obs(0.5, "SELL", -0.02)
    hold = _obs(0.5, "HOLD", 0.02)

    assert jev_calibrate.net_label(buy_win) is True
    assert jev_calibrate.net_label(buy_loss) is False
    assert jev_calibrate.net_label(sell_win) is True
    assert jev_calibrate.net_label(hold) is None
    assert jev_calibrate.net_return_pct(buy_win) == pytest.approx(1.9)
    assert jev_calibrate.net_return_pct(sell_win) == pytest.approx(1.9)
    assert jev_calibrate.net_return_pct(hold) is None
