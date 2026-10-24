"""Weekly JEV tradeability policy calibration (conservative, no live effect).

Reads the observation log produced by ``scripts/jev_calibration.py``, keeps the
last ``--window-days`` of states, fits an isotonic map from ``noul`` to the
probability that a base trade is profitable after costs, and proposes a small
weekly threshold move. Promotion is deliberately slow:

* the candidate must satisfy coverage, trade-count and positive net-benefit
  constraints;
* the weekly threshold is ``clamp(0.7 * current + 0.3 * candidate, 0.35, 0.65)``;
* the move is capped at ``0.05`` per week;
* it is promoted only when the out-of-sample (last 7 days) net benefit improves
  by at least 2 %;
* fewer than ``--min-states`` observations or a policy older than 14 days falls
  back to 0.5.

The signal service is NOT wired to this file in this PR. The policy is written
to a gitignored runtime path for the planned integration documented in
``docs/jev-policy.md``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.signal_service.contract import iso_z, parse_iso  # noqa: E402

logger = logging.getLogger(__name__)

FALLBACK_THRESHOLD = 0.5
WINDOW_DAYS = 60
MIN_STATES = 200
DEFAULT_COST_PCT = 0.10
THRESHOLD_GRID = tuple(round(0.30 + 0.05 * step, 2) for step in range(9))  # 0.30 .. 0.70
MIN_COVERAGE = 0.40
MIN_TRADES = 20
MIN_IMPROVEMENT = 0.02
MAX_WEEKLY_DELTA = 0.05
THRESHOLD_FLOOR = 0.35
THRESHOLD_CEILING = 0.65
OOS_DAYS = 7
POLICY_MAX_AGE_DAYS = 14
HIGH_VOL_OFFSET = 0.05
LOW_VOL_OFFSET = -0.05
TRADE_HORIZONS = (1, 2, 3, 4)


# ------------------------------------------------------------------ observations
@dataclass
class Observation:
    """One calibrated state from the observation log."""

    candle_open_time: datetime
    pair: str
    noul: float
    base_action: str
    forward: Dict[int, float] = field(default_factory=dict)
    bias: Optional[str] = None
    regime: Optional[str] = None
    atr_pct: Optional[float] = None
    cost_pct: float = DEFAULT_COST_PCT

    @property
    def is_directional(self) -> bool:
        """True when the base decision is a BUY or SELL."""
        return self.base_action in ("BUY", "SELL")


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float coercion; None when not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def observation_from_line(line: Dict[str, Any]) -> Optional[Observation]:
    """Parse one JSONL observation; return None for unusable lines."""
    noul = _as_float(line.get("noul"))
    raw_time = line.get("candle_open_time")
    if noul is None or not raw_time:
        return None
    try:
        opened = parse_iso(str(raw_time))
    except (TypeError, ValueError):
        return None
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    forward: Dict[int, float] = {}
    for horizon in (1, 2, 3, 4):
        value = _as_float(line.get(f"forward_{horizon}h"))
        if value is not None:
            forward[horizon] = value
    return Observation(
        candle_open_time=opened,
        pair=str(line.get("pair") or ""),
        noul=max(0.0, min(1.0, noul)),
        base_action=str(line.get("base_action") or "HOLD").upper(),
        forward=forward,
        bias=line.get("bias"),
        regime=line.get("regime"),
        atr_pct=_as_float(line.get("atr_pct")),
        cost_pct=_as_float(line.get("cost_pct")) or DEFAULT_COST_PCT,
    )


def load_observations(path: str) -> List[Observation]:
    """Load a JSONL observation log, skipping malformed lines.

    A missing log is not an error: the first weekly run (or a run after
    cleanup) has zero observations and must still produce the 0.5 fallback
    policy instead of crashing cron.
    """
    observations: List[Observation] = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        logger.warning("observation log not found (%s); treating it as empty", path)
        return []
    with handle:
        for number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except ValueError:
                logger.warning("skipping malformed observation on line %d", number)
                continue
            if not isinstance(payload, dict):
                continue
            observation = observation_from_line(payload)
            if observation is None:
                logger.warning("skipping unusable observation on line %d", number)
                continue
            observations.append(observation)
    return observations


def filter_window(
    observations: Sequence[Observation], now: datetime, window_days: int = WINDOW_DAYS
) -> List[Observation]:
    """Keep observations inside ``[now - window_days, now]``.

    Both bounds matter: a candle after ``now`` has not happened yet and must
    never enter the fit.
    """
    cutoff = now - timedelta(days=window_days)
    return [
        observation for observation in observations
        if cutoff <= observation.candle_open_time <= now
    ]


def last_days(
    observations: Sequence[Observation], reference: datetime, days: int = OOS_DAYS
) -> List[Observation]:
    """Keep observations in ``[reference - days, reference]``."""
    cutoff = reference - timedelta(days=days)
    return [
        observation for observation in observations
        if cutoff <= observation.candle_open_time <= reference
    ]


def split_train_oos(
    observations: Sequence[Observation],
    now: datetime,
    window_days: int = WINDOW_DAYS,
    oos_days: int = OOS_DAYS,
) -> Tuple[List[Observation], List[Observation]]:
    """Split the bounded window into pre-OOS training and held-out slices.

    The last ``oos_days`` are never used to fit the ATR rules or to choose the
    candidate threshold; they only validate promotion.
    """
    window = filter_window(observations, now, window_days)
    cutoff = now - timedelta(days=oos_days)
    train = [observation for observation in window if observation.candle_open_time < cutoff]
    oos = [observation for observation in window if observation.candle_open_time >= cutoff]
    return train, oos


# -------------------------------------------------------------------- isotonic
def fit_isotonic(probabilities: Sequence[float], labels: Sequence[bool]) -> List[Tuple[float, float]]:
    """Fit a monotone map with pool-adjacent-violators.

    Returns ordered ``(mean_probability, observed_rate)`` blocks. An empty
    sample returns an empty model.
    """
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    grouped: Dict[float, List[int]] = {}
    for probability, label in zip(probabilities, labels):
        counts = grouped.setdefault(float(probability), [0, 0])
        counts[0] += 1 if label else 0
        counts[1] += 1
    blocks: List[List[float]] = []
    for probability in sorted(grouped):
        positives, total = grouped[probability]
        blocks.append([probability, positives / total, float(total)])
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            left = blocks[-2]
            right = blocks.pop()
            left[1] = (left[1] * left[2] + right[1] * right[2]) / (left[2] + right[2])
            left[2] += right[2]
            left[0] = (left[0] * (left[2] - right[2]) + right[0] * right[2]) / left[2]
    return [(round(block[0], 6), round(block[1], 6)) for block in blocks]


def isotonic_predict(model: Sequence[Tuple[float, float]], value: float) -> float:
    """Predict with the piecewise-constant isotonic model."""
    if not model:
        return FALLBACK_THRESHOLD
    xs = [point[0] for point in model]
    index = bisect.bisect_right(xs, float(value)) - 1
    index = max(0, min(len(model) - 1, index))
    return model[index][1]


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Brier score for calibrated probabilities (lower is better)."""
    if not probabilities:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


# ---------------------------------------------------------------------- labels
def _trade_cost(observation: Observation, cost_pct: Optional[float]) -> float:
    """Return the explicit cost override, else the observation's stored cost."""
    return observation.cost_pct if cost_pct is None else float(cost_pct)


def net_label(
    observation: Observation, horizon: int = 1, cost_pct: Optional[float] = None
) -> Optional[bool]:
    """Trade-P&L label: the base trade beats costs on this horizon.

    Directional states only (BUY/SELL). HOLD states have no trade direction and
    return None so they are excluded from the trade-P&L label sample.
    """
    result = observation.forward.get(horizon)
    if result is None or not observation.is_directional:
        return None
    gross = result * 100.0 if observation.base_action == "BUY" else -result * 100.0
    return gross - _trade_cost(observation, cost_pct) > 0.0


def net_return_pct(
    observation: Observation, horizon: int = 1, cost_pct: Optional[float] = None
) -> Optional[float]:
    """Directional trade P&L in percent, net of the configured cost."""
    result = observation.forward.get(horizon)
    if result is None or not observation.is_directional:
        return None
    gross = result * 100.0 if observation.base_action == "BUY" else -result * 100.0
    return gross - _trade_cost(observation, cost_pct)


# ------------------------------------------------------------------- vol rules
def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Linear-interpolation percentile (0-100), None for an empty sample."""
    if not values:
        return None
    if not 0.0 <= quantile <= 100.0:
        raise ValueError("quantile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_offset_rules(observations: Sequence[Observation]) -> Dict[str, Any]:
    """Derive per-pair ATR cutoffs and the two volatility offset rules."""
    by_pair: Dict[str, List[float]] = {}
    for observation in observations:
        if observation.atr_pct is not None:
            by_pair.setdefault(observation.pair, []).append(observation.atr_pct)
    high_cutoff = {
        pair: round(value, 6)
        for pair, values in by_pair.items()
        if (value := percentile(values, 80)) is not None
    }
    low_cutoff = {
        pair: round(value, 6)
        for pair, values in by_pair.items()
        if (value := percentile(values, 20)) is not None
    }
    return {
        "high_vol": {
            "offset": HIGH_VOL_OFFSET,
            "atr_percentile_gte": 80,
            "regimes": ["high_vol"],
            "combine": "any",
            "atr_cutoff": high_cutoff,
        },
        "low_vol": {
            "offset": LOW_VOL_OFFSET,
            "atr_percentile_lte": 20,
            "regimes": ["range"],
            "combine": "all",
            "atr_cutoff": low_cutoff,
        },
    }


def vol_offset(observation: Observation, rules: Optional[Dict[str, Any]]) -> float:
    """Return +0.05, -0.05 or 0.0 for one state.

    High volatility fires on ``regime == high_vol`` OR ATR above the pair's
    80th percentile. Low volatility fires only on ``regime == range`` AND ATR
    below the pair's 20th percentile.
    """
    if not rules:
        return 0.0
    high = rules.get("high_vol") or {}
    high_cutoff = (high.get("atr_cutoff") or {}).get(observation.pair)
    high_regime = observation.regime in (high.get("regimes") or [])
    high_atr = (
        high_cutoff is not None
        and observation.atr_pct is not None
        and observation.atr_pct >= high_cutoff
    )
    if high_regime or high_atr:
        return float(high.get("offset") or 0.0)

    low = rules.get("low_vol") or {}
    low_cutoff = (low.get("atr_cutoff") or {}).get(observation.pair)
    low_regime = observation.regime in (low.get("regimes") or [])
    low_atr = (
        low_cutoff is not None
        and observation.atr_pct is not None
        and observation.atr_pct <= low_cutoff
    )
    if low_regime and low_atr:
        return float(low.get("offset") or 0.0)
    return 0.0


# ------------------------------------------------------------------ evaluation
def effective_threshold(
    observation: Observation, threshold: float, rules: Optional[Dict[str, Any]]
) -> float:
    """Apply the volatility offset to the base threshold for one state."""
    return threshold + vol_offset(observation, rules)


def _bias_conflict(observation: Observation) -> bool:
    """Mirror the post-gate bias conflict check."""
    return (
        (observation.base_action == "BUY" and observation.bias == "short")
        or (observation.base_action == "SELL" and observation.bias == "long")
    )


def evaluate_threshold(
    observations: Sequence[Observation],
    threshold: float,
    rules: Optional[Dict[str, Any]] = None,
    horizon: int = 1,
    cost_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Coverage, trades and realised trade P&L at one base threshold."""
    scored = [observation for observation in observations if observation.forward.get(horizon) is not None]
    covered = [
        observation for observation in scored
        if observation.noul >= effective_threshold(observation, threshold, rules)
    ]
    # Match JevGate.check_pre: only an explicit ``none`` bias blocks; a
    # missing or unparseable bias (None) is allowed and the trade is taken.
    taken = [
        observation for observation in covered
        if observation.is_directional
        and observation.bias != "none"
        and not _bias_conflict(observation)
    ]
    returns = [value for value in (net_return_pct(observation, horizon, cost_pct) for observation in taken)
               if value is not None]
    return {
        "threshold": round(float(threshold), 4),
        "coverage": round(len(covered) / len(scored), 4) if scored else None,
        "trades": len(taken),
        "net_benefit": round(sum(returns), 4) if returns else 0.0,
        "mean_net_pct": round(statistics.fmean(returns), 4) if returns else None,
        "win_rate": (
            round(sum(1 for value in returns if value > 0) / len(returns), 4)
            if returns else None
        ),
        "sample": len(scored),
    }


def threshold_curve(
    observations: Sequence[Observation],
    thresholds: Sequence[float] = THRESHOLD_GRID,
    rules: Optional[Dict[str, Any]] = None,
    horizon: int = 1,
    cost_pct: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Evaluate every candidate threshold."""
    return [
        evaluate_threshold(observations, threshold, rules, horizon, cost_pct)
        for threshold in thresholds
    ]


def select_candidate(
    curve: Sequence[Dict[str, Any]],
    min_coverage: float = MIN_COVERAGE,
    min_trades: int = MIN_TRADES,
) -> Optional[Dict[str, Any]]:
    """Best net-benefit row that meets coverage and trade-count constraints."""
    feasible = [
        row for row in curve
        if row["coverage"] is not None
        and row["coverage"] >= min_coverage
        and row["trades"] >= min_trades
        and row["net_benefit"] > 0
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda row: (row["net_benefit"], row["threshold"]))


# ------------------------------------------------------------------ smoothing
def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into ``[low, high]``."""
    if low > high:
        raise ValueError("low must be <= high")
    return max(low, min(high, value))


def smooth_threshold(
    current: float,
    candidate: float,
    floor: float = THRESHOLD_FLOOR,
    ceiling: float = THRESHOLD_CEILING,
) -> float:
    """Weekly EWMA move, clamped to the allowed threshold band."""
    return round(clamp(0.7 * current + 0.3 * candidate, floor, ceiling), 4)


def limit_delta(current: float, target: float, max_delta: float = MAX_WEEKLY_DELTA) -> float:
    """Cap the weekly move to ``max_delta`` in either direction."""
    if max_delta < 0:
        raise ValueError("max_delta must be >= 0")
    return round(current + clamp(target - current, -max_delta, max_delta), 4)


def relative_improvement(current: float, new: float) -> float:
    """Relative net-benefit improvement; infinite when leaving a non-positive base."""
    if current > 0:
        return (new - current) / current
    if new > 0:
        return math.inf
    return 0.0


def should_promote(
    current_net_benefit: float,
    new_net_benefit: float,
    min_improvement: float = MIN_IMPROVEMENT,
) -> bool:
    """Promote only when out-of-sample net benefit improves enough."""
    return relative_improvement(current_net_benefit, new_net_benefit) >= min_improvement


# -------------------------------------------------------------------- policy
def _parse_policy_time(policy: Dict[str, Any]) -> Optional[datetime]:
    """Read generated_at from a policy; None when absent or malformed."""
    raw = policy.get("generated_at")
    if not raw:
        return None
    try:
        value = parse_iso(str(raw))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def policy_is_stale(
    policy: Optional[Dict[str, Any]],
    now: datetime,
    max_age_days: int = POLICY_MAX_AGE_DAYS,
) -> bool:
    """True when the policy is missing, malformed or older than max_age_days."""
    if not isinstance(policy, dict):
        return True
    generated = _parse_policy_time(policy)
    if generated is None:
        return True
    return now - generated > timedelta(days=max_age_days)


def resolve_threshold(
    policy: Optional[Dict[str, Any]],
    now: datetime,
    fallback: float = FALLBACK_THRESHOLD,
) -> float:
    """Runtime rule: use the policy threshold unless missing or stale."""
    if policy_is_stale(policy, now):
        return fallback
    value = _as_float((policy or {}).get("threshold"))
    if value is None or not 0.0 < value < 1.0:
        return fallback
    return value


def validate_policy(policy: Dict[str, Any]) -> List[str]:
    """Return a list of schema problems; empty means valid."""
    problems: List[str] = []
    required = ("version", "generated_at", "window_days", "min_states", "threshold",
                "offset_rules", "label", "metrics", "fallback", "expire_at")
    for key in required:
        if key not in policy:
            problems.append(f"missing {key}")
    if problems:
        return problems
    if not isinstance(policy["version"], int) or policy["version"] < 1:
        problems.append("version must be a positive int")
    if not isinstance(policy["window_days"], int) or policy["window_days"] < 1:
        problems.append("window_days must be a positive int")
    if not isinstance(policy["min_states"], int) or policy["min_states"] < 1:
        problems.append("min_states must be a positive int")
    threshold = _as_float(policy["threshold"])
    if threshold is None or not 0.0 < threshold < 1.0:
        problems.append("threshold must be in (0, 1)")
    fallback = _as_float(policy["fallback"])
    if fallback is None or not 0.0 < fallback < 1.0:
        problems.append("fallback must be in (0, 1)")
    if not isinstance(policy["offset_rules"], dict):
        problems.append("offset_rules must be an object")
    label = policy["label"]
    if not isinstance(label, dict):
        problems.append("label must be an object")
    else:
        if label.get("type") != "trade_pnl_directional":
            problems.append("label.type must be trade_pnl_directional")
        if not isinstance(label.get("horizon"), int) or label["horizon"] < 1:
            problems.append("label.horizon must be a positive int")
        cost = _as_float(label.get("cost_pct"))
        if cost is None or cost < 0:
            problems.append("label.cost_pct must be >= 0")
    if not isinstance(policy["metrics"], dict):
        problems.append("metrics must be an object")
    for key in ("generated_at", "expire_at"):
        try:
            parse_iso(str(policy[key]))
        except (TypeError, ValueError):
            problems.append(f"{key} must be ISO-8601")
    return problems


def load_policy(path: str) -> Optional[Dict[str, Any]]:
    """Load a policy file; None when absent or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("cannot read policy %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _iso(value: datetime) -> str:
    """Serialize an aware datetime."""
    return iso_z(value.astimezone(timezone.utc))


def _label_block(trade_horizon: int, cost_pct: float, horizons: Sequence[int]) -> Dict[str, Any]:
    """Describe the trade-P&L label stored with every policy."""
    return {
        "type": "trade_pnl_directional",
        "horizon": trade_horizon,
        "horizons_evaluated": list(horizons),
        "cost_pct": round(float(cost_pct), 4),
        "description": (
            "forward return in the base action direction over the horizon, "
            "net of round-trip cost; HOLD states are excluded"
        ),
    }


def _trade_sample(
    observations: Sequence[Observation], horizon: int, cost_pct: float
) -> Dict[str, Any]:
    """Isotonic fit and Brier scores for the trade-P&L label."""
    labels = [net_label(observation, horizon, cost_pct) for observation in observations]
    usable = [
        (observation.noul, bool(label))
        for observation, label in zip(observations, labels)
        if label is not None
    ]
    probabilities = [probability for probability, _ in usable]
    targets = [label for _, label in usable]
    model = fit_isotonic(probabilities, targets)
    calibrated = [isotonic_predict(model, probability) for probability in probabilities]
    return {
        "states_used": len(usable),
        "states_excluded": len(observations) - len(usable),
        "brier": brier_score(calibrated, targets),
        "raw_brier": brier_score(probabilities, targets),
    }


def _by_horizon_metrics(
    observations: Sequence[Observation],
    threshold: float,
    rules: Optional[Dict[str, Any]],
    horizons: Sequence[int],
    cost_pct: float,
) -> Dict[str, Any]:
    """Trade-P&L label count, Brier and net benefit for every horizon."""
    result: Dict[str, Any] = {}
    for horizon in horizons:
        sample = _trade_sample(observations, horizon, cost_pct)
        evaluated = evaluate_threshold(observations, threshold, rules, horizon, cost_pct)
        result[f"h{horizon}"] = {
            "states_used": sample["states_used"],
            "states_excluded": sample["states_excluded"],
            "brier": _round_optional(sample["brier"]),
            "raw_brier": _round_optional(sample["raw_brier"]),
            "coverage": evaluated["coverage"],
            "trades": evaluated["trades"],
            "net_benefit": evaluated["net_benefit"],
            "mean_net_pct": evaluated["mean_net_pct"],
            "win_rate": evaluated["win_rate"],
        }
    return result


def _round_optional(value: Optional[float]) -> Optional[float]:
    """Round an optional float for JSON stability."""
    return None if value is None else round(float(value), 4)


def _fallback_policy(
    previous: Optional[Dict[str, Any]],
    now: datetime,
    train: Sequence[Observation],
    oos: Sequence[Observation],
    reason: str,
    min_states: int,
    window_days: int,
    rules: Dict[str, Any],
    label: Dict[str, Any],
    trade_horizon: int,
    cost_pct: float,
    horizons: Sequence[int],
) -> Dict[str, Any]:
    """Policy used when there are not enough trade-labeled training states."""
    threshold = FALLBACK_THRESHOLD
    offset_rules = rules
    version = 1
    if previous and not policy_is_stale(previous, now):
        threshold = resolve_threshold(previous, now)
        offset_rules = previous.get("offset_rules") or rules
        version = int(previous.get("version") or 1) + 1
    sample = _trade_sample(train, trade_horizon, cost_pct)
    evaluated = evaluate_threshold(train, threshold, offset_rules, trade_horizon, cost_pct)
    oos_evaluated = evaluate_threshold(oos, threshold, offset_rules, trade_horizon, cost_pct)
    curve = threshold_curve(train, THRESHOLD_GRID, offset_rules, trade_horizon, cost_pct)
    return {
        "version": version,
        "generated_at": _iso(now),
        "window_days": window_days,
        "min_states": min_states,
        "threshold": round(threshold, 4),
        "fallback": FALLBACK_THRESHOLD,
        "expire_at": _iso(now + timedelta(days=POLICY_MAX_AGE_DAYS)),
        "offset_rules": offset_rules,
        "label": label,
        "metrics": {
            "brier": _round_optional(sample["brier"]),
            "raw_brier": _round_optional(sample["raw_brier"]),
            "coverage": evaluated["coverage"],
            "net_benefit": evaluated["net_benefit"],
            "mean_net_pct": evaluated["mean_net_pct"],
            "win_rate": evaluated["win_rate"],
            "trades": evaluated["trades"],
            "sample": len(train),
            "window_sample": len(train) + len(oos),
            "oos_sample": oos_evaluated["sample"],
            "oos_coverage": oos_evaluated["coverage"],
            "oos_net_benefit": oos_evaluated["net_benefit"],
            "label_states_used": sample["states_used"],
            "label_states_excluded": sample["states_excluded"],
            "reason": reason,
            "threshold_curve": curve,
            "trade_pnl_by_horizon": _by_horizon_metrics(
                train, threshold, offset_rules, horizons, cost_pct
            ),
        },
        "status": "fallback",
    }


def calibrate_policy(
    observations: Sequence[Observation],
    previous: Optional[Dict[str, Any]],
    now: datetime,
    window_days: int = WINDOW_DAYS,
    min_states: int = MIN_STATES,
    min_coverage: float = MIN_COVERAGE,
    min_trades: int = MIN_TRADES,
    min_improvement: float = MIN_IMPROVEMENT,
    max_delta: float = MAX_WEEKLY_DELTA,
    cost_pct: float = DEFAULT_COST_PCT,
    trade_horizon: int = 1,
    horizons: Sequence[int] = TRADE_HORIZONS,
) -> Dict[str, Any]:
    """Build the weekly policy from the trade-P&L label.

    ``min_states`` applies to trade-labeled states (directional base actions),
    not to the raw observation count: HOLD states carry no trade P&L and are
    excluded.
    """
    train, oos = split_train_oos(observations, now, window_days, OOS_DAYS)
    rules = build_offset_rules(train)
    label = _label_block(trade_horizon, cost_pct, horizons)
    sample = _trade_sample(train, trade_horizon, cost_pct)
    if sample["states_used"] < min_states:
        return _fallback_policy(
            previous, now, train, oos,
            f"insufficient trade-labeled states: {sample['states_used']} < {min_states}",
            min_states, window_days, rules, label, trade_horizon, cost_pct, horizons,
        )
    if previous and policy_is_stale(previous, now):
        logger.info("previous policy is stale; current threshold falls back to %.2f", FALLBACK_THRESHOLD)
        previous = None

    # Every fit below uses the training slice only; ``oos`` stays untouched
    # until the promotion check.
    curve = threshold_curve(train, THRESHOLD_GRID, rules, trade_horizon, cost_pct)
    candidate = select_candidate(curve, min_coverage, min_trades)
    current = resolve_threshold(previous, now) if previous else FALLBACK_THRESHOLD

    oos_current = evaluate_threshold(oos, current, rules, trade_horizon, cost_pct)
    promoted = False
    final = current
    if candidate is not None:
        smoothed = smooth_threshold(current, candidate["threshold"])
        proposed = limit_delta(current, smoothed, max_delta)
        oos_proposed = evaluate_threshold(oos, proposed, rules, trade_horizon, cost_pct)
        if should_promote(oos_current["net_benefit"], oos_proposed["net_benefit"], min_improvement):
            final = proposed
            promoted = True
        logger.info(
            "candidate=%.2f smoothed=%.4f proposed=%.4f final=%.4f oos_nb=%.2f->%.2f",
            candidate["threshold"], smoothed, proposed, final,
            oos_current["net_benefit"], oos_proposed["net_benefit"],
        )
    else:
        logger.info("no feasible candidate; keeping current threshold %.2f", current)

    metrics = evaluate_threshold(train, final, rules, trade_horizon, cost_pct)
    oos_final = evaluate_threshold(oos, final, rules, trade_horizon, cost_pct)
    version = int((previous or {}).get("version") or 0) + 1
    policy = {
        "version": version,
        "generated_at": _iso(now),
        "window_days": window_days,
        "min_states": min_states,
        "threshold": round(final, 4),
        "fallback": FALLBACK_THRESHOLD,
        "expire_at": _iso(now + timedelta(days=POLICY_MAX_AGE_DAYS)),
        "offset_rules": rules,
        "label": label,
        "metrics": {
            "brier": _round_optional(sample["brier"]),
            "raw_brier": _round_optional(sample["raw_brier"]),
            "coverage": metrics["coverage"],
            "net_benefit": metrics["net_benefit"],
            "mean_net_pct": metrics["mean_net_pct"],
            "win_rate": metrics["win_rate"],
            "trades": metrics["trades"],
            "sample": metrics["sample"],
            "window_sample": metrics["sample"] + oos_current["sample"],
            "label_states_used": sample["states_used"],
            "label_states_excluded": sample["states_excluded"],
            "candidate_threshold": candidate["threshold"] if candidate else None,
            "promoted": promoted,
            "oos_net_benefit": oos_final["net_benefit"],
            "oos_net_benefit_current": oos_current["net_benefit"],
            "oos_sample": oos_current["sample"],
            "oos_coverage": oos_final["coverage"],
            "threshold_curve": curve,
            "trade_pnl_by_horizon": _by_horizon_metrics(
                train, final, rules, horizons, cost_pct
            ),
        },
        "status": "active",
    }
    problems = validate_policy(policy)
    if problems:
        raise ValueError(f"generated policy is invalid: {problems}")
    return policy


def write_policy(policy: Dict[str, Any], path: str) -> None:
    """Write the policy JSON atomically enough for a local runtime file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2)
        handle.write("\n")


# ------------------------------------------------------------------------ CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weekly JEV tradeability policy calibration")
    parser.add_argument(
        "--observations",
        default=os.path.join(_REPO_ROOT, "runtime", "jev_bench", "observations.jsonl"),
    )
    parser.add_argument(
        "--policy",
        default=os.path.join(_REPO_ROOT, "runtime", "jev_calibration", "policy.json"),
    )
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--min-states", type=int, default=MIN_STATES)
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    parser.add_argument("--min-improvement", type=float, default=MIN_IMPROVEMENT)
    parser.add_argument("--max-delta", type=float, default=MAX_WEEKLY_DELTA)
    parser.add_argument(
        "--cost-pct", type=float, default=DEFAULT_COST_PCT,
        help="Round-trip trading cost in percent applied to the trade-P&L label",
    )
    parser.add_argument(
        "--trade-horizon", type=int, default=1,
        help="Horizon (candles) used for the policy label and net benefit",
    )
    parser.add_argument(
        "--now",
        default=None,
        help="Reference time (ISO-8601) for reproducible weekly runs; defaults to now UTC",
    )
    return parser


def _print_summary(policy: Dict[str, Any], path: str) -> None:
    metrics = policy["metrics"]
    print(f"policy: {path}")
    print(f"status={policy['status']} version={policy['version']} threshold={policy['threshold']} "
          f"expire_at={policy['expire_at']}")
    label = policy.get("label") or {}
    print(f"label={label.get('type')} horizon={label.get('horizon')} "
          f"cost_pct={label.get('cost_pct')} "
          f"states_used={metrics.get('label_states_used')} "
          f"states_excluded={metrics.get('label_states_excluded')}")
    print(f"sample={metrics.get('sample')} coverage={metrics.get('coverage')} "
          f"trades={metrics.get('trades')} net_benefit={metrics.get('net_benefit')} "
          f"brier={metrics.get('brier')}")
    print(f"candidate={metrics.get('candidate_threshold')} promoted={metrics.get('promoted')} "
          f"oos_net_benefit={metrics.get('oos_net_benefit')} "
          f"oos_current={metrics.get('oos_net_benefit_current')}")
    for name, rule in (policy.get("offset_rules") or {}).items():
        print(f"offset {name}: {rule.get('offset')} regimes={rule.get('regimes')} "
              f"atr_cutoff={rule.get('atr_cutoff')}")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: read observations, write the weekly policy."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)
    now = parse_iso(args.now) if args.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    observations = load_observations(args.observations)
    previous = load_policy(args.policy)
    policy = calibrate_policy(
        observations,
        previous,
        now,
        window_days=args.window_days,
        min_states=args.min_states,
        min_coverage=args.min_coverage,
        min_trades=args.min_trades,
        min_improvement=args.min_improvement,
        max_delta=args.max_delta,
        cost_pct=args.cost_pct,
        trade_horizon=args.trade_horizon,
    )
    problems = validate_policy(policy)
    if problems:
        raise SystemExit(f"invalid policy: {problems}")
    write_policy(policy, args.policy)
    _print_summary(policy, args.policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
