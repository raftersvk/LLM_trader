# JEV weekly tradeability policy

A conservative weekly layer on top of the raw `noul` threshold. It never
changes the signal service by itself: the service still uses 0.5 in this PR.
This document records the observation format, the calibration rules and the
planned (not yet wired) integration.

## Files

| Path | Tracked | Purpose |
| --- | --- | --- |
| `runtime/jev_bench/observations.jsonl` | no (gitignored) | One JSON line per calibrated state |
| `runtime/jev_calibration/policy.json` | no (gitignored) | Current weekly policy |
| `scripts/jev_calibration.py` | yes | Produces observations (`--observations`) |
| `scripts/jev_calibrate.py` | yes | Reads observations, writes the policy |

Both runtime paths are configurable through `--observations` / `--policy`.

## Observation schema

```json
{
  "candle_open_time": "2026-10-15T09:00:00Z",
  "pair": "BTC/USDT",
  "noul": 0.1702,
  "bias": "none",
  "regime": "range",
  "atr_pct": 0.0075,
  "atr_percentile": 94.47,
  "base_action": "HOLD",
  "cost_pct": 0.1,
  "forward_1h": -0.0027,
  "forward_2h": -0.0035,
  "forward_3h": -0.0088,
  "forward_4h": -0.013
}
```

Forward returns are fractions; `cost_pct` is a round-trip percentage (default
0.10 %). Observations are written only for states with a real `noul` answer.

## Weekly rules

1. Keep the last 60 days of observations (`--window-days`).
2. Require at least 200 states (`--min-states`). Otherwise preserve the
   previous policy, or fall back to 0.5 when there is none.
3. Fit isotonic regression `noul -> P(base trade is profitable after costs)`
   with pool-adjacent-violators. The label is the base action's direction
   (`BUY`: `forward_1h`, `SELL`: `-forward_1h`) minus `cost_pct`; HOLD states
   have no label.
4. Evaluate thresholds 0.30-0.70 by 0.05 with the volatility offset applied
   per state. A candidate must have coverage >= 40 %, >= 20 trades and a
   positive net benefit.
5. Smooth: `clamp(0.7 * current + 0.3 * candidate, 0.35, 0.65)`.
6. Cap the weekly move at 0.05.
7. Promote only if the proposed threshold improves out-of-sample net benefit
   (last 7 days) by at least 2 % relative.
8. A policy older than 14 days resolves to 0.5.

Volatility offsets:

- `+0.05` when `regime == high_vol` **or** `atr_pct >= pair 80th percentile`;
- `-0.05` when `regime == range` **and** `atr_pct <= pair 20th percentile`.

Per-pair ATR cutoffs are stored in the policy so the runtime does not need to
recompute percentiles.

## Initial policy (2026-10-24, 400 states)

Produced from the 200 BTC + 200 ETH calibration states. Command:

```bash
python scripts/jev_calibrate.py \
  --observations runtime/jev_bench/observations.jsonl \
  --policy runtime/jev_calibration/policy.json \
  --now 2026-10-24T20:00:00Z
```

```json
{
  "version": 1,
  "generated_at": "2026-10-24T20:00:00Z",
  "window_days": 60,
  "min_states": 200,
  "threshold": 0.5,
  "fallback": 0.5,
  "expire_at": "2026-11-07T20:00:00Z",
  "offset_rules": {
    "high_vol": {"offset": 0.05, "regimes": ["high_vol"],
                 "atr_cutoff": {"BTC/USDT": 0.0056, "ETH/USDT": 0.00842}},
    "low_vol":  {"offset": -0.05, "regimes": ["range"],
                 "atr_cutoff": {"BTC/USDT": 0.00358, "ETH/USDT": 0.0059}}
  },
  "metrics": {
    "brier": 0.2391,
    "raw_brier": 0.3033,
    "coverage": 0.33,
    "net_benefit": -1.89,
    "trades": 41,
    "sample": 400,
    "candidate_threshold": null,
    "promoted": false
  },
  "status": "active"
}
```

At the initial window, every threshold with coverage >= 40 % had a negative
net benefit (0.30: -6.07, 0.35: -3.01, 0.40: -2.90). The only positive points
(0.65 / 0.70: +0.35) sat at ~20 % coverage. No feasible candidate existed, so
the policy keeps 0.5.

## Planned signal-service integration (not in this PR)

Nothing in `src/signal_service` reads the policy yet. The intended integration:

1. Add a config/env path (`JEV_POLICY_PATH`, default
   `runtime/jev_calibration/policy.json`).
2. At gate construction (or on a periodic reload), load the policy and expose
   `threshold` plus the offset rules.
3. Per state, compute the effective threshold as
   `policy.threshold + vol_offset(state.atr_pct, state.regime)` and compare
   `noul` against it.
4. If the file is missing, malformed or older than 14 days, use 0.5.
5. Never restart a running service for a policy update; reload on the next
   analysis cycle.

`scripts/jev_calibrate.py` is safe to run from cron weekly: it only writes the
gitignored policy file. Promotion requires a fresh observation log and the
out-of-sample improvement rule above.
