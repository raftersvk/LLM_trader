# JEV weekly tradeability policy

A conservative weekly layer on top of the raw `noul` threshold. It never
changes the signal service by itself: the service still uses 0.5 in this PR.
This document records the observation format, the trade-P&L label, the
calibration rules and the planned (not yet wired) integration.

## Files

| Path | Tracked | Purpose |
| --- | --- | --- |
| `runtime/jev_bench/observations.jsonl` | no (gitignored) | Merged JSONL log, one line per calibrated state |
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

Each calibration run merges into the existing log: new records are appended and
a duplicate `(pair, candle_open_time)` replaces the older line in place. An
aborted run never replaces a non-empty log with its partial states. Running the
harness per symbol therefore accumulates a cross-pair history; no manual
concatenation is required.

A missing observation log is treated as zero states: the first weekly run (or
one after cleanup) still writes the 0.5 fallback policy instead of failing.

`cost_pct` defaults to 0.10 % for a fresh calibration run and is stored in the
report dataset. A flag-free `--aggregate-from` carries it over (including an
explicit zero); an explicit `--cost-pct` overrides it.

## Trade-P&L label

The policy uses the label that matches the actual trade, not the absolute
move label from the calibration report:

```text
BUY  label(h) = forward_h * 100 - cost_pct > 0
SELL label(h) = -forward_h * 100 - cost_pct > 0
HOLD label    = none (excluded)
```

Horizons 1-4 are all evaluated and stored in
`metrics.trade_pnl_by_horizon`. The policy threshold and net benefit use
`--trade-horizon` (default 1) and `--cost-pct` (default 0.10). Net benefit is
the sum of the directional net P&L in percent for the trades that survive the
gate.

The absolute `abs(forward_h1) >= 1 %` label remains the **secondary** label in
`docs/jev-calibration.md`; it is not used to build the policy. That document no
longer treats Brier alone as a headline: the label is rare (1-4 %), so Brier
must be read with Brier skill scores and ranking metrics, not against a
constant predictor alone.

`--min-states` (default 200) applies to trade-labeled states, not raw
observations. HOLD states have no trade direction and are excluded; the policy
metrics report both `label_states_used` and `label_states_excluded`.

## Weekly rules

1. Keep observations inside `[now - 60 days, now]`. Anything after `now` is
   excluded, not just anything before the cutoff.
2. Hold out the last 7 days as out-of-sample (OOS). The earlier part of the
   window is the training set. ATR percentiles, the isotonic model and the
   candidate threshold are fitted **only** on training data.
3. Require at least 200 trade-labeled training states (`--min-states`).
   Otherwise preserve the previous policy, or fall back to 0.5 when there is
   none.
4. Fit isotonic regression `noul -> P(trade is profitable net of cost)` with
   pool-adjacent-violators on the selected horizon.
5. Evaluate training thresholds 0.30-0.70 by 0.05 with the volatility offset
   applied per state. A candidate must have coverage >= 40 %, >= 20 trades and
   a positive net benefit.
6. Smooth: `clamp(0.7 * current + 0.3 * candidate, 0.35, 0.65)`.
7. Cap the weekly move at 0.05.
8. Promote only if the proposed threshold improves held-out OOS net benefit
   (last 7 days) by at least 2 % relative.
9. A policy older than 14 days resolves to 0.5.

Volatility offsets:

- `+0.05` when `regime == high_vol` **or** `atr_pct >= pair 80th percentile`;
- `-0.05` when `regime == range` **and** `atr_pct <= pair 20th percentile`.

Per-pair ATR cutoffs are stored in the policy so the runtime does not need to
recompute percentiles. They are computed from the training slice only, so the
held-out week cannot leak into the rules.

## Initial policy (2026-10-24)

Command:

```bash
python scripts/jev_calibrate.py \
  --observations runtime/jev_bench/observations.jsonl \
  --policy runtime/jev_calibration/policy.json \
  --now 2026-10-24T20:00:00Z \
  --cost-pct 0.1 --trade-horizon 1
```

**Status: fallback, threshold 0.5.** The 60-day window holds **400
observations**, split into **118 training** states (older than 7 days) and
**282 held-out OOS** states (last 7 days). The training slice contains only
**21 trade-labeled states** (directional base action); the other 97 are HOLD.
21 < 200, so the data cannot support a trade-P&L threshold yet and the policy
correctly keeps 0.5. The OOS slice is not used for any fit.

```json
{
  "version": 1,
  "generated_at": "2026-10-24T20:00:00Z",
  "window_days": 60,
  "min_states": 200,
  "threshold": 0.5,
  "fallback": 0.5,
  "expire_at": "2026-11-07T20:00:00Z",
  "label": {
    "type": "trade_pnl_directional",
    "horizon": 1,
    "horizons_evaluated": [1, 2, 3, 4],
    "cost_pct": 0.1
  },
  "offset_rules": {
    "high_vol": {"offset": 0.05, "regimes": ["high_vol"],
                 "atr_cutoff": {"BTC/USDT": 0.007417, "ETH/USDT": 0.010305}},
    "low_vol":  {"offset": -0.05, "regimes": ["range"],
                 "atr_cutoff": {"BTC/USDT": 0.002768, "ETH/USDT": 0.004118}}
  },
  "metrics": {
    "brier": 0.2101,
    "raw_brier": 0.2818,
    "coverage": 0.2627,
    "net_benefit": -0.7721,
    "trades": 10,
    "sample": 118,
    "window_sample": 400,
    "oos_sample": 282,
    "oos_net_benefit": -1.2106,
    "label_states_used": 21,
    "label_states_excluded": 97,
    "reason": "insufficient trade-labeled states: 21 < 200"
  },
  "status": "fallback"
}
```

Diagnostic per-horizon trade P&L on the **training slice** (10 surviving
trades):

| Horizon | States used | Brier | Net benefit | Mean net % | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| h1 | 21 | 0.2101 | -0.77 | -0.077 | 0.400 |
| h2 | 21 | 0.1508 | -0.55 | -0.055 | 0.600 |
| h3 | 21 | 0.1508 | +1.06 | +0.106 | 0.600 |
| h4 | 21 | 0.1440 | +0.99 | +0.099 | 0.600 |

Even if the 200-state minimum were waived, no training threshold with coverage
>= 40 % has a positive trade-P&L net benefit: at 0.30-0.40 the net benefit is
-3.87 to -0.97. The positive points (0.60-0.70: +1.42/+1.46) sit at 12-19 %
coverage. At the retained 0.5 threshold the held-out OOS net benefit is -1.21
over 282 states.

## Calibration evidence (2026-10-24 audit)

The independent audit of the committed calibration reports changes how the
policy layer must be interpreted. The evidence, reproduced from
`runtime/jev_bench/calibration_{btc,eth}_200.json`:

- **Class imbalance:** the h1 worth-trading label has BTC 2/200 (1.0 %) and ETH
  8/200 (4.0 %) positives. A constant base-rate predictor scores Brier 0.0099
  (BTC) / 0.0384 (ETH), which is trivially strong at that prevalence; raw noul
  Brier is 0.2504 / 0.2271. These two numbers are **not** comparable as a
  head-to-head model verdict.
- **Ranking, not calibration:** noul ranks worth-trading h1 states with AUC
  0.854 (BTC) / 0.652 (ETH), but Brier skill against both the constant and a
  logistic baseline is strongly negative. noul is a ranking signal, not a
  calibrated probability.
- **No directional edge:** directional AUC (noul predicting
  `base_direction_correct`) is 0.526 (BTC) / 0.545 (ETH) on h1, with bootstrap
  CIs spanning 0.5. The policy must not treat noul as a direction predictor.
- **No demonstrated edge per symbol:** over all labeled states at
  `noul >= 0.5`, BTC has 18 trades, net **-2.87 %** at 0.10 % cost and is
  negative at 0.0/0.5/0.65/0.8 thresholds and at 0.05/0.10/0.20 % costs. ETH
  has 29 trades, **+1.41 %** at 0.10 % but **-1.49 %** at 0.20 %, a
  cost-sensitive sign flip. Per the reworked report, both are "no demonstrated
  edge".
- **Per-symbol OOS only:** the calibration report now evaluates economics
  separately per symbol on the last 30 % of scored states. It never pools
  symbols and never uses the in-sample slice to claim an edge.

Consequences for this policy layer:

1. Keep the **0.5 fallback** and do not change the signal service in this
   rework. No gate or threshold behavior changes.
2. Do not promote a weekly threshold on the strength of in-sample precision or
   a single cost assumption. Promotion must survive the existing held-out OOS
   improvement rule, and a cost-sensitive sign flip is a rejection. The
   calibration report only calls a positive result an edge when every tested
   cost bucket's bootstrap net CI excludes zero; a positive point estimate with
   a CI spanning zero is reported as "no demonstrated edge (CI includes zero)".
3. Do not read `metrics.brier` / `metrics.raw_brier` in a policy as evidence of
   skill by themselves; the policy label is the directional trade-P&L label,
   but the sample is still small and the calibration doc's class-imbalance
   caveat applies.
4. A real edge would require a larger multi-regime sample, positive net OOS at
   the highest tested cost (0.20 %), a bootstrap net CI excluding zero with a
   stable sign across costs, and directional AUC with a CI above 0.5.

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
