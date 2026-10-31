# Regime snapshot 1h (P0-A / P0-B)

Deterministic, token-free regime input for the 1h signal service. It derives
everything from data the analyzer already collected; it adds no data layer, no
Open Interest and no external call.

## Scope

- `src/signal_service/regime_snapshot.py` returns `regime`, `confidence`,
  `no_trade`, `htf_alignment`, `extension_atr`, `rs_btc_7d/30d` (when a BTC
  close series is already exposed) and the allowed setup families.
- The prompt receives one compact `REGIME SNAPSHOT` block.
- JEV receives the same fields inside its state. The JEV payload and the
  signal contract are unchanged.

## Ex-ante rule

The snapshot only sees candles closed at or before the analysed candle close.
The adapter truncates the 1h series before building it.

The report-33 figure `aligned 20 / +2.91 vs misaligned 16 / -8.60` is **not**
used: it labels a side against the pair's full-week return, which is a
look-ahead. P0-B recomputed it ex-ante and the edge disappeared. HTF alignment
is computed only from the daily/weekly blocks available before the decision.

## 1h regime priority (P0-B)

1. `high_vol`: ATR% > max(3 %, 2.5 x median 90 bars) and ADX < 25;
2. `trend_up`: ADX >= 25, +DI > -DI, close > SMA20;
3. `trend_down`: ADX >= 25, -DI > +DI, close < SMA20;
4. `low_vol`: ATR% < 0.55 x median 90 bars and choppiness > 55;
5. `range`: choppiness > 61.8 or ADX < 20;
6. `unclear`: everything else.

`unclear`, `low_vol` and `high_vol` are marked `tradeable=no` in the prompt
block and expose no allowed setup family. Enforcing a hard gate on that flag is
left to the gate policy, not to the snapshot.

## JEV gate (P0-B)

- The long noul threshold stays **0.50** everywhere. In particular, **range is
  not lowered**: the P0-B arm degraded from -11.01 R to -15.73 R.
- A short in a deterministic `trend_down` is admitted from **noul >= 0.10**.
  In `pre_post` mode the pre-gate lets the state through at 0.10 and the
  post-gate applies 0.10 to SELL and keeps 0.50 to BUY. In **pre-only** mode
  there is no post-gate to re-apply 0.50, so the pre-gate keeps 0.50 for every
  side.
- A bias conflict vetoes only when the regime is `unclear` or the setup is
  `momentum_*`, symmetrically for BUY/short and SELL/long.
- The setup taxonomy and contract fields remain out of scope: P1-B only,
  optional fields.

## P1-A price-preserving exits (not implemented here)

The 1h ATR is about 0.52x the 4h ATR at entry. Copying 4h multiples halves the
price distance and stopped the ARB shorts before their TP. P1-A must preserve
price distance:

| Setup | SL | TP | TTL |
| --- | --- | --- | --- |
| trend / rally_failure / momentum_short | 2.5 ATR 1h | 4.5-5.0 ATR 1h | 24 bars |
| momentum_* | 2.5 ATR 1h | 4.0 ATR 1h | 12 bars |
| range_* | 2.0 ATR 1h | opposite edge / mid | 8-12 bars |

No trail/BE by default: the P0-B sample showed the trail cutting winners before
TP.

## Go / no-go

No positive configuration outside BEAMX on the 7-day sample: **NO-GO for live
or 1h production**, and **OOS 30 days is required** before any performance
claim. This module is a reproducible risk input, not a proven edge.
