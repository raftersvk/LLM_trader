# Regime snapshot 1h (P0-A / P0-B)

Deterministic, token-free regime input for the 1h signal service. It derives
everything from data the analyzer already collected; it adds no data layer, no
Open Interest and no external call.

## Scope

- `src/signal_service/regime_snapshot.py` returns `regime`, `confidence`,
  `no_trade`, `htf_alignment`, `extension_atr`, `rs_btc_7d/30d` (when a BTC
  close series is already exposed) and the allowed setup families.
- The prompt receives one compact `REGIME SNAPSHOT` block.
- JEV receives the same fields inside its state. P0-A left the JEV payload and
  the signal contract unchanged; P1-B adds them as optional v1.2 fields (below).

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

## JEV gate (post-OOS: bias only, asymmetric short)

- The gate filters on **JEV bias only**. A **SELL** is admitted only when
  `jev.bias == "short"`; a **BUY is never filtered by the JEV bias**.
- `noul` is **logged only** (and kept in the advisory payload). There is no
  noul threshold, no `trend_down` short exemption and no regime gate: those
  three P0-B rules are retired.
- A `none` or `long` bias vetoes a short. A missing/unparseable bias fails open
  instead of blocking.
- The pre-gate has no side yet, so it only consults and logs. In `pre` mode the
  short-only rule is applied by the adapter risk layer once the LLM decision is
  known; in `post`/`pre_post` the post-gate applies it.
- The OOS decision behind this policy: short + short bias over 28 trades was
  +14.05 R at 64.3 % WR (`rally_failure` 22 trades, +12.91 R at 68 % WR);
  long + bias was -9.14 R and unstable; `noul >= 0.50` degraded the arm and the
  global bias rule was a NO-GO. Only short-only survives as research.
- No setup is wired into the JEV state in P0: `_jev_state` only carries the
  regime fields, and the gate does not read `setup`. The former `momentum_*`
  bias-conflict branch was removed together with the regime gate; the setup
  taxonomy and contract fields remain P1-B only, optional fields.

## P1-B contract and taxonomy (v1.2)

P1-B makes the taxonomy explicit end to end, additively:

- `signal.schema.json` accepts `schema_version` `1.2` and optional top-level
  `setup`, `regime`, `htf_alignment`. `jev` optionally accepts
  `regime_confidence`, `bias_confidence`, `tradeability_probability`,
  `confidence` (all in `[0, 1]`).
- `contract.py` validates the same enums. `1.0` and `1.1` records without the
  new keys remain valid; a producer that emits no taxonomy still writes `1.0`.
- The decision JSON asks for `setup`, `regime`, `htf_alignment` only when the
  prompt target is 1h; the shared 4h/live template is unchanged. The adapter
  emits `1.2` only when the model actually emitted `setup` or `regime`; a
  deterministic snapshot alone never upgrades a production signal. It prefers
  the snapshot's `regime`/`htf_alignment` (even when ADX is missing) and drops a
  `setup` outside that regime's allowed families, so the model cannot relabel
  the computed regime.
- The Decision Rules system prompt lists all nine setups, the six regimes and
  the three alignments, with the regime -> allowed-family mapping, and only for
  a 1h target. It states that the supplied `REGIME SNAPSHOT` is authoritative,
  that indicators must not be recomputed or refetched, and that only the latest
  closed 1h candle may be used (no future candle, no retrospective week return).

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
claim. The post-OOS result above only supports **short-only research**; the
global bias rule remains a NO-GO. This module is a reproducible risk input, not
a proven edge.
