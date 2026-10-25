# JEV spot and USDT-M futures calibration campaign

**Question:** can the JEV "tradeability" score (`noul`) support a reliable
*rejection* threshold — a rule that skips the lowest-scoring states and keeps
the rest — and does the answer change on perpetual futures?

**Short answer: not today, in either market, and futures does not change the
conclusion.** Across 7 spot markets and 6 USDT-M perpetual futures markets
(2,600 real JEV calls), no rank-based rejection rule survives its own
confidence interval *and* the multiple-comparison burden. Every spot-vs-futures
metric difference is within noise. One spot symbol (NEXO, no perp available)
has a nominally significant 30 % cut on six events; futures has no nominal
rejection hit at all. Production stays at the **0.5** fallback and no service
code is changed.

- **Spot:** 7 symbols, 700 aligned 1h Binance spot candles each,
  `2026-09-26T02:00Z` → `2026-10-25T05:00Z`.
- **USDT-M futures:** the 6 symbols that have a Binance USDT-M perpetual
  (BTC, ETH, SOL, AXS, AVAX, ARB), same 700 candles over the **same calendar
  window**, from `binanceusdm` contracts such as `SOL/USDT:USDT`.
- **NEXO is skipped on futures:** Binance has no `NEXO/USDT:USDT` perpetual.
  Its spot result is still reported.
- **Scored states:** 200 per symbol-market, spanning
  `2026-10-16T18:00Z` → `2026-10-25T01:00Z` (≈8.3 days).
- **Event:** `abs(forward 1h return) >= 1.0 %`.
- **Provider:** Decision 3.0 `vllm-sr/d3-flash`, one call per state, 0.5 s
  pacing.
- Every report records `dataset.market_type = "spot"` or `"futures"`.
- Machine inputs: `runtime/jev_bench/<BASE>_USDT_1h.csv`,
  `runtime/jev_bench/<BASE>_USDT_1h_futures.csv`,
  `runtime/jev_bench/calibration_<base>_200.json`,
  `runtime/jev_bench/calibration_<base>_200_futures.json`.
  Aggregated metrics: `runtime/jev_bench/campaign_summary.json`.

## Definitions (plain language)

- **Event / worth trading:** the price moves at least 1 % (up or down) over the
  next hour. The label does not care about direction.
- **JEV score (`noul`):** a 0–1 number from JEV. It is treated here as a
  *ranking* score, not as the probability of an event.
- **Top / bottom group:** the 30 % of states with the highest scores and the
  30 % with the lowest scores.
- **AUC (ranking quality):** the chance that a randomly chosen event state
  scores higher than a randomly chosen non-event state. 0.5 is a coin flip,
  1.0 is a perfect ranking. An AUC is "proven at 95 %" only when its 95 %
  confidence interval excludes 0.5.
- **95 % confidence interval (CI):** the range of values compatible with the
  data under repeated sampling. For a difference, a CI that excludes zero means
  the difference is larger than the noise the sample can explain.
- **Rank rejection:** throw away the lowest-scoring X % of states and only
  consider the kept states.

## Calls and failures

Every symbol-market ran exactly 200 states, sequentially, one pass, no retry
loop. There were **no 429 rate-limit responses and no failures**: 2,600/2,600
scored.

| Symbol | Spot calls / scored / failed | Futures calls / scored / failed |
| --- | ---: | ---: |
| BTC/USDT | 200 / 200 / 0 | 200 / 200 / 0 |
| ETH/USDT | 200 / 200 / 0 | 200 / 200 / 0 |
| SOL/USDT | 200 / 200 / 0 | 200 / 200 / 0 |
| ARB/USDT | 200 / 200 / 0 | 200 / 200 / 0 |
| AXS/USDT | 200 / 200 / 0 | 200 / 200 / 0 |
| AVAX/USDT | 200 / 200 / 0 | 200 / 200 / 0 |
| NEXO/USDT | 200 / 200 / 0 | no market (skipped) |
| **Total** | **1,400 / 1,400 / 0** | **1,200 / 1,200 / 0** |

Latency stayed in the same band as the spot run (p50 156–160 ms; p95 162–401
ms; futures ETH had the widest tail). A separate one-call credential check was
made before the first campaign; the 2,600 calls above are the measured work.

## 1. How often is a state worth trading?

| Symbol | Spot event rate (95 % CI) | Futures event rate (95 % CI) |
| --- | ---: | ---: |
| ARB/USDT | 28.5 % (22.7–35.1 %) | 28.0 % (22.2–34.6 %) |
| AVAX/USDT | 9.5 % (6.2–14.4 %) | 9.5 % (6.2–14.4 %) |
| AXS/USDT | 8.5 % (5.4–13.2 %) | 8.0 % (5.0–12.6 %) |
| BTC/USDT | 0.5 % (0.1–2.8 %) | 0.5 % (0.1–2.8 %) |
| ETH/USDT | 4.0 % (2.0–7.7 %) | 4.0 % (2.0–7.7 %) |
| NEXO/USDT | 3.0 % (1.4–6.4 %) | no perp |
| SOL/USDT | 3.5 % (1.7–7.1 %) | 4.0 % (2.0–7.7 %) |

The event rate is essentially identical on spot and futures: the largest
difference is 0.5 percentage points (ARB, AXS, SOL), and every difference CI
includes zero. That is expected because the perpetual price tracks the spot
price closely over one hour.

## 2. Does a high score identify those states?

"Top" and "bottom" are the highest and lowest 30 % of scores (60 states each).
The difference is top minus bottom, with a bootstrap 95 % CI that re-draws the
rank cut on every resample.

### Spot

| Symbol | Top 30 % | Bottom 30 % | Difference (95 % CI) | AUC (95 % CI) | Proven at 95 %? |
| --- | ---: | ---: | ---: | ---: | --- |
| ARB/USDT | 31.7 % | 23.3 % | +8.3 pp (−8.3 to +25.0) | 0.554 (0.459–0.644) | No |
| AVAX/USDT | 11.7 % | 10.0 % | +1.7 pp (−10.0 to +11.7) | 0.510 (0.359–0.665) | No |
| AXS/USDT | 13.3 % | 5.0 % | +8.3 pp (−1.7 to +18.3) | 0.608 (0.469–0.749) | No |
| BTC/USDT | 1.7 % | 0.0 % | +1.7 pp (0.0 to +5.0) | 0.975 (0.950–0.995) | Formally yes, but see below |
| ETH/USDT | 6.7 % | 3.3 % | +3.3 pp (−5.0 to +11.7) | 0.547 (0.270–0.814) | No |
| NEXO/USDT | 3.3 % | 0.0 % | +3.3 pp (0.0 to +10.0) | 0.656 (0.419–0.914) | No |
| SOL/USDT | 5.0 % | 1.7 % | +3.3 pp (−3.3 to +10.0) | 0.587 (0.317–0.865) | No |

### USDT-M perpetual futures

| Symbol | Top 30 % | Bottom 30 % | Difference (95 % CI) | AUC (95 % CI) | Proven at 95 %? |
| --- | ---: | ---: | ---: | ---: | --- |
| ARB/USDT | 36.7 % | 20.0 % | +16.7 pp (−1.7 to +30.0) | 0.583 (0.486–0.672) | No |
| AVAX/USDT | 10.0 % | 10.0 % | 0.0 pp (−10.0 to +11.7) | 0.520 (0.363–0.675) | No |
| AXS/USDT | 11.7 % | 5.0 % | +6.7 pp (−5.0 to +15.0) | 0.595 (0.446–0.741) | No |
| BTC/USDT | 1.7 % | 0.0 % | +1.7 pp (0.0 to +5.0) | 0.990 (0.975–1.000) | Formally yes, but see below |
| ETH/USDT | 6.7 % | 3.3 % | +3.3 pp (−5.0 to +11.7) | 0.553 (0.269–0.824) | No |
| SOL/USDT | 6.7 % | 3.3 % | +3.3 pp (−6.7 to +10.0) | 0.545 (0.293–0.799) | No |

**BTC is the trap in both markets.** Its AUC CI excludes 0.5, so it is
formally "proven" at 95 %, but there is exactly **one** worth-trading hour in
each 200-state sample. The AUC is then just the rank of that single event; it
does not establish a repeatable rule. In practical terms BTC's ranking is
**not** established on either market. Every other symbol-market AUC CI
includes 0.5.

No top-vs-bottom difference has a CI strictly above zero in either market:
BTC and NEXO (spot) reach a lower bound that rounds to 0.0, and futures ARB
reaches −1.7 pp.

## 3. Does the score say anything about direction (up vs down)?

The score is used to rank *up hours* (`forward return > 0`) against *down
hours*. AUC 0.5 means no directional information.

| Symbol | Spot direction AUC (95 % CI) | Futures direction AUC (95 % CI) |
| --- | ---: | ---: |
| ARB/USDT | 0.540 (0.461–0.617) | 0.547 (0.469–0.623) |
| AVAX/USDT | 0.514 (0.432–0.595) | 0.525 (0.445–0.607) |
| AXS/USDT | 0.512 (0.431–0.592) | 0.527 (0.448–0.605) |
| BTC/USDT | 0.514 (0.433–0.593) | 0.535 (0.454–0.616) |
| ETH/USDT | 0.486 (0.405–0.567) | 0.481 (0.402–0.562) |
| NEXO/USDT | 0.455 (0.374–0.537) | no perp |
| SOL/USDT | 0.512 (0.431–0.590) | 0.507 (0.426–0.585) |

**The score says nothing about direction in either market.** Every CI includes
0.5. NEXO spot is the lowest point estimate (0.455, higher scores slightly more
often before a down hour), but its CI includes 0.5, so even that is not proven.

## 4. Rejection by rank (the practical proposal)

Rule: skip the lowest 30 %, 40 % or 50 % of scores and keep the rest. Positive
means the kept states had a higher event rate than the skipped ones.

### Spot (30 % cut shown in detail)

| Symbol | Skipped | Kept rate | Rejected rate | Diff @30 % (95 % CI) | Diff @40 % | Diff @50 % | Beats noise? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ARB/USDT | 60 | 30.7 % | 23.3 % | +7.4 pp (−8.3 to +18.8) | +1.7 pp | +5.0 pp | No |
| AVAX/USDT | 60 | 9.3 % | 10.0 % | −0.7 pp (−10.7 to +7.4) | −0.8 pp | +1.0 pp | No |
| AXS/USDT | 60 | 10.0 % | 5.0 % | +5.0 pp (−1.7 to +12.9) | +3.8 pp | +3.0 pp | No |
| BTC/USDT | 60 | 0.7 % | 0.0 % | +0.7 pp (0.0 to +2.1) | +0.8 pp | +1.0 pp | No |
| ETH/USDT | 60 | 4.3 % | 3.3 % | +0.9 pp (−5.2 to +6.2) | −1.7 pp | 0.0 pp | No |
| NEXO/USDT | 60 | 4.3 % | 0.0 % | +4.3 pp (+0.2 to +7.9) | +2.9 pp | 0.0 pp | 30 % cut only |
| SOL/USDT | 60 | 4.3 % | 1.7 % | +2.6 pp (−3.8 to +7.1) | −0.4 pp | +1.0 pp | No |

### USDT-M perpetual futures (30 % cut shown in detail)

| Symbol | Skipped | Kept rate | Rejected rate | Diff @30 % (95 % CI) | Diff @40 % | Diff @50 % | Beats noise? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ARB/USDT | 60 | 31.4 % | 20.0 % | +11.4 pp (−4.1 to +22.6) | +7.1 pp | +6.0 pp | No |
| AVAX/USDT | 60 | 9.3 % | 10.0 % | −0.7 pp (−11.4 to +6.7) | −0.8 pp | +3.0 pp | No |
| AXS/USDT | 60 | 9.3 % | 5.0 % | +4.3 pp (−5.0 to +10.5) | +2.9 pp | +6.0 pp | No |
| BTC/USDT | 60 | 0.7 % | 0.0 % | +0.7 pp (0.0 to +2.1) | +0.8 pp | +1.0 pp | No |
| ETH/USDT | 60 | 4.3 % | 3.3 % | +0.9 pp (−5.2 to +6.2) | +0.4 pp | 0.0 pp | No |
| SOL/USDT | 60 | 4.3 % | 3.3 % | +1.0 pp (−6.7 to +5.5) | +0.4 pp | 0.0 pp | No |

### Spot vs futures, side by side

| Symbol | Spot: top−bottom / reject-30 | Futures: top−bottom / reject-30 | Spot vs futures deltas |
| --- | ---: | ---: | --- |
| ARB/USDT | +8.3 pp / +7.4 pp | +16.7 pp / +11.4 pp | All metric deltas within noise; no delta CI excludes zero |
| AVAX/USDT | +1.7 pp / −0.7 pp | 0.0 pp / −0.7 pp | All within noise |
| AXS/USDT | +8.3 pp / +5.0 pp | +6.7 pp / +4.3 pp | All within noise |
| BTC/USDT | +1.7 pp / +0.7 pp | +1.7 pp / +0.7 pp | All within noise; one event in each |
| ETH/USDT | +3.3 pp / +0.9 pp | +3.3 pp / +0.9 pp | All within noise |
| NEXO/USDT | +3.3 pp / +4.3 pp (nominal) | no perp | — |
| SOL/USDT | +3.3 pp / +2.6 pp | +3.3 pp / +1.0 pp | All within noise |

**Consistency: no, in either market, and spot vs futures does not separate
them.** Only 1 of the 21 spot cut/symbol comparisons has a CI above zero
(NEXO at 30 %, which collapses at 40 % and 50 %). Futures has **zero** nominal
rejection cuts. ARB futures has the largest point estimates (top−bottom
+16.7 pp, reject-30 +11.4 pp) but its CIs include zero, and the direct
spot-vs-futures AUC difference is −0.03 (95 % CI −0.16 to +0.10), i.e. noise.
No symbol is proven positive at all three cuts in either market, and no common
cut works across the 13 symbol-markets.

## 5. What this means, per symbol

| Symbol | Spot verdict | Futures verdict | Spot vs futures |
| --- | --- | --- | --- |
| ARB/USDT | No reliable rule; AUC 0.55, rejection diffs inside noise. | No reliable rule; AUC 0.58, CIs still include 0.5. | Same; differences within noise. |
| AVAX/USDT | No reliable rule; AUC 0.51 is a coin flip. | No reliable rule; AUC 0.52, rejection flat. | Same; differences within noise. |
| AXS/USDT | No reliable rule; AUC 0.61 is suggestive but CI includes 0.5. | No reliable rule; AUC 0.60, CI includes 0.5. | Same; differences within noise. |
| BTC/USDT | No reliable rule; one event in 200, high AUC is that hour's rank. | No reliable rule; same single-event caveat. | Same; differences within noise. |
| ETH/USDT | No reliable rule; AUC 0.55 with a wide CI. | No reliable rule; AUC 0.55, rejection does nothing. | Same; differences within noise. |
| NEXO/USDT | No reliable rule today; 30 % cut nominal on six events, collapses at other cuts. | No perpetual market; skipped. | Only spot evaluated. |
| SOL/USDT | No reliable rule; AUC 0.59 with a CI spanning 0.32–0.87. | No reliable rule; AUC 0.55. | Same; differences within noise. |

**Overall:** the JEV score does not support a reliable rejection threshold on
spot or USDT-M perpetual futures in this campaign. Futures is not better in any
statistically meaningful way; every spot-vs-futures delta is within noise.
Production stays at the 0.5 fallback; this is a measurement result, not a
threshold change.

## Limits (read before quoting any single row)

- **Short window.** 200 hourly states per symbol-market ≈ 8.3 days, one market
  regime. That is long enough to describe these two weeks, not long enough to
  establish a durable rule.
- **Few events in most symbols.** BTC has 1 event in each market, NEXO 6 (spot
  only), SOL 7 spot / 8 futures, ETH 8 each. AUC and event-rate differences
  with so few positives are fragile even when the bootstrap CI looks tight.
- **Funding and leverage are not modelled.** Perpetual funding/borrow costs are
  not in the candle data, and no leverage is modelled. Any economic statement
  would therefore be indicative only; this document makes no P&L claim at all.
  The saved reports still contain per-symbol OOS economics at 0.05 / 0.10 /
  0.20 % cost, but they do not include funding and are not used for a verdict.
- **13 symbol-market comparisons are not independent.** Seven spot and six
  futures evaluations are correlated (the same assets, the same dates, and
  futures tracks spot), and each carries AUC plus three rejection checks. The
  campaign runs 65 correlated metric checks in total. At 95 %, several nominal
  hits are expected by chance; a single symbol looking good (NEXO spot, ARB
  futures) is not evidence. The aggregate conclusion must be read across
  markets, not cherry-picked per row.
- **Base decision is a momentum proxy, not the LLM.** The event label here is
  raw forward price movement, so the base decision does not enter the event
  definition. Do not read these numbers as LLM decision quality.
- **Same-calendar-window alignment is deliberate**, so cross-market comparison
  is not contaminated by different dates. It does not remove the shared market
  factor.
- **Reliability/Brier are not headline metrics.** The 1 % event is rare in most
  symbols, so a constant base-rate predictor has a trivially low Brier score.
  The campaign judges ranking (AUC and rank-rejection differences), never a
  Brier-vs-constant headline.

## Reproduction

```bash
# 1. Spot candles (committed CSVs are the exact inputs used)
python scripts/fetch_jev_bench_candles.py \
  --symbols BTC/USDT,ETH/USDT,SOL/USDT,AXS/USDT,AVAX/USDT,NEXO/USDT,ARB/USDT \
  --limit 700 --end 2026-10-25T05:00:00Z --output-dir runtime/jev_bench

# 2. USDT-M perpetual futures candles over the same window
python scripts/fetch_jev_bench_candles.py --market futures \
  --symbols BTC/USDT,ETH/USDT,SOL/USDT,AXS/USDT,AVAX/USDT,ARB/USDT \
  --limit 700 --end 2026-10-25T05:00:00Z --output-dir runtime/jev_bench

# 3. One calibration run per symbol-market (0.5 s pacing, no retries)
for pair in BTC:btc ETH:eth SOL:sol AXS:axs AVAX:avax NEXO:nexo ARB:arb; do
  sym="${pair%%:*}"; base="${pair##*:}"
  python scripts/jev_calibration.py \
    --candles "runtime/jev_bench/${sym}_USDT_1h.csv" \
    --symbol "${sym}/USDT" --market-type spot --timeframe 1h --limit 200 \
    --delay-seconds 0.5 \
    --output "runtime/jev_bench/calibration_${base}_200.json"
done
for pair in BTC:btc ETH:eth SOL:sol ARB:arb AXS:axs AVAX:avax; do
  sym="${pair%%:*}"; base="${pair##*:}"
  python scripts/jev_calibration.py \
    --candles "runtime/jev_bench/${sym}_USDT_1h_futures.csv" \
    --symbol "${sym}/USDT" --market-type futures --timeframe 1h --limit 200 \
    --delay-seconds 0.5 \
    --output "runtime/jev_bench/calibration_${base}_200_futures.json"
done

# 4. Aggregate both markets and compare (bootstrap CIs, rank rejection)
python scripts/jev_campaign_analysis.py \
  --reports runtime/jev_bench/calibration_*_200.json \
  --futures-reports runtime/jev_bench/calibration_*_200_futures.json \
  --output runtime/jev_bench/campaign_summary.json
```

Seeded bootstrap (2,000 resamples, seed `20261024`) makes the intervals and the
summary deterministic.
