# JEV 7-symbol calibration campaign

**Question:** can the JEV "tradeability" score (`noul`) support a reliable
*rejection* threshold — a rule that skips the lowest-scoring states and keeps
the rest?

**Short answer: not today.** Across seven symbols and 1,400 real JEV calls, no
rank-based rejection rule survives its own confidence interval *and* the
multiple-comparison burden. One symbol (NEXO) shows a nominally significant
difference at the 30 % cut only, with six events total; that is exactly the
kind of single hit that testing 21 rejection rules can produce by chance. The
production fallback threshold stays at **0.5** and no service code is changed.

- Dataset: 700 aligned 1h Binance spot candles per symbol,
  `2026-09-26T02:00Z` → `2026-10-25T05:00Z`. The 200 scored states per symbol
  span `2026-10-16T18:00Z` → `2026-10-25T01:00Z` (≈8.3 days).
- Symbols: BTC/USDT and ETH/USDT as references, plus SOL/USDT, AXS/USDT,
  AVAX/USDT, NEXO/USDT and ARB/USDT.
- Provider: Decision 3.0 `vllm-sr/d3-flash`, one call per state, 0.5 s pacing.
- Worth-trading event: `abs(forward 1h return) >= 1.0 %`.
- Machine inputs: `runtime/jev_bench/<BASE>_USDT_1h.csv` and
  `runtime/jev_bench/calibration_<base>_200.json`. Aggregated metrics:
  `runtime/jev_bench/campaign_summary.json`.

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

Every symbol ran exactly 200 states, sequentially, one pass, no retry loop.
There were **no 429 rate-limit responses and no failures**.

| Symbol | Calls | Scored | Failed | 429 | p50 / p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT | 200 | 200 | 0 | 0 | 158 / 193 ms |
| ETH/USDT | 200 | 200 | 0 | 0 | 157 / 218 ms |
| SOL/USDT | 200 | 200 | 0 | 0 | 159 / 168 ms |
| AXS/USDT | 200 | 200 | 0 | 0 | 159 / 204 ms |
| AVAX/USDT | 200 | 200 | 0 | 0 | 157 / 166 ms |
| NEXO/USDT | 200 | 200 | 0 | 0 | 159 / 202 ms |
| ARB/USDT | 200 | 200 | 0 | 0 | 160 / 187 ms |
| **Total** | **1,400** | **1,400** | **0** | **0** | |

A separate one-call credential check was made before the campaign; the campaign
itself is the 1,400 calls above.

## 1. How often is a state worth trading?

| Symbol | Event states | Event rate (95 % CI) |
| --- | ---: | ---: |
| ARB/USDT | 57 / 200 | 28.5 % (22.7–35.1 %) |
| AVAX/USDT | 19 / 200 | 9.5 % (6.2–14.4 %) |
| AXS/USDT | 17 / 200 | 8.5 % (5.4–13.2 %) |
| BTC/USDT | 1 / 200 | 0.5 % (0.1–2.8 %) |
| ETH/USDT | 8 / 200 | 4.0 % (2.0–7.7 %) |
| NEXO/USDT | 6 / 200 | 3.0 % (1.4–6.4 %) |
| SOL/USDT | 7 / 200 | 3.5 % (1.7–7.1 %) |

The event is common in ARB (about one hour in three and a half) and rare in BTC
(one hour in 200). This spread alone means a threshold or rejection fraction
imported from one symbol cannot be assumed to transfer to another.

## 2. Does a high score identify those states?

"Top" and "bottom" are the highest and lowest 30 % of scores (60 states each).
The difference is top minus bottom, with a bootstrap 95 % CI that re-draws the
rank cut on every resample.

| Symbol | Top 30 % event rate | Bottom 30 % event rate | Difference (95 % CI) | AUC (95 % CI) | Proven at 95 %? |
| --- | ---: | ---: | ---: | ---: | --- |
| ARB/USDT | 31.7 % | 23.3 % | +8.3 pp (−8.3 to +25.0) | 0.554 (0.459–0.644) | No |
| AVAX/USDT | 11.7 % | 10.0 % | +1.7 pp (−10.0 to +11.7) | 0.510 (0.359–0.665) | No |
| AXS/USDT | 13.3 % | 5.0 % | +8.3 pp (−1.7 to +18.3) | 0.608 (0.469–0.749) | No |
| BTC/USDT | 1.7 % | 0.0 % | +1.7 pp (0.0 to +5.0) | 0.975 (0.950–0.995) | Formally yes, but see below |
| ETH/USDT | 6.7 % | 3.3 % | +3.3 pp (−5.0 to +11.7) | 0.547 (0.270–0.814) | No |
| NEXO/USDT | 3.3 % | 0.0 % | +3.3 pp (0.0 to +10.0) | 0.656 (0.419–0.914) | No |
| SOL/USDT | 5.0 % | 1.7 % | +3.3 pp (−3.3 to +10.0) | 0.587 (0.317–0.865) | No |

**BTC is the trap in this table.** Its AUC CI excludes 0.5, so it is formally
"proven" at 95 %, but there is exactly **one** worth-trading hour in its 200
states. The AUC is then just the rank of that single event; it does not
establish a repeatable rule. In practical terms BTC's ranking is **not**
established. Every other symbol's AUC CI includes 0.5, so none of them is
proven.

No top-vs-bottom difference has a CI strictly above zero: BTC and NEXO reach a
lower bound that rounds to 0.0, which is not enough.

## 3. Does the score say anything about direction (up vs down)?

Here the score is used to rank *up hours* (`forward return > 0`) against *down
hours*. AUC 0.5 means no directional information.

| Symbol | Up hours | Direction AUC (95 % CI) | Direction proven at 95 %? |
| --- | ---: | ---: | --- |
| ARB/USDT | 90 / 200 | 0.540 (0.461–0.617) | No |
| AVAX/USDT | 91 / 200 | 0.514 (0.432–0.595) | No |
| AXS/USDT | 83 / 200 | 0.512 (0.431–0.592) | No |
| BTC/USDT | 104 / 200 | 0.514 (0.433–0.593) | No |
| ETH/USDT | 107 / 200 | 0.486 (0.405–0.567) | No |
| NEXO/USDT | 87 / 200 | 0.455 (0.374–0.537) | No |
| SOL/USDT | 104 / 200 | 0.512 (0.431–0.590) | No |

**The score says nothing about direction in this campaign.** Every CI includes
0.5. NEXO's point estimate is below 0.5 (higher scores slightly more often
preceded a down hour), but its CI includes 0.5, so even that is not proven.

## 4. Rejection by rank (the practical proposal)

Rule: skip the lowest 30 %, 40 % or 50 % of scores and keep the rest. The table
shows the 30 % cut in detail (60 states skipped out of 200 for every symbol)
and the point difference for the larger cuts. Positive means the kept states
had a higher event rate than the skipped ones.

| Symbol | Skipped at 30 % | Kept event rate @30 % | Rejected event rate @30 % | Difference @30 % (95 % CI) | Difference @40 % | Difference @50 % | Beats noise? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ARB/USDT | 60 | 30.7 % | 23.3 % | +7.4 pp (−8.3 to +18.8) | +1.7 pp | +5.0 pp | No |
| AVAX/USDT | 60 | 9.3 % | 10.0 % | −0.7 pp (−10.7 to +7.4) | −0.8 pp | +1.0 pp | No |
| AXS/USDT | 60 | 10.0 % | 5.0 % | +5.0 pp (−1.7 to +12.9) | +3.8 pp | +3.0 pp | No |
| BTC/USDT | 60 | 0.7 % | 0.0 % | +0.7 pp (0.0 to +2.1) | +0.8 pp | +1.0 pp | No |
| ETH/USDT | 60 | 4.3 % | 3.3 % | +0.9 pp (−5.2 to +6.2) | −1.7 pp | 0.0 pp | No |
| NEXO/USDT | 60 | 4.3 % | 0.0 % | +4.3 pp (+0.2 to +7.9) | +2.9 pp | 0.0 pp | 30 % cut only |
| SOL/USDT | 60 | 4.3 % | 1.7 % | +2.6 pp (−3.8 to +7.1) | −0.4 pp | +1.0 pp | No |

**Consistency across the seven symbols: no.** Only one of the 21 cut/symbol
comparisons has a CI above zero (NEXO at 30 %), and it does not hold at 40 %
or 50 %, where the difference falls to +2.9 pp and 0.0 pp with CIs that
include zero. ARB and AXS have positive point estimates at all three cuts but
every CI includes zero. AVAX is flat or slightly negative. There is no symbol
where a rank rejection is proven positive at all three cuts, and no common cut
that works across symbols.

## 5. What this means, per symbol

| Symbol | Verdict |
| --- | --- |
| ARB/USDT | **No reliable rejection rule.** Event is common (28.5 %) but the score barely ranks it (AUC 0.55, CI includes 0.5); rejection differences are inside noise. |
| AVAX/USDT | **No reliable rejection rule.** AUC 0.51 is a coin flip and rejection does not raise the kept event rate. |
| AXS/USDT | **No reliable rejection rule, but the closest suggestion.** AUC 0.61 and kept 10 % vs rejected 5 %; the CI still includes zero, so it is a hypothesis for a larger sample, not a rule. |
| BTC/USDT | **No reliable rejection rule.** Only one event in 200 hours; the high AUC is that one hour's rank, not evidence. |
| ETH/USDT | **No reliable rejection rule.** AUC 0.55 with a very wide CI, and rejection does nothing at 40–50 %. |
| NEXO/USDT | **No reliable rejection rule today.** The 30 % cut is nominally significant (0 % rejected vs 4.3 % kept, CI +0.2 to +7.9 pp), but it rests on six events, collapses at the other cuts, and is 1 hit among 21 tests. Do not deploy it on this evidence. |
| SOL/USDT | **No reliable rejection rule.** AUC 0.59 with a CI spanning 0.32–0.87; rejection differences include zero. |

**Overall:** the JEV score does not support a reliable rejection threshold in
this campaign. Production stays at the 0.5 fallback; this is a measurement
result, not a threshold change.

## Limits (read before quoting any single row)

- **Short window.** 200 hourly states per symbol ≈ 8.3 days, one market regime.
  That is long enough to describe these two weeks, not long enough to establish
  a durable rule.
- **Few events in most symbols.** BTC has 1 event, NEXO 6, SOL 7, ETH 8. AUC
  and event-rate differences with so few positives are fragile even when the
  bootstrap CI looks tight.
- **Seven symbols are not seven independent experiments.** Crypto pairs move
  together, and the campaign checks 7 tradeability AUCs, 7 directional AUCs and
  21 rejection cuts. At 95 %, roughly one false positive per 20 checks is
  expected; NEXO's single hit is entirely consistent with noise.
- **Base decision is a momentum proxy, not the LLM.** The event label here is
  raw forward price movement, so the base decision does not enter the event
  definition. Do not read these numbers as LLM decision quality.
- **No economic claim.** The question is classification (does a 1 % move
  happen?), not profitability. The saved reports still contain per-symbol OOS
  economics at 0.05 / 0.10 / 0.20 % cost, but no P&L conclusion is drawn here.
- **Reliability/Brier are not headline metrics.** The 1 % event is rare in most
  symbols, so a constant base-rate predictor has a trivially low Brier score.
  The campaign therefore judges ranking (AUC and rank-rejection differences),
  never a Brier-vs-constant headline.
- **Same-calendar-window alignment is deliberate**, so cross-symbol comparison
  is not contaminated by different market dates. It does not remove the shared
  market factor.

## Reproduction

```bash
# 1. Aligned candles (committed CSVs are the exact inputs used)
python scripts/fetch_jev_bench_candles.py \
  --symbols BTC/USDT,ETH/USDT,SOL/USDT,AXS/USDT,AVAX/USDT,NEXO/USDT,ARB/USDT \
  --limit 700 --end 2026-10-25T05:00:00Z --output-dir runtime/jev_bench

# 2. One calibration run per symbol (0.5 s pacing, no retries)
for pair in BTC:btc ETH:eth SOL:sol AXS:axs AVAX:avax NEXO:nexo ARB:arb; do
  sym="${pair%%:*}"; base="${pair##*:}"
  python scripts/jev_calibration.py \
    --candles "runtime/jev_bench/${sym}_USDT_1h.csv" \
    --symbol "${sym}/USDT" --timeframe 1h --limit 200 --delay-seconds 0.5 \
    --output "runtime/jev_bench/calibration_${base}_200.json"
done

# 3. Aggregate (bootstrap CIs, top/bottom groups, rank rejection)
python scripts/jev_campaign_analysis.py \
  --reports runtime/jev_bench/calibration_*_200.json \
  --output runtime/jev_bench/campaign_summary.json
```

Seeded bootstrap (2,000 resamples, seed `20261024`) makes the intervals and the
summary deterministic.
