# JEV question set v2: separating magnitude from direction

**Question:** the production question set (v1) asks one vague "is a short-term
directional trade worth taking" question, which mixes *whether price will move*
with *which way*. Does asking two separate questions help — one for magnitude
(`noul`) and one for direction (`bias`) — with a small past-only calibration
block added to the state?

**Short answer: no, not for any decision-relevant metric.** On the same 1,400
states the v2 wording does not rank 1 %-move states better than v1, does not
predict direction, and does not produce a usable rank-rejection rule at 30 / 40
/ 50 %. It does move the raw `noul` probability level much closer to the
observed event rate, but that calibration gain does not translate into a better
filter. **Production keeps v1 and the 0.5 threshold.** No service code changes.

- **Baseline:** the committed v1 reports from PR #17
  (`runtime/jev_bench/calibration_<base>_200.json`), not re-run.
- **Variant:** `--question-set v2 --calibration-context` on the same 7 symbols,
  same 700-candle inputs, same 200 states per symbol, same h1 horizon and
  `abs(forward return) >= 1.0 %` label, 0.5 s pacing.
- **Window:** states span `2026-10-16T18:00Z` → `2026-10-25T01:00Z` (≈8.3 days,
  one market regime).
- **Provider:** Decision 3.0 `vllm-sr/d3-flash`, one call per state.
- **Calls:** 1,400/1,400 scored, 0 errors, 0 rate-limit (429) responses.
- **Smoke first:** one call per question set per symbol = 14/14 accepted, no
  `400`, no `invalid_question` (`runtime/jev_bench/v2_question_smoke.json`).
- **Machine outputs:**
  `runtime/jev_bench/calibration_<base>_200_v2.json`,
  `runtime/jev_bench/campaign_summary_v2.json`,
  `runtime/jev_bench/v2_comparison.json`.

## What changed in v2 (and what did not)

1. **`tradeability` (noul) asks magnitude only.** `true` = the absolute price
   change over the next hour is likely at least 1.0 %, regardless of direction;
   `false` = such a move is unlikely. The question never mentions direction.
2. **`bias` asks direction only.** The instructions say it is direction only,
   independent of move size; `none` means no directional edge.
3. **`regime` and `confidence` are byte-identical to v1.**
4. **Optional calibration block in the state**, computed only from closed
   candles before `as_of`:
   - `base_rate_move_1pct`: share of the last up-to-200 hourly simple returns
     with `abs(return) >= 1 %` (4 decimals);
   - `realized_vol_24h_pct`: 100 × the population standard deviation of the last
     up-to-24 hourly simple returns (6 decimals), not annualised;
   - the lookback counts and threshold are included in the block.

The block is off by default, so a v1 run stays byte-comparable. This experiment
bundles the new wording **and** the block: it cannot separate their effects.

## 1. Does v2 rank "big move" states better?

No. Tradeability AUC against the `abs(forward h1 return) >= 1 %` label, with
bootstrap 95 % CIs. The last column is the **paired** v2-minus-v1 bootstrap
interval (same states resampled for both versions).

| Symbol | Events / states | v1 AUC (95 % CI) | v2 AUC (95 % CI) | v2 − v1 (paired 95 % CI) |
| --- | ---: | --- | --- | --- |
| ARB/USDT | 57/200 (28.5 %) | 0.554 (0.459–0.644) | 0.513 (0.416–0.602) | −0.041 (−0.119 to +0.032) |
| AVAX/USDT | 19/200 (9.5 %) | 0.510 (0.359–0.665) | 0.506 (0.349–0.656) | −0.004 (−0.118 to +0.104) |
| AXS/USDT | 17/200 (8.5 %) | 0.608 (0.469–0.749) | 0.605 (0.433–0.771) | −0.003 (−0.174 to +0.144) |
| BTC/USDT | 1/200 (0.5 %) | 0.975 (0.950–0.995) | 0.980 (0.959–0.995) | +0.005 (−0.015 to +0.025) |
| ETH/USDT | 8/200 (4.0 %) | 0.547 (0.270–0.814) | 0.549 (0.236–0.852) | +0.002 (−0.138 to +0.130) |
| NEXO/USDT | 6/200 (3.0 %) | 0.655 (0.419–0.914) | 0.741 (0.465–0.975) | +0.086 (−0.162 to +0.352) |
| SOL/USDT | 7/200 (3.5 %) | 0.587 (0.317–0.865) | 0.522 (0.172–0.852) | −0.065 (−0.364 to +0.174) |

- v2 is higher on 3 symbols and lower on 4.
- **No symbol's paired interval excludes zero** in either direction.
- Median paired delta: **−0.003**.
- BTC's 0.98 AUC in both versions rests on **one positive state**, so it is not
  evidence of ranking skill in either version.

**Answer: no measurable ranking improvement.** The point estimates are scattered
around zero, and the direction of the scatter slightly favours v1.

## 2. Does v2 predict direction?

No. The direction question is `bias` (long=1, short=0, `none` excluded), scored
against `forward h1 return > 0`.

| Symbol | v1 bias AUC (95 % CI) | v2 bias AUC (95 % CI) | v2 − v1 (paired 95 % CI) |
| --- | --- | --- | --- |
| ARB/USDT | 0.490 (0.414–0.563) | 0.483 (0.408–0.554) | −0.007 (−0.029 to +0.014) |
| AVAX/USDT | 0.432 (0.359–0.513) | 0.437 (0.364–0.517) | +0.005 (−0.024 to +0.036) |
| AXS/USDT | 0.465 (0.385–0.545) | 0.459 (0.379–0.537) | −0.006 (−0.044 to +0.031) |
| BTC/USDT | 0.467 (0.381–0.550) | 0.487 (0.400–0.573) | +0.020 (−0.014 to +0.055) |
| ETH/USDT | 0.494 (0.409–0.579) | 0.490 (0.402–0.573) | −0.004 (−0.037 to +0.026) |
| NEXO/USDT | 0.366 (0.284–0.450) | 0.365 (0.287–0.447) | −0.001 (−0.030 to +0.030) |
| SOL/USDT | 0.427 (0.340–0.509) | 0.459 (0.381–0.537) | +0.032 (−0.009 to +0.074) |

- v2 is higher on 3 symbols and lower on 4; **no paired interval excludes zero**;
  median delta **−0.001**.
- **NEXO is the only bias AUC whose interval excludes 0.5, and it is below 0.5**
  in both versions (0.366 / 0.365). The direction answer was systematically
  wrong there, not useful. With 7 symbols that is best read as a nominal
  multiple-comparison artefact, not a discovered anti-signal.
- The direction-answer coverage is similar in both versions (v2 gives long or
  short on 62–75 % of states), so v2 did not simply become more abstinent.

The direction signal of `noul` (not the intended channel) moved up on all 7
symbols but by a trivial amount: median paired delta **+0.021**, and **no**
interval excludes zero. Since v2's `noul` question is explicitly
direction-blind, that is noise, not direction skill.

**Answer: v2 does not predict direction, and the separation did not unlock it.**

## 3. Does v2 produce a usable rank-rejection rule?

No. The table shows kept-minus-rejected event rate after rejecting the lowest
30 / 40 / 50 % of `noul` states. `*` marks an interval whose 95 % lower bound is
above zero.

| Symbol | v1 30 % | v2 30 % | v1 40 % | v2 40 % | v1 50 % | v2 50 % | v2 cuts usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARB/USDT | +0.074 | −0.045 | +0.017 | −0.046 | +0.050 | +0.010 | 0/3 |
| AVAX/USDT | −0.007 | +0.017 | −0.008 | −0.008 | +0.010 | −0.050 | 0/3 |
| AXS/USDT | +0.050 | +0.026 | +0.037 | +0.037 | +0.030 | +0.050 | 0/3 |
| BTC/USDT | +0.007 | +0.007 | +0.008 | +0.008 | +0.010 | +0.010 | 0/3 |
| ETH/USDT | +0.010 | −0.014 | −0.017 | −0.017 | 0.000 | 0.000 | 0/3 |
| NEXO/USDT | +0.043 * | +0.019 | +0.029 | +0.029 | 0.000 | +0.020 | 0/3 |
| SOL/USDT | +0.026 | −0.021 | −0.004 | −0.004 | +0.010 | +0.010 | 0/3 |
| **Symbols with a usable cut** | **1/7** | **0/7** | **0/7** | **0/7** | **0/7** | **0/7** | **0/7** |

- v2 has **no** symbol where all three cuts separate, and **no single cut** on
  any symbol clears zero.
- v1's one nominal hit (NEXO, 30 %) disappears under v2; it was the only
  nominal hit out of 21 v1 checks, which is what chance produces.
- The paired deltas at the cuts are also all within noise: at 30 % v2 is worse
  on 5/7 symbols (median −0.024), and at 40 / 50 % the medians are 0.000.
- The standalone v2 campaign summary reaches the same verdict: "no reliable
  rejection rule today" for every symbol.

**Answer: no usable rank-based rejection rule, at any tested cut, on any
symbol.**

## 4. What v2 did fix: probability level, not ranking

`noul` under v2 is a far better *level* estimate: its mean is close to the
observed event rate, whereas v1 was heavily overconfident.

| Symbol | Event rate | v1 median / mean | v1 states ≥ 0.5 | v2 median / mean | v2 states ≥ 0.5 |
| --- | ---: | --- | ---: | --- | ---: |
| ARB/USDT | 28.5 % | 0.311 / 0.422 | 68/200 | 0.204 / 0.293 | 43/200 |
| AVAX/USDT | 9.5 % | 0.166 / 0.262 | 32/200 | 0.063 / 0.092 | 3/200 |
| AXS/USDT | 8.5 % | 0.209 / 0.305 | 43/200 | 0.082 / 0.105 | 2/200 |
| BTC/USDT | 0.5 % | 0.337 / 0.411 | 65/200 | 0.020 / 0.029 | 1/200 |
| ETH/USDT | 4.0 % | 0.262 / 0.365 | 57/200 | 0.023 / 0.044 | 2/200 |
| NEXO/USDT | 3.0 % | 0.164 / 0.275 | 36/200 | 0.038 / 0.068 | 3/200 |
| SOL/USDT | 3.5 % | 0.239 / 0.331 | 46/200 | 0.029 / 0.047 | 2/200 |

The Brier skill against a constant base-rate predictor moves from strongly
negative under v1 (−0.35 to −45.3) to near zero under v2 (−0.48 to +0.06). Near
zero is the honest reading: **v2 is close to predicting the base rate, not
close to discriminating events.** The previous campaign already warned that
Brier-vs-constant is not a ranking metric.

This also means the production 0.5 threshold cannot be copied to v2: only
1–43 of 200 v2 states per symbol clear it, versus 32–68 under v1. Any future
variant would need its own threshold calibration, not a wording swap.

## Overall verdict

| Metric (v2 − v1) | v2 better | v1 better | Outside noise | Median delta |
| --- | ---: | ---: | ---: | ---: |
| Tradeability AUC | 3/7 | 4/7 | 0 | −0.003 |
| Direction AUC of `noul` | 7/7 | 0/7 | 0 | +0.021 |
| Direction AUC of `bias` | 3/7 | 4/7 | 0 | −0.001 |
| Rejection diff 30 % | 1/7 | 5/7 | 0 | −0.024 |
| Rejection diff 40 % | 0/7 | 1/7 | 0 | 0.000 |
| Rejection diff 50 % | 2/7 | 2/7 | 0 | 0.000 |

**Verdict: no improvement; keep v1.** v2 is not better for ranking, for
direction, or for a rank-rejection rule. Its one measurable gain is that the
raw probability level is much closer to the event base rate, but a gate filters
by rank/threshold, and that gain does not survive as a decision rule.

The result is consistent with the earlier campaign: the engine can rank
big-move states a little on some symbols, but no symbol supports a rejection
rule, and the direction question remains unanswered regardless of how it is
phrased. Production keeps `build_questions()` v1 and
`tradeability_threshold = 0.5`.

## Limits (read before quoting any single row)

- **Short window.** 200 hourly states per symbol ≈ 8.3 days, one market regime.
  This describes these two weeks; it does not establish a durable rule.
- **Very few events.** BTC has **1** event per 200 states; ETH 8, NEXO 6, SOL 7.
  AUCs and rejection differences with so few positives are fragile even when a
  bootstrap interval looks narrow (for BTC many bootstrap resamples contain no
  positive at all).
- **Seven symbols, many checks.** There are 7 symbols × 6 correlated checks
  (tradeability, two direction views, three rejection cuts) ≈ 42 comparisons.
  At 95 %, roughly two nominal hits are expected by chance. Here **no**
  comparison clears zero in v2's favour, and the only nominal direction hit
  (NEXO) points the wrong way. Do not cherry-pick a row.
- **Bundled treatment.** v2 changes both the question wording and adds the
  calibration block. This run cannot say which part (if either) caused the
  calibration-level shift; a future ablation would need separate runs.
- **Time gap between campaigns.** v1 reports are the committed baseline and
  were not re-run. The model id is the same (`vllm-sr/d3-flash`), but provider
  behaviour can drift between campaign dates; a same-day v1 re-run would remove
  that ambiguity.
- **The base decision is a momentum proxy, not the LLM.** The label is raw
  forward price movement, so the base decision does not enter the AUC,
  direction or rejection definitions. Nothing here is an LLM decision-quality
  or P&L result.
- **Threshold and economics not re-derived.** This document judges ranking and
  rank rejection only. It makes no P&L claim and does not propose a v2
  threshold; the saved reports contain per-symbol OOS economics for audit.

## Reproduction

```bash
# 1. One acceptance call per question set per symbol (14 calls)
python scripts/jev_question_smoke.py \
  --csv-dir runtime/jev_bench \
  --output runtime/jev_bench/v2_question_smoke.json

# 2. v2 campaign, same inputs/window/label as v1, 200 states per symbol
for pair in BTC:btc ETH:eth SOL:sol AVAX:avax ARB:arb AXS:axs NEXO:nexo; do
  sym="${pair%%:*}"; base="${pair##*:}"
  python scripts/jev_calibration.py \
    --candles "runtime/jev_bench/${sym}_USDT_1h.csv" \
    --symbol "${sym}/USDT" --market-type spot --timeframe 1h --limit 200 \
    --delay-seconds 0.5 --question-set v2 --calibration-context \
    --output "runtime/jev_bench/calibration_${base}_200_v2.json"
done

# 3. Standalone v2 metrics in the same format as the v1 campaign summary
python - <<'PY'
import glob, json, sys
sys.path.insert(0, "scripts")
from jev_campaign_analysis import build_summary
paths = [p for p in sorted(glob.glob("runtime/jev_bench/calibration_*_200_v2.json"))
         if "_futures" not in p]
summary = build_summary(paths, ())
with open("runtime/jev_bench/campaign_summary_v2.json", "w") as fh:
    json.dump(summary, fh, indent=2)
PY

# 4. Paired v1-versus-v2 comparison (2000 resamples, seed 20261024)
python scripts/jev_question_set_comparison.py \
  --v1-reports runtime/jev_bench/calibration_*_200.json \
  --v2-reports runtime/jev_bench/calibration_*_200_v2.json \
  --output runtime/jev_bench/v2_comparison.json
```

All bootstrap intervals are deterministic (2,000 resamples, seed `20261024`).
