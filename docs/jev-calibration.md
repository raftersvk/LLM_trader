# JEV tradeability calibration (honest metrics)

Measure what the JEV gate's `noul` actually predicts, using the production
no-look-ahead state and the production `build_questions` payload. This document
supersedes the earlier "Brier vs constant base rate" headline, which was
misleading for a rare label and is not a valid basis for a threshold decision.

- Dataset: 600 closed 1h candles per symbol; the 200 most recent evaluable
  states were scored (`BTC/USDT`, `ETH/USDT`).
- Provider: Decision 3.0 `vllm-sr/d3-flash`, one call per state, 0.5 s pacing.
- Calls: 400/400 succeeded. 0 errors, 0 rate-limit (429) responses.
- Latency: BTC p50 159 ms / p95 204 ms; ETH p50 153 ms / p95 160 ms.
- Worth-trading label: `abs(forward_return_h1) >= 1.0 %`.
- Base decision: rule-based 4-candle momentum proxy, **not an LLM**.
- Repro: `scripts/jev_calibration.py`; machine reports in
  `runtime/jev_bench/calibration_{btc,eth}_200.json`, observation log via
  `--observations runtime/jev_bench/observations.jsonl`.
- Weekly policy layer: see [`jev-policy.md`](jev-policy.md). The policy uses a
  trade-P&L label (base direction, net of cost) over h1-h4.

## What changed in this revision

The audit found three problems with the old report:

1. The headline compared a model's Brier with a constant base-rate predictor.
   At 1-4 % prevalence the constant predictor is trivially strong, so the
   comparison says almost nothing about ranking skill.
2. Economics were shown only in aggregate, at one cost, without an OOS split.
3. The report ended with one joint threshold recommendation (0.65 across both
   symbols), which the data do not support.

The reworked report:

- reports prevalence, Brier, Brier skill score against **both** a constant
  base-rate predictor and a one-feature logistic-regression baseline, AUC,
  PR-AUC and a reliability table, per symbol and horizon;
- evaluates economics per symbol on a chronological OOS tail only (last 30 %
  of scored states), at 0.05 / 0.10 / 0.20 % cost, with a seeded bootstrap 95 %
  CI for net;
- treats `noul` as a tradeability **ranking** signal, reports directional AUC
  with a CI, and makes **no joint threshold recommendation**.

The gate/threshold behavior and the signal service are unchanged: the runtime
fallback remains `tradeability_threshold = 0.5`.

## noul distribution

| Bin | BTC | ETH |
| --- | ---: | ---: |
| 0.0-0.1 | 5 | 7 |
| 0.1-0.2 | 44 | 49 |
| 0.2-0.3 | 32 | 43 |
| 0.3-0.4 | 28 | 15 |
| 0.4-0.5 | 21 | 16 |
| 0.5-0.6 | 16 | 22 |
| 0.6-0.7 | 11 | 11 |
| 0.7-0.8 | 18 | 13 |
| 0.8-0.9 | 9 | 14 |
| 0.9-1.0 | 16 | 10 |
| **min** | 0.0639 | 0.0488 |
| **median** | 0.3538 | 0.3089 |
| **mean** | 0.4336 | 0.4047 |
| **max** | 0.9672 | 0.9631 |

JEV is confident: mean noul is 0.43 / 0.40 while only 1 % / 4 % of states meet
the 1 %-move label. Both medians sit below the production fallback of 0.5.

## Class imbalance caveat

The h1 label is rare: **BTC 2/200 (1.0 %)**, **ETH 8/200 (4.0 %)**. When
prevalence is below 5 %, predicting the base rate for every state gives a very
low Brier score. That is an artefact of imbalance, not evidence that the
constant predictor is useful. The report emits an explicit warning for every
such horizon:

> prevalence < 5 %: a constant base-rate predictor's Brier is trivially strong;
> never compare a model's Brier head-to-head with it.

Use Brier skill scores and ranking metrics (AUC, PR-AUC) to judge the model,
and read the reliability table to judge calibration.

## Per-horizon metrics

`BSS` is the Brier skill score, `1 - Brier / Brier_reference`; 0 means the
model ties the reference, negative means it is worse. The constant reference is
`p * (1 - p)`. The logistic reference is a one-feature logistic regression on
`noul`, fitted in-sample (a reference, not a validated model).

### BTC/USDT

| Horizon | Positives | Prevalence | Brier | Constant Brier | BSS const | Logistic Brier | BSS logistic | AUC | PR-AUC | Directional AUC (95 % CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1 | 2/200 | 0.0100 | 0.2504 | 0.0099 | -24.29 | 0.0097 | -24.76 | 0.8535 | 0.0814 | 0.5255 (0.292-0.761) |
| h2 | 7/200 | 0.0350 | 0.2572 | 0.0338 | -6.62 | 0.0338 | -6.62 | 0.6018 | 0.0473 | 0.5677 (0.323-0.791) |
| h3 | 16/200 | 0.0800 | 0.2741 | 0.0736 | -2.72 | 0.0734 | -2.73 | 0.4677 | 0.0744 | 0.4792 (0.255-0.703) |
| h4 | 27/200 | 0.1350 | 0.2692 | 0.1168 | -1.31 | 0.1167 | -1.31 | 0.5237 | 0.1605 | 0.3904 (0.181-0.615) |

### ETH/USDT

| Horizon | Positives | Prevalence | Brier | Constant Brier | BSS const | Logistic Brier | BSS logistic | AUC | PR-AUC | Directional AUC (95 % CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h1 | 8/200 | 0.0400 | 0.2271 | 0.0384 | -4.91 | 0.0377 | -5.03 | 0.6517 | 0.2837 | 0.5449 (0.377-0.705) |
| h2 | 23/200 | 0.1150 | 0.2495 | 0.1018 | -1.45 | 0.1017 | -1.45 | 0.5372 | 0.1492 | 0.6092 (0.445-0.768) |
| h3 | 39/200 | 0.1950 | 0.2676 | 0.1570 | -0.70 | 0.1570 | -0.70 | 0.5100 | 0.2214 | 0.5700 (0.410-0.729) |
| h4 | 48/200 | 0.2400 | 0.2663 | 0.1824 | -0.46 | 0.1820 | -0.46 | 0.5296 | 0.3218 | 0.6055 (0.438-0.759) |

Interpretation:

- **Ranking is the only demonstrated signal.** On h1, `noul` ranks
  worth-trading states well above chance: AUC **0.854 BTC**, **0.652 ETH**.
  PR-AUC is low in absolute terms (0.081 / 0.284) because positives are rare,
  but it is far above the 1 % / 4 % prevalence floor.
- **noul is not calibrated as a probability.** Brier skill is deeply negative
  against both references, and the reliability table below shows predicted
  rates far above observed rates.
- **Direction is not predicted.** Directional AUC is **0.526 BTC** and
  **0.545 ETH** on h1, with CIs that include 0.5. Higher noul does not mean the
  momentum proxy is more often right about direction.
- The h3/h4 BTC AUCs (0.468 / 0.524) do not support a ranking claim either.

### Reliability, h1 (1 % move label)

| Bin | BTC count | BTC predicted | BTC observed | ETH count | ETH predicted | ETH observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0-0.2 | 49 | 0.155 | 0.000 | 56 | 0.143 | 0.000 |
| 0.2-0.4 | 60 | 0.292 | 0.000 | 58 | 0.270 | 0.069 |
| 0.4-0.6 | 37 | 0.494 | 0.000 | 38 | 0.504 | 0.026 |
| 0.6-0.8 | 29 | 0.711 | 0.034 | 24 | 0.705 | 0.000 |
| 0.8-1.0 | 25 | 0.908 | 0.040 | 24 | 0.883 | 0.125 |

The observed rate never approaches the predicted rate. A noul of 0.5 does not
mean "50 % chance this state is worth a 1 % move"; it only means the state
ranks above lower-noul states.

## Economics (per symbol, OOS only)

The report evaluates each symbol separately on the chronological held-out tail
of scored states (last 30 %; 140 train / 60 OOS here). No symbol is ever
pooled. Trade selection is a base BUY/SELL with `noul >= 0.5`, matching the
audit that invalidated the old comparison. The bootstrap is seeded
(`20261024`) so regeneration is deterministic.

### BTC/USDT OOS h1 (`noul >= 0.5`)

| Cost % | Trades | Net % | Mean % | Win rate | Net 95 % CI |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 6 | +0.086 | +0.014 | 0.333 | [-1.080, +1.551] |
| 0.10 | 6 | -0.214 | -0.036 | 0.333 | [-1.380, +1.251] |
| 0.20 | 6 | -0.814 | -0.136 | 0.167 | [-1.980, +0.651] |

**Verdict: no demonstrated edge (cost-sensitive sign flip).**

### ETH/USDT OOS h1 (`noul >= 0.5`)

| Cost % | Trades | Net % | Mean % | Win rate | Net 95 % CI |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 7 | -0.110 | -0.016 | 0.429 | [-2.275, +1.979] |
| 0.10 | 7 | -0.460 | -0.066 | 0.429 | [-2.625, +1.629] |
| 0.20 | 7 | -1.160 | -0.166 | 0.429 | [-3.325, +0.929] |

**Verdict: no demonstrated edge (non-positive at all tested costs).**

The wider OOS sample is small (6-7 trades), so the CIs are wide. That is the
honest result: with these data, neither symbol demonstrates a net edge.

### Audit baseline on all labeled states (pre-rework evidence)

The audit that motivated this rework computed economics over all 200 states,
not the OOS tail. Those numbers are retained here as the "before" evidence,
because they are what the old report should have shown but did not. They are
**not** the report's OOS verdict.

| Symbol | Threshold | Trades | Net @ 0.05 % | Net @ 0.10 % | Net @ 0.20 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC | 0.00 | 28 | -2.89 | -4.29 | -7.09 |
| BTC | 0.50 | 18 | -1.97 | **-2.87** | -4.67 |
| BTC | 0.65 | 15 | -0.32 | -1.07 | -2.57 |
| BTC | 0.80 | 11 | -0.33 | -0.88 | -1.98 |
| ETH | 0.00 | 50 | +0.51 | -1.99 | -6.99 |
| ETH | 0.50 | 29 | +2.86 | **+1.41** | -1.49 |
| ETH | 0.65 | 20 | +3.50 | +2.50 | +0.50 |
| ETH | 0.80 | 13 | +4.07 | +3.42 | +2.12 |

BTC is negative at every tested threshold and cost. ETH is positive at
0.10 % but negative at 0.20 % at the 0.5 threshold, which is exactly the
cost-sensitive sign flip the report now flags. No pooled or joint conclusion is
valid.

## No threshold recommendation

This document deliberately makes **no** single joint threshold
recommendation. The earlier recommendation of 0.65 was based on a marginal
in-sample precision change inside sampling noise and is withdrawn.

- Keep the production fallback at **0.5**; no gate, policy, or service code is
  changed by this revision.
- Treat `noul` as a tradeability ranking signal (AUC/PR-AUC), not as a
  calibrated probability.
- Do not use `noul` as a directional signal: directional AUC is ~0.53 / 0.55
  on h1 with CIs spanning 0.5.
- A higher threshold is a coverage trade-off choice, not an evidence-based
  improvement, until OOS economics show a positive net with a CI that excludes
  zero at realistic costs.

## What a real edge would require

1. A larger, multi-regime sample per symbol; 6-7 OOS trades cannot establish an
   edge even if the point estimate is positive.
2. Positive net P&L OOS at the highest tested cost (0.20 %), not only at the
   lowest.
3. A bootstrap CI for net that excludes zero, with the sign stable across the
   cost range.
4. Directional skill separate from tradeability: directional AUC with a CI
   above 0.5.
5. Calibration evidence that is not an artefact of a rare label: reliability
   and Brier skill against a proper reference, never against the constant
   predictor alone.

## Caveats

- Small sample: 200 states per symbol, one 600-candle (25 day) 1h window and
  one market regime. OOS economics contain only 6-7 trades per symbol.
- The base decision is a momentum proxy, not the production LLM. Its unfiltered
  directional AUC is ~0.53-0.55, so no pipeline variant demonstrates a
  directional edge here.
- The 1 % / h1 label is rare (1 % BTC, 4 % ETH). Brier and reliability are
  dominated by this rare-event base rate; the constant baseline is trivially
  strong and must not be used as a head-to-head headline.
- The logistic baseline is fitted in-sample and is a reference, not a
  cross-validated model.
- Coverage uses noul alone; the threshold curve in the JSON report additionally
  applies the production pre-gate rules. It is a diagnostic, not a
  recommendation.
- No LLM decisions were run in this calibration.
