# JEV tradeability calibration

Measure whether the JEV gate default threshold (`tradeability_threshold = 0.5`)
is calibrated, using the production no-look-ahead state and the production
`build_questions` payload.

- Dataset: 600 closed 1h candles per symbol; the 200 most recent evaluable
  states were scored (`BTC/USDT`, `ETH/USDT`).
- Provider: Decision 3.0 `vllm-sr/d3-flash`, one call per state, 0.5 s pacing.
- Calls: 400/400 succeeded. 0 errors, 0 rate-limit (429) responses.
- Latency: BTC p50 159 ms / p95 476 ms; ETH p50 154 ms / p95 163 ms.
- Worth-trading label: `abs(forward_return_h1) >= 1.0 %`.
- Base decision: rule-based 4-candle momentum proxy, **not an LLM**.
- Repro: `scripts/jev_calibration.py`; machine reports in
  `runtime/jev_bench/calibration_{btc,eth}_200.json`.

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
| **mean** | 0.4337 | 0.4047 |
| **max** | 0.9672 | 0.9631 |

JEV is confident: mean noul is 0.43 / 0.40 while only 1 % / 4 % of states meet
the 1 %-move label. Both medians sit below the production threshold of 0.5.

## Brier score and reliability

Brier = mean `(noul - worth_trading)^2`, lower is better. A constant forecast
at the base rate `p(1-p)` is the no-skill reference.

| Horizon | Worth rate | Brier | Constant baseline |
| --- | ---: | ---: | ---: |
| BTC h1 | 0.010 | 0.2505 | 0.0099 |
| BTC h2 | 0.035 | 0.2573 | 0.0338 |
| BTC h3 | 0.080 | 0.2742 | 0.0736 |
| BTC h4 | 0.135 | 0.2693 | 0.1168 |
| ETH h1 | 0.040 | 0.2271 | 0.0384 |
| ETH h2 | 0.115 | 0.2495 | 0.1018 |
| ETH h3 | 0.195 | 0.2676 | 0.1570 |
| ETH h4 | 0.240 | 0.2663 | 0.1824 |

On every horizon the Brier score is **worse than the constant base-rate
forecast**: noul is systematically overconfident for this label.

Reliability bins (predicted vs observed worth-trading rate):

| Bin | BTC count | BTC predicted | BTC observed | ETH count | ETH predicted | ETH observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0-0.2 | 49 | 0.155 | 0.000 | 56 | 0.143 | 0.000 |
| 0.2-0.4 | 60 | 0.292 | 0.000 | 58 | 0.270 | 0.069 |
| 0.4-0.6 | 37 | 0.494 | 0.000 | 38 | 0.504 | 0.026 |
| 0.6-0.8 | 29 | 0.712 | 0.034 | 24 | 0.705 | 0.000 |
| 0.8-1.0 | 25 | 0.909 | 0.040 | 24 | 0.883 | 0.125 |

The observed rate never reaches one tenth of the predicted rate. A noul of
0.5 does not mean "50 % chance this state is worth a 1 % move".

## Threshold curve (h1, 1 % move label)

Coverage is the share of scored states with `noul >= threshold`.
Trade count and precision apply the full pre-gate rule to base BUY/SELL
decisions. False-hold rate is computed over base trades that the gate blocked.

### BTC/USDT

| Threshold | Coverage | Trades | Base-trade precision | False-hold rate | Directional AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.755 | 21 | 0.4286 | 0.0000 | 0.4809 |
| 0.25 | 0.655 | 21 | 0.4286 | 0.0000 | 0.4878 |
| 0.30 | 0.595 | 21 | 0.4286 | 0.0000 | 0.4744 |
| 0.35 | 0.510 | 20 | 0.4500 | 0.0000 | 0.4744 |
| 0.40 | 0.455 | 17 | 0.4706 | 0.0000 | 0.4882 |
| 0.45 | 0.400 | 17 | 0.4706 | 0.0000 | 0.4770 |
| 0.50 | 0.350 | 16 | 0.5000 | 0.0000 | 0.4802 |
| 0.55 | 0.320 | 15 | 0.5333 | 0.0000 | 0.5170 |
| 0.60 | 0.270 | 14 | 0.5000 | 0.0000 | 0.4792 |
| 0.65 | 0.240 | 13 | 0.5385 | 0.0667 | 0.5000 |
| 0.70 | 0.215 | 13 | 0.5385 | 0.0667 | 0.5000 |

Unfiltered base precision: 0.5000 over 28 trades.

### ETH/USDT

| Threshold | Coverage | Trades | Base-trade precision | False-hold rate | Directional AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.720 | 37 | 0.5135 | 0.0000 | 0.5296 |
| 0.25 | 0.595 | 35 | 0.5143 | 0.0000 | 0.5163 |
| 0.30 | 0.505 | 32 | 0.4688 | 0.0000 | 0.5143 |
| 0.35 | 0.465 | 30 | 0.5000 | 0.0000 | 0.5464 |
| 0.40 | 0.430 | 29 | 0.5172 | 0.0476 | 0.5344 |
| 0.45 | 0.375 | 26 | 0.5385 | 0.0417 | 0.5449 |
| 0.50 | 0.350 | 26 | 0.5385 | 0.0417 | 0.5265 |
| 0.55 | 0.295 | 22 | 0.5000 | 0.0357 | 0.4676 |
| 0.60 | 0.240 | 19 | 0.5263 | 0.0323 | 0.4500 |
| 0.65 | 0.215 | 18 | 0.5556 | 0.0312 | 0.4722 |
| 0.70 | 0.185 | 17 | 0.5294 | 0.0303 | 0.4667 |

Unfiltered base precision: 0.5200 over 50 trades.

## Recommendation

**Recommended threshold: 0.65 (provisional).**

At 0.65 base-trade precision is the joint best on both symbols (BTC 0.5385,
ETH 0.5556) at 24 % / 21.5 % coverage, with false-hold rates of 6.7 % / 3.1 %.
Raising 0.50 -> 0.65 removes 3 (BTC) and 8 (ETH) trades, but the precision gain
is 1-2 correct trades and sits well inside sampling noise; 0.50 remains
defensible when coverage is the priority.

The stronger conclusion is that **0.5 is not a calibrated probability for this
label**. Directional AUC stays at 0.47-0.55 across the whole curve, so the
threshold mostly trades coverage for a marginal precision change rather than
separating genuinely tradeable states.

Even at the high end, observed worth rate is far below noul: at 0.9 the worth
rate is 6.25 % (BTC) and 30 % (ETH) on 16 / 10 states. Tightening the threshold
can improve the hit rate but cannot make the probability itself honest.

## Caveats

- Small sample: 200 states per symbol, one 600-candle (25 day) 1h window and
  one market regime. Confidence intervals on 13-26 trades are wide.
- The base decision is a momentum proxy, not the production LLM. Its unfiltered
  precision is ~0.50-0.52, so no pipeline variant demonstrates an edge here.
- The 1 % / h1 label is rare (1 % BTC, 4 % ETH). Brier and reliability are
  dominated by this rare-event base rate; noul could still be useful for a
  different tradeability definition or horizon.
- Coverage uses noul alone; trade count and false holds additionally apply the
  bias-none and bias-conflict rules of the production pre-gate.
- No LLM decisions were run in this calibration; rerun with `--base llm` style
  plumbing only after an LLM baseline is available.
