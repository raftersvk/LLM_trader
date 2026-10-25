# JEV post-gate veto impact report

Read-only measurement of the JEV **post** gate as validator of `risk_approved`, the last stop before an approved BUY/SELL can execute.

## Sample

- Signals directory: `<signals-dir>` (read-only)
- Cost assumption: 0.10% per round trip
- Horizons: 1h, 2h
- Target sample: 3 BUY/SELL signals

| Item | Value |
| --- | ---: |
| JSONL files read | 1 |
| Lines read | 4 |
| BUY/SELL signals collected | 3 |
| HOLD signals ignored | 1 |
| Malformed lines skipped | 0 |
| Duplicate decisions skipped | 0 |

## Signal counts per symbol

| Symbol | BUY/SELL | Approved | Vetoed | Tradeability veto | Bias veto | Other veto | Other rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT | 3 | 1 | 2 | 1 | 1 | 0 | 0 |

## Counterfactual P&L, all symbols (net of cost)

Avoided net is the P&L the vetoed trades would have produced: negative means the veto saved money. False-veto cost is the sum of the vetoed trades that would have been profitable.

| Horizon | Approved evaluated | Approved net | Vetoed evaluated | Avoided net | False vetoes | False-veto cost | Loss avoided |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1h | 1 | +2.99% | 2 | +2.69% | 1 | +4.90% | -2.21% |
| 2h | 1 | +7.12% | 2 | -2.46% | 1 | +2.90% | -5.36% |

## P&L by symbol and horizon

| Symbol | Horizon | Approved net | Vetoed evaluated | Avoided net | False vetoes | False-veto cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT | 1h | +2.99% | 2 | +2.69% | 1 | +4.90% |
| BTC/USDT | 2h | +7.12% | 2 | -2.46% | 1 | +2.90% |

## Worst false vetoes

Vetoed signals that would have been profitable, largest net P&L first.

| Symbol | Action | Candle open (UTC) | Horizon | Net P&L | Veto kind | noul |
| --- | --- | --- | ---: | ---: | --- | ---: |
| BTC/USDT | SELL | 2026-01-01 00:00 | 1h | +4.90% | tradeability | 0.200 |

## Verdict on the sample

Collected **3** BUY/SELL signals against a target of **3**. The sample is sufficient to judge the validator.

Decision rule per horizon:

- 1h: **NOT VALIDATED** (fewer than 20 vetoed signals with candles); avoided +2.69%, false-veto cost +4.90%.
- 2h: **NOT VALIDATED** (fewer than 20 vetoed signals with candles); avoided -2.46%, false-veto cost +2.90%.

A veto role is validated only if (a) net avoided P&L is clearly negative over a sufficient sample and (b) false-veto cost stays small; otherwise the validator is not demonstrated.

## Conclusions

- **Sample is at target.** 3 BUY/SELL signals were collected, enough to apply the decision rule, though more data always helps.
- **At 1h the veto cost money.** Across 2 vetoed trades the counterfactual net was +2.69% (negative means the vetoed trades would have made money, so the veto gave up that gain).
- **False vetoes.** 1 of 2 vetoed trades would have been profitable, an opportunity cost of +4.90%.
- **Direction mix.** The sample contains BUY, SELL; results may differ per direction and should be split once the sample is larger.
- **Costs and limits.** P&L subtracts a fixed 0.10% round trip and ignores slippage and funding; rerun weekly until the sample is sufficient and both decision-rule conditions can be checked.
