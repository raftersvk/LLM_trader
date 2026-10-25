# JEV post-gate veto impact report

`scripts/jev_validator_report.py` is a read-only measurement tool for **one**
role of JEV: the `post` gate acting as the last validator of `risk_approved`
before a BUY/SELL signal can reach Freqtrade.

It does not change any production code, threshold or service, and it never
writes into the signals directory.

## What it measures

In `post` mode the signal adapter calls JEV only for BUY and SELL decisions.
When JEV vetoes, the signal keeps its action but `risk_approved` becomes
`false` and the reasoning gets a `[JEV veto: <reason>]` suffix. The veto is
therefore visible after the fact in two ways:

- the persisted `jev` object: `model`, `tradeability`, `bias`, `regime`;
- the veto reason in `reasoning`, which is the only place where the `noul`
  probability appears (e.g. `noul=0.273 < 0.5`). The tool parses `noul` from
  that text when present and never invents it.

For every BUY/SELL signal the tool simulates the trade that would have been
executed: enter at the close of the signal's own candle, in the signal's
direction (BUY = long, SELL = short), exit at the close `h` hours later, and
subtract a fixed round-trip cost. Candle data comes from Binance hourly
candles through ccxt, cached locally.

It then reports, per symbol and horizon (`--horizons`, default 1, 2, 4 hours):

- how many signals were approved, vetoed for tradeability, vetoed for a bias
  conflict, vetoed for another reason, or rejected by a non-JEV risk rule;
- the two numbers that judge the validator (below);
- the worst false vetoes (signals JEV blocked that would have been profitable);
- a plain-language verdict on whether the sample is large enough to judge.

## How to run it weekly

From the repository root, with the project venv:

```bash
.venv/bin/python scripts/jev_validator_report.py \
  --signals-dir /home/djelinux/TradingBots/trading-stack/runtime/signals \
  --cost-pct 0.10 \
  --horizons 1,2,4 \
  --target 200 \
  --out runtime/jev_bench/jev_validator_report.md
```

Weekly routine:

1. Run the command above. The explicit `--signals-dir` points at the live
   stack; without it the tool uses `$LLM_TRADER_SIGNALS_DIR` or the repo-local
   `data/signals`, like the other signal readers. The directory is read-only.
2. Open `runtime/jev_bench/jev_validator_report.md`.
3. Read the sample table first: if `BUY/SELL signals collected` is below
   `--target`, do not draw a conclusion.
4. Compare the avoided-net and false-veto-cost columns at 1h, 2h and 4h.
5. Keep the report with the date of the run if you want a week-over-week
   history; the tool itself is deterministic for a fixed input and candle
   cache.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--signals-dir` | Directory of per-pair `*.jsonl` files (read-only). |
| `--cost-pct` | Round-trip cost in percentage points (default 0.10). |
| `--horizons` | Comma-separated holding periods in **hours** (default `1,2,4`). A horizon is measured in wall-clock hours: on a 15m signal it is four candles, and it is skipped (counted in the report) when it is not a whole number of candles for the signal's timeframe (e.g. 1h on a 4h signal). |
| `--target` | BUY/SELL signals needed for a verdict (default 200). |
| `--out` | Markdown output path. |
| `--candles-cache` | Local candle cache directory (default `runtime/jev_bench/candles_cache`). |

The report and the candle cache are refused if they point inside
`--signals-dir`. The signals directory is only read.

## How to read the two numbers

Both numbers are in percentage points and are computed on vetoed signals only.

**Avoided net P&L** — the counterfactual net P&L of all vetoed signals, summed.

- **Negative** means the vetoed trades would have lost money, so the veto
  **saved** money. More negative is better.
- **Positive** means the veto blocked trades that would have made money, so it
  **cost** money.
- The per-trade round-trip cost is already subtracted, so a "saved" trade is
  credited only if it would have lost more than the cost.

**False-veto cost** — the sum and count of vetoed signals that would have been
profitable. It is always positive (or zero) and is a subset of the avoided
number: the vetoed winners. It measures the opportunity cost of blocking good
trades. The report also lists the largest individual false vetoes.

Example: avoided net = -3.96% with false-veto cost = +2.67% means the vetoed
trades would have lost 3.96% in total; 2.67% of that is missed profit from
vetoed winners, while the rest is losses that were avoided.

## Decision rule

A veto role is **validated only if both conditions hold**:

1. **Net avoided P&L is clearly negative over a sufficient sample.** The tool
   requires at least `--target` BUY/SELL signals and at least 20 vetoed signals
   with matching candles per horizon, and `avoided net < 0`.
2. **False-veto cost stays small.** The report requires the false-veto cost to
   be at most 25% of the avoided losses (`|loss avoided|`). Blocking many
   profitable trades is a red flag even when the net is negative.

If either condition fails, the validator is **not demonstrated**. A small
positive avoided net or a false-veto cost that rivals the avoided losses is
not evidence that the role works; it is a reason to collect more data.

## Limits to keep in mind

- **Sample size.** Around 15-20 BUY/SELL signals is a first reading, not a
  verdict. The live mix also contains many HOLD rows, which the tool ignores.
- **Signal mix.** The LLM signals and any rule-based/momentum signals are
  pooled. A veto can look good simply because the underlying signal was bad in
  that window, not because JEV added information.
- **One direction per window.** If all collected signals are SELL (as in the
  first reading), the report only describes the veto in that market phase.
- **Cost model.** A fixed `--cost-pct` ignores slippage, spread variation and
  funding. It is a simplifying assumption, not an execution simulator.
- **Timeframe alignment.** Horizons are hours, converted to candles for each
  signal's timeframe; a horizon that is not a whole number of candles for that
  timeframe is skipped and counted as `Skipped (not whole candles)` instead of
  being rounded or silently held for a different duration.
- **Signal timestamp.** The entry is the close of the candle stored in
  `candle_open_time`, not the wall-clock time the signal was written. A signal
  produced late for an older candle is still measured on that candle.
- **Missing candles.** Signals whose entry or exit candle is unavailable (for
  example the newest signal whose +4h candle has not closed yet) are counted
  as evaluated only for the horizons that have data; they are never treated as
  zero P&L.

## First reading (2026-10-25)

Against the live directory (BTC/USDT and ETH/USDT, 72 lines, 17 BUY/SELL
signals, all SELL, all vetoed, 55 HOLD ignored):

| Horizon | Vetoed evaluated | Avoided net | False vetoes | False-veto cost |
| --- | ---: | ---: | ---: | ---: |
| 1h | 16 | -1.79% | 5 | +1.06% |
| 2h | 16 | -3.28% | 3 | +1.32% |
| 4h | 13 | -3.96% | 3 | +2.67% |

The direction is consistent with the veto helping, but 17 signals is far below
the 200 target, so the validator is **not demonstrated**. The 4h false-veto
cost (+2.67%) is also close to the avoided loss, which is exactly the kind of
number that needs a larger sample before it can be trusted.
