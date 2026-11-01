# External regime/taxonomy configuration

The 1h regime snapshot, the setup taxonomy rendered into the prompt and the
JEV gate policy are driven by one external document. This removes the setup
lists and thresholds that used to be hard-coded in
`src/signal_service/regime_snapshot.py` and
`src/analyzer/prompts/template_manager.py`.

No signal contract, LLM prompt contract or live-bot behaviour changes in this
step: the built-in defaults reproduce the shipped behaviour byte for byte.

## Location and overriding

| Priority | Source |
| --- | --- |
| 1 | `--taxonomy-config <path>` on `python -m src.signal_service run` / `replay` |
| 2 | `$TAXONOMY_CONFIG_PATH` (legacy alias: `$REGIME_TAXONOMY_CONFIG`) |
| 3 | `/home/djelinux/TradingBots/trading-stack/freqtrade/user_data/regime_taxonomy.yaml` |

`config/regime_taxonomy.example.yaml` mirrors the built-in defaults and is the
recommended starting point. YAML is canonical; the loader also accepts JSON
(a `.json` path, or a JSON document when PyYAML is unavailable).

Loader semantics:

- **File absent** (including the canonical default): the built-in defaults
  apply. This is the normal state before an operator deploys a policy file.
- **File present and valid**: the document is validated strictly and overrides
  the defaults section by section.
- **File present and invalid**: `TaxonomyConfigError` is raised with the path
  and the offending field. A typo never silently changes trading policy.

Validation rejects unknown top-level/section keys, wrong types, non-finite
numbers, an inverted ADX pair, an empty `enabled_timeframes`, unknown regimes
and `regime_setups` entries that reference an undefined setup.

## Schema

```yaml
version: "1.0"
enabled_timeframes: [1h]        # 4h stays off until P1-A exit validation

regime_thresholds:
  adx_trend: 25.0               # ADX >= this: directional trend
  adx_weak: 20.0                # ADX < this: range
  range_choppiness: 61.8        # choppiness > this: range
  high_vol_atr_pct_floor: 3.0   # ATR% floor for high volatility
  high_vol_atr_multiple: 2.5    # ATR% > this x median 90 bars: high vol
  low_vol_atr_multiple: 0.55    # ATR% < this x median 90 bars: low vol
  low_vol_choppiness: 55.0      # choppiness > this for low vol

regime_setups:                  # allowed setup families per regime
  trend_up: [trend_pullback, breakout_retest, momentum_continuation]
  trend_down: [rally_failure, breakdown_retest, momentum_short]
  range: [range_fade_extreme, range_mean_reversion]
  low_vol: []                   # empty = no-trade
  high_vol: []
  unclear: []

setups:                         # one definition per setup family
  trend_pullback:
    side: long                  # long | short | both
    description: orderly pullback to SMA20/structure support in an uptrend with a rejection close
    entry: orderly pullback to SMA20/structure support in an uptrend with a rejection close
    invalidation: invalidated below the pullback low
    regimes: [trend_up]
    sl_atr: 2.5
    tp_atr: 4.5
    max_hold_bars: 24
    trail_atr: 0.0              # 0 = no trail by default
    be_at_r: 0.0                # 0 = no break-even move by default

pair_overrides:                 # optional, keyed by "BTC/USDT" or "BTCUSDT"
  BTC/USDT:
    regime_thresholds:
      adx_trend: 30.0
    regime_setups:
      trend_up: [trend_pullback]

jev:
  bias_short_only: true         # post-OOS asymmetric short rule
  log_decisions: false          # append JSONL audit records
  log_path: /home/djelinux/TradingBots/trading-stack/freqtrade/user_data/jev_decisions.jsonl
```

### Sections

- `version`: free-form document version (string or number).
- `enabled_timeframes`: timeframes for which the taxonomy may be applied.
  Only `1h` is enabled by default. The prompt taxonomy and the snapshot's
  `allowed_setups` are empty for any other timeframe.
- `regime_thresholds`: numeric cut-offs consumed by
  `build_regime_snapshot()`. Defaults match the P0-B rules documented in
  `docs/regime-snapshot-1h.md`.
- `regime_setups`: `regime -> [setup names]`. An empty list marks the regime
  as no-trade. Referenced names must exist in `setups`.
- `setups`: per-family `side`, `description`, `entry`, `invalidation`,
  `regimes`, `sl_atr`, `tp_atr`, `max_hold_bars`, `trail_atr`, `be_at_r`.
  `description`/`invalidation` are rendered into the prompt taxonomy;
  `entry` and the numeric bracket fields are carried for downstream exit
  wiring and are not yet enforced by the execution layer.
- `pair_overrides`: optional per-symbol overrides of `regime_thresholds`
  and/or `regime_setups`. The compact `BTCUSDT` key also matches `BTC/USDT`.
- `jev`: `bias_short_only` keeps the shipped post-OOS rule when `true`;
  `log_decisions` appends one JSON record per JEV consultation to `log_path`.

## Wiring

- `src/config/regime_taxonomy.py`: schema, validation, loader, defaults and
  the taxonomy renderer.
- `src/signal_service/regime_snapshot.py`: thresholds and `allowed_setups`
  come from the effective config (`symbol` selects pair overrides); a
  disabled timeframe is no-trade.
- `src/analyzer/prompts/template_manager.py`: the setup taxonomy block and the
  optional JSON taxonomy fields are generated from the config and gated by
  `enabled_timeframes`.
- `src/signal_service/jev_gate.py`: the asymmetric short veto and optional
  decision log come from the `jev` section.
- `src/signal_service/__main__.py`: `--taxonomy-config` loads the document
  once and injects it into every `PairAnalyzer` and the JEV gate.

## Tests

`tests/test_regime_taxonomy_config.py` covers absent/valid/invalid documents,
threshold overrides, pair overrides, rendered descriptions, 1h enabled, 4h
disabled and the JEV policy flags. The example file is asserted to match the
built-in defaults so the two cannot drift.
