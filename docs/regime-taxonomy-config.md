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

The canonical document is owned by the trading-stack repository
(`freqtrade/user_data/regime_taxonomy.yaml`, PR #15). This repository must not
ship a second canonical YAML: the loader reads that single file. YAML is
canonical; the loader also accepts JSON (a `.json` path, or a JSON document
when PyYAML is unavailable).

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
version: 1
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

pair_overrides: {}              # optional flat execution overrides (W3 schema)
# BTC/USDT:USDT:
#   sl_atr: 3.0
#   tp_atr: 5.5
#   max_hold_bars: 12
#   max_hold_hours: 6
#   max_stake: 30.0
#   leverage: 2
#   trail_atr: 0.5
#   be_at_r: 1.0
#   trail_after_r: 1.5

jev:
  bias_short_only: false        # W3 canonical: short veto disabled
  log_decisions: true           # append JSONL audit records
  log_path: runtime/logs/jev_decisions.jsonl
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
  wiring and are not yet enforced by the execution layer. Setup names are
  restricted to `ALLOWED_SETUPS` in the signal contract, so the YAML can never
  announce a setup the contract cannot carry.
- `pair_overrides`: optional flat per-pair execution overrides using the W3
  field set (`sl_atr`, `tp_atr`, `max_hold_bars`, `max_hold_hours`,
  `max_stake`, `leverage`, `trail_atr`, `be_at_r`, `trail_after_r`). These are
  consumed by the Freqtrade strategy, not by the regime snapshot. Both
  `BTC/USDT:USDT` and `BTC/USDT` keys resolve to the signal symbol.
- `jev`: the W3 canonical file sets `bias_short_only: false` and
  `log_decisions: true`. `bias_short_only: true` keeps the shipped post-OOS
  short veto (the built-in fallback value when no file is present);
  `log_decisions` appends one JSON record per JEV consultation to `log_path`.

## Wiring

- `src/config/regime_taxonomy.py`: schema, validation, loader, defaults and
  the taxonomy renderer.
- `src/signal_service/regime_snapshot.py`: thresholds and `allowed_setups`
  come from the config; a disabled timeframe is no-trade.
- `src/analyzer/prompts/template_manager.py`: the setup taxonomy block and the
  optional JSON taxonomy fields are generated from the config and gated by
  `enabled_timeframes`. When a taxonomy config is loaded, the prompt metadata
  appends `+taxonomy.<version>` to `prompt_version`.
- `src/signal_service/jev_gate.py`: the asymmetric short veto and optional
  decision log come from the `jev` section.
- `src/signal_service/analyzer_adapter.py`: `SIGNAL_TAXONOMY_OUTPUT_CONTRACT`
  is rebuilt from the same config and gated by `enabled_timeframes`, so the
  output contract and the prompt template enable/disable a timeframe together.
  Its setup list only names setups present in the effective `regime_setups`.
- `src/signal_service/__main__.py`: `--taxonomy-config` loads the document
  once and injects it into every `PairAnalyzer` and the JEV gate.

## Tests

`tests/test_regime_taxonomy_config.py` covers absent/valid/invalid documents,
threshold overrides, the W3 flat pair overrides, rendered descriptions, 1h
enabled, 4h disabled and the JEV policy flags. It embeds the W3 canonical
document for CI and additionally validates the installed canonical file when
the sibling trading-stack checkout is present.
