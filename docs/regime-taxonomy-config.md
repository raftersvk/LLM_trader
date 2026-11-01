# External regime/taxonomy configuration

The 1h regime snapshot, the setup taxonomy rendered into the prompt and the
JEV gate policy are driven by one external document. This removes the setup
lists and thresholds that used to be hard-coded in
`src/signal_service/regime_snapshot.py` and
`src/analyzer/prompts/template_manager.py`.

No signal contract schema change. The built-in defaults mirror the canonical
document, so an absent file behaves exactly like the deployed policy; the live
bot's non-taxonomy prompt path is unchanged.

## Location and overriding

| Priority | Source |
| --- | --- |
| 1 | `--taxonomy-config <path>` on `python -m src.signal_service run` / `replay` |
| 2 | `$TAXONOMY_CONFIG_PATH` (legacy alias: `$REGIME_TAXONOMY_CONFIG`) |
| 3 | `/home/djelinux/TradingBots/trading-stack/freqtrade/user_data/regime_taxonomy.yaml` |

The canonical document is owned by the trading-stack repository
(`freqtrade/user_data/regime_taxonomy.yaml`, PR #15 head `fa1c67f`, sha256
`5035922df4f976f520993fbd85406c78234051c4d7f5fee7046b6a03e9124ca7`). This repository must not ship a second canonical YAML: the
loader reads that single file. YAML is canonical; the loader also accepts JSON
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
    description: Orderly pullback to SMA20/structure support in an uptrend with a rejection close.
    entry: Pullback into SMA20/support with a rejection close.
    invalidation: Close below the pullback low.
    regimes: [trend_up]
    sl_atr: 2.5                 # optional; omit to keep the signal's absolute SL
    tp_atr: 4.8                 # optional; omit to keep the signal's absolute TP
    max_hold_bars: 24
    # trail_atr / be_at_r omitted on purpose: a value would shadow the legacy
    # per-setup trailing config instead of letting it apply.

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
  bias_short_only: true         # W3 canonical: shorts only (P0-B default)
  log_decisions: true           # append one JSONL audit record per analysis
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
  wiring and are not yet enforced by the execution layer. `sl_atr` and
  `tp_atr` are optional: when omitted, the signal's absolute SL/TP is kept.
  `trail_atr` and `be_at_r` are optional and must stay absent unless a value is
  explicitly wanted, because a value would shadow legacy trailing config.
  Setup names are restricted to `ALLOWED_SETUPS` in the signal contract, so the
  YAML can never announce a setup the contract cannot carry.
- `pair_overrides`: optional flat per-pair execution overrides using the W3
  field set (`sl_atr`, `tp_atr`, `max_hold_bars`, `max_hold_hours`,
  `max_stake`, `leverage`, `trail_atr`, `be_at_r`, `trail_after_r`). These are
  consumed by the Freqtrade strategy, not by the regime snapshot. Both
  `BTC/USDT:USDT` and `BTC/USDT` keys resolve to the signal symbol.
- `jev`: the W3 canonical file sets `bias_short_only: true` (shorts only,
  P0-B default) and `log_decisions: true`; set `bias_short_only: false` to
  disable the post-OOS short veto. `log_decisions` appends one structured
  JSONL record per analysis to `log_path`, including runs where the gate is
  disabled (`--jev-mode off`) or never consulted. When `log_path` is omitted
  it defaults to `runtime/logs/jev_decisions.jsonl` under the repository root.

### Decision audit log

Each line describes one analysis and is written by
`src/signal_service/audit_log.py` after the contract signal is built:

```json
{
  "ts": "2026-11-01T06:01:02Z",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "decision_id": "btc-usdt-1h-2026-11-01t05:00:00z",
  "candle_close_time": "2026-11-01T06:00:00Z",
  "signal_action": "SELL",
  "signal_confidence": 0.85,
  "risk_approved": false,
  "jev": {
    "tradeability": 0.82,
    "bias": "long",
    "regime": "trend",
    "bias_confidence": 0.8,
    "regime_confidence": 0.7,
    "tradeability_confidence": 0.66,
    "confidence": 0.75
  },
  "gate": {
    "policy": "bias_short_only",
    "bias_short_only": true,
    "min_confidence": null,
    "used": true,
    "allowed": false,
    "reason": "JEV veto: SELL requires short bias (bias=long)"
  },
  "setup": "trend_pullback",
  "regime": "trend_up",
  "htf_alignment": "aligned"
}
```

- `ts` is the write time; `candle_close_time` is the close of the candle that
  was analysed, so the record never describes a future candle.
- `jev.tradeability` is the raw noul probability. The `*_confidence` fields
  are diagnostics only; the gate policy still filters on `bias` alone.
- `gate.policy` is derived from the gate that actually ran:
  `bias_short_only` when the asymmetric short veto is enabled,
  `bias_filter_disabled` when the W3 taxonomy turns it off, and a
  `+min_confidence` suffix when a JEV confidence threshold is configured.
  `gate.bias_short_only` and `gate.min_confidence` carry the effective flag
  values, and the policy is `not_configured` when JEV is off and no gate
  object exists.
- `gate.used` records whether the gate consulted JEV. It stays false when the
  mode is off or the post-gate exempts a `HOLD`; in `pre`/`pre_post` the
  pre-gate is consulted before the LLM call, so it is true even when the
  final action is `HOLD`.
- `setup`, `regime` and `htf_alignment` are emitted only when the signal
  actually carries them.
- The raw LLM response, prompt, headers and API keys are never written.

Replay logging is opt-in: `replay` writes an audit record only when
`--audit-log PATH` is given, or when both the gate is enabled and an explicit
`--taxonomy-config` (with `log_decisions: true`) is supplied. A replay never
appends to the configured live audit file implicitly.

## Wiring

- `src/config/regime_taxonomy.py`: schema, validation, loader, defaults and
  the taxonomy renderer.
- `src/signal_service/regime_snapshot.py`: thresholds and `allowed_setups`
  come from the config; a disabled timeframe is no-trade.
- `src/analyzer/prompts/template_manager.py`: the setup taxonomy block and the
  optional JSON taxonomy fields are generated from the config and gated by
  `enabled_timeframes`. When a taxonomy config is loaded, the prompt metadata
  appends `+taxonomy.<version>` to `prompt_version`.
- `src/signal_service/jev_gate.py`: the asymmetric short veto comes from the
  `jev` section.
- `src/signal_service/audit_log.py`: builds the per-analysis JSONL record from
  the final signal, the gate result and the `jev` section. The live `run`
  command injects it from the config independently of the gate mode, so an
  audit run with JEV off is still recorded.
- `src/signal_service/analyzer_adapter.py`: `SIGNAL_TAXONOMY_OUTPUT_CONTRACT`
  is rebuilt from the same config and gated by `enabled_timeframes`, so the
  output contract and the prompt template enable/disable a timeframe together.
  Its setup list only names setups present in the effective `regime_setups`.
  The audit writer is injected explicitly and never rebuilt from the taxonomy.
- `src/signal_service/__main__.py`: `--taxonomy-config` loads the document
  once and injects it into every `PairAnalyzer` and the JEV gate. `replay`
  takes `--audit-log PATH` as the explicit opt-in for replay audit records.

## Tests

`tests/test_regime_taxonomy_config.py` covers absent/valid/invalid documents,
threshold overrides, the W3 flat pair overrides, rendered descriptions, 1h
enabled, 4h disabled and the JEV policy flags. It embeds the current W3
canonical document byte for byte and pins its source ref (`fa1c67f`) and
sha256 (`5035922df4f976f520993fbd85406c78234051c4d7f5fee7046b6a03e9124ca7`), so a stale copy cannot silently return.
