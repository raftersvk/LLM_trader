# LLM-focus v2 preparation

Status: **prepared, default off, not wired into the live service.**
Branch: `feat/llm-focus-v2-prep`.

This document describes the staged v2 architecture: the LLM owns the
qualitative thesis and proposes absolute stop-loss/take-profit levels, while
deterministic code owns regime, position size, validation of those levels and
the decision to spend another LLM call. Every flag defaults to `false`, so the
current v1 behavior is unchanged unless an operator opts in.

## Layered architecture

```
Market data + indicators
        |
        v
P0-A / JEV deterministic context  (regime, allowed setups, ATR/volatility, bias)
        |
        v
Shortlist v2 (flag: shortlist_v2)  -> candidates + explicit rejection reasons
        |
        v
LLM qualitative review (flag: llm_focus_v2)
  action, setup, entry thesis, invalidation, confidence, reasoning/news,
  proposed absolute stop_loss / take_profit
        |
        v
Deterministic risk layer (P0-A/JEV/strategy)
  regime, position size, fail-closed validation of LLM SL/TP
        |
        v
Event-driven position management (flag: event_positions_v2)
  decide when an LLM position review is actually needed
```

### Decision ownership and target geometry

- `docs/adr/ADR-0001-decision-ownership.md` fixes the authority for regime,
  direction, SL/TP and size, with the JEV disagreement/precision evidence, the
  numeric regime switch criterion and the policy-C fee caveat.
- **Policy C (confirmed):** the LLM proposes absolute `stop_loss`/`take_profit`
  levels; the code validates them (correct side, entry present, 0.5-10%
  distance bounds, R/R floor). Invalid levels reject the signal
  (`risk_approved=false` with an explicit reason) and are never replaced by ATR
  targets.
- `docs/adr/ADR-0002-target-geometry.md` keeps optional per-setup `tp_mode`
  (`atr_multiple` | `structure`) and the structure sub-modes (`mid_range`,
  `vwap`, `measured_move`) as an **explicitly documented fallback only** for
  signals that carry no levels at all. It is never used when the LLM supplied
  levels. The canonical taxonomy stays in trading-stack (PR #15).
- `src/llm_focus/target_geometry.py` implements the fallback resolver and the
  Python target validation (correct side, 0.5-10% bounds, configurable R/R).
- `src/llm_focus/fees.py` holds the Kraken reference fees (source and date in
  the module), uses taker by default and exposes venue/tier/liquidity
  sensitivity. Spot Tier 1 is **1.60% round trip** (0.80% + 0.80%), which
  disqualifies most intra-hour strategies. Earlier backtests at 0.05/0.10/0.20%
  materially understated spot costs. Futures Tier 1 is 0.02/0.05%, stablecoin/FX
  is 0.20/0.20%.

### Phase 1: v2 prompt/schema (`llm_focus_v2`)

- New pure module: `src/llm_focus/prompts.py`.
- The LLM output keeps `action`, `setup`, `entry_thesis`, `invalidation`,
  `confidence`, `reasoning`, proposed absolute `stop_loss`/`take_profit` and
  optional `news`.
- The LLM no longer self-reports market regime/higher-timeframe bias or
  position size.
- `to_v1_compatible()` maps `action` to the legacy `signal` field and carries
  the proposed levels across for execution/validation.
- Deterministic ownership lives in `src/llm_focus/deterministic_risk.py`:
  regime normalization, allowed setups, confidence-based size, and fail-closed
  validation of the LLM levels. The geometry/ATR path is a documented fallback
  used only when no levels are present; it never overrides the LLM.
- Integration point: `TemplateManager` branches to the v2 system prompt,
  analysis steps and response template only when the flag is on.

### Phase 2: recall-oriented shortlist (`shortlist_v2`)

- New pure module: `src/llm_focus/shortlist.py`.
- `build_shortlist()` consumes P0-A regime/allowed setups, ATR/volatility, open
  positions and JEV bias/tradeability.
- Hard filters produce explicit rejection reasons:
  `regime_unknown`, `no_allowed_setups`, `setup_not_allowed`, `open_position`,
  `atr_unavailable`, `atr_too_low`, `atr_too_high`.
- JEV is **advisory only**: `jev_not_tradeable` and `jev_bias_mismatch` are
  attached to accepted candidates; JEV never silently removes a signal.
- `maybe_build_shortlist()` returns `None` when the flag is off, which is the
  isolation boundary for the legacy full-universe path.
- Offline A/B: `scripts/ab_shortlist.py` (never calls an LLM).

### Phase 3: event-driven position management (`event_positions_v2`)

- New pure module: `src/llm_focus/position_events.py`.
- Triggers: `regime_flip`, `near_stop_loss`, `near_take_profit`, `atr_shock`,
  `stale_position`.
- The existing strategy trailing/TP/TTL logic remains the default. The module
  only decides whether an LLM position review is worth requesting.
- `maybe_review_position()` returns `None` when the flag is off.

## Flags

Add these to `[llm_focus]` in `config.ini` (see `config/config.ini.example`).
All default to false.

| Flag | Default | Effect |
| --- | --- | --- |
| `llm_focus_v2` | `false` | Use the v2 prompt/schema; LLM emits absolute SL/TP, code validates; LLM stops emitting regime/size. |
| `shortlist_v2` | `false` | Build the recall-oriented pre-LLM shortlist. |
| `event_positions_v2` | `false` | Trigger LLM position reviews only on meaningful events. |

The loader exposes `LLM_FOCUS_V2`, `SHORTLIST_V2`, `EVENT_POSITIONS_V2`.
`src/llm_focus/flags.py` also accepts mappings and test doubles, and treats
unknown values as off.

## Rollout order

1. **Schema first (`llm_focus_v2`).** Validate that v2 responses parse, map to
   the v1 `signal` field, and that deterministic risk validates the LLM SL/TP
   (valid levels pass verbatim, invalid levels are rejected, never substituted)
   and produces a sane size. Do not enable shortlist or event positions yet.
2. **Shortlist A/B (`shortlist_v2`).** Run `scripts/ab_shortlist.py` against
   archived signals/backtest data. Require the recall guardrail to pass before
   using the shortlist to skip any LLM calls.
3. **Event positions (`event_positions_v2`).** Start in shadow mode: compute
   review decisions but keep executing the current strategy-owned trailing/TP/
   TTL path. Compare trigger frequency and review cost against the current
   position-review cadence before acting on triggers.

## A/B metrics (offline)

The A/B script compares the current full-universe path with the shortlist:

- **Recall**: archived BUY/SELL reference signals captured by the shortlist.
- **Missed candidates**: reference signals not shortlisted, with rejection
  reasons.
- **LLM calls saved**: full-universe calls minus shortlist calls, per cycle.
- **Guardrail**: recall must be at least `--min-recall` (default 95%).
- **Rejection reason counts**: audit of why the universe shrank.

Example command:

```bash
python scripts/ab_shortlist.py \
  --dataset data/archived_signals.jsonl \
  --full-pairs 10 \
  --min-recall 0.95 \
  --json-out /tmp/ab_shortlist.json \
  --fail-below-guardrail
```

The script is offline: it imports only the pure shortlist helpers and never
constructs a model manager. `--full-pairs 10` (the default) models the current
10-pair path; pass another value for a different universe size.

## Risks

- **Recall loss**: a hard filter can drop a valid signal. The recall guardrail
  and explicit missed-with-reasons output are the mitigation; JEV stays
  advisory so it cannot silently hide signals.
- **Regime/setup drift**: P0-A inputs may be stale or incomplete. Unknown regime
  and missing ATR fail closed (rejected) rather than guessing.
- **Prompt regression**: v2 changes the output contract. Existing consumers must
  keep using `to_v1_compatible()` / `action` fallback until introspection is
  migrated.
- **Trigger noise**: ATR shocks and stale positions can over-trigger. Run event
  positions in shadow mode first and tune thresholds from observed frequency.
- **Parameter duplication**: deterministic risk mirrors the live RiskManager
  bounds. Keep both in sync or extract shared constants before wiring v2 live.
- **Proposed-level quality**: policy C trusts the LLM to propose usable levels;
  the validator only rejects bad geometry. Weak levels still pass if they are
  internally coherent, which is why the ADR evidence remains "least bad".
- **Venue costs**: the C-vs-A comparison used 0.10% round-trip (futures-like).
  At spot Tier 1 (1.60% round trip) no intra-hour policy is profitable; see
  ADR-0001.
- **Default-off discipline**: no live wiring is included in this preparation
  branch; enabling a flag alone must not alter behavior until the explicit
  integration step is reviewed.

## Next test plan

1. Run the focused unit suites:
   `pytest tests/test_llm_focus_flags.py tests/test_llm_focus_prompts.py tests/test_llm_focus_deterministic_risk.py tests/test_shortlist.py tests/test_position_events.py tests/test_ab_shortlist.py`
2. Run the signal-service/template suites:
   `pytest tests/test_template_manager.py tests/test_context_builder.py`.
3. Capture a baseline full-suite run and track known environment/baseline
   failures separately from v2 regressions.
4. Collect 30-60 days of archived signals with regime/setup/ATR labels, run the
   A/B at 10 pairs, and require >=95% recall with reported missed reasons.
5. Shadow-run event triggers for one release cycle before letting them schedule
   LLM reviews.
6. Only then add a guarded live wiring PR (schema -> shortlist -> event
   positions), with kill switches and a rollback to v1.
