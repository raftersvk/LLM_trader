# ADR-0002 — Géométrie des cibles

- Statut : accepté pour la préparation v2, drapeaux par défaut `false`
- Date : 2026-11-02
- Portée : `tp_mode` par setup, replis, validation de cible

## Contexte

La taxonomie canonique vit dans **trading-stack (PR #15)** et n'est pas modifiée ici. Certains setups de range ont une description qui contredit le `tp_atr = 4.5` actuel : une cible ATR de 4,5 peut sortir du range au lieu de revenir vers son milieu. La v2 lit un champ optionnel `tp_mode` et retombe sur `atr_multiple` quand il est absent.

## Décision

Nouveau champ par setup :

- `tp_mode = atr_multiple | structure`
- sous-mode `structure = mid_range | vwap | measured_move`

| Setup | `tp_mode` v2 | Cible | Repli |
| --- | --- | --- | --- |
| `range_mean_reversion` | `structure` | `mid_range` en priorité, puis `vwap` | ATR (`tp_atr`) si mesure indisponible ou du mauvais côté |
| `range_fade_extreme` | `structure` | retour dans le range (`mid_range`), puis `vwap` | ATR (`tp_atr`) |
| Setups de tendance | `atr_multiple` | `entry ± tp_atr × ATR` | — |
| `breakout_retest`, `rally_failure`, `breakdown_retest` | pas de `sl_atr`/`tp_atr` | **niveaux du LLM conservés** | — |

`measured_move` est supporté uniquement sur demande explicite ; c'est une **piste non validée**, pas un défaut.

## Règles de repli

1. `tp_mode` absent ou inconnu → `atr_multiple`.
2. `structure` sans sous-mode explicite → `mid_range` puis `vwap` pour les deux setups de range ; sinon repli ATR.
3. Mesure impossible (`range_high <= range_low`, VWAP absent, cible du mauvais côté) → repli ATR documenté.
4. `tp_atr` absent → `effective_mode = llm_level`, cible `None`, le niveau du LLM est conservé.
5. `measured_move` n'est utilisé que si explicitement demandé et fourni ; il reste marqué non validé.

## Validation Python

`validate_target()` vérifie :

- côté correct : LONG `SL < entry < TP`, SHORT `TP < entry < SL` ;
- bornes de distance **0,5 % à 10 %** pour le SL et le TP ;
- R/R minimum paramétrable (1,0 par défaut) ;
- valeurs finies et strictement positives.

## Suivi

- Ajouter `tp_mode`/`tp_submode` à la taxonomie canonique est un **suivi trading-stack**, hors de ce dépôt.
- Valider `measured_move` par backtest avant de l'activer par défaut.
- Implémentation : `src/llm_focus/target_geometry.py`.
