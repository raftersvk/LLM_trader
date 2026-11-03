# ADR-0002 — Géométrie des cibles

- Statut : accepté pour la préparation v2, drapeaux par défaut `false`
- Date : 2026-11-02 ; aligné sur la politique C le 2026-11-03
- Portée : niveaux LLM, validation de cible, repli `tp_mode`/structure

## Contexte

La décision humaine du 2026-11-03 retient la **politique C** : le LLM propose les niveaux absolus de stop-loss et de take-profit, et le code les valide. La géométrie `tp_mode`/`structure` n'est donc plus une politique de cible normale ; elle est conservée uniquement comme **repli documenté** pour un signal qui ne porte aucun niveau (cas attendu inexistant).

La taxonomie canonique vit dans **trading-stack (PR #15)** et n'est pas modifiée ici. Certains setups de range ont une description qui contredit le `tp_atr = 4.5` actuel : une cible ATR de 4,5 peut sortir du range au lieu de revenir vers son milieu. Cette remarque ne concerne plus que le repli.

## Décision

1. **Source primaire (C)** : `stop_loss` et `take_profit` absolus du LLM. Ils sont utilisés verbatim après validation.
2. **Jamais de substitution** : si les niveaux LLM sont invalides, le signal est rejeté (`risk_approved=false`, raison explicite). Il n'est pas remplacé par une cible ATR ou structure.
3. **Repli uniquement sans niveaux** : si et seulement si le signal ne porte ni SL ni TP, le résolveur peut produire une cible de repli à partir de la taxonomie. Les niveaux fournis par le LLM ne déclenchent jamais ce chemin.
4. Champ de repli par setup :
   - `tp_mode = atr_multiple | structure`
   - sous-mode `structure = mid_range | vwap | measured_move`

| Setup (repli seulement) | `tp_mode` v2 | Cible | Repli interne |
| --- | --- | --- | --- |
| `range_mean_reversion` | `structure` | `mid_range` en priorité, puis `vwap` | ATR (`tp_atr`) si mesure indisponible ou du mauvais côté |
| `range_fade_extreme` | `structure` | retour dans le range (`mid_range`), puis `vwap` | ATR (`tp_atr`) |
| Setups de tendance | `atr_multiple` | `entry ± tp_atr × ATR` | — |
| `breakout_retest`, `rally_failure`, `breakdown_retest` | pas de `sl_atr`/`tp_atr` | cible de repli ATR générique | — |

`measured_move` est supporté uniquement sur demande explicite ; c'est une **piste non validée**, pas un défaut.

## Règles du repli

1. Le repli n'est calculé que lorsque `stop_loss` et `take_profit` sont tous deux absents. Un seul niveau présent est une erreur qui rejette le signal, sans repli.
2. `tp_mode` absent ou inconnu → `atr_multiple`.
3. `structure` sans sous-mode explicite → `mid_range` puis `vwap` pour les deux setups de range ; sinon repli ATR.
4. Mesure impossible (`range_high <= range_low`, VWAP absent, cible du mauvais côté) → repli ATR documenté.
5. `tp_atr` absent → cible ATR générique du plan de repli, jamais un remplacement d'un niveau LLM.
6. `measured_move` n'est utilisé que si explicitement demandé et fourni ; il reste marqué non validé.

## Validation Python

`validate_target()` vérifie, pour les niveaux LLM comme pour un repli :

- entrée présente, finie et strictement positive ;
- côté correct : LONG `SL < entry < TP`, SHORT `TP < entry < SL` ;
- bornes de distance **0,5 % à 10 %** pour le SL et le TP ;
- R/R minimum paramétrable (1,0 par défaut) ;
- valeurs finies et strictement positives.

Après ces contrôles, la **porte de frais** de l'ADR-0001 s'applique (edge 1,5x et R/R ajusté aux frais, selon `[costs]`/taxonomie). Si une vérification échoue : `risk_approved=false` et raison explicite (`fees_exceed_edge` ou `fees_break_rr` inclus). Aucun niveau n'est corrigé, arrondi ou remplacé pour forcer l'approbation.

## Suivi

- **Cross-repo trading-stack** : `SignalStrategy` résout `sl_atr`/`tp_atr` avec la précédence **pair > setup > global**, ce qui écrase les niveaux absolus du signal. Pour que C soit effectif en production, la précédence doit devenir **niveaux du signal > taxonomie (repli)**. Changement trading-stack avec sa propre revue, non implémenté ici.
- Ajouter `tp_mode`/`tp_submode` à la taxonomie canonique reste un suivi trading-stack, hors de ce dépôt.
- Valider `measured_move` par backtest avant toute activation.
- Implémentation : `src/llm_focus/target_geometry.py` (repli) et `src/llm_focus/deterministic_risk.py` (validation C).
