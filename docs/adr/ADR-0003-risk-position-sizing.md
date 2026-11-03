# ADR-0003 — Taille de position risque / volatilité

- Statut : implémenté pour la préparation v2, drapeaux par défaut `false`
- Date : 2026-11-03 ; niveau de risque recommandé à 0.25 % le 2026-11-03
- Portée : niveau de risque (`risk_per_trade_pct`), taille dérivée et exposition

## Contexte

ADR-0001 fixe la cible : la taille ne doit pas dépendre de la confiance auto-déclarée
du LLM, mais d'un budget de risque et de la distance au stop validé (politique C).

Le **levier de réglage est le niveau de risque**, pas un plafond de taille. Une
première version appliquait un plafond global de 10 % par trade : sur **111 signaux
BUY/SELL, 108 étaient plafonnés à 10 %** et seulement 3 ne l'étaient pas. La formule
risque/volatilité dégénérait donc en taille fixe de 10 %. Ce plafond a été retiré :
la taille est une **conséquence** du risque choisi et du stop, et l'exposition
globale est bornée par un garde-fou de portefeuille (filet de sécurité, pas le
levier de réglage).

## Décision

```text
size_pct = risk_per_trade_pct / sl_distance_pct
```

`risk_per_trade_pct` et `sl_distance_pct` sont en pourcentage. La taille est une
fraction du capital. Le défaut recommandé est **`risk_per_trade_pct = 0.25 %`**.

### Table de dimensionnement (le risque est le levier)

| Risque par trade | Stop 2 % | Stop 1 % | Stop 4 % |
| --- | --- | --- | --- |
| **0.25 % (défaut recommandé)** | **12.5 %** | **25 %** | **6.25 %** |
| 0.50 % | 25 % | 50 % | 12.5 % |

Le même tableau figure dans `docs/regime-taxonomy-config.md` et
`config/config.ini.example`. Un stop plus large réduit mécaniquement la taille ;
on ne plafonne pas la taille, on baisse le risque si l'exposition devient trop
grande.

### Overrides par paire et par régime

- **Par paire** : `risk.pair_overrides` dans le bloc `risk` de la taxonomie,
  p. ex. `BEAMX/USDT: 0.10` ou `0.15` pour une paire volatile/illiquide. L'override
  de paire gagne sur le défaut global. Les clés acceptent `BEAMX/USDT` et
  `BEAMX/USDT:USDT`.
- **Par régime** : `risk.regime_risk_multipliers`, appliqué au risque effectif
  (après l'override de paire). Défaut vide = multiplicateur `1.0` neutre. Exemple :

```yaml
risk:
  risk_per_trade_pct: 0.25
  pair_overrides:
    BEAMX/USDT: 0.10
    TRX/USDT: 0.15
  regime_risk_multipliers:
    high_vol: 0.5
    low_vol: 1.0
  min_size_pct: 0.1
  max_leverage: 1.0
  max_portfolio_exposure_pct: 100.0
  exposure_mode: scale_down
```

Le nombre de positions concurrentes n'est pas réglé par le risque : il est borné
par le garde-fou d'exposition portefeuille.

### Freins réels (ordre d'application)

1. **Aucun plafond de taille par trade par défaut.** Un `max_stake` absolu
   optionnel par paire (`pair_overrides.max_stake`, en devise de cotation) reste
   honoré comme garde-fou de sécurité s'il est configuré :
   `pair_max_stake / capital`.
2. **Garde-fou d'exposition portefeuille** (`max_portfolio_exposure_pct`) : la
   somme des tailles ouvertes ne doit pas dépasser ce plafond (défaut **100 % sur
   spot**, **200 % sur futures**). Si un nouveau signal dépasserait :
   - `exposure_mode = scale_down` (défaut) : taille ramenée à la place restante,
     avec une ligne de log `Portfolio exposure scale-down: ...` ;
   - `exposure_mode = reject` : refus avec `portfolio_exposure_exceeded`.
3. **Limites de compte par trade** : spot refuse au-delà de **100 % du capital**
   (`spot_exposure_exceeded`) ; futures refuse au-delà de `max_leverage`
   (`leverage_exceeded`).
4. **Plancher** `min_size_pct` (défaut **0.1 %**) : en dessous, le trade est refusé
   avec `size_below_min` ; jamais d'arrondi au plancher.
5. Distance de stop nulle, négative ou non finie → `invalid_stop_distance`, jamais
   de taille infinie.

Ordre : distance valide → risque effectif (paire × régime) → `risque / distance`
→ `max_stake` de paire optionnel → limite de compte → exposition portefeuille →
plancher.

### Configuration et précédence

Le bloc `risk:` de la taxonomie porte `risk_per_trade_pct`, `min_size_pct`,
`max_leverage`, `max_portfolio_exposure_pct`, `exposure_mode`, `pair_overrides`
et `regime_risk_multipliers`. `[risk_sizing]` de `config.ini` surcharge les clés
plates. Précédence : **config.ini > taxonomie > défauts intégrés**. Les valeurs
effectives sont loguées au démarrage (`Effective risk sizing: ...`). Le flag v2
désactivé conserve la taille par confiance : aucun changement legacy.

### API

- `src/config/regime_taxonomy.py` : `RiskSettings`, parsing du bloc `risk`,
  `resolve_risk_config()`, `format_risk_summary()`, résolution
  `risk_for_symbol()` / `regime_multiplier()` / `effective_risk_per_trade_pct()`.
- `src/llm_focus/risk_sizing.py` : `compute_risk_based_size()` pur, avec
  `PositionSizeDecision` (`size_pct`, `requested_size_pct`, `risk_per_trade_pct`
  effectif, `base_risk_per_trade_pct`, `regime_multiplier`, `reason`, `capped_by`,
  `leverage`, exposition courante/résultante, risque en % et en devise).
- `src/llm_focus/deterministic_risk.py` : `build_risk_plan(..., risk_sizing=False,
  symbol=None, regime=None, current_exposure_pct=0.0)` conserve la taille par
  confiance quand le flag est absent ; avec `risk_sizing=True`, un plan approuvé
  est redimensionné et un refus de portefeuille/plancher/levier devient
  `risk_approved=false` avec la raison explicite.

## Évaluation sur les signaux archivés (111 BUY/SELL)

Même jeu de **111 signaux BUY/SELL** que les ADR précédentes (47 BUY, 64 SELL,
2026-10-24 → 2026-11-02), entrée = clôture de la bougie Binance 1h du signal,
aucun appel LLM. Capital de référence : **10 000 $**, défauts `min_size_pct = 0.1`,
spot.

### Distribution sans plafond de taille

| Mesure | 0.25 % (recommandé) | 0.50 % (précédent) |
| --- | --- | --- |
| Taille minimale | **3.09 %** | 6.17 % |
| Taille p10 | **15.70 %** | 31.39 % |
| Taille médiane | **23.30 %** | 46.59 % |
| Taille moyenne | **22.05 %** | 44.10 % |
| Taille p90 | **25.40 %** | 50.79 % |
| Taille maximale | **54.24 %** | 108.49 % |
| Tailles > 100 % (limite spot) | **0 / 111** | 1 / 111 |
| Risque moyen par trade | **25.00 $** | 50.00 $ |
| Risque médian par trade | **25.00 $** | 50.00 $ |
| Somme du risque sur 111 trades | **2 775.00 $** | 5 550.00 $ |

Le risque divise exactement les tailles par deux ; la distribution n'est plus
écrasée et aucune taille ne touche la limite spot.

### Exposition si toutes les positions restaient ouvertes (borne supérieure)

Scénario **irréaliste** : les 111 signaux sont exécutés et aucune position ne se
ferme jamais ; c'est une borne supérieure, pas un modèle de concurrence.

| Mesure | 0.25 % | 0.50 % |
| --- | --- | --- |
| Somme brute des tailles | **2 447.65 %** | 4 895.29 % |
| Notionnel sur 10 000 $ | **244 764.73 $** (~24.5×) | 489 529.46 $ (~49×) |
| Spot : positions avant plafond 100 % | **5** (4 pleines + 1 réduite) | 3 (2 pleines + 1 réduite) |
| Spot : signaux sans place ensuite | **106** | 107 |
| Spot : risque moyen des trades approuvés | **22.67 $** | 40.50 $ |
| Futures 200 % : positions avant plafond | **9** (8 pleines + 1 réduite) | 5 (4 pleines + 1 réduite) |
| Futures : signaux sans place ensuite | **102** | 106 |
| Futures : risque moyen des trades approuvés | **24.77 $** | 45.34 $ |

Dans ce modèle sans fermeture, **exactement 1 position est réduite** au moment
où le plafond est franchi ; les suivantes sont refusées faute de place. Le
garde-fou borne donc l'exposition totale à 100 % (spot) / 200 % (futures), mais
le nombre réel de réductions dépend de la concurrence et des fermetures.

## Statut d'évaluation

Le mode reste **derrière les drapeaux v2 désactivés** et n'est pas câblé dans le
service live. Aucune preuve de PnL n'est disponible : cette ADR ne prétend pas
améliorer le PnL par trade. L'effet attendu porte sur la **cohérence du risque**
(risque par trade borné et indépendant du stop et de la confiance) et sur
l'**exposition agrégée**, pas sur le rendement par trade.

## Questions ouvertes

- Le défaut 0.25 % et les overrides 0.10-0.15 % pour les paires volatiles sont-ils
  les bons niveaux sur un échantillon plus large ?
- Le plafond d'exposition (100 % spot / 200 % futures) doit-il dépendre du nombre
  de paires actives et de leur corrélation ?
- Le mode `scale_down` par défaut est-il préférable à `reject` quand le portefeuille
  est déjà chargé ? Faut-il une priorité/ancienneté de signal ?
- Le champ canonique `pair_overrides.max_stake` reste un montant absolu en devise
  de cotation ; conserver ce garde-fou optionnel ou le remplacer par un plafond de
  risque par paire ?
- Le plancher `min_size_pct` doit-il être exprimé en pourcentage du capital ou en
  notionnel minimal d'exchange (min notional) ?
- Le sizing risque/volatilité n'a pas encore été évalué en futures réels.

## Vérification

- Tests ciblés : `tests/test_llm_focus_risk_sizing.py`,
  `tests/test_llm_focus_deterministic_risk.py`,
  `tests/test_regime_taxonomy_config.py`.
- Rapport archivé : script hors dépôt, bougies Binance 1h, aucun appel LLM ;
  les chiffres ci-dessus reproduisent le snapshot de 111 signaux.
- `ruff` propre sur les fichiers ajoutés/modifiés ; `src/config/loader.py` garde
  ses erreurs baseline de `multi-pair-trading`.
