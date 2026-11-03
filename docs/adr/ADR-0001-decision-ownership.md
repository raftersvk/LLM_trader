# ADR-0001 — Propriété des décisions

- Statut : accepté pour la préparation v2, drapeaux par défaut `false`
- Date : 2026-11-02 ; décision SL/TP C confirmée le 2026-11-03
- Portée : régime, direction, SL/TP, taille

## Contexte

La v2 sépare la décision qualitative du LLM des paramètres déterministes. Cette ADR fixe l'autorité de chaque décision et les preuves disponibles. Les preuves viennent des rapports d'analyse JEV et des backtests SL/TP.

La décision humaine du 2026-11-03 retient **définitivement la politique C** pour les cibles : le LLM propose les niveaux absolus de stop-loss et de take-profit, et le code les valide.

## Décision

| Décision | Autorité | Preuve / règle | Statut |
| --- | --- | --- | --- |
| RÉGIME | **P0-A déterministe** (seuils de la taxonomie) | JEV n'est pas un arbitre indépendant : accord seulement **14,4 % sur 111 BUY/SELL** ; précision directionnelle JEV **56,8 % en h1**, IC incluant 50 %. | Acté |
| DIRECTION | **LLM propose, JEV applique un veto de biais** | Veto short **et** long, uniquement si le biais est explicitement contraire. `none` = neutre, pas de veto. AUC directionnelle ~**0,53** ; aucun gain prouvé. | En évaluation |
| SL/TP | **LLM propose les niveaux absolus, le code VALIDE (politique C)** | A (multiples ATR taxonomie) : **-47,41 $ / -2,50 %** par trade. C : **-19,81 $ / -1,09 %**. B (structure) : **-41,09 $**. Sur **41 entrées communes**, B et C sont équivalentes et toutes deux significativement meilleures que A (**+0,43 pt**). A est **abandonnée**. B n'est pas retenue : une seule source de niveaux est plus simple. | Acté (C) |
| TAILLE | **Risque / volatilité** : `risque% / distance au SL%`, plafonnée par `max_stake` | Cible déterministe, indépendante du `position_size` LLM. | En évaluation, non implémentée |

## Politique C — le LLM propose, le code valide

1. Les champs absolus `stop_loss` et `take_profit` de la réponse LLM sont **l'unique source des niveaux de cible**.
2. Le layer déterministe ne les remplace **jamais** par des multiples ATR de la taxonomie : c'était la politique A, abandonnée.
3. La validation est **fail-closed** :
   - prix d'entrée présent et strictement positif ;
   - côté correct : `stop_loss < entry < take_profit` pour un BUY, `take_profit < entry < stop_loss` pour un SELL ;
   - R/R supérieur ou égal au minimum configuré ;
   - valeurs finies et strictement positives ;
   - **pas de plafond absolu de 10 % sur les niveaux LLM**. La distance du stop doit être dans une bande **relative à la volatilité** : `min_sl_atr_multiple × ATR <= sl_distance <= max_sl_atr_multiple × ATR` (défauts **0,25x** et **12x**, configurables dans le bloc `costs`/risk de la taxonomie et dans `[costs]` de `config.ini`). Hors bande → rejet `implausible_stop`, sans clamp ni ajustement silencieux ;
   - l'ATR doit être disponible et positif, sinon rejet `atr_unavailable`.
4. La borne absolue historique **0,5 %–10 %** ne s'applique plus qu'au **chemin de repli ATR/déterministe**, qui synthétise des niveaux ATR ; elle ne s'applique jamais aux niveaux fournis par le LLM. Ce plafond de 10 % était un circuit breaker legacy copié du calcul ATR, pas une règle adaptée à des stops structurels larges (haute volatilité, range large).
5. Un stop plus large n'est pas automatiquement dangereux : la taille est calculée par `risque% / distance au SL%`, donc un stop de 15 % produit simplement une position plus petite. La porte de frais protège l'économie de l'entrée.
6. Si la validation échoue, le signal est **REJETÉ pour exécution** : `risk_approved=false` avec une raison explicite. Les niveaux proposés peuvent être conservés pour l'audit, mais ils ne sont **jamais** remplacés silencieusement par une cible ATR.
7. Le module de géométrie `tp_mode`/`structure` (`src/llm_focus/target_geometry.py`) ne sert plus que de **repli explicitement documenté** pour un signal qui ne porte aucun niveau (cas attendu inexistant). Il n'est jamais consulté quand le LLM a fourni des niveaux.
8. Tout reste derrière les drapeaux v2 désactivés par défaut : aucun changement du comportement legacy.

## Preuves et limites

- Les trois politiques SL/TP restent **négatives**. C est retenue comme **la moins mauvaise**, pas comme une preuve d'edge validé.
- L'échantillon est **mince** : **35 à 44 trades**, environ **9 jours**, **spot uniquement**. Les conclusions statistiques restent fragiles.
- A est **abandonnée**. B (structure) a été testée et est statistiquement équivalente à C, mais n'est pas choisie : une seule source de niveaux (le LLM) est plus simple à opérer et à auditer.

## Frais — configuration et porte d'entrée

Les frais sont une **configuration partagée** et un **critère de validité** de la position, pas seulement une hypothèse de backtest.

```yaml
costs:
  venue: futures        # spot | futures
  tier: 1               # Kraken tier
  entry_order_type: taker   # taker (market) | maker (limit)
  maker_pct: 0.02
  taker_pct: 0.05
  min_edge_multiple: 1.5    # expected move >= 1.5x round-trip cost
  min_sl_atr_multiple: 0.25 # LLM stop plausibility floor (ATR multiples)
  max_sl_atr_multiple: 12.0 # LLM stop plausibility ceiling (relative, not %)
```

- **Précédence** : `config.ini` > taxonomie YAML > défauts intégrés. Le bloc `[costs]` de `config.ini` existe parce que venue/tier sont propres au compte. Les valeurs effectives sont loguées au démarrage (`Effective costs: ...`).
- Les nombres Kraken viennent de `src/llm_focus/fees.py` (spot T1 0,40/0,80 ; T2 0,30/0,60 ; T3 0,22/0,38 ; stablecoin 0,20/0,20 ; futures T1 0,02/0,05). Le YAML et `config.ini` les surchargent.
- La porte d'entrée, appliquée **après** les validations de C (côté, 0,5–10 %, entrée présente), exige les deux conditions :
  - `tp_distance_pct >= min_edge_multiple * round_trip` sinon `fees_exceed_edge` ;
  - `(tp_distance_pct - round_trip) / (sl_distance_pct + round_trip) >= min_rr` sinon `fees_break_rr`.
- La sortie est supposée **taker** sauf sortie maker documentée. Un échec est explicite (`risk_approved=false`) et les niveaux ne sont jamais remplacés.

### Chiffres historiques

La comparaison C-vs-A initiale utilisait un coût de **0,10 % aller-retour**, ce qui correspond aux frais **futures** Kraken Tier 1 (taker 0,05 % par côté). Ce n'est pas l'hypothèse spot.

- **Spot Tier 1** : taker **0,80 % par côté**, soit **1,60 % aller-retour**. Le résultat C net publié (-1,09 % par trade) intègre 0,10 % de coût, donc le brut est d'environ **-0,99 %**. Avec 1,60 % de coût, C devient environ **-2,59 % par trade** (environ **-47 $** par trade au notionnel implicite d'environ 1 817 $). **À ces frais spot, aucune politique intra-horaire ne survit** : A, B et C sont toutes largement négatives. Le spot est donc **disqualifié** pour cette stratégie.
- **Futures Tier 1** : taker **0,05 % par côté**, soit **0,10 % aller-retour**. C'est l'hypothèse pertinente ; C reste alors négative (voir ci-dessous).

### Recomputation de la porte sur les signaux archivés (futures T1)

Même jeu de **111 signaux BUY/SELL archivés** (47 BUY, 64 SELL, 2026-10-24 → 2026-11-02), mêmes bougies Binance 1h, aucun appel LLM :

- **Règle 1,5x seule** (`tp_distance >= 0,15 %`) : **111/111 passent**, 0 rejet `fees_exceed_edge`.
- **Porte complète** (1,5x + R/R ajusté aux frais, `min_rr = 1,5`) : **84/111 passent, 27 rejetés**. Répartition : **27 `fees_break_rr`**, **0 `fees_exceed_edge`**.
- Après application des autres validations C (porte d'entrée, R/R, distance), le chemin C avec porte de frais prend **79 trades**, dont **73 clôturés** et 6 ouverts en fin de données. PnL net **-21,14 $**, moyenne **-0,53 %/trade** (futures, 0,10 % aller-retour). La même replay C sans porte de frais sur ces 111 signaux donne **-25,13 $** sur 88 clôturés : la porte retire 15 trades exécutables (**tous `fees_break_rr`**) et réduit la perte en dollars, sans rendre la stratégie positive.
- Conclusion inchangée : C est la moins mauvaise **sous frais futures**, pas une stratégie rentable en spot Tier 1. L'échantillon reste minuscule et négatif.
- Le tooling C-vs-A n'est **pas** dans ce dépôt ; la recomputation ci-dessus a réutilisé le cache/moteur hors dépôt de l'étude initiale. Le résolveur de coûts est paramétré par venue/marché/tier via `src/llm_focus/fees.py`.

## Critère de bascule du régime

P0-A reste l'autorité tant que **l'accord indépendant JEV/P0-A est < 70 %** ou que la **borne basse de l'IC95 % de la précision directionnelle JEV est ≤ 50 %**. Une bascule vers JEV exigerait simultanément un accord ≥ 70 % et une borne basse IC95 % > 50 %. Les mesures actuelles (14,4 % d'accord, IC incluant 50 %) ne satisfont aucun des deux critères.

## Conséquences

- Aucun auto-rapport de régime/HTF dans la sortie v2 : P0-A le fournit.
- Le LLM fournit désormais les **niveaux absolus SL/TP** ; le code les valide et rejette le signal si invalides. La taille reste déterministe.
- JEV reste un veto de biais explicite, avec `none` neutre ; son apport directionnel reste à démontrer.
- **Suivi cross-repo trading-stack (non implémenté ici)** : `SignalStrategy` résout actuellement `sl_atr`/`tp_atr` avec la précédence **pair > setup > global** et écrase donc les niveaux absolus du signal pour les setups qui définissent des multiples ATR. Pour que C soit effectif en production, cette précédence doit changer : **les niveaux du signal gagnent, la taxonomie ne sert que de repli**. C'est un changement trading-stack avec sa propre revue.

## Reste ouvert

- Paires et levier.
- Horaires de trading.
- Seuil de confiance.
- Implémentation de la taille risque/volatilité.
- Validation d'un edge sur un échantillon plus large et hors spot-only.
