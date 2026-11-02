# ADR-0001 — Propriété des décisions

- Statut : accepté pour la préparation v2, drapeaux par défaut `false`
- Date : 2026-11-02
- Portée : régime, direction, SL/TP, taille

## Contexte

La v2 sépare la décision qualitative du LLM des paramètres déterministes. Cette ADR fixe l'autorité de chaque décision et les preuves disponibles. Les preuves viennent des rapports d'analyse JEV et des backtests SL/TP.

## Décision

| Décision | Autorité | Preuve / règle | Statut |
| --- | --- | --- | --- |
| RÉGIME | **P0-A déterministe** (seuils de la taxonomie) | JEV n'est pas un arbitre indépendant : accord seulement **14,4 % sur 111 BUY/SELL** ; précision directionnelle JEV **56,8 % en h1**, IC incluant 50 %. | Acté |
| DIRECTION | **LLM propose, JEV applique un veto de biais** | Veto short **et** long, uniquement si le biais est explicitement contraire. `none` = neutre, pas de veto. AUC directionnelle ~**0,53** ; aucun gain prouvé. | En évaluation |
| SL/TP | **LLM propose, le code VALIDE (politique C)** | A (multiples ATR taxonomie) : **-47,41 $ / -2,50 %** par trade. C : **-19,81 $ / -1,09 %**. B (structure) : **-41,09 $**. Sur **41 entrées communes**, B et C sont équivalentes et toutes deux significativement meilleures que A (**+0,43 pt**). A est **abandonnée**. | Acté (C) |
| TAILLE | **Risque / volatilité** : `risque% / distance au SL%`, plafonnée par `max_stake` | Cible déterministe, indépendante du `position_size` LLM. | En évaluation, non implémentée |

## Critère de bascule du régime

P0-A reste l'autorité tant que **l'accord indépendant JEV/P0-A est < 70 %** ou que **la borne basse de l'IC95 % de la précision directionnelle JEV est ≤ 50 %**. Une bascule vers JEV exigerait simultanément un accord ≥ 70 % et une borne basse IC95 % > 50 %. Les mesures actuelles (14,4 % d'accord, IC incluant 50 %) ne satisfont aucun des deux critères.

## Conséquences

- Aucun auto-rapport de régime/HTF dans la sortie v2 : P0-A le fournit.
- Le LLM ne fournit ni taille ni SL/TP absolus dans le schéma v2 ; le code valide les cibles proposées.
- Les trois politiques SL/TP restent négatives : C est retenue comme **la moins mauvaise**, pas comme une preuve de rentabilité.
- JEV reste un veto de biais explicite, avec `none` neutre ; son apport directionnel reste à démontrer.

## Reste ouvert

- Paires et levier.
- Horaires de trading.
- Seuil de confiance.
