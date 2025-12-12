# Dota Data — Roadmap v4 (technique/ML)

## Objectifs et périmètre
- Prédire au niveau map (pré-match, pré-draft) : `win/lose`, `duration`, `first blood`, `first tower`, `first roshan`, combo `fb+ft+fs+win` (forte corrélation exploitable pour paris).
- Sources : tables parquet `data/processed` comme vérité terrain; réutilisation possible des calculs/méthodes `data/metrics` (élos, firsts, buckets, series) pour produire de nouvelles features pré-match; enrichir si besoin mais éviter de dépendre du brut hors parquet.
- Construire une feature store avec Feast (offline parquet, potentielle online store légère si besoin) et tracer toutes les expériences via MLflow (datasets, features, versions de modèle, métriques, artefacts).
- Pipeline entraînement avec splits walk-forward chrono (pas de fuite temporelle), et inférence via une interface (sélection date/heure, teams, side/pick, BO, map_num, lineup par défaut modifiable).

## Design Elo (proposé)
- **Team Elo** : global + patch. Base 1500, side_adv explicite (ex: Radiant +25 Elo ou équivalent offset dans l’expected). K factor décroissant (40/25/15 selon volume), importance capée (major/lan/qualif/online). Patch init = 0.75*global + 0.25*1500; possibilité de shrink si longue inactivité ou roster change massif.
- **Player Elo** : global + patch. Même K tiers mais éventuellement plus rapides au début (40/25/15 avec seuils plus bas). Expected = probabilité de win de l’équipe (donc même E pour les 5), update par joueur ; recency via K ou decay temporel.
- **Player Elo par rôle** : 5 ratings par joueur (1..5). Init role = Elo global joueur; pour un rôle jamais joué, fallback sur global. Partage du delta via α (ex: 0.3 global, 0.7 role). Optionnel : micro-signal par rôle vs counterpart (lane outcome) pour ajuster le delta role.
- **Weights/importance** : grille proposée (capée, multiplié au K moyen du match) : Major/LAN 1.3, Tier1 1.15, Qualif 1.05, Online/Tier2-3 1.0. Nécessite un mapping `leagueid -> tier/lan`. Toujours plafonner (ex: w<=1.3) pour éviter explosion.
- **Recency/decay** : appliquer un facteur de fraîcheur (ex: exp(-Δt/τ) avec τ ~ 90 jours) sur le K ou en shrink périodique vers 1500 si inactivité prolongée.
- **Lineup churn** : si variation de lineup >=2 joueurs depuis la dernière map, shrink partiel du Team Elo vers la moyenne des 5 Elo joueurs (ou vers global 1500) pour éviter de conserver un rating obsolète. Stand-in/alias gérés via mapping + warnings.
- **Outputs features pré-match** (par side) : team Elo global + patch; player Elo global + patch; player Elo role-based + patch (5 rôles, fallback global si non joué); delta patch-global; recency indicators (dernier match, matches joués sur patch).

## Observations récentes (données locales)
- `league` présent dans le brut (`data/raw/data_v2.json`) avec `leagueid`, `name`, `tier` (principalement `professional`, quelques `premium` type TI). Pas de flag LAN/online explicite -> mapping à construire (heuristique nom + table externe).
- Répartition leagueids : 222 ligues distinctes; tiers `professional` majoritaires, `premium` minoritaire.
- Side / first pick (14124 maps avec draft meta) : radiant win rate ≈ 50.95% (~+7 Elo), first pick win rate ≈ 51.26% (~+9 Elo), Radiant first pick ≈ 53.27% (~+23 Elo). Variation par patch : rad WR de ~49% (patch 55) à ~52.4% (patch 57/58); FP WR de ~50.2% (patch 57) à ~52% (patch 54/56). => Fixer un `side_adv` autour de 20-25 Elo pour Radiant+FP, ~5-10 Elo pour FP seul, et recalibrer par patch si besoin.

## État actuel / assets
- Parquets `matches/players/objectives/teamfights` sous `data/processed`; métriques dérivées sous `data/metrics` (elo, firsts, roshan, buckets gold/xp, series, draft_meta, adv_snapshots, pick_outcomes) déjà utilisées dans Streamlit et réutilisables pour features pré-match.
- Picks/bans exclus des parquets (dans le brut) : on reste pré-draft, donc non requis pour le scope actuel; à conserver pour une version post-draft si besoin.
- Alias équipes (`data/team_aliases.csv`) surtout utiles pour ingestion/Streamlit, pas critiques pour le modeling si les parquets sont déjà canonisés.
- Série/BO : `series_id`, `series_type`, `map_num` présents dans `matches.parquet` et déjà exploités dans le dashboard (logique à réutiliser pour l’ordre des maps).

## Cibles (définitions)
- `win/lose` : binaire par side (POV équipe) -> attention aux maps sans vainqueur (abandons?).
- `duration` : régression (minutes) + quantiles/buckets.
- `fb/ft/fs` : binaire par side, cohérence source (players.firstblood_claimed, objectives.building_kill, objectives.CHAT_MESSAGE_ROSHAN_KILL).
- `combo fb+ft+fs+win` : conjonction des 4 events (snowball). Possibilité future de multi-label joint si volume suffisant, mais priorité à la conjonction pour la valeur paris.

## Entités Feast (proposées)
- `team` (team_id), `player` (account_id), `hero` (hero_id).
- Entités composées : `team_lineup` (team_id + hash lineup + event_time), `team_hero` (team_id, hero_id), `player_hero` (account_id, hero_id), `series_map` (series_id + map_num + side).
- Event timestamp = `start_time` (UTC) du match/map; TTL pour features de forme (ex: 120 jours) vs historiques longues (all-time).
- Offline store : fichiers parquet (ou Delta) versionnés; registry Feast dans repo; option online store (SQLite/Redis) si besoin d’inférence temps réel (sinon pré-matérialiser des features matchup fréquents).

## Backlog features (pré-match uniquement)
- **Team-level (historique)** : elo/delta récent (avec side_adv et importance), winrate par side/BO type, force de calendrier (elo moyen adversaires), repos (jours depuis dernier match), record par map_num dans le BO, performance vs first/second pick, online vs LAN/qualif/major (mapping), tendance (rolling slopes).
- **Roster/lineup** : stabilité (nb de maps avec même 5), expériences cumulées duo/triades, churn récent, remplacements par rôle, minutes/matches joués par lineup avant `start_time`, qualité lineup (moyenne elo joueurs et roles), gestion des lineups partielles (fallback role/global).
- **Player-level** (rolling N maps) : gpm/xpm, lane outcome (lh@10/dn@10, networth@10 si présent), KDA, kill participation, deaths/10 min, damage share, heal share, wards posés/détruits, participation first blood, buyback discipline, tilt (early deaths), forme récente vs long terme.
- **Hero-level** : pool joueurs (diversité, confort), winrate héros global par patch et par side, winrate héro-joueur, bans subis, priorité de pick/ban vs adversaire, synergies/counters (top 5 combos/counters par héros, simplifiés en embeddings ou stats). Même sans draft courante, utiliser tendances de pick/ban par patch/meta et par matchup.
- **Patch/meta** : features du patch courant (indicateur patch, rolling par patch, stats par macro-période), effets de transition de patch sur perf équipe/lineup/héros.
- **Draft/meta** : first pick flag, radiant/dire side; garder stockage picks/bans pour futures versions post-draft.
- **Series/BO** : score actuel dans la série, probabilité historique de clutch/close-out (map d’avance) ou de comeback (map de retard), perf en map décisive, adaptation entre maps (delta durée, delta kills), choix side par map.
- **Tempo/avantage** (historique) : buckets gold/xp @ 5/10/12/15/20 min (`adv_snapshots`), probabilité de convertir un lead donné, capacité de comeback (winrate malgré -5k/-10k), risque de throw (défaites malgré lead).
- **Qualité données** : nombre de maps supportant la feature, horizon max, indicateurs de fraîcheur; fallback simple si data insuffisante.

## Pipeline entraînement / évaluation
- Splits walk-forward : train jusqu’à T, valid sur fenêtre suivante (ex: 1 mois ou 200 maps), test sur bloc final; pas de fuite entre maps d’une même série (bloquer par `series_id`).
- Targets indépendantes mais possibilité multi-task (win + events) à tester; calibrer proba (Platt/Isotonic).
- Baselines rapides (logreg, XGBoost/LightGBM/CatBoost) + modèles plus riches si volume (GBDT multioutput, réseaux tabulaires) ; durée : régression ou survival/quantiles.
- Gestion déséquilibre (class weights, focal, undersampling contrôlé) ; métriques : AUC/PR, Brier, calibration, logloss; pour durée : MAE/RMSE/Pinball.
- Feature governance : listes de features par cible (inclusion/exclusion), hash des jeux de features loggué dans MLflow; importance/SHAP pour debug.
- Data leakage check : cutoff strict avant `start_time`, exclure stats du match courant, éviter le double comptage des maps du même BO dans train/valid.
- Tracking MLflow : run name = cible + fenêtre, tags (dataset hash, featureset, code commit, horizon), artefacts (conf Feast, importance, calibration plots, courbes temps).

## Inférence & interface
- Entrée : date/heure (UTC), team A/B, side A/B, first pick, type BO, num_map, flag online/lan (si dispo), lineup par défaut (dernière lineup connue avant date) modifiable joueur par rôle; plus tard : picks/bans (optionnel pour post-draft).
- Flow : résolution des team_id/alias -> récupération lineup par défaut -> requêtes Feast -> assemblage features -> modèle -> sorties multi-cibles; stratégie de fallback si features manquantes (reversion sur agrégats équipe ou global/role, prise en charge des lineups partielles).
- Sorties : probas par cible + intervalles pour durée; log des requêtes (feature vector hash, modèle utilisé) dans MLflow ou table dédiée pour audit.
- UX : validation de cohérence (lineup match side, joueurs dupliqués), avertir quand data trop ancienne ou faible volume, possibilité de verrouiller date de cutoff pour reproduire un scénario historique.

## Plan incrémental (proposition)
1) Audit données : vérifier schemas parquets (patch présent, champs rôle/lane/lh@10, chat roshan), fréquence des colonnes clés; identifier ce qui manque pour online/lan; bench sampler.
2) Design Feast : définir entités + feature views prioritaires (team, team_hero, player) et TTL; config offline store parquet, registrer dans repo, tests unitaires sur joins avec TTL et `start_time`.
3) Ingestion features : pipelines Polars pour rolling stats (team/player/hero), table lineup historique (top 5 joueurs par match) avec snapshots et fallback partiel; réutiliser/adapter calculs metrics existants (elo, firsts, buckets, series) pour pré-match.
4) Baselines modelling : targets win/fb/ft/fs + durée + combo; splits walk-forward; log MLflow; rapport calibration et leakage-check.
5) Extension features : patch/meta, série/BO avancées, tempo/comeback, online/lan, synergies/counters; variantes per-target.
6) Inférence/UX : service léger ou notebook + intégration Streamlit (sélecteurs + fallback), contrôle de cohérence, logging inference; pré-matérialiser features si besoin de latence faible sans online store (sinon backfill offline + store online sur la queue récente).

## Points de vigilance / risques
- Fiabilité `series_id`/`map_num` : logique existante dans le dashboard pour l’ordre des maps à réutiliser/valider.
- Patch/meta : risque de mélange inter-patch; besoin de features patch-spécifiques et d’une stratégie de gouvernance (voir questions).
- Roster : account_id manquants/aliases/stand-ins à détecter; warnings/fallbacks dans les features.
- Online vs LAN/qualif/major : pas de champ direct; nécessite un mapping `leagueid` -> tier/lan/online avec cap sur les poids.
- Draft features : pré-draft uniquement; conserver stockage picks/bans pour future version post-draft.
- Volume limité sur certaines équipes -> risque de surapprentissage; prévoir backoff vers stats globales + incertitude.
- Latence inférence : si pas d’online store, anticiper la matérialisation de features (matchups fréquents) pour éviter le scan parquet coûteux.

## Décisions/clarifications actées
- Scope pré-match, pré-draft : features héros limitées aux tendances historiques (par patch/matchup), pas de draft courante.
- Cibles firsts calculées par side/pick : global, Radiant FP/LP, Dire FP/LP; side/pick impact à inclure.
- Picks/bans : non utilisés pour la V1 pré-draft, mais conservés pour futurs use-cases post-draft.
- Série/BO : on peut réutiliser la logique Streamlit existante pour l’ordre des maps (`series_id`, `series_type`, `map_num`).
- Alias équipes : non prioritaires pour le modeling si les parquets sont déjà propres.
- Combo fb+ft+fs+win : viser la conjonction pour maximiser la valeur (bookmakers ne modélisent pas la corrélation).
- Elo à produire en features : team global/patch, player global/patch, player role-based global/patch (5 rôles, fallback global si jamais joué), plus deltas patch-global et indicateurs de fraîcheur.
- Importance des matchs : grille Major/LAN/T1/T2-T3/online, pondération capée appliquée au K.
- Lineup churn : shrink du team Elo vers la moyenne des Elo joueurs ou vers 1500 en cas de gros changement de roster.
- Online store : par défaut on backfill offline et on pousse la queue récente en online store Feast; si pas d’online, matérialiser des snapshots par date pour l’inférence.

## Points à éclaircir (inputs attendus / à investiguer)
- Mapping `leagueid` -> tier/lan/online (source OpenDota ou table externe) pour alimenter `w_importance`.
- Paramétrage fin de side_adv (Radiant/first pick) et des constantes (α partage role/global, τ decay, seuils K) après un petit grid-search ou backtest.
- Stratégie patch governance finale : modèle unique + feature patch (préféré) vs modèles par macro-période si rupture forte observée.
- Règles de fallback lineups partielles (poids du joueur manquant, shrink global) et seuils d’historique par feature (maps min avant d’exposer la feature).
