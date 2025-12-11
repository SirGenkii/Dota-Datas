# Dota Data — Roadmap v4 (technique/ML)

## Objectifs et périmètre
- Prédire au niveau map : `win/lose`, `duration`, `first blood`, `first tower`, `first roshan`, combo `fb+ft+fs+win`.
- Exploiter les tables parquet `data/processed` + métriques pré-calculées `data/metrics` + brut JSON (picks/bans, timelines gold/xp, chat) si nécessaire.
- Construire une feature store avec Feast (offline parquet, potentielle online store légère) et tracer toutes les expériences via MLflow (datasets, features, versions de modèle, métriques, artefacts).
- Pipeline entraînement avec splits walk-forward chrono (pas de fuite temporelle), et inférence via une interface (sélection date/heure, teams, side/pick, BO, map_num, lineup par défaut modifiable).

## État actuel / assets
- Parquets `matches/players/objectives/teamfights` sous `data/processed`; métriques dérivées sous `data/metrics` (elo, firsts, roshan, buckets gold/xp, series, draft_meta, adv_snapshots, pick_outcomes).
- Les picks/bans ne sont pas dans les parquets de base (exclus dans `io.EXCLUDED_MATCH_KEYS`) -> à tirer soit du brut JSON soit à générer une table dédiée.
- Alias équipes gérés via `data/team_aliases.csv`; tracked teams `data/teams_to_look.csv`; compare list pour l’app Streamlit sous `data/interim/compare_list.csv`.
- Série/BO : `series_id`, `series_type` et `map_num` présents dans `matches.parquet` ; ordre à confirmer (`start_time` vs `match_seq_num`).

## Cibles (définitions)
- `win/lose` : binaire par side (POV équipe) -> attention aux maps sans vainqueur (abandons?).
- `duration` : régression (minutes) + quantiles/buckets.
- `fb/ft/fs` : binaire par side, cohérence source (players.firstblood_claimed, objectives.building_kill, objectives.CHAT_MESSAGE_ROSHAN_KILL).
- `combo fb+ft+fs+win` : soit multi-label joint (4 bits) soit conjonction booléenne -> à choisir selon volume.

## Entités Feast (proposées)
- `team` (team_id), `player` (account_id), `hero` (hero_id).
- Entités composées : `team_lineup` (team_id + hash lineup + event_time), `team_hero` (team_id, hero_id), `player_hero` (account_id, hero_id), `series_map` (series_id + map_num + side).
- Event timestamp = `start_time` (UTC) du match/map; TTL pour features de forme (ex: 120 jours) vs historiques longues (all-time).
- Offline store : fichiers parquet (ou Delta) versionnés; registry Feast dans repo; option online store (SQLite/Redis) si besoin d’inférence temps réel.

## Backlog features (pré-match uniquement)
- **Team-level (historique)** : elo/delta récent, winrate par side/BO type, force de calendrier (elo moyen adversaires), repos (jours depuis dernier match), record par map_num dans le BO, performance vs first/second pick, online vs LAN (flag à dériver du tournoi/lieu), tendance (rolling slopes).
- **Roster/lineup** : stabilité (nb de maps avec même 5), expériences cumulées duo/triades, churn récent, remplacements par rôle, minutes/matches joués par lineup avant `start_time`, qualité lineup (moyenne elo joueurs si dispo, sinon proxy via winrate lineup).
- **Player-level** (rolling N maps) : gpm/xpm, lane outcome (lh@10/dn@10, networth@10 si présent), KDA, kill participation, deaths/10 min, damage share, heal share, wards posés/détruits, participation first blood, buyback discipline, tilt (early deaths), forme récente vs long terme.
- **Hero-level** : pool joueurs (diversité, confort), winrate héros global par patch et par side, winrate héro-joueur, bans subis, priorité de pick/ban vs adversaire, synergies/counters (top 5 combos/counters par héros, simplifiés en embeddings ou stats).
- **Draft/meta** : first pick flag, radiant/dire side, draft order (ban/pick sequencing), cheese picks (surprises <2% pickrate), mirror picks, denies (héros pickés par l’adversaire habituel), flexibilité (nb de héros joués par rôle).
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
- Entrée : date/heure (UTC), team A/B, side A/B, first pick, type BO, num_map, flag online/lan (si dispo), lineup par défaut (dernière lineup connue avant date) modifiable joueur par rôle; plus tard : picks/bans (optionnel selon scénario de prédiction pré ou post-draft).
- Flow : résolution des team_id/alias -> récupération lineup par défaut -> requêtes Feast -> assemblage features -> modèle -> sorties multi-cibles; stratégie de fallback si features manquantes (reversion sur agrégats équipe ou global).
- Sorties : probas par cible + intervalles pour durée; log des requêtes (feature vector hash, modèle utilisé) dans MLflow ou table dédiée pour audit.
- UX : validation de cohérence (lineup match side, joueurs dupliqués), avertir quand data trop ancienne ou faible volume, possibilité de verrouiller date de cutoff pour reproduire un scénario historique.

## Plan incrémental (proposition)
1) Audit données : vérifier schemas parquets vs brut (picks/bans, patch, tournoi/lan flag), fréquence des colonnes clés (role, lane, lh@10, chat roshan). Bench sampler.
2) Design Feast : définir entités + feature views prioritaires (team, team_hero, player), config offline store parquet, registrer dans repo, tests unitaires sur joins avec TTL et `start_time`.
3) Ingestion features : pipelines Polars pour rolling stats (team/player/hero), tables draft (picks/bans par ordre), table lineup historique (top 5 joueurs par match) avec snapshots.
4) Baselines modelling : targets win/fb/ft/fs + durée; splits walk-forward; log MLflow; rapport calibration et leakage-check.
5) Extension features : série/BO, tempo/comeback, online/lan, synergies/counters; variantes per-target.
6) Inférence/UX : service léger ou notebook + intégration Streamlit (sélecteurs + fallback), contrôle de cohérence, logging inference.

## Points de vigilance / risques
- Fiabilité `series_id`/`map_num` selon sources OpenDota; besoin de reconstruire l’ordre via `start_time` ou `match_seq_num`.
- Patch/versions : non explicitement exploités; risque de mélanger métas; envisager un champ patch ou période dans features.
- Roster : les account_id manquants/aliases compliquent les stats joueur; besoin de mapping alias et détection des stand-ins.
- Online vs LAN : pas de champ direct; peut nécessiter mapping `leagueid` -> LAN/online/region via source externe.
- Draft features : besoin d’un stockage picks/bans structuré; attention aux matches sans draft complet.
- Volume limité sur certaines équipes -> risque de surapprentissage; prévoir backoff vers stats globales + incertitude.
- Evaluation temps réel : si online store absent, latence liée au scan parquet; prévoir pré-matérialisations pour les matchups fréquents.

## Questions à clarifier
- Scénario de prédiction : avant draft ou après draft (picks/bans connus)? Les features héros doivent-elles se limiter aux tendances (pas à la draft courante)?
- Les cibles firsts sont-elles définies par side ou par équipe nommée (A/B)? On prédira pour un côté ou pour chaque équipe séparément?
- Source/format pour flag online vs LAN : mapping tournois existant ou à construire?
- Peut-on exploiter des features in-match early (ex: draft + 0-2 minutes) ou seulement pré-match?
- Faut-il supporter des lineups partielles (4 joueurs connus + 1 inconnu) et comment fallback dans ce cas?
- Taille minimale d’historique par joueur/lineup avant d’autoriser une feature? (ex: >=10 maps sinon fallback global).
- Governance des patches : faut-il entraîner un modèle par macro-période/meta ou un modèle unique avec feature patch?
