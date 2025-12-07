# Dota Data — Roadmap v1

## Aperçu des données (`data/raw/data.json`)
- Format: tableau de 50 matches avec deux clés: `json` (payload principal, ~58 champs) et `pairedItem.item` (toujours 0, probablement inutile).
- Stat clés: 58 champs top-level; 148 champs côté joueurs; champs d’objectifs observés: `time`, `type`, `value`, `key`, `team`, `killer`, `slot`, `player_slot`, `unit`; max 46 objectifs/match sur l’échantillon.
- Info match: `match_id`, `start_time`, `duration`, `cluster`, `region`, `patch`, `game_mode`, `lobby_type`, `radiant_win`, scores, séries (`series_id`, `series_type`), équipes (`*_team_id`, `*_name`, `*_captain`, logos), `replay_url`.
- Timelines/événements: `radiant_gold_adv`, `radiant_xp_adv`, `objectives` (ex: CHAT_MESSAGE_COURIER_LOST, roshan_kill, building_kill, buyback), `chat`, `pauses`, `draft_timings`, `picks_bans`.
- Joueurs: tableau de 10 objets (~148 champs chacun) incluant identifiants (`account_id`, `player_slot`, `hero_id`), stats (`kills`, `deaths`, `assists`, `gold_per_min`, `xp_per_min`, `last_hits`, `denies`, `net_worth`, `stuns`), items (`item_0-5`, `backpack_0-2`, `neutral_item`, `purchase_log`), capacités (`ability_upgrades_arr`, `ability_uses`, `damage_inflictor`), position/role (`lane`, `lane_role`, `role`, `is_roaming`, `benchmarks`), vision (`obs_log`, `sen_log`, `obs_left_log`, `sen_left_log`), ressources temporelles (`lh_t`, `dn_t`, `gold_t`, `xp_t`, `purchase`, `runes`), teamfight (`teamfight_participation`, `kills_log`, `deaths_log`).
- Teamfights: liste de combats avec `start`, `end`, `last_death` + stats par joueur (damage, heal, ability_uses, décès).
- Points de vigilance: certains champs nulls (`metadata`), types mêlés (bool/int), timestamps en secondes à aligner avec `pre_game_duration`.

## Setup environnement
- Déjà ajouté: `.venv`, `requirements.txt`, `Makefile` (cibles: `venv`, `install`, `activate`, `notebook`, `deps-clean`); dossiers `src/dota_data/`, `scripts/`, `notebooks/`, `data/processed/`.
- Installation: `pip install pandas polars pyarrow` OK (install complète requirements à reprendre si besoin de MLflow/Feast/plots).
- À stabiliser: `reports/`, conventions de nommage parquet, config `.env` (chemins data).
- Version Python cible >= 3.10; dépendances data/ML (pandas/polars, matplotlib/seaborn, scikit-learn, mlflow, feast, etc.).

## Phase 0 — Audit & ingestion
- Implémenté: `src/dota_data/io.py` avec helpers `load_raw_matches`, `matches_table`, `players_table`, `objectives_table`, `teamfights_table`, `write_parquet_tables`, `summarize_raw`. CLI: `python -m src.dota_data.io --raw data/raw/data.json --out data/processed`. Les champs non-scalaires sont sérialisés en JSON string pour éviter les structs vides (à désérialiser si besoin).
- Les notebooks vérifient la présence des parquet et les génèrent via `write_parquet_tables` si absents (détection automatique du chemin projet même lancés depuis `notebooks/`).
- Script de validation du JSON: à compléter pour normalisation types (ints vs bools), détection des listes vides/anormales.
- Conversion en tables parquet partitionnées: `matches`, `players`, `objectives/events`, `teamfights`, `drafts`, avec clés (`match_id`, `player_slot`, `team`/`is_radiant`, timestamp `time`).
- Documentation du schéma cible (dtypes + contraintes) pour réutilisation dans les notebooks et transformations.

## Phase 1 — Notebooks EDA
- `01_data_overview.ipynb`: charge les parquet, winrate global/par patch, distribution durée, patch/modes fréquents, top valeurs manquantes, distribution kill participation.
- `02_match_flow.ipynb`: explore un match (gold/xp advantage depuis le JSON brut, timelines objectifs tour/roshan/buyback, aperçu teamfights). Param `match_id` à ajuster dans la première cellule.
- `03_hero_lane.ipynb`: lane outcomes (LH/DN minute 10, GPM/XPM 10), comparaisons par lane_role, top héros par GPM moyen (base pour item timings à venir).
- `05_dictionaries.ipynb`: génère les dictionnaires équipes/joueurs (parquet) et affiche les premiers enregistrements.
- `06_data_explorer.ipynb`: explorer baselines (winrate, first blood, first tower), timeline objectifs d’un match, aperçu dictionnaires teams/players/heroes (counts).
- Streamlit: `app/streamlit_app.py` pour une interface interactive (filtres équipe/héros, métriques winrate/first blood/first tower, dictionnaires, timeline d’un match).
- `04_team_matchups.ipynb`: drafts (picks/bans), synergies/contre, efficacité teamfights, vision/wards, contrôle neutres.
- Les notebooks s’appuient sur modules `src/dota_data` pour éviter la duplication.

## Phase 2 — Modules réutilisables (`src/dota_data`)
- `io.py`: charge JSON brut -> DataFrame/Arrow, export parquet partitionné, helpers de sampling (par match_id). Non-scalaires sérialisés en JSON.
- `transforms.py`: ajouté `read_processed_tables`, `lane_phase_features(minute=10)`, `match_header`. TODO: features temps (rolling gold/xp, deltas 10/20/30), extraction des timings d’items/objectifs (désérialiser les logs).
- `stats.py`: ajouté `kill_participation`, `vision_stats`, `team_scores`.
- `viz.py`: helpers simples `objectives_timeline`, `teamfights_for_match`. TODO: courbes gold/xp, heatmaps wards.
- `metadata.py`: ajouté `build_team_dictionary` (team_id/name/tag/logo), `build_player_dictionary` (account_id + noms + nb de matches), `build_hero_counts` (freq hero_id). `load_hero_dictionary` permet de charger un mapping `hero_id -> name` si un fichier `data/dictionaries/heroes.(json|csv)` est présent.
- Tests unitaires ciblés (extraction events, merges clés, conversions temps) + docstring courtes.

## Phase 3 — Feature Store (Feast)
- Offline store parquet local; entities: `match_id`, `team_id`, `player_id`/`player_slot`, `hero_id`.
- Feature views proposées: `team_pre_game` (draft comps, bans, winrate patch), `lane_phase` (GPM/XPM10, LH/DN10, first_blood involvement), `mid_game` (roshan/torres capturés, kill diff 20m), `vision` (wards posées/détruites, sentry ratio).
- Pipeline: production des features -> matérialisation vers offline -> option online (file/redis) pour tests; validation des freshness TTL.

## Phase 4 — Modèles + MLflow
- Cibles: `radiant_win`; temps de prise des tours T1/T2; nombre d’objectifs neutres (roshan/tormentor/lotus); diff de kills à 25m.
- Baselines: régressions/logreg, gradient boosting, éventuellement séquences légères (features par tranche de temps). Utiliser splits temporels + calibration.
- MLflow: tracking exp, params/features versions, artefacts (figures, modèles). CLI d’orchestration (train/eval/register) + stockage local d’artefacts.

## Questions ouvertes / risques
- Identifiants équipes/capitaines suffisants pour relier les matches? fallback sur noms si ID manquant.
- Données d’un seul patch ou multiples? à intégrer comme variable/contrôle.
- Précision des objectifs neutres: champs `team`/`killer` cohérents pour roshan/torres? à valider.
- `account_id` parfois nul? prévoir anonymisation ou mapping interne.
- Alignement temporel: `start_time` vs `pre_game_duration` pour synchroniser timelines objectives/achats.

Prochaines actions immédiates
- Étendre `06_data_explorer.ipynb` et l’app Streamlit (`app/streamlit_app.py`) avec filtres team_id/hero_id/match picker, overlay tours/roshan sur gold/xp, vues early/mid/late.
- Ajouter un dictionnaire `data/dictionaries/heroes.json` ou `.csv` pour mapper `hero_id -> name/localized_name`.
- Ajouter désérialisation ciblée (purchase_log, lh_t/xp_t) dans `transforms.py` + features lane/tempo.
- Préparer les features baselines pour paris first blood/first tower/win et un notebook dédié ou interface légère pour les afficher.
