# Roadmap — Migration vers une app web + Postgres (Next.js + FastAPI)

## Objectif (en 1 phrase)
Passer d’un projet “fichiers + Streamlit” à une **application web Docker** où **toutes les données de scraping et les métriques sont stockées en Postgres**, avec des jobs **scrap/precompute déclenchables depuis l’UI**, tout en gardant un **workflow CLI local** via venv.

---

## Pourquoi c’est intéressant (bénéfices)
- **Partage / collaboration** : plus besoin de déplacer des parquets, tout le monde lit la même source (DB).
- **Reproductibilité** : chaque run de scraping/precompute est **versionné** (`run_id`, timestamp, config).
- **UI plus libre** : Next.js permet une UX propre (layout, graphs, filtres) sans contraintes Streamlit.
- **Déploiement simple** : Docker + CI/CD → push, migrate, restart.

---

## Proposition d’architecture (vulgarisée)

### Les briques
- **Postgres** : la base de données (source de vérité).
- **API (FastAPI)** : “le serveur Python” qui expose des endpoints (ex: leaderboard, stats, lancement d’un job).
- **Worker (Python)** : “l’ouvrier” qui fait les tâches lourdes (scrap / precompute) en arrière-plan.
- **Front (Next.js)** : “le site web” (pages, graphiques, filtres, boutons).
- **Queue (Redis recommandé)** : file d’attente pour exécuter les jobs sans bloquer l’API.
  - Même si “tout en DB” est l’objectif data, la queue Redis reste souvent la solution la plus simple/fiable.
  - Alternative 100% Postgres possible (table `jobs` + polling/locks), mais c’est plus délicat.

### Diagramme mental
UI (Next) → API (FastAPI) → (crée un job) → Queue → Worker → écrit en DB → UI relit via API.

---

## Principe clé : “runs” + versioning
On garde l’idée de “run” qu’on avait pour `update` :
- `runs` : un run = une exécution (scrap ou precompute)
- chaque table de métriques a un `run_id`
- “latest” = une vue (ou requête) qui prend le dernier `run_id` valide

Bénéfice : on peut comparer avant/après, rollback, auditer.

---

## Schéma de données Postgres (MVP puis extension)

## Schéma DB détaillé (basé sur nos parquets actuels)
L’idée est de partir d’un schéma DB qui couvre **100% des champs déjà présents** dans :
- `data/processed/*.parquet` (tables “source”)
- `data/metrics/*.parquet` (tables “precompute”)

Ensuite, on peut “nettoyer” (normaliser, typer plus finement, partitionner) sans perdre la compatibilité.

### Conventions proposées
- `*_id` : `BIGINT`
- timestamps :
  - `start_time` (epoch seconds) : `BIGINT`
  - `start_dt` (datetime) : `TIMESTAMPTZ`
- chaînes JSON sérialisées aujourd’hui dans les parquets : `JSONB` en DB (au lieu de `TEXT`)
- arrays :
  - `teams: BIGINT[]`
  - `radiant_gold_adv` / `radiant_xp_adv` : `DOUBLE PRECISION[]` (ou `JSONB` si on veut garder le format brut)

### Tables “processed” (source-of-truth issue du scraping)
Ces tables correspondent directement à `data/processed/*.parquet`.

#### `matches` (PK: `match_id`)
Champs (mapping 1:1 depuis `data/processed/matches.parquet`) :
- Identité : `match_id`, `version`
- Contexte : `leagueid`, `series_id`, `series_type`, `series_type_raw`, `bo_type`, `teams_in_series`, `winner_team_id`
- Temps : `start_time`, `duration`, `pre_game_duration`, `match_seq_num`, `first_blood_time`
- Résultat : `radiant_win`, `radiant_score`, `dire_score`, `comeback`, `stomp`, `throw`, `loss`
- Teams : `radiant_team_id`, `dire_team_id`, `radiant_name`, `dire_name`, `radiant_team`, `dire_team`, `radiant_logo`, `dire_logo`, `radiant_team_complete`, `dire_team_complete`, `radiant_captain`, `dire_captain`
- Match config : `cluster`, `replay_salt`, `lobby_type`, `human_players`, `game_mode`, `flags`, `engine`, `patch`, `region`, `skill`, `can_be_archived`
- Buildings : `tower_status_radiant`, `tower_status_dire`, `barracks_status_radiant`, `barracks_status_dire`
- Text/metadata : `od_data`, `league`, `league_name`, `tournament_name`, `tournament_slug`, `tournament_tier`, `tournament_location`, `all_word_counts`, `my_word_counts`, `metadata`, `replay_url`
- Counts : `players_count`, `objectives_count`, `teamfights_count`
- Votes : `negative_votes`, `positive_votes`

Indexes recommandés :
- `matches(start_time)`
- `matches(leagueid, series_id)` (pour join séries)
- `matches(radiant_team_id)`, `matches(dire_team_id)`

#### `players` (PK: `(match_id, player_slot)`)
Mapping depuis `data/processed/players.parquet`.

Note : la table est grosse (620MB actuellement). On peut :
- soit tout stocker en colonnes (beaucoup de `JSONB`/`TEXT`),
- soit stocker `match_raw` en JSONB + une table `players_core` avec les champs “query-friendly”, et garder le reste en JSONB.

Champs principaux (sélection “query-friendly”) :
- Identité : `match_id`, `player_slot`, `account_id`, `hero_id`, `team_number`, `team_slot`, `is_radiant`, `isRadiant`
- Résultat : `radiant_win`, `win`, `lose`
- KDA / farm : `kills`, `deaths`, `assists`, `last_hits`, `denies`, `gold_per_min`, `xp_per_min`, `net_worth`, `level`
- Items : `item_0..item_5`, `backpack_0..backpack_2`, `item_neutral`, `item_neutral2`, `aghanims_scepter`, `aghanims_shard`, `moonshard`
- Stats : `hero_damage`, `tower_damage`, `hero_healing`, `stuns`, `teamfight_participation`
- Vision/support : `obs_placed`, `sen_placed`, `purchase_ward_observer`, `purchase_ward_sentry`, `observer_uses`, `sentry_uses`
- Lane : `lane`, `lane_role`, `is_roaming`, `lane_efficiency`, `lane_efficiency_pct`, `lane_pos`
- Match context (dupliqué) : `start_time`, `duration`, `cluster`, `lobby_type`, `game_mode`, `patch`, `region`

Champs “logs / blobs” (à mettre en `JSONB` ou `TEXT` selon parsing) :
- timelines : `times`, `gold_t`, `lh_t`, `dn_t`, `xp_t`
- logs : `purchase_log`, `kills_log`, `buyback_log`, `runes_log`, `connection_log`, `neutral_tokens_log`, `neutral_item_history`
- dicts JSON : `purchase`, `gold_reasons`, `xp_reasons`, `killed`, `item_uses`, `ability_uses`, `ability_targets`, `damage_targets`, `hero_hits`, `damage`, `damage_taken`, `damage_inflictor`, `runes`, `killed_by`, `kill_streaks`, `multi_kills`, `life_state`, `healing`, `damage_inflictor_received`, `permanent_buffs`, `cosmetics`, `benchmarks`, `performance_others`, `additional_units`
- wards logs : `obs_log`, `sen_log`, `obs_left_log`, `sen_left_log`, `obs`, `sen`
- autres : `max_hero_hit`, `ability_upgrades_arr`, `purchase_time`, `first_purchase_time`, `item_win`, `item_usage`

#### `objectives` (PK: `(match_id, objective_index)`)
Mapping depuis `data/processed/objectives.parquet` :
- `match_id`, `objective_index`, `time`, `type`, `value`, `killer`, `team`, `slot`, `key`, `player_slot`, `unit`

Index : `objectives(match_id, time)`

#### `teamfights` (PK: `(match_id, teamfight_index, teamfight_player_index)`)
Mapping depuis `data/processed/teamfights.parquet` :
- `match_id`, `teamfight_index`, `teamfight_player_index`
- `start`, `end`, `last_death`
- `deaths`, `buybacks`, `gold_delta`, `xp_delta`, `xp_start`, `xp_end`, `healing`, `damage`
- blobs : `ability_uses`, `item_uses`, `ability_targets`, `deaths_pos`, `killed` (JSONB/TEXT)

#### `extras` (PK: `match_id`)
Mapping depuis `data/processed/extras.parquet` :
- `match_id`
- `radiant_gold_adv` (array)
- `radiant_xp_adv` (array)
- `picks_bans` (JSONB)

#### `series` (PK logique: `(leagueid, series_id)`)
Mapping depuis `data/processed/series.parquet` :
- Identité : `leagueid`, `series_id`
- Tournoi : `league_name`, `tournament_name`, `tournament_slug`, `tournament_tier`, `tournament_location`
- Format : `series_type_raw`, `bo_type`, `max_wins`, `match_count`, `series_type`
- Teams : `teams` (array), `teams_in_series`, `team_a`, `team_b`
- Score : `score_team_a`, `score_team_b`, `winner_team_id`
- Temps : `start_time_min`, `start_time_max`

### Tables “metrics” (precompute, versionnées par run)
Ces tables existent aujourd’hui sous `data/metrics/*.parquet`. En DB, on ajoute :
- `run_id` (FK vers `runs`)
- `generated_at`
- des indexes orientés UI (leaderboard, fenêtres de rank, time-series)

#### `ratings_elo_latest_all` / `ratings_elo_latest`
Depuis `elo_latest_all.parquet` / `elo_latest.parquet` :
- `team_id`, `elo`, `elo_rank`

#### `ratings_elo_timeseries`
Depuis `elo_timeseries.parquet` :
- `match_id`, `start_time`, `start_dt`
- `team_id`, `opponent_id`, `team_is_radiant`, `team_win`
- `rating_pre`, `rating_post`, `expected`
- `rating_pre_patch`, `rating_post_patch`, `expected_patch`
- `tournament_tier`, `tournament_location`, `weight`, `patch`

#### `ratings_glicko2_latest_all` / `ratings_glicko2_latest`
Depuis `glicko2_latest_all.parquet` / `glicko2_latest.parquet` :
- `team_id`, `rating`, `rd`, `sigma`, `score`, `rank`, `score_rank`

#### `ratings_glicko2_timeseries`
Depuis `glicko2_timeseries.parquet` :
- `period` (ex: day/week), `team_id`, `rating`, `rd`, `sigma`, `series`

#### `series_results` (input de Glicko-2)
Depuis `series_results.parquet` :
- clé logique `(leagueid, series_id)`
- `start_time`, `start_dt`
- `team_a_id`, `team_b_id`, `wins_a`, `wins_b`, `result_a`
- `maps`, `bo_type`, `series_type`, `weight`

#### `draft_meta`
Depuis `draft_meta.parquet` :
- `match_id`, `first_pick_team_id`, `last_pick_team_id`

#### `adv_snapshots`
Depuis `adv_snapshots.parquet` :
- `match_id`, `team_id`, `opponent_id`, `team_is_radiant`
- `minute`
- `gold_adv`, `gold_bucket`, `xp_adv`, `xp_bucket`
- `start_time`

#### `pick_outcomes`
Depuis `pick_outcomes.parquet` :
- `team_id`, `label`, `matches`, `winrate`
- `first_blood_rate`, `first_blood_count`
- `first_tower_rate`, `first_tower_count`
- `first_roshan_rate`, `first_roshan_count`
- `combo_for_rate`, `combo_against_rate`
- `aegis_steal_rate`, `aegis_steal_against_rate`

#### `firsts`
Depuis `firsts.parquet` :
- `team_id`, `team_is_radiant`, `matches`
- `first_blood_rate`, `first_tower_rate`, `first_roshan_rate`
- `first_blood_count`, `first_tower_count`, `first_roshan_count`

#### `roshan`
Depuis `roshan.parquet` :
- `team_id`, `matches`
- `roshan_kills_avg`, `aegis_claims_avg`, `first_roshan_rate`
- `steals_total`, `steals_rate`

#### `gold_buckets` / `xp_buckets`
Depuis `gold_buckets.parquet` / `xp_buckets.parquet` :
- `team_id`, `minute`, `bucket`
- `winrate`, `matches`, `adv_avg`

#### `series_team_stats`
Depuis `series_team_stats.parquet` :
- `team_id`, `map_num`, `bo_type`
- `winrate`, `maps_played`

#### `series_maps`
Depuis `series_maps.parquet` :
- clé logique `(leagueid, series_id, map_num)`
- `match_id`, `start_time`
- `radiant_team_id`, `dire_team_id`, `radiant_win`
- `series_type`, `bo_type`

#### `tracked_teams`
Depuis `tracked_teams.parquet` :
- `team_id`, `name`

### Tables “core” (MVP)
1) `teams`
   - `team_id` (PK), `name`, `logo_url`, `is_tracked`, `created_at`, `updated_at`
2) `team_aliases`
   - `alias_team_id` → `canonical_team_id`
3) `series`
   - clé logique: `(league_id, series_id)` (comme tu l’as noté, `series_id` seul n’est pas unique)
   - `league_id`, `series_id`, `start_time`, `team_a_id`, `team_b_id`, `team_a_wins`, `team_b_wins`, `bo_type`, …
4) `matches`
   - `match_id` (PK), `start_time`, `league_id`, `series_id`, `radiant_team_id`, `dire_team_id`, `radiant_win`, …

### Tables “raw / provenance” (option très utile)
5) `fetch_events`
   - `id`, `run_id`, `endpoint`, `requested_at`, `status_code`, `duration_ms`, `error`, …
6) `match_raw`
   - `match_id` (PK), `fetched_at`, `payload` (JSONB)
   - permet de reparser sans rescraper si on change le parser.

### Tables “big / détaillées” (phase 2)
7) `players` (potentiellement énorme)
8) `objectives`
9) `teamfights`
10) `drafts` / `picks_bans`
11) `advantage_snapshots` (ou stockage JSONB/array selon usage)

Notes perf :
- On indexe fort : `matches(start_time)`, `matches(radiant_team_id)`, `matches(dire_team_id)`, `matches(league_id, series_id)`.
- Si `players` devient massif : **partitionnement** par mois sur `matches.start_time` ou partition by `match_id` range.

---

## Mécanique d’ingestion (scraping → DB)

### Étape A — “Discovery”
- on récupère les `match_id` récents par team (comme aujourd’hui)
- on filtre en DB : ne garder que `match_id` inconnus ou plus récents que la dernière date qu’on veut conserver

### Étape B — “Fetch details”
- pour chaque `match_id`, fetch `/matches/{id}`
- on écrit :
  - `match_raw` (JSONB) (optionnel mais recommandé)
  - `matches`, `series` (upsert), + tables détaillées si activées

### Étape C — “Quality / healthcheck”
- checks: taux de couverture `draft`, `advantage`, etc.
- on marque le run `valid=true/false` selon seuils

---

## Precompute en DB (Elo + Glicko-2)

### Point important : ranking sur les **séries**, pas les matchs
On construit les “events” de rating depuis `series` :
- player/team = `team_id`
- outcome = win/lose la série
- time = `series.start_time`

On stocke :
- `ratings_glicko2`
  - `run_id`, `period`, `team_id`, `rating`, `rd`, `sigma`, `score`, `score_rank`, `last_series_dt`, `series_played`
- (optionnel) `ratings_elo`
- des tables agrégées (ex: buckets gold/xp) avec `run_id`

Pourquoi conserver `run_id` :
- tu peux comparer les changements de classement d’un run à l’autre
- tu peux relancer un run avec une nouvelle config (ex: `tau`) sans écraser l’historique

---

## API (FastAPI) — endpoints (MVP)

### Lecture (UI)
- `GET /api/teams`
- `GET /api/leaderboard?system=glicko2&run=latest`
- `GET /api/teams/{id}/summary?run=latest`
- `GET /api/teams/{id}/performance?rank_min=...&rank_max=...&run=latest`
- `GET /api/runs?type=scrape|precompute`
- `GET /api/runs/{run_id}`

### Jobs
- `POST /api/jobs/scrape` (payload: teams/tracked, since, max_new, options)
- `POST /api/jobs/precompute` (payload: config glicko2 + options)
- `GET /api/jobs/{job_id}` (status + logs)

### Auth (simple au début)
MVP : basic auth / token fixe dans `.env` (admin-only).
Plus tard : OAuth / login.

---

## UI (Next.js) — pages (MVP)
- “Dashboard” : Team A/B compare (reprend l’esprit Streamlit)
- “Leaderboard” : top teams + filtres (period, region, tracked only)
- “Jobs” : bouton “Run scraping”, “Run precompute”, progression, logs
- “Runs” : liste des runs + détails (stats, coverage, deltas)

---

## Jobs / queue (choix technique recommandé)
### Option recommandée (simple) : RQ (Redis Queue) ou Celery
- API crée un job dans Redis
- worker consomme et écrit DB

### Option “100% Postgres”
Possible : table `jobs` + worker qui “claim” via `SELECT ... FOR UPDATE SKIP LOCKED`.
Avantage : moins de composants.
Inconvénient : plus de pièges (retry, timeouts, locking).

Je recommande Redis au début, quitte à “consolider” plus tard.

---

## Docker (dev + prod)
### `docker-compose.yml` (dev)
- `postgres`
- `redis` (si queue)
- `api` (FastAPI)
- `worker` (même image que api, autre commande)
- `web` (Next.js)

### Prod (VPS)
- reverse-proxy (Caddy/Traefik) + TLS
- volumes persistants (PG data)

---

## CI/CD (GitHub)
Pipeline type :
1) lint + tests (python + node)
2) build images Docker
3) push registry (GHCR)
4) deploy via SSH sur VPS (pull + compose up)
5) migrations DB (Alembic) avant redémarrage API/worker

---

## Roadmap par phases (avec livrables)

### Phase 0 — Design + fondations (1–2 jours)
- décider : queue Redis (recommandé) vs jobs Postgres
- définir le schéma DB v1 (MVP) + indexes
- choisir ORM/migrations : SQLAlchemy + Alembic

Livrables :
- `docker-compose.yml` (pg + redis optionnel)
- migrations Alembic v1

### Phase 1 — Ingestion minimale (3–7 jours)
- importer `teams_to_look.csv` + aliases → `teams`, `team_aliases`
- job “discover + fetch match details”
- upsert `matches` + `series` + `match_raw` (JSONB)
- run tracking (`runs`, `fetch_events`)

Livrables :
- `POST /jobs/scrape`
- `GET /runs`, `GET /jobs/{id}`
- logs job consultables

### Phase 2 — Precompute Glicko-2 en DB (3–7 jours)
- builder series outcomes depuis `series`
- compute glicko2 (conserver ton implémentation actuelle en module réutilisable)
- écriture `ratings_glicko2` + vue “latest”
- endpoint leaderboard + endpoint team summary

Livrables :
- `POST /jobs/precompute`
- `GET /leaderboard?system=glicko2&run=latest`

### Phase 3 — UI web MVP (Next.js) (5–10 jours)
- page leaderboard
- page team compare (A/B) avec sliders + graphs
- page jobs / runs

Livrables :
- un site utilisable (même si pas 100% feature-parity)

### Phase 4 — Tables détaillées (players/objectives/adv) (itératif)
- ingestion des tables “big”
- optimisation perf (bulk insert, partition si nécessaire)
- endpoints et UI pour les blocs avancés (draft splits, buckets, etc.)

### Phase 5 — CI/CD + prod stable
- déploiement auto sur VPS
- sauvegardes Postgres (cron + retention)
- monitoring basique (health endpoints, logs)

---

## Compat CLI local (venv) — important
On garde ton usage local :
- le code Python devient une “lib” commune (`src/dota_data/...`)
- on ajoute des commandes CLI (Typer) :
  - `dota-data scrape --since ...`
  - `dota-data precompute --config ...`
- CLI peut :
  1) soit appeler directement la DB locale/remote via `DATABASE_URL`
  2) soit appeler l’API (`POST /jobs/...`)

Recommandation :
- dev local → DB locale via docker compose
- prod → CLI appelle API (plus safe)

---

## Risques / points à surveiller
- **volume** (players/tableaux énormes) → bulk insert + partitions + indexes
- **qualité des données** (draft/adv manquants) → coverage gating par run
- **coût API OpenDota** (rate limit) → cache JSONB + retry/backoff
- **concurrence** (2 jobs en même temps) → lock de job + “single flight”

---

## “Definition of done” (migration réussie)
- un coworker clone, lance `docker compose up`, ouvre le site
- clique “Scrape” → données en DB
- clique “Precompute” → leaderboard + team compare cohérents
- plus besoin de partager des gros parquets

---

## Prochaine étape (si tu veux que je code derrière)
Je peux partir sur Phase 0 + Phase 1 en créant :
- `docker-compose.yml` (postgres + redis)
- `api/` (FastAPI) + `worker/`
- migrations Alembic v1
- endpoints jobs + run tracking

Dis-moi juste : tu veux du **Redis** pour la queue (recommandé) ou un système de jobs “Postgres-only” ? 
