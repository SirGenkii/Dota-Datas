# Dota Data — Roadmap v2

## Objectifs
- Relancer l ingestion a partir de l API OpenDota (key OPENDOTA_KEY dans `.env`) pour toutes les equipes listees dans `data/teams_to_look.csv`, en ne gardant que les matches a partir du 1 janv 2025.
- Reconstituer un brut JSON au format actuel (`[{"json": {match_payload}, "pairedItem": {"item": 0}}, ...]`) pour que les modules existants (`src/dota_data/io.py`, `transforms.py`, `app/streamlit_app.py`) restent compatibles.
- Regenerer les parquets `matches/players/objectives/teamfights` via `write_parquet_tables` et valider que les schemas/colonnes restent identiques aux donnees v1.
- Etendre l app Streamlit pour couvrir le backlog `todo.md` (metrics Roshan/Aegis, firsts, scenarios de series, POV radiant/dire, minutes cle) et preparer l affichage/UX associe.
- Externaliser les calculs lourds (elo equipe dans le temps, stats minutes/buckets, agrégats Roshan/series) dans des jobs offline et les sauvegarder en CSV/Parquet que Streamlit charge sans recalcul.

## Decisions prises / clarifications
- Aegis/Roshan: detection via `objectives` avec `type` = `CHAT_MESSAGE_AEGIS`, `CHAT_MESSAGE_ROSHAN_KILL`, `building_kill` (infos deja exploitees dans les infobulles Streamlit).
- Buckets gold/xp: vus dans la perspective de l equipe selectionnee (radiant/dire flippe si besoin), mais garder une vue globale par cote pour comparer les moyennes Radiant vs Dire.
- UX Streamlit: onglets par theme (ex: Overview / Roshan & Aegis / Firsts & Early / Series / Ratings) pour limiter la surcharge.
- Precomputation: calculs elo, buckets minutes, stats Roshan/Aegis, firsts, scenarios series stockes en fichiers (CSV/Parquet) charges par Streamlit.

## Backlog ingestion OpenDota (v2)
- Setup/env: charger OPENDOTA_KEY depuis `.env`; ajouter helper pour charger `opendata_api.json` si besoin de mapping params; fixer la date de depart en timestamp (2025-01-01 00:00:00 UTC).
- Collecte des matches par equipe:
  - Lire `data/teams_to_look.csv` (TeamID) et iterer sur `/teams/{team_id}/matches` avec la key API.
  - Appliquer un filtre de date cote client sur `start_time` (>= 2025-01-01) si l endpoint ne filtre pas nativement; deduper les `match_id`.
  - Gérer la pagination/limit (si dispo) et le throttling (sleep ou rate-limit). Logguer les erreurs et reprendre en cas d echec.
- Recup detail par match:
  - Pour chaque `match_id`, appeler `/matches/{match_id}` pour obtenir les donnees avancees (players, objectives, teamfights, timelines gold/xp).
  - Option: fallback `/request/{match_id}` si une partie manque (parse on-demand) et re-tenter le fetch.
- Agrégation et format brut:
  - Construire un loader Python (ex `scripts/fetch_matches.py`) qui renvoie une liste d objets `{"json": <payload_match_complet>, "pairedItem": {"item": 0}}`.
  - Sauvegarder en `data/raw/data.json` (ou fichier date) + checksum/metadata (nb matches, teams couvertes, periode).
  - Garder un cache local des `match_id` deja recuperes pour eviter les doubles appels.
- Validation/qualite:
  - Compare les cles/colonnes vs l ancien brut (`summarize_raw`) pour garantir compatibilite.
  - Compter les evenements Roshan/Aegis/first tower pour verifier que les champs attendus (`objectives`, `radiant_gold_adv`, `radiant_xp_adv`, `picks_bans`, etc.) sont presents.
  - Ajouter un smoke test sur `write_parquet_tables` avec le nouveau brut pour valider les parquets generes et leur taille.
- Orchestration/automation:
  - Ajouter un script CLI ou cible Makefile pour lancer la collecte (params: date min, sortie, dry-run, rate-limit).
  - Prevoir un cron simple (ou GitHub Action) si on veut rafraichir regulierement apres v2.

## Streamlit v2 (backlog oriente `todo.md`)
- Roshan/Aegis: afficher part de first Roshan, steals d Aegis, nombre moyen de Roshans par match, par equipe et split radiant/dire; utiliser `objectives` (CHAT_MESSAGE_ROSHAN_KILL, CHAT_MESSAGE_AEGIS, building_kill) et `chat`.
- Firsts: calculer taux de first blood / first tower / first Roshan et impact sur winrate (map). Ajouter visuals et filtres par cote (radiant/dire) et par serie (series_id/series_type).
- Scenarios de series:
  - BO3 en 2-1: quand la serie est a 1-1, mesurer les taux de victoire map1 vs map2; afficher la decomposition par cote.
  - Series a 1-1 (draw): pour une equipe, frequence de win map1 et map2 (avec POV radiant/dire).
- Minutes interessantes (5,10,12,15,20):
  - Afficher gold/xp advantage a ces minutes et probas de win associees (buckets 0/1/5/10k).
  - Ajouter vue table/graph pour filtrer par equipe ou H2H.
- Elo temporel: suivre un elo/simple rating par equipe dans le temps (serie temporelle + delta).
- UX: conserver filtres equipe A/B existants, ajouter toggles pour POV radiant/dire, selection de serie, export CSV/PNG pour les nouvelles metriques.

## Risques/points de vigilance
- Limite API: `/teams/{team_id}/matches` peut imposer un limit/pagination non documente dans l openapi (voir tests), et le parse `/matches/{match_id}` peut etre lent si replay non parse -> prevoir retries/backoff.
- Champs objectifs: les codes d evenements Roshan/Aegis/steal peuvent varier (CHAT_MESSAGE_AEGIS, CHAT_MESSAGE_ROSHAN_KILL, building_kill roshan?); valider sur un echantillon recent.
- Format brut: bien conserver les cles top-level et types pour ne pas casser la generation parquet ni l app (ex: `team` vs `building_is_radiant`, `picks_bans` arrays, `players` 10 elements).
- Volume: matches post-2025 pourraient etre nombreux; prevoir un decoupage par fichiers (mois?) si besoin pour eviter un JSON trop volumineux.

## Questions a clarifier
- Series/BO: identifier comment grouper les matches d une meme serie et dans quel ordre (examiner `series_id`, `series_type`, `match_seq_num`, `start_time`, `leagueid`). A investiguer sur des matches recents pour reconstruire les sequences.

## Observations data (notebook 02_filter_and_sample)
- `series_id` et `series_type` sont bien présents dans les matches détaillés; l ordre des maps peut se reconstruire en groupant par `series_id` (et `leagueid`) puis en triant par `start_time` ou `match_seq_num`. `series_type` ressemble à un code BO (à confirmer: 0=BO1, 1=BO3, 2=BO5).
- Objectifs/timelines:
  - Roshan: `CHAT_MESSAGE_ROSHAN_KILL` a `team`=2 (Radiant) ou 3 (Dire); `CHAT_MESSAGE_AEGIS` fournit `slot` et `player_slot` du porteur.
  - Buildings: `building_kill` expose `key` avec `goodguys` (Radiant) / `badguys` (Dire); `slot`/`player_slot` identifient le killer (0-4 Radiant, 5-9 Dire; `player_slot` <128 Radiant, >=128 Dire).
  - First blood: `CHAT_MESSAGE_FIRSTBLOOD` sans `team` mais avec `slot`/`player_slot` pour le côté.
- Dedup match_id: les endpoints teams renvoient le même match pour plusieurs équipes sources; après filtre date (5391 lignes), on obtient 3879 `match_id` uniques -> normal, il faut dédupliquer avant fetch détail.
