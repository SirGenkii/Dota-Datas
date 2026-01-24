# Roadmap — pipeline incrémentale (raw chunks → parquet → métriques)

## Contexte / objectif
Aujourd’hui la collecte “full” produit un `data/raw/data_v2.json` très volumineux (plusieurs Go), puis on génère des parquets et on recalcule des métriques (Elo, firsts, roshan, buckets…).

Objectif: rendre le pipeline **incrémental** pour pouvoir:
- **détecter et télécharger** uniquement les matchs “récents” manquants pour les teams “source” (celles de `data/teams_to_look.csv`)
- **ajouter** ces matchs aux parquets existants (`data/processed/`)
- **relancer les calculs** (script `scripts/precompute_metrics.py`) sans avoir à “refaire” un gros `data.json`
- produire des **parquets à partir des réponses JSON** (chunk files) et garder une trace “raw” exploitable

## Choix techniques (et pourquoi)
### 1) Raw = chunks (bronze)
- Au lieu d’un unique `data_v2.json`, on persiste des lots JSON “chunks” sous `data/raw/updates/`.
- Chaque run crée un dossier `data/raw/updates/run_YYYYMMDD_HHMMSS/` avec:
  - `matches_chunk_0000.json`, `matches_chunk_0001.json`, … (liste de matchs wrapped: `{ "json": {…match…}, ... }`)
  - `matches_chunk_retry.json` si retry
  - `summary.json`, `errors.json`, `run_metadata.json`

Avantages:
- pas de ré-écriture d’un fichier géant
- reprise facile, auditabilité (on sait ce qui a été téléchargé à chaque run)
- conversion parquet “batch” simple

### 2) Processed = parquets appendés (silver)
- On génère les tables (matches/players/objectives/teamfights) pour le batch et on append aux parquets existants.
- `matches.parquet` est dédupliqué sur `match_id`.
- Le résumé `series.parquet` est **recalculé** depuis `matches.parquet` après l’append.

Note BO/series:
- Le champ `bo_type` est inféré “par série” via `series_id+leagueid`. Après append, on le **ré-inférer** sur l’ensemble de `matches.parquet` pour éviter les incohérences si une série se complète en plusieurs runs.

### 3) Metrics (gold) = support multi-sources raw
Certaines métriques utilisent des champs exclus des parquets “match header” (ex: `radiant_gold_adv`, `radiant_xp_adv`, `picks_bans`).

Solution pragmatique:
- `scripts/precompute_metrics.py` accepte maintenant **plusieurs sources raw** (`--raw` repeatable), fichier ou répertoire.
- Par défaut, il essaie `data/raw/data_v2.json` + `data/raw/updates/` (si présent).

Cela évite d’avoir à append un `data_v2.json` géant, tout en couvrant les nouveaux matchs.

## Pipeline concrète (nouveaux points d’entrée)
### Mise à jour incrémentale (raw + processed)
- Commande Make:
  - `make update OUT=data/processed`
- Options utiles:
  - `MAX_NEW=200` (cap sur les matchs téléchargés, plus récents d’abord)
  - `DRY_RUN=1` (ne télécharge pas, écrit seulement `run_metadata.json`)
  - `make healthcheck OUT=data/processed` (sanity check + traces match/série du dernier run)
- Ce que ça fait:
  1) compare les `match_id` existants (via `data/processed/matches.parquet`)
  2) interroge `/teams/{id}/matches` pour trouver les matchs récents non présents
  3) télécharge `/matches/{match_id}` en chunks dans `data/raw/updates/run_.../`
  4) append les parquets dans `data/processed/`

### Recalcul des métriques
- `make precompute OUT=data/processed METRICS_OUT=data/metrics`
- Le script chargera automatiquement `data/raw/data_v2.json` + `data/raw/updates/` (sauf si `--raw` est fourni).

## Impact sur l’existant
- `make parquet` / `python -m src.dota_data.io` restent valides (conversion depuis un raw JSON unique).
- Nouveau module: `src/dota_data/update.py` (sync incrémentale + option `--apply-parquet`).
- `scripts/precompute_metrics.py` change d’interface: `--raw` devient repeatable (compatible si on ne passe rien).
- Les dashboards continuent à lire `data/processed/*.parquet` (on reste sur des fichiers uniques pour l’instant).

## Roadmap (phases)
### Phase A — déjà en place (MVP incrémental)
- [x] Découverte de nouveaux matchs vs `matches.parquet`
- [x] Téléchargement en chunks sous `data/raw/updates/run_.../`
- [x] Append batch → `data/processed/*.parquet` + recompute `series.parquet`
- [x] Precompute support `--raw` multi-sources (fichier + répertoires)

### Phase B — fiabilisation & DX
- [x] Ajouter un `--max-new` (cap) et un `--dry-run` côté Make (optionnel)
- [x] Écrire une petite “healthcheck” (ex: % de raw_map couverte, nb de matchs sans adv arrays)
- [x] Logging plus clair + résumé final (nb nouveaux matchs / nb erreurs / temps)

### Phase C — sortir complètement du “gros raw”
Option 1 (recommended):
- [x] Extraire dès la conversion parquet un `extras.parquet` (match_id + `radiant_gold_adv`/`radiant_xp_adv`/`picks_bans` en JSON string)
- [x] Faire pointer `precompute_metrics.py` sur `extras.parquet` (plus besoin de raw JSON)
- [ ] (Une fois complet) supprimer la dépendance à `data/raw/data_v2.json`

Notes migration:
- Pour backfill `extras.parquet` sans réécrire `players/objectives/...`, utiliser `make extras RAW=data/raw/data_v2.json OUT=data/processed` (streaming).

Option 2:
- [ ] Migrer l’historique: convertir `data/raw/data_v2.json` vers une arbo `data/raw/chunks_full/` (une fois) et ne garder que les chunks

### Phase D — scalabilité (si on grossit beaucoup)
- [ ] Passer `data/processed/*` en dataset partitionné (ex: `players/part-*.parquet`) + compaction périodique
- [ ] Lecture via `scan_parquet` (lazy) dans les notebooks et le dashboard

## Snippets “notebook” pour tester petit
### 1) Découvrir les nouveaux `match_id` (sans télécharger)
```python
from pathlib import Path
from src.dota_data.update import download_new_matches

res = download_new_matches(
    teams_csv=Path("data/teams_to_look.csv"),
    processed_dir=Path("data/processed"),
    raw_updates_dir=Path("data/raw/updates"),
    limit=50,
    max_pages=2,
    dry_run=True,
)
res
```

### 2) Télécharger + append parquets (petit batch)
```python
from pathlib import Path
from src.dota_data.update import download_new_matches, apply_raw_run_to_processed

res = download_new_matches(
    teams_csv=Path("data/teams_to_look.csv"),
    processed_dir=Path("data/processed"),
    raw_updates_dir=Path("data/raw/updates"),
    limit=50,
    max_pages=2,
    chunk_size=25,
    sleep_match_detail=0.5,
    dry_run=False,
)

apply_raw_run_to_processed(
    run_dir=Path(res["run_dir"]),
    processed_dir=Path("data/processed"),
    aliases=Path("data/team_aliases.csv"),
)
```

### 3) Valider rapidement l’append
```python
import polars as pl

matches = pl.read_parquet("data/processed/matches.parquet")
players = pl.read_parquet("data/processed/players.parquet")

matches.select(pl.len(), pl.col("match_id").n_unique())
players.select(pl.len(), pl.col("match_id").n_unique())
```

### 4) Recompute métriques en incluant updates
```bash
make precompute OUT=data/processed METRICS_OUT=data/metrics
```
