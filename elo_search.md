Non, tu peux garder **exactement la même base Elo** (logistique), et jouer uniquement sur **qui tu updates** (team / joueur / rôle / patch) et **comment tu répartis le delta**.
Je te fais un plan “spéc technique” pour ta roadmap.

---

## 0. Notations générales

* Tous les ratings sont à **l’échelle Elo classique** : base 1500.
* On travaille au **niveau de la map** (1 ligne = 1 map).
* Pour chaque map `m` :

  * `A`, `B` : id des équipes.
  * `S_A` : score de l’équipe A (1 = A gagne la map, 0 = A perd).
  * `patch(m)` : version de patch.
  * `players(A, m)` : liste des 5 joueurs de A sur la map.
  * Pour chaque joueur `i` de la map :

    * `team(i, m)` ∈ {A, B}
    * `role(i, m)` ∈ {1, 2, 3, 4, 5} (ou ton mapping support/carry/etc).
    * `patch(m)` idem ci-dessus.

Base Elo (commune à tout) :

* Proba attendue que A gagne :

[
E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}
]

* Update générique :

[
R'_A = R_A + K \times (S_A - E_A)
]

---

## 1. Team Elo (global + par patch)

### 1.1. États stockés

* `TeamEloGlobal[team] = (R_team, games_team)`
* `TeamEloPatch[patch][team] = (R_team_patch, games_team_patch)`

Initialisation (si team jamais vue) :

* `R_team = 1500`, `games_team = 0`
* pour `TeamEloPatch[p]`, initialisation à la **première map de la team sur le patch p** (cf. 1.3).

### 1.2. K-factor team global

Fonction recommandée :

```text
K_team_global(games) =
    40 si games < 30
    25 si 30 ≤ games < 80
    15 si games ≥ 80
```

Pour le patch, même logique mais appliquée à `games_team_patch`.

### 1.3. Initialisation par patch

Quand une team `T` joue pour la première fois sur un patch `p` :

* `R_team_patch[T, p] = β_team * R_team_global[T] + (1 - β_team) * 1500`
* `games_team_patch[T, p] = 0`

Avec par ex. `β_team = 0.75`.

### 1.4. Update par map

Pour une map `m` entre A et B, sur patch `p` (trié chronologiquement) :

1. **Global**

   * Récupérer `R_A, R_B, games_A, games_B` dans `TeamEloGlobal`.
   * Calculer `E_A_global` via la formule Elo.
   * Déterminer `K_A_global`, `K_B_global` via `K_team_global`.
   * Définir `K_match_global = (K_A_global + K_B_global) / 2`.
   * Eventuellement multiplier par un facteur d’importance `w_importance` (LAN / Major etc.).
   * `Δ_global = K_match_global * w_importance * (S_A - E_A_global)`
   * Mettre à jour :

     * `R_A += Δ_global`, `R_B -= Δ_global`
     * `games_A += 1`, `games_B += 1`

2. **Patch p**

   * Si A ou B n’ont pas encore de rating sur ce patch, faire l’init (1.3).
   * Idem global mais avec `R_A_patch, R_B_patch, games_*_patch`.
   * Calculer `E_A_patch` avec `R_team_patch`.
   * `K_match_patch` comme en global mais basé sur `games_team_patch`.
   * `Δ_patch = K_match_patch * w_importance_patch * (S_A - E_A_patch)`
   * Mettre à jour `R_A_patch`, `R_B_patch`, `games_*_patch`.

👉 Résultat : pour chaque team tu as :

* un Elo global (tous matchs),
* un Elo par patch (reset soft à chaque patch).

---

## 2. Player Elo overall (global + patch)

Idée : chaque joueur est “récompensé/pénalisé” selon la **probabilité de victoire de son équipe**, pas besoin de recalc une proba à partir de ses propres Elo.

### 2.1. États stockés

Pour chaque joueur `i` :

* Global :

  * `PlayerGlobalOverall[i] = (G_i, games_i_global)`
* Par patch `p` :

  * `PlayerPatchOverall[p][i] = (P_i_p, games_i_patch)`

Initialisation (jamais vu global) :
`G_i = 1500`, `games_i_global = 0`.

Initialisation patch `p` (première map de `i` sur `p`) :

* `P_i_p = β_player * G_i + (1 - β_player) * 1500`
* `games_i_patch = 0`
  avec par ex. `β_player = 0.75`.

### 2.2. K-factor player

Même principe que pour les teams, mais éventuellement avec d’autres seuils :

```text
K_player_global(games_i_global) =
    40 si games < 30
    25 si 30 ≤ games < 80
    15 si games ≥ 80
```

Patch :

```text
K_player_patch(games_i_patch) =
    40 si games < 20
    25 si 20 ≤ games < 50
    15 si games ≥ 50
```

### 2.3. Update par map

Pour une map `m`, sur patch `p`, on a déjà calculé :

* `E_side_global(team)` via `TeamEloGlobal`.
* `E_side_patch(team)` via `TeamEloPatch[p]`.

Pour chaque joueur `i` sur la map :

* `t = team(i, m)`
* `S_i = 1` si `t` a gagné la map, sinon `0`.

**Global overall :**

* Récupérer `G_i, games_i_global`.
* `K_i_global = K_player_global(games_i_global)`
* `E_i_global = E_side_global(t)` (proba de win de son équipe, réutilisée pour tous les joueurs de cette équipe).
* `Δ_i_global = K_i_global * (S_i - E_i_global)`
* Update :

  * `G_i += Δ_i_global`
  * `games_i_global += 1`

**Patch overall :**

* Initialiser `P_i_p` si besoin.
* `K_i_patch = K_player_patch(games_i_patch)`
* `E_i_patch = E_side_patch(t)`
* `Δ_i_patch = K_i_patch * (S_i - E_i_patch)`
* Update :

  * `P_i_p += Δ_i_patch`
  * `games_i_patch += 1`

👉 Ça te donne une **force globale du joueur** (tous rôles confondus) et une force par patch.

---

## 3. Player Elo par rôle (global + patch)

On ajoute une couche “spécialisation” par rôle.
Idée : une partie du signal va dans le rating global du joueur, l’autre dans son rating spécifique au rôle.

### 3.1. États stockés

Pour chaque joueur `i` et rôle `r` :

* Global :

  * `PlayerGlobalRole[i][r] = R_i_r_global`
* Patch `p` :

  * `PlayerPatchRole[p][i][r] = R_i_r_patch`

Initialisation :

* Global : première apparition du joueur dans le rôle `r` :

  * `R_i_r_global = G_i` (ou 1500 si tu préfères).
* Patch `p`, rôle `r`, première apparition :

  * `R_i_r_patch = β_role * R_i_r_global + (1 - β_role) * 1500`
    (par ex. `β_role = 0.75`).

### 3.2. Partage du delta global vs rôle

On réutilise les `Δ_i_global` et `Δ_i_patch` calculés en 2.3 pour le joueur `i`.

Définir un coefficient de partage `α ∈ [0, 1]`, par ex. `α = 0.3` :

* `α` = part du delta qui va dans l’overall du joueur.
* `1 - α` = part du delta qui va dans le rating spécifique au rôle.

Pour une map `m` où le joueur `i` joue le rôle `r = role(i, m)` :

**Global :**

* `G_i += α * Δ_i_global`
* `R_i_r_global += (1 - α) * Δ_i_global`

**Patch p :**

* `P_i_p += α * Δ_i_patch`
* `R_i_r_patch += (1 - α) * Δ_i_patch`

👉 Résultat :

* `G_i` ≈ skill overall du joueur, tous rôles confondus.
* `R_i_r_global` = skill spécifique quand il joue le rôle `r` (car supply, mid, offlane, etc.).
* Idem pour `P_i_p` et `R_i_r_patch` mais **restreint au patch p**.

---

## 4. Résumé logique (pour l’implémentation)

Pour chaque map `m` **dans l’ordre chronologique** :

1. Lire : `A, B, patch, S_A`, lineups, rôles.
2. **Teams :**

   * Initialiser / charger `TeamEloGlobal` et `TeamEloPatch[patch]`.
   * Calculer `E_A_global`, `E_A_patch`.
   * Mettre à jour ratings team global et patch.
3. **Players :**

   * Pour chaque joueur `i` :

     * Initialiser / charger `G_i`, `P_i_patch`, `R_i_r_global`, `R_i_r_patch`.
     * Calculer `Δ_i_global = K_i_global * (S_i - E_side_global(team(i)))`.
     * Calculer `Δ_i_patch = K_i_patch * (S_i - E_side_patch(team(i)))`.
     * Appliquer les updates :

       * `G_i += α * Δ_i_global`
       * `R_i_role_global += (1 - α) * Δ_i_global`
       * `P_i_patch += α * Δ_i_patch`
       * `R_i_role_patch += (1 - α) * Δ_i_patch`
     * Incrémenter `games_i_global` / `games_i_patch`.

Formules Elo = **identiques partout**, tu changes seulement :

* la “clé” du rating (team vs joueur vs joueur+role),
* le “scope” (global vs patch),
* et le partage du delta (α entre overall et rôle).

---

Si tu veux, quand tu auras placé ça dans ta roadmap, on pourra faire ensemble :

* les signatures de fonctions / classes (genre `EloEngine`, `update_team`, `update_player`),
* puis voir comment exploiter ces rating (prévision de match, top players par patch et par rôle, etc.).
