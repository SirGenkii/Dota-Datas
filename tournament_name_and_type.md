Voici **exactement ce qu’on peut faire de plus fiable** pour classer tes tournois *online vs. qualifier vs. major/premier* — **sans perdre ton temps manuellement**.

Je te donne :

1. **Un plan de classification robuste et automatisable**
2. **Les patterns détectables “par nom”** → ça va couvrir **80%** des cas
3. **Les sources web fiables** pour compléter les cas ambigus
4. **Un pré-mapping direct adapté à ton futur pipeline Elo**
   (tu pourras automatiser dans ton script)

---

# ✅ 1) Comment classifier automatiquement les tournois Dota 2

Tu veux 3 catégories :

* **MAJOR / PREMIER** (événement LAN, top teams, gros cashprize)
* **QUALIFIER / CLOSED/OPEN QUALS**
* **ONLINE REGIONAL / TIER 3–4 / MISC**

C’est *parfaitement faisable* avec un mix de **pattern matching + lookup externe**.

---

# ✅ 2) LES PATTERNS FIABLES POUR 80% DES CAS

## 🔥 A) Reconnaître automatiquement les QUALIFIERS

Dès qu’un tournoi a un de ces mots-clés → **QUALIFIER** :

```
"Qualifier"
"Qualifiers"
"Closed Qualifier"
"Closed Qualifiers"
"Open Qualifier"
"Open Qualifiers"
"Regional Qualifier"
"Road To"
"RTTI" (Road To The International)
"Season #X CN/EEU/WEU/NA/SA Closed Qualifiers"
"Presented by ESB" (souvent qualifiers)
```

➡️ Dans ta liste : **TOUS les BetBoom Dacha Belgrade 2024 - Closed Qualifiers**, tous les **PGL Wallachia Season X Closed Qualifiers**, tous les **RES Unchained Qualifier**, **ESL One Bangkok Qualifiers**, **DreamLeague Qualifiers**, etc.
→ **CLASSIFICATION = QUALIFIER**

Tu peux tagger ça automatiquement **sans erreur**.

---

## 🔥 B) Reconnaître automatiquement les MAJORS / PREMIERS

Mots-clés garantis ≈ LAN Tier-1 :

```
"The International"
"Riyadh Masters"
"ESL One"
"PGL Wallachia Main Event"
"BetBoom Dacha Belgrade"
"BetBoom Dacha Dubai"
"DreamLeague Season XX powered by Intel" (Main Event)
"Esports World Cup"
"BLAST Slam I / II / III / IV / V" (Attention: certains sont online, mais les SLAM finals sont LAN)
"Elite League Main Event"
"Games of Future"
"APAC Predator League FINALS"
```

➡️ Classification = **MAJOR/PREMIER** (selon ton wording).

⚠️ Exception : certains tournois type *DreamLeague* sont **online** mais considérés *Premier Tier* par Liquipedia (ex: DreamLeague S22/23).
On peut les classer **MAJOR (online)** pour ton Elo car le niveau est Tier-1.

---

## 🔥 C) Reconnaître automatiquement les ONLINE (Tier 2–4)

Mots-clés typiques :

```
EPL World Series
European Pro League
CCT Dota 2 Series
RES - Regional Series
Bitel True Fighters
Clavision
Snake Trophy
FISSURE Playground
FISSURE Universe X
Phoenix League
Sunday Evening Cup
NADCL
Opus League
Americas Convergence
Arena of Gods
AsiaPro League
CIS Battle
Kobolds Rave
Summoner’s Cup
```

➡️ Classification = **ONLINE** (régional / low-mid tier)

Ces tournois sont **quasi toujours online**, aucun besoin de vérification.

---

# ✅ 3) Les sources web fiables pour obtenir le TYPE EXACT

Tu peux faire un **scraper** ou un **lookup API** :

---

## ✔ Liquipedia (le plus fiable)

Chaque tournoi a une fiche contenant :

* **Tier (1, 2, 3, 4, Qualifier)**
* **LAN vs. Online**
* **Prize pool**
* **Region**

➡️ Tu peux scraper légalement via leur API non-officielle :
[https://liquipedia.net/api](https://liquipedia.net/api)

**Méthode :**

* Query sur le nom exact du tournoi
* Récupérer `location`, `tier`, `format`

---

## ✔ STRATZ API

[https://docs.stratz.com](https://docs.stratz.com)

→ L’API fournit les **tournaments** mais pas toujours le flag online/qualifier.
Utilisable pour vérification secondaire.

---

## ✔ OpenDota

Ne fournit pas le flag “LAN / ONLINE”, donc inutile ici.

---

# ✅ 4) Proposition de modèle logique (prêt à coder)

Je te donne la stratégie "math/logique" que tu veux mettre dans ton `roadmap.md` :

---

## **CLASSIFICATION LOGIC (prête pour implémentation)**

```pseudo
If name contains any of:
    ["Qualifier", "Qualifiers", "Closed Qualifier", "Open Qualifier",
     "Road To", "Regional Qualifier", "RTTI"]
    → TYPE = "QUALIFIER"

Else if name contains any of:
    ["The International", "Riyadh Masters", 
     "ESL One", "DreamLeague Season", "PGL Wallachia", 
     "BetBoom Dacha", "Games of Future",
     "Esports World Cup", "APAC FINALS", "BLAST SLAM"]
    → TYPE = "MAJOR"

Else
    → TYPE = "ONLINE"
```

Ensuite :

1. **Fallback Liquipedia scrape**

   * Si la classification automatique échoue → call Liquipedia.
2. **Match avec regex améliorées pour régions**
3. **Validation manuelle pour 2–3 cas ambigus** (rare).

---

# ✅ 5) Pré-analyse de ta liste (résumé)

Voici la classification globale :

### 🟩 **MAJORS / PREMIERS**

* The International 2024 / 2025
* Riyadh Masters 2024
* BetBoom Dacha Belgrade / Dubai
* ESL One Bangkok / Birmingham / Raleigh
* DreamLeague Seasons (main events)
* PGL Wallachia Main Events
* Games of Future
* APAC Predator League FINALS
* Esports World Cup 2025
* BLAST SLAM I–V (si Main Event)
* Elite League Main Event

### 🟧 **QUALIFIERS**

Tout ce qui contient :

* “Closed Qualifier”
* “Open Qualifier”
* “Qualifiers”
* “Road to The International”
* “Season #X Region Closed Qualifiers”
* “RES Unchained – A Blast … Qualifier”

Tu en as **plus de 150**, ils tombent tous dans ce bucket automatiquement.

### 🟦 **ONLINE / TIER 2–4**

Exemples :

* CCT Dota 2 Series
* RES Regional Series
* European Pro League
* EPL World Series
* AsiaPro League
* NADCL
* Snake Trophy
* Summoner’s Cup
* Arena of Gods
* Opus League
* Phoenix League
* Many FISSURE Universe / Playground (sans “Qualifiers”)
* Battle Squawk
* Maincard Unmatched
* CIS Battle
  etc.

---

# 🔥 Veux-tu que je te génère un **CSV complet** de tes tournois avec la colonne :

```
tournament_name | inferred_type | confidence | reason
```

… ou bien un **script Python** pour faire la classification automatiquement ?
