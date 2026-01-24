from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

# Glicko-2 constants
_SCALE = 173.7178
_PI2 = math.pi**2

# We reuse the same weighting philosophy as the Elo implementation.
TIER_WEIGHTS = {"major": 1.25, "qualifier": 1.05, "regular": 1.0}
LOCATION_WEIGHTS = {"lan": 1.05, "online": 1.0}


def match_weight(tier: str | None, location: str | None) -> float:
    tier_w = TIER_WEIGHTS.get(str(tier or "").lower(), 1.0)
    loc_w = LOCATION_WEIGHTS.get(str(location or "").lower(), 1.0)
    return float(tier_w * loc_w)


@dataclass(frozen=True)
class Glicko2Config:
    base_rating: float = 1500.0
    init_rd: float = 350.0
    init_sigma: float = 0.06
    tau: float = 0.5
    # How we batch updates (important for RD drift / inactivity).
    period: str = "day"  # one of: "day", "week", "series"
    # Conservative score used for ranking (handles uncertainty for sparse teams).
    score_rd_mult: float = 2.0


def _to_mu(rating: float) -> float:
    return (rating - 1500.0) / _SCALE


def _to_phi(rd: float) -> float:
    return rd / _SCALE


def _from_mu(mu: float, base_rating: float) -> float:
    return mu * _SCALE + base_rating


def _from_phi(phi: float) -> float:
    return phi * _SCALE


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + (3.0 * phi * phi) / _PI2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _update_sigma(phi: float, sigma: float, delta: float, v: float, tau: float, eps: float = 1e-6) -> float:
    # Glicko-2 volatility update (Glickman 2012), solved by iteration.
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return (num / den) - ((x - a) / (tau * tau))

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1.0
        B = A - k * tau
        while f(B) < 0.0:
            k += 1.0
            B = A - k * tau

    fA = f(A)
    fB = f(B)
    # Illinois / regula falsi variant recommended in the spec
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB < 0.0:
            A = B
            fA = fB
        else:
            fA = fA / 2.0
        B = C
        fB = fC

    return float(math.exp(A / 2.0))


def _rate_one_period(
    mu: float,
    phi: float,
    sigma: float,
    games: Sequence[Tuple[float, float, float, float]],
    tau: float,
) -> Tuple[float, float, float]:
    """
    Update one player for a rating period.
    games: list of (opp_mu, opp_phi, s, weight) where s in {0, 0.5, 1}.
    """
    # If no games, only RD increases (uncertainty grows).
    if not games:
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return mu, phi_star, sigma

    # v = 1 / Σ (w * g^2 * E(1-E))
    v_inv = 0.0
    delta_num = 0.0
    for mu_j, phi_j, s, w in games:
        g_j = _g(phi_j)
        E_j = 1.0 / (1.0 + math.exp(-g_j * (mu - mu_j)))
        v_inv += w * (g_j * g_j) * E_j * (1.0 - E_j)
        delta_num += w * g_j * (s - E_j)

    if v_inv <= 0.0:
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return mu, phi_star, sigma

    v = 1.0 / v_inv
    delta = v * delta_num
    sigma_prime = _update_sigma(phi=phi, sigma=sigma, delta=delta, v=v, tau=tau)

    phi_star = math.sqrt(phi * phi + sigma_prime * sigma_prime)
    phi_prime = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
    mu_prime = mu + (phi_prime * phi_prime) * delta_num
    return mu_prime, phi_prime, sigma_prime


def _period_key(ts: int, period: str) -> str:
    dt = datetime.fromtimestamp(int(ts or 0), tz=timezone.utc)
    if period == "series":
        return f"{dt.isoformat()}"
    if period == "week":
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return dt.date().isoformat()


def build_series_results(matches: pl.DataFrame) -> pl.DataFrame:
    """
    Build one row per series (unique by (leagueid, series_id)).

    Notes:
    - Some matches have null series_id => treated as BO1 series where series_id := match_id.
    - series_id alone is not globally unique; leagueid is required.
    - Winner/score computed from per-map radiant_win.
    """
    needed = [
        "match_id",
        "start_time",
        "leagueid",
        "series_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
        "bo_type",
        "series_type",
        "tournament_tier",
        "tournament_location",
    ]
    cols = [c for c in needed if c in matches.columns]
    if not cols:
        return pl.DataFrame([])
    df = matches.select(cols)

    series_map: Dict[Tuple[int, int], Dict[str, object]] = {}
    for r in df.iter_rows(named=True):
        mid = r.get("match_id")
        st = r.get("start_time")
        rid = r.get("radiant_team_id")
        did = r.get("dire_team_id")
        rw = r.get("radiant_win")
        if None in (mid, st, rid, did, rw):
            continue
        try:
            mid_i = int(mid)
            st_i = int(st)
            rid_i = int(rid)
            did_i = int(did)
        except Exception:  # noqa: BLE001
            continue
        leagueid = r.get("leagueid")
        series_id = r.get("series_id")
        try:
            league_i = int(leagueid) if leagueid is not None else 0
        except Exception:  # noqa: BLE001
            league_i = 0
        try:
            series_i = int(series_id) if series_id is not None else mid_i
        except Exception:  # noqa: BLE001
            series_i = mid_i

        key = (league_i, series_i)
        entry = series_map.get(key)
        if entry is None:
            entry = {
                "leagueid": league_i,
                "series_id": series_i,
                "start_time": st_i,
                "teams": set(),
                "wins": {},
                "maps": 0,
                "bo_type": r.get("bo_type"),
                "series_type": r.get("series_type"),
                "weight_sum": 0.0,
                "weight_n": 0,
            }
            series_map[key] = entry
        # Track last map time
        entry["start_time"] = max(int(entry.get("start_time") or 0), st_i)
        teams: set = entry["teams"]  # type: ignore[assignment]
        teams.add(rid_i)
        teams.add(did_i)
        wins: Dict[int, int] = entry["wins"]  # type: ignore[assignment]
        winner = rid_i if bool(rw) else did_i
        wins[winner] = int(wins.get(winner, 0)) + 1
        entry["maps"] = int(entry.get("maps") or 0) + 1

        tier = r.get("tournament_tier")
        loc = r.get("tournament_location")
        w = match_weight(tier, loc)
        entry["weight_sum"] = float(entry.get("weight_sum") or 0.0) + w
        entry["weight_n"] = int(entry.get("weight_n") or 0) + 1

        # Keep the first non-null BO info as hint for analysis/debug.
        if entry.get("bo_type") is None and r.get("bo_type") is not None:
            entry["bo_type"] = r.get("bo_type")
        if entry.get("series_type") is None and r.get("series_type") is not None:
            entry["series_type"] = r.get("series_type")

    rows: List[Dict[str, object]] = []
    for (_league_i, _series_i), entry in series_map.items():
        teams = sorted(int(t) for t in entry["teams"])  # type: ignore[index]
        if len(teams) != 2:
            continue
        a, b = int(teams[0]), int(teams[1])
        wins: Dict[int, int] = entry["wins"]  # type: ignore[assignment]
        wa = int(wins.get(a, 0))
        wb = int(wins.get(b, 0))
        if wa == wb:
            s_a = 0.5
        else:
            s_a = 1.0 if wa > wb else 0.0
        weight_n = int(entry.get("weight_n") or 0)
        weight = float(entry.get("weight_sum") or 0.0) / max(1, weight_n)
        start_time = int(entry.get("start_time") or 0)
        rows.append(
            {
                "leagueid": int(entry.get("leagueid") or 0),
                "series_id": int(entry.get("series_id") or 0),
                "start_time": start_time,
                "start_dt": datetime.fromtimestamp(start_time, tz=timezone.utc),
                "team_a_id": a,
                "team_b_id": b,
                "wins_a": wa,
                "wins_b": wb,
                "result_a": float(s_a),
                "maps": int(entry.get("maps") or 0),
                "bo_type": entry.get("bo_type"),
                "series_type": entry.get("series_type"),
                "weight": float(weight),
            }
        )

    if not rows:
        return pl.DataFrame([])
    return pl.DataFrame(rows, strict=False).sort(["start_time", "leagueid", "series_id"])


def compute_glicko2_from_series(
    series_results: pl.DataFrame,
    tracked_team_ids: Optional[Iterable[int]] = None,
    config: Optional[Glicko2Config] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Compute Glicko-2 ratings from *series* results (not per-map).

    Returns: (timeseries_tracked, latest_tracked, latest_all)
    """
    if config is None:
        config = Glicko2Config()
    tracked_set = set(int(x) for x in tracked_team_ids) if tracked_team_ids is not None else set()

    if series_results is None or series_results.is_empty():
        empty_ts = pl.DataFrame([], schema={"period": pl.Utf8, "team_id": pl.Int64, "rating": pl.Float64, "rd": pl.Float64, "sigma": pl.Float64, "series": pl.Int64})
        empty_latest = pl.DataFrame([], schema={"team_id": pl.Int64, "rating": pl.Float64, "rd": pl.Float64, "sigma": pl.Float64})
        return empty_ts, empty_latest, empty_latest

    needed = {"team_a_id", "team_b_id", "result_a", "start_time", "weight"}
    if not needed.issubset(set(series_results.columns)):
        raise ValueError(f"series_results missing required columns: {sorted(needed - set(series_results.columns))}")

    teams: set[int] = set()
    for r in series_results.select(["team_a_id", "team_b_id"]).iter_rows(named=True):
        if r.get("team_a_id") is not None:
            teams.add(int(r["team_a_id"]))
        if r.get("team_b_id") is not None:
            teams.add(int(r["team_b_id"]))

    # State in Glicko-2 scale.
    state: Dict[int, Tuple[float, float, float]] = {}
    for tid in teams:
        mu0 = _to_mu(config.base_rating)
        phi0 = _to_phi(config.init_rd)
        state[int(tid)] = (mu0, phi0, float(config.init_sigma))

    # Group series into rating periods.
    period = config.period
    if period not in {"day", "week", "series"}:
        raise ValueError("Glicko2Config.period must be one of: day, week, series")

    rows = series_results.select(["team_a_id", "team_b_id", "result_a", "start_time", "weight"]).iter_rows(named=True)
    by_period: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        ts = int(r.get("start_time") or 0)
        pk = _period_key(ts, period=period)
        by_period.setdefault(pk, []).append(r)

    periods_sorted = sorted(by_period.keys())
    ts_rows: List[Dict[str, object]] = []

    for pk in periods_sorted:
        batch = by_period[pk]
        snapshot = {tid: state[tid] for tid in state}
        games_by_team: Dict[int, List[Tuple[float, float, float, float]]] = {tid: [] for tid in state}
        series_count: Dict[int, int] = {tid: 0 for tid in state}

        for r in batch:
            a = r.get("team_a_id")
            b = r.get("team_b_id")
            s_a = r.get("result_a")
            w = float(r.get("weight") or 1.0)
            if None in (a, b, s_a):
                continue
            try:
                a_i = int(a)
                b_i = int(b)
                s = float(s_a)
            except Exception:  # noqa: BLE001
                continue
            if a_i not in snapshot or b_i not in snapshot:
                continue
            mu_a, phi_a, _sig_a = snapshot[a_i]
            mu_b, phi_b, _sig_b = snapshot[b_i]
            games_by_team[a_i].append((mu_b, phi_b, s, w))
            games_by_team[b_i].append((mu_a, phi_a, 1.0 - s, w))
            series_count[a_i] += 1
            series_count[b_i] += 1

        for tid in state:
            mu, phi, sigma = state[tid]
            games = games_by_team.get(tid, [])
            mu_p, phi_p, sigma_p = _rate_one_period(mu, phi, sigma, games, tau=float(config.tau))
            state[tid] = (mu_p, phi_p, sigma_p)

        # Persist timeseries for tracked teams only (keeps file size reasonable).
        if tracked_set:
            for tid in tracked_set:
                if tid not in state:
                    continue
                mu, phi, sigma = state[tid]
                ts_rows.append(
                    {
                        "period": pk,
                        "team_id": int(tid),
                        "rating": float(_from_mu(mu, config.base_rating)),
                        "rd": float(_from_phi(phi)),
                        "sigma": float(sigma),
                        "series": int(series_count.get(tid, 0)),
                    }
                )

    latest_rows: List[Dict[str, object]] = []
    for tid, (mu, phi, sigma) in state.items():
        rating = float(_from_mu(mu, config.base_rating))
        rd = float(_from_phi(phi))
        latest_rows.append({"team_id": int(tid), "rating": rating, "rd": rd, "sigma": float(sigma)})

    latest_all = pl.DataFrame(latest_rows, strict=False)
    latest_all = latest_all.with_columns(
        (pl.col("rating") - float(config.score_rd_mult) * pl.col("rd")).alias("score")
    ).with_columns(
        pl.col("rating").rank(method="dense", descending=True).alias("rank"),
        pl.col("score").rank(method="dense", descending=True).alias("score_rank"),
    ).sort("rank")

    latest_tracked = latest_all.filter(pl.col("team_id").is_in(list(tracked_set))) if tracked_set else pl.DataFrame([])
    ts_tracked = pl.DataFrame(ts_rows, strict=False) if ts_rows else pl.DataFrame([])
    return ts_tracked, latest_tracked, latest_all


def evaluate_glicko2_series(
    series_results: pl.DataFrame,
    config: Optional[Glicko2Config] = None,
    warmup_frac: float = 0.2,
) -> Dict[str, float]:
    """
    Walk-forward evaluation on series results.

    - Predict p(win) for team_a at the start of each rating period.
    - Update ratings at the end of each period.
    - Skip scoring during the first warmup_frac of series (stabilization phase).
    """
    if config is None:
        config = Glicko2Config()
    if series_results is None or series_results.is_empty():
        return {"series": 0.0, "logloss": float("nan"), "brier": float("nan")}

    df = series_results.select(["team_a_id", "team_b_id", "result_a", "start_time", "weight"]).sort("start_time")
    rows = list(df.iter_rows(named=True))
    total_series = len(rows)
    warmup_n = int(max(0, min(total_series, round(float(warmup_frac) * total_series))))
    warmup_cut_ts = None
    if warmup_n > 0:
        warmup_cut_ts = int(rows[warmup_n - 1].get("start_time") or 0)

    teams: set[int] = set()
    for r in rows:
        if r.get("team_a_id") is not None:
            teams.add(int(r["team_a_id"]))
        if r.get("team_b_id") is not None:
            teams.add(int(r["team_b_id"]))

    mu0 = _to_mu(config.base_rating)
    phi0 = _to_phi(config.init_rd)
    state: Dict[int, Tuple[float, float, float]] = {int(tid): (mu0, phi0, float(config.init_sigma)) for tid in teams}

    by_period: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        ts = int(r.get("start_time") or 0)
        pk = _period_key(ts, period=config.period)
        by_period.setdefault(pk, []).append(r)

    logloss_sum = 0.0
    brier_sum = 0.0
    scored = 0
    weight_sum = 0.0

    for pk in sorted(by_period.keys()):
        batch = by_period[pk]
        snapshot = {tid: state[tid] for tid in state}
        games_by_team: Dict[int, List[Tuple[float, float, float, float]]] = {tid: [] for tid in state}

        for r in batch:
            a = r.get("team_a_id")
            b = r.get("team_b_id")
            s_a = r.get("result_a")
            ts = int(r.get("start_time") or 0)
            w = float(r.get("weight") or 1.0)
            if None in (a, b, s_a):
                continue
            a_i = int(a)
            b_i = int(b)
            s = float(s_a)
            if a_i not in snapshot or b_i not in snapshot:
                continue

            mu_a, phi_a, _sig_a = snapshot[a_i]
            mu_b, phi_b, _sig_b = snapshot[b_i]
            p = _E(mu_a, mu_b, phi_b)

            if warmup_cut_ts is None or ts > warmup_cut_ts:
                # Weighted losses.
                eps = 1e-15
                p_clip = min(1.0 - eps, max(eps, p))
                logloss_sum += w * (-(s * math.log(p_clip) + (1.0 - s) * math.log(1.0 - p_clip)))
                brier_sum += w * ((s - p) ** 2)
                scored += 1
                weight_sum += w

            games_by_team[a_i].append((mu_b, phi_b, s, w))
            games_by_team[b_i].append((mu_a, phi_a, 1.0 - s, w))

        for tid in state:
            mu, phi, sigma = state[tid]
            games = games_by_team.get(tid, [])
            mu_p, phi_p, sigma_p = _rate_one_period(mu, phi, sigma, games, tau=float(config.tau))
            state[tid] = (mu_p, phi_p, sigma_p)

    if scored <= 0:
        return {"series": float(total_series), "logloss": float("nan"), "brier": float("nan")}

    denom = weight_sum if weight_sum > 0 else float(scored)
    return {
        "series": float(total_series),
        "scored": float(scored),
        "weight_sum": float(weight_sum),
        "logloss": float(logloss_sum / denom),
        "brier": float(brier_sum / denom),
    }
