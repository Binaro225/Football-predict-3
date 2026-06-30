"""
Construit le vecteur de features attendu par les modèles ML à partir des
données brutes récupérées via football_data_client.

En mode démo (sans clé API), ou lorsque l'historique réel est insuffisant
pour calculer des statistiques fiables, on complète avec des estimations
raisonnables basées sur les moyennes du championnat plutôt que d'inventer
des valeurs arbitraires, et on le signale dans `data_quality`.
"""
import random
from dataclasses import dataclass

from app.services.football_data_client import FootballDataClient


LEAGUE_AVG = {
    "xg": 1.35, "xga": 1.35, "shots": 12.5, "sot": 4.5,
    "possession": 50.0, "corners": 5.0, "fouls": 11.0,
    "yellow": 1.9, "rank": 10,
}


@dataclass
class TeamFeatureSet:
    form_pts5: float
    xg_avg: float
    xga_avg: float
    shots_avg: float
    shots_on_target_avg: float
    possession_avg: float
    corners_avg: float
    fouls_avg: float
    yellow_avg: float
    red_rate: int
    rank: int
    win_streak: int
    injuries_count: int
    rest_days: int
    elo: float
    data_quality: str  # "real" | "estimated"


def _form_points(results: list[str]) -> float:
    """results: liste de 'W'/'D'/'L', les plus récents en premier."""
    pts_map = {"W": 3, "D": 1, "L": 0}
    last5 = results[:5]
    return sum(pts_map.get(r, 0) for r in last5)


def _win_streak(results: list[str]) -> int:
    streak = 0
    for r in results:
        if r == "W":
            streak += 1
        else:
            break
    return streak


async def build_team_features(
    client: FootballDataClient, team_id: int, team_name: str, fallback_rank: int = 10,
) -> TeamFeatureSet:
    raw_matches = await client.get_team_recent_form(team_id, limit=10)

    if not raw_matches:
        return _estimated_features(fallback_rank)

    goals_for, goals_against, results = [], [], []
    shots_proxy, corners_proxy = [], []

    for m in raw_matches:
        score = m.get("score", {}).get("fullTime", {})
        gh, ga = score.get("home"), score.get("away")
        if gh is None or ga is None:
            continue
        is_home = m.get("homeTeam", {}).get("id") == team_id
        gf = gh if is_home else ga
        gc = ga if is_home else gh
        goals_for.append(gf)
        goals_against.append(gc)
        if gf > gc:
            results.append("W")
        elif gf == gc:
            results.append("D")
        else:
            results.append("L")
        # Proxies raisonnables dérivés des buts (l'API gratuite ne fournit pas xG/tirs détaillés)
        shots_proxy.append(8 + gf * 2.5)
        corners_proxy.append(4 + gf * 0.8)

    if not goals_for:
        return _estimated_features(fallback_rank)

    n = len(goals_for)
    avg_gf = sum(goals_for) / n
    avg_ga = sum(goals_against) / n

    elo = 1500 + (avg_gf - avg_ga) * 60 - (fallback_rank - 10) * 8

    return TeamFeatureSet(
        form_pts5=_form_points(results),
        xg_avg=round(max(0.3, avg_gf * 0.95), 2),
        xga_avg=round(max(0.3, avg_ga * 0.95), 2),
        shots_avg=round(sum(shots_proxy) / n, 1),
        shots_on_target_avg=round(sum(shots_proxy) / n * 0.38, 1),
        possession_avg=round(LEAGUE_AVG["possession"] + (avg_gf - avg_ga) * 2.5, 1),
        corners_avg=round(sum(corners_proxy) / n, 1),
        fouls_avg=LEAGUE_AVG["fouls"],
        yellow_avg=LEAGUE_AVG["yellow"],
        red_rate=0,
        rank=fallback_rank,
        win_streak=_win_streak(results),
        injuries_count=0,
        rest_days=6,
        elo=round(elo, 1),
        data_quality="real",
    )


def _estimated_features(rank: int) -> TeamFeatureSet:
    """Estimation basée sur le rang d'équipe quand aucune donnée fraîche n'est disponible."""
    rank_factor = (10 - rank) / 10  # >0 si meilleur que la moyenne, <0 sinon
    return TeamFeatureSet(
        form_pts5=round(8 + rank_factor * 4, 1),
        xg_avg=round(LEAGUE_AVG["xg"] + rank_factor * 0.4, 2),
        xga_avg=round(LEAGUE_AVG["xga"] - rank_factor * 0.4, 2),
        shots_avg=round(LEAGUE_AVG["shots"] + rank_factor * 3, 1),
        shots_on_target_avg=round(LEAGUE_AVG["sot"] + rank_factor * 1.5, 1),
        possession_avg=round(LEAGUE_AVG["possession"] + rank_factor * 6, 1),
        corners_avg=round(LEAGUE_AVG["corners"] + rank_factor * 1.2, 1),
        fouls_avg=LEAGUE_AVG["fouls"],
        yellow_avg=LEAGUE_AVG["yellow"],
        red_rate=0,
        rank=rank,
        win_streak=1 if rank_factor > 0.3 else 0,
        injuries_count=1,
        rest_days=6,
        elo=round(1500 + rank_factor * 250, 1),
        data_quality="estimated",
    )


def to_feature_vector(home: TeamFeatureSet, away: TeamFeatureSet, h2h_home_win_rate: float = 0.45) -> dict:
    """Construit le dict de features dans l'ordre attendu par le modèle (FEATURE_COLUMNS)."""
    return {
        "home_strength_elo": home.elo,
        "away_strength_elo": away.elo,
        "home_form_pts5": home.form_pts5,
        "away_form_pts5": away.form_pts5,
        "home_xg_avg": home.xg_avg,
        "away_xg_avg": away.xg_avg,
        "home_xga_avg": home.xga_avg,
        "away_xga_avg": away.xga_avg,
        "home_shots_avg": home.shots_avg,
        "away_shots_avg": away.shots_avg,
        "home_shots_on_target_avg": home.shots_on_target_avg,
        "away_shots_on_target_avg": away.shots_on_target_avg,
        "home_possession_avg": home.possession_avg,
        "away_possession_avg": away.possession_avg,
        "home_corners_avg": home.corners_avg,
        "away_corners_avg": away.corners_avg,
        "home_fouls_avg": home.fouls_avg,
        "away_fouls_avg": away.fouls_avg,
        "home_yellow_avg": home.yellow_avg,
        "away_yellow_avg": away.yellow_avg,
        "home_red_rate": home.red_rate,
        "away_red_rate": away.red_rate,
        "home_rank": home.rank,
        "away_rank": away.rank,
        "home_injuries_count": home.injuries_count,
        "away_injuries_count": away.injuries_count,
        "home_rest_days": home.rest_days,
        "away_rest_days": away.rest_days,
        "home_win_streak": home.win_streak,
        "away_win_streak": away.win_streak,
        "h2h_home_win_rate": h2h_home_win_rate,
    }
