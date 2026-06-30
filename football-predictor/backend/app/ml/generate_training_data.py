"""
Générateur de données d'entraînement synthétiques.

En production, ce module est remplacé par de vraies données historiques
récupérées via football_data_client.py (API football-data.org) et stockées
en base / CSV avant l'entraînement (voir train_models.py).

Pour permettre un entraînement immédiat et reproductible sans dépendre
d'une clé API externe, on génère ici un jeu de données statistiquement
cohérent avec les distributions réelles du football européen (xG, forme,
possession, etc.), en y injectant un signal contrôlé liant les features
au résultat afin que les modèles aient quelque chose de réel à apprendre.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_SAMPLES = 6000


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    # Force d'équipe latente (Elo-like), détermine en grande partie tout le reste
    home_strength = RNG.normal(1500, 180, n_samples)
    away_strength = RNG.normal(1500, 180, n_samples)

    home_advantage = 60  # avantage du terrain, en points Elo équivalents

    strength_diff = (home_strength + home_advantage) - away_strength

    # Forme récente (points sur 5 derniers matchs, 0-15), corrélée à la force
    home_form = _clip(RNG.normal(8 + strength_diff / 120, 3, n_samples), 0, 15)
    away_form = _clip(RNG.normal(8 - strength_diff / 120, 3, n_samples), 0, 15)

    # xG moyen par match (saison), corrélé à la force
    home_xg = _clip(RNG.normal(1.4 + strength_diff / 900, 0.35, n_samples), 0.3, 3.5)
    away_xg = _clip(RNG.normal(1.2 - strength_diff / 900, 0.35, n_samples), 0.2, 3.2)

    home_xga = _clip(RNG.normal(1.2 - strength_diff / 900, 0.3, n_samples), 0.2, 3.0)
    away_xga = _clip(RNG.normal(1.4 + strength_diff / 900, 0.3, n_samples), 0.3, 3.2)

    home_shots = _clip(RNG.normal(13 + strength_diff / 100, 3.5, n_samples), 3, 28)
    away_shots = _clip(RNG.normal(11 - strength_diff / 100, 3.5, n_samples), 3, 26)

    home_shots_on_target = _clip(home_shots * RNG.uniform(0.32, 0.45, n_samples), 1, 14)
    away_shots_on_target = _clip(away_shots * RNG.uniform(0.32, 0.45, n_samples), 1, 14)

    home_possession = _clip(RNG.normal(50 + strength_diff / 40, 7, n_samples), 28, 75)
    away_possession = 100 - home_possession

    home_corners = _clip(RNG.normal(5.5 + strength_diff / 250, 2.0, n_samples), 0, 14)
    away_corners = _clip(RNG.normal(4.5 - strength_diff / 250, 2.0, n_samples), 0, 14)

    home_fouls = _clip(RNG.normal(11, 2.5, n_samples), 4, 22)
    away_fouls = _clip(RNG.normal(11.5, 2.5, n_samples), 4, 22)

    home_yellow = _clip(RNG.poisson(1.8, n_samples), 0, 6)
    away_yellow = _clip(RNG.poisson(2.0, n_samples), 0, 6)
    home_red = (RNG.uniform(0, 1, n_samples) < 0.05).astype(int)
    away_red = (RNG.uniform(0, 1, n_samples) < 0.06).astype(int)

    home_rank = _clip(RNG.normal(10 - strength_diff / 35, 5, n_samples), 1, 20).round()
    away_rank = _clip(RNG.normal(10 + strength_diff / 35, 5, n_samples), 1, 20).round()

    home_injuries = RNG.poisson(1.5, n_samples)
    away_injuries = RNG.poisson(1.5, n_samples)

    home_rest_days = RNG.integers(2, 10, n_samples)
    away_rest_days = RNG.integers(2, 10, n_samples)

    home_win_streak = _clip(RNG.poisson(1.2 + max(0, 1) * (strength_diff > 0) * 0.5, n_samples), 0, 8)
    away_win_streak = _clip(RNG.poisson(1.0, n_samples), 0, 8)

    h2h_home_win_rate = _clip(RNG.normal(0.4 + strength_diff / 1000, 0.15, n_samples), 0, 1)

    # --- Simulation du résultat via un modèle de buts type Poisson, influencé par les features ---
    lambda_home = _clip(home_xg * 0.6 + away_xga * 0.4 + (strength_diff / 1000), 0.2, 4.0)
    lambda_away = _clip(away_xg * 0.6 + home_xga * 0.4 - (strength_diff / 1000), 0.15, 3.5)

    goals_home = RNG.poisson(lambda_home)
    goals_away = RNG.poisson(lambda_away)

    result = np.where(goals_home > goals_away, "H", np.where(goals_home < goals_away, "A", "D"))
    btts = ((goals_home > 0) & (goals_away > 0)).astype(int)
    total_goals = goals_home + goals_away
    over_2_5 = (total_goals > 2.5).astype(int)

    df = pd.DataFrame({
        "home_strength_elo": home_strength,
        "away_strength_elo": away_strength,
        "home_form_pts5": home_form,
        "away_form_pts5": away_form,
        "home_xg_avg": home_xg,
        "away_xg_avg": away_xg,
        "home_xga_avg": home_xga,
        "away_xga_avg": away_xga,
        "home_shots_avg": home_shots,
        "away_shots_avg": away_shots,
        "home_shots_on_target_avg": home_shots_on_target,
        "away_shots_on_target_avg": away_shots_on_target,
        "home_possession_avg": home_possession,
        "away_possession_avg": away_possession,
        "home_corners_avg": home_corners,
        "away_corners_avg": away_corners,
        "home_fouls_avg": home_fouls,
        "away_fouls_avg": away_fouls,
        "home_yellow_avg": home_yellow,
        "away_yellow_avg": away_yellow,
        "home_red_rate": home_red,
        "away_red_rate": away_red,
        "home_rank": home_rank,
        "away_rank": away_rank,
        "home_injuries_count": home_injuries,
        "away_injuries_count": away_injuries,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_win_streak": home_win_streak,
        "away_win_streak": away_win_streak,
        "h2h_home_win_rate": h2h_home_win_rate,
        "goals_home": goals_home,
        "goals_away": goals_away,
        "result": result,
        "btts": btts,
        "over_2_5": over_2_5,
    })
    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("app/data/training_data.csv", index=False)
    print(f"Dataset généré: {len(data)} lignes -> app/data/training_data.csv")
    print(data["result"].value_counts(normalize=True))
