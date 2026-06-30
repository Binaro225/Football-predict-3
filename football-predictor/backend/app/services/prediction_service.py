"""
Charge les artefacts ML entraînés (modèle résultat, BTTS, Over/Under,
scaler, encodeur) et expose une fonction de prédiction unique.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"

_result_model = None
_btts_model = None
_over_model = None
_scaler = None
_feature_columns = None
_label_encoder = None
_model_comparison = None


def _load_artifacts():
    global _result_model, _btts_model, _over_model, _scaler, _feature_columns, _label_encoder, _model_comparison
    if _result_model is not None:
        return
    _result_model = joblib.load(ML_DIR / "artifacts_result_model.joblib")
    _btts_model = joblib.load(ML_DIR / "artifacts_btts_model.joblib")
    _over_model = joblib.load(ML_DIR / "artifacts_over_model.joblib")
    _scaler = joblib.load(ML_DIR / "artifacts_scaler.joblib")
    _feature_columns = joblib.load(ML_DIR / "artifacts_feature_columns.joblib")
    _label_encoder = joblib.load(ML_DIR / "artifacts_label_encoder.joblib")
    comparison_path = ML_DIR / "model_comparison.json"
    if comparison_path.exists():
        import json
        _model_comparison = json.loads(comparison_path.read_text())


def get_model_comparison() -> dict | None:
    _load_artifacts()
    return _model_comparison


def predict_match(feature_dict: dict) -> dict:
    """
    feature_dict: dict avec les clés définies dans FEATURE_COLUMNS
    (voir feature_builder.to_feature_vector).

    Retourne un dict prêt à être sérialisé en réponse API.
    """
    _load_artifacts()

    X = pd.DataFrame([feature_dict])[_feature_columns]
    X_scaled = _scaler.transform(X)

    # --- Résultat (H/D/A) ---
    result_proba = _result_model.predict_proba(X_scaled)[0]
    result_classes = _label_encoder.inverse_transform(_result_model.classes_)
    proba_by_class = dict(zip(result_classes, result_proba))

    home_win_p = float(proba_by_class.get("H", 0))
    draw_p = float(proba_by_class.get("D", 0))
    away_win_p = float(proba_by_class.get("A", 0))

    predicted_label = max(proba_by_class, key=proba_by_class.get)
    confidence = float(max(proba_by_class.values()))

    # --- BTTS ---
    btts_proba = _btts_model.predict_proba(X_scaled)[0]
    btts_yes_p = float(btts_proba[1]) if len(btts_proba) > 1 else float(btts_proba[0])

    # --- Over/Under 2.5 ---
    over_proba = _over_model.predict_proba(X_scaled)[0]
    over_p = float(over_proba[1]) if len(over_proba) > 1 else float(over_proba[0])

    # --- Score probable (estimation simple basée sur xG fournis en entrée) ---
    home_xg = feature_dict.get("home_xg_avg", 1.3)
    away_xg = feature_dict.get("away_xg_avg", 1.1)
    likely_home_goals = round(home_xg)
    likely_away_goals = round(away_xg)

    # --- Facteurs explicatifs (feature importance si dispo, sinon heuristique) ---
    factors = _explain_factors(feature_dict)

    label_map = {"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur"}

    return {
        "prediction": label_map[predicted_label],
        "prediction_code": predicted_label,
        "confidence": round(confidence, 3),
        "probabilities": {
            "home_win": round(home_win_p, 3),
            "draw": round(draw_p, 3),
            "away_win": round(away_win_p, 3),
        },
        "btts": {
            "prediction": "Oui" if btts_yes_p >= 0.5 else "Non",
            "probability_yes": round(btts_yes_p, 3),
        },
        "over_under_2_5": {
            "prediction": "Over 2.5" if over_p >= 0.5 else "Under 2.5",
            "probability_over": round(over_p, 3),
        },
        "likely_score": f"{likely_home_goals}-{likely_away_goals}",
        "key_factors": factors,
        "model_used": _model_comparison.get("best_model") if _model_comparison else "unknown",
    }


def _explain_factors(f: dict) -> list[str]:
    """Heuristique simple et lisible expliquant les facteurs dominants de la prédiction."""
    factors = []

    elo_diff = f["home_strength_elo"] - f["away_strength_elo"]
    if abs(elo_diff) > 80:
        leader = "l'équipe à domicile" if elo_diff > 0 else "l'équipe à l'extérieur"
        factors.append(f"Différence de niveau marquée en faveur de {leader}")

    form_diff = f["home_form_pts5"] - f["away_form_pts5"]
    if abs(form_diff) >= 4:
        leader = "domicile" if form_diff > 0 else "extérieur"
        factors.append(f"Meilleure forme récente pour l'équipe à {leader}")

    xg_diff = f["home_xg_avg"] - f["away_xg_avg"]
    if abs(xg_diff) > 0.5:
        leader = "domicile" if xg_diff > 0 else "extérieur"
        factors.append(f"Production offensive (xG) supérieure côté {leader}")

    if f["home_rank"] <= 6 and f["away_rank"] >= 14:
        factors.append("Écart de classement important en championnat")
    elif f["away_rank"] <= 6 and f["home_rank"] >= 14:
        factors.append("Écart de classement important en championnat")

    if f["h2h_home_win_rate"] > 0.6:
        factors.append("Historique des confrontations favorable au domicile")
    elif f["h2h_home_win_rate"] < 0.25:
        factors.append("Historique des confrontations défavorable au domicile")

    if f["home_win_streak"] >= 3:
        factors.append("Série de victoires en cours à domicile")
    if f["away_win_streak"] >= 3:
        factors.append("Série de victoires en cours à l'extérieur")

    if not factors:
        factors.append("Match équilibré, aucun facteur dominant identifié")

    return factors[:5]
