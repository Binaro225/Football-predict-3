"""
Entraîne plusieurs modèles de classification pour prédire le résultat
d'un match (H/D/A), compare leurs performances par validation croisée,
puis entraîne également des modèles auxiliaires pour BTTS et Over/Under 2.5.

Sauvegarde :
  - le meilleur modèle "résultat" (+ tous les modèles comparés, pour transparence)
  - le modèle BTTS
  - le modèle Over/Under
  - le scaler
  - un rapport de comparaison (model_comparison.json)
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


FEATURE_COLUMNS = [
    "home_strength_elo", "away_strength_elo",
    "home_form_pts5", "away_form_pts5",
    "home_xg_avg", "away_xg_avg",
    "home_xga_avg", "away_xga_avg",
    "home_shots_avg", "away_shots_avg",
    "home_shots_on_target_avg", "away_shots_on_target_avg",
    "home_possession_avg", "away_possession_avg",
    "home_corners_avg", "away_corners_avg",
    "home_fouls_avg", "away_fouls_avg",
    "home_yellow_avg", "away_yellow_avg",
    "home_red_rate", "away_red_rate",
    "home_rank", "away_rank",
    "home_injuries_count", "away_injuries_count",
    "home_rest_days", "away_rest_days",
    "home_win_streak", "away_win_streak",
    "h2h_home_win_rate",
]


def build_candidate_models() -> dict:
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0,
        ),
        "SVM": SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "NeuralNet": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="mlogloss", n_jobs=-1,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_state=42, verbose=False,
        )
    return models


def main():
    df = pd.read_csv("app/data/training_data.csv")
    X = df[FEATURE_COLUMNS]

    label_encoder = LabelEncoder()
    y_result = pd.Series(label_encoder.fit_transform(df["result"]))  # H/D/A -> 0/1/2
    y_btts = df["btts"]
    y_over = df["over_2_5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_result, test_size=0.2, random_state=42, stratify=y_result,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = build_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    fitted_models = {}

    print(f"Comparaison de {len(candidates)} modèles sur résultat (H/D/A)...\n")
    for name, model in candidates.items():
        start = time.time()
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, preds)
        test_f1 = f1_score(y_test, preds, average="macro")
        try:
            proba = model.predict_proba(X_test_scaled)
            ll = log_loss(y_test, proba, labels=model.classes_)
        except Exception:
            ll = None
        elapsed = time.time() - start

        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_macro": round(float(test_f1), 4),
            "log_loss": round(float(ll), 4) if ll is not None else None,
            "train_time_sec": round(elapsed, 2),
        }
        fitted_models[name] = model
        print(f"  {name:20s} CV-acc={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})  "
              f"Test-acc={test_acc:.4f}  F1={test_f1:.4f}  time={elapsed:.1f}s")

    # Sélection du meilleur modèle sur la base du score CV (robuste à l'overfitting)
    best_name = max(results, key=lambda n: results[n]["cv_accuracy_mean"])
    best_model = fitted_models[best_name]
    print(f"\n>> Meilleur modèle: {best_name} (CV-acc={results[best_name]['cv_accuracy_mean']})")

    # --- Modèles auxiliaires : BTTS et Over/Under 2.5 (XGBoost si dispo, sinon GradientBoosting) ---
    aux_model_cls = XGBClassifier if HAS_XGB else GradientBoostingClassifier
    aux_kwargs = (
        dict(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss", n_jobs=-1)
        if HAS_XGB else dict(n_estimators=200, max_depth=3, learning_rate=0.05)
    )

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X, y_btts, test_size=0.2, random_state=42, stratify=y_btts,
    )
    btts_model = aux_model_cls(**aux_kwargs, random_state=42)
    btts_model.fit(scaler.transform(Xb_train), yb_train)
    btts_acc = accuracy_score(yb_test, btts_model.predict(scaler.transform(Xb_test)))
    print(f"BTTS model accuracy: {btts_acc:.4f}")

    Xo_train, Xo_test, yo_train, yo_test = train_test_split(
        X, y_over, test_size=0.2, random_state=42, stratify=y_over,
    )
    over_model = aux_model_cls(**aux_kwargs, random_state=42)
    over_model.fit(scaler.transform(Xo_train), yo_train)
    over_acc = accuracy_score(yo_test, over_model.predict(scaler.transform(Xo_test)))
    print(f"Over/Under 2.5 model accuracy: {over_acc:.4f}")

    # --- Sauvegarde ---
    joblib.dump(best_model, "app/ml/artifacts_result_model.joblib")
    joblib.dump(scaler, "app/ml/artifacts_scaler.joblib")
    joblib.dump(btts_model, "app/ml/artifacts_btts_model.joblib")
    joblib.dump(over_model, "app/ml/artifacts_over_model.joblib")
    joblib.dump(FEATURE_COLUMNS, "app/ml/artifacts_feature_columns.joblib")
    joblib.dump(label_encoder, "app/ml/artifacts_label_encoder.joblib")

    report = {
        "best_model": best_name,
        "models_compared": results,
        "auxiliary_models": {
            "btts_model": aux_model_cls.__name__,
            "btts_test_accuracy": round(float(btts_acc), 4),
            "over_under_model": aux_model_cls.__name__,
            "over_under_test_accuracy": round(float(over_acc), 4),
        },
        "feature_count": len(FEATURE_COLUMNS),
        "training_samples": len(df),
    }
    with open("app/ml/model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nArtefacts sauvegardés dans app/ml/. Rapport: app/ml/model_comparison.json")


if __name__ == "__main__":
    main()
