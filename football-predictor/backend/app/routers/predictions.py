from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import MatchSummary, PredictionRequest, PredictionResponse
from app.services.football_data_client import COMPETITIONS, FootballDataClient
from app.services.feature_builder import build_team_features, to_feature_vector
from app.services.prediction_service import get_model_comparison, predict_match

router = APIRouter()


@router.get("/competitions")
async def list_competitions():
    return [{"code": code, "name": name} for code, name in COMPETITIONS.items()]


@router.get("/matches", response_model=list[MatchSummary])
async def list_upcoming_matches(competition: str = Query("PL", description="Code compétition, ex: PL, PD, BL1")):
    client = FootballDataClient()
    try:
        matches = await client.get_upcoming_matches(competition=competition)
    finally:
        await client.close()
    if not matches:
        raise HTTPException(status_code=404, detail="Aucun match trouvé pour cette compétition.")
    return matches


@router.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    client = FootballDataClient()
    try:
        home_features = await build_team_features(
            client, req.home_team_id, req.home_team_name, fallback_rank=req.home_rank or 10,
        )
        away_features = await build_team_features(
            client, req.away_team_id, req.away_team_name, fallback_rank=req.away_rank or 10,
        )
    finally:
        await client.close()

    feature_vector = to_feature_vector(home_features, away_features)
    result = predict_match(feature_vector)
    result["data_quality"] = {
        "home_team": home_features.data_quality,
        "away_team": away_features.data_quality,
        "api_mode": "demo" if client.demo_mode else "live",
    }
    return result


@router.get("/model-info")
async def model_info():
    comparison = get_model_comparison()
    if not comparison:
        raise HTTPException(status_code=503, detail="Modèles non encore entraînés.")
    return comparison
