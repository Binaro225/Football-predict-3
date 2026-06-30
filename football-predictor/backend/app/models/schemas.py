from typing import Optional

from pydantic import BaseModel, Field


class MatchSummary(BaseModel):
    id: int
    competition: str
    utc_date: str
    home_team: str
    home_team_id: int
    away_team: str
    away_team_id: int
    matchday: Optional[int] = None


class PredictionRequest(BaseModel):
    home_team_id: int = Field(..., description="ID de l'équipe à domicile (football-data.org)")
    away_team_id: int = Field(..., description="ID de l'équipe à l'extérieur")
    home_team_name: str
    away_team_name: str
    competition: str = "PL"
    home_rank: Optional[int] = Field(default=10, ge=1, le=24)
    away_rank: Optional[int] = Field(default=10, ge=1, le=24)


class ProbabilitiesOut(BaseModel):
    home_win: float
    draw: float
    away_win: float


class BttsOut(BaseModel):
    prediction: str
    probability_yes: float


class OverUnderOut(BaseModel):
    prediction: str
    probability_over: float


class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: str
    confidence: float
    probabilities: ProbabilitiesOut
    btts: BttsOut
    over_under_2_5: OverUnderOut
    likely_score: str
    key_factors: list[str]
    model_used: str
    data_quality: dict
