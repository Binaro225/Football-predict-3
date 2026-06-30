from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predictions

app = FastAPI(
    title="Football Predictor API",
    description="API de prédiction de matchs de football basée sur le Machine Learning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ouvert pour permettre l'appel depuis le frontend Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, prefix="/api", tags=["predictions"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "football-predictor-api"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}
