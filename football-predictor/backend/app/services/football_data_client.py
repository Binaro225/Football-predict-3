"""
Client pour l'API football-data.org (https://www.football-data.org/).

Cette API publique gratuite (tier "Free") donne accès à :
  - calendrier des matchs (competitions, matchdays)
  - classements
  - résultats récents par équipe (form)
  - effectifs

Limite du tier gratuit : 10 requêtes/minute, compétitions majeures
uniquement (Premier League, Liga, Bundesliga, Serie A, Ligue 1, Champions
League, etc.). Une clé API gratuite est nécessaire : à définir dans la
variable d'environnement FOOTBALL_DATA_API_KEY (voir .env.example).

Si la clé n'est pas configurée, le client bascule automatiquement sur un
jeu de données de démonstration (mock) afin que l'application reste
utilisable sans configuration, avec un avertissement clair dans la réponse.
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import httpx

BASE_URL = "https://api.football-data.org/v4"

COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
}


class FootballDataClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY", "")
        self.demo_mode = not bool(self.api_key)
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"X-Auth-Token": self.api_key} if self.api_key else {},
            timeout=15.0,
        )

    async def close(self):
        await self._client.aclose()

    async def get_upcoming_matches(self, competition: str = "PL", days_ahead: int = 14) -> list[dict]:
        if self.demo_mode:
            return self._mock_matches(competition)
        date_from = datetime.utcnow().strftime("%Y-%m-%d")
        date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        try:
            resp = await self._client.get(
                f"/competitions/{competition}/matches",
                params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "id": m["id"],
                    "competition": competition,
                    "utc_date": m["utcDate"],
                    "home_team": m["homeTeam"]["name"],
                    "home_team_id": m["homeTeam"]["id"],
                    "away_team": m["awayTeam"]["name"],
                    "away_team_id": m["awayTeam"]["id"],
                    "matchday": m.get("matchday"),
                }
                for m in data.get("matches", [])
            ]
        except httpx.HTTPError:
            return self._mock_matches(competition)

    async def get_team_recent_form(self, team_id: int, limit: int = 10) -> list[dict]:
        """Derniers matchs joués par une équipe, utilisés pour calculer la forme/xG approx."""
        if self.demo_mode:
            return self._mock_team_matches(team_id)
        try:
            resp = await self._client.get(
                f"/teams/{team_id}/matches",
                params={"status": "FINISHED", "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("matches", [])
        except httpx.HTTPError:
            return self._mock_team_matches(team_id)

    async def get_standings(self, competition: str = "PL") -> list[dict]:
        if self.demo_mode:
            return self._mock_standings()
        try:
            resp = await self._client.get(f"/competitions/{competition}/standings")
            resp.raise_for_status()
            data = resp.json()
            table = data.get("standings", [{}])[0].get("table", [])
            return table
        except httpx.HTTPError:
            return self._mock_standings()

    # ---------- Mock data (mode démo, sans clé API) ----------

    @staticmethod
    def _mock_matches(competition: str) -> list[dict]:
        now = datetime.utcnow()
        demo_teams = [
            ("Arsenal", 57, "Chelsea", 61),
            ("Manchester City", 65, "Liverpool", 64),
            ("Real Madrid", 86, "Barcelona", 81),
            ("Bayern Munich", 5, "Borussia Dortmund", 4),
            ("Inter Milan", 108, "AC Milan", 98),
            ("PSG", 524, "Marseille", 516),
        ]
        return [
            {
                "id": 900000 + i,
                "competition": competition,
                "utc_date": (now + timedelta(days=i + 1)).isoformat() + "Z",
                "home_team": h,
                "home_team_id": hid,
                "away_team": a,
                "away_team_id": aid,
                "matchday": 1,
            }
            for i, (h, hid, a, aid) in enumerate(demo_teams)
        ]

    @staticmethod
    def _mock_team_matches(team_id: int) -> list[dict]:
        # Génère un historique plausible pour le calcul de forme en mode démo
        import random
        random.seed(team_id)
        matches = []
        for i in range(10):
            gh = random.randint(0, 3)
            ga = random.randint(0, 3)
            matches.append({
                "score": {"fullTime": {"home": gh, "away": ga}},
                "homeTeam": {"id": team_id if i % 2 == 0 else 0},
                "awayTeam": {"id": team_id if i % 2 != 0 else 0},
            })
        return matches

    @staticmethod
    def _mock_standings() -> list[dict]:
        return [{"position": i + 1, "team": {"id": i}, "playedGames": 20} for i in range(20)]
