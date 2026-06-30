export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Competition = { code: string; name: string };

export type Match = {
  id: number;
  competition: string;
  utc_date: string;
  home_team: string;
  home_team_id: number;
  away_team: string;
  away_team_id: number;
  matchday: number | null;
};

export type Prediction = {
  prediction: string;
  prediction_code: "H" | "D" | "A";
  confidence: number;
  probabilities: { home_win: number; draw: number; away_win: number };
  btts: { prediction: string; probability_yes: number };
  over_under_2_5: { prediction: string; probability_over: number };
  likely_score: string;
  key_factors: string[];
  model_used: string;
  data_quality: { home_team: string; away_team: string; api_mode: string };
};

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Erreur API (${res.status}): ${body || res.statusText}`);
  }
  return res.json();
}

export function getCompetitions() {
  return jsonFetch<Competition[]>("/api/competitions");
}

export function getMatches(competition: string) {
  return jsonFetch<Match[]>(`/api/matches?competition=${encodeURIComponent(competition)}`);
}

export function predictMatch(payload: {
  home_team_id: number;
  away_team_id: number;
  home_team_name: string;
  away_team_name: string;
  competition: string;
}) {
  return jsonFetch<Prediction>("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
