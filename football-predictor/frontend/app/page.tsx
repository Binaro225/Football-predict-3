"use client";

import { useState } from "react";
import MatchPicker from "@/components/MatchPicker";
import Scoreboard from "@/components/Scoreboard";
import { predictMatch, type Match, type Prediction } from "@/lib/api";

export default function Home() {
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    if (!selectedMatch) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const result = await predictMatch({
        home_team_id: selectedMatch.home_team_id,
        away_team_id: selectedMatch.away_team_id,
        home_team_name: selectedMatch.home_team,
        away_team_name: selectedMatch.away_team,
        competition: selectedMatch.competition,
      });
      setPrediction(result);
    } catch (e) {
      setError(
        "Impossible de générer la prédiction pour le moment. Réessayez dans un instant.",
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSelect(match: Match | null) {
    setSelectedMatch(match);
    setPrediction(null);
    setError(null);
  }

  return (
    <main className="flex-1 flex flex-col max-w-xl w-full mx-auto px-4 sm:px-6 py-8 sm:py-12">
      <header className="mb-8 text-center">
        <p className="font-mono text-[10px] tracking-[0.3em] text-amber uppercase mb-2">
          Intelligence Artificielle · Football
        </p>
        <h1 className="font-display text-3xl sm:text-4xl font-semibold text-chalk leading-tight">
          Pronostic IA
        </h1>
        <p className="font-body text-sm text-chalk-dim mt-2">
          Sélectionnez un match. L&apos;analyse se fait automatiquement.
        </p>
      </header>

      <section className="mb-6">
        <MatchPicker onSelect={handleSelect} selectedMatch={selectedMatch} />
      </section>

      <button
        onClick={handlePredict}
        disabled={!selectedMatch || loading}
        className="font-display text-lg tracking-wide uppercase w-full py-4 rounded-sm bg-amber text-pitch font-semibold disabled:opacity-30 disabled:cursor-not-allowed transition-opacity active:scale-[0.99] mb-8"
      >
        {loading ? "Analyse en cours…" : "Prédire"}
      </button>

      {error && (
        <p className="text-center text-sm text-away-red font-body mb-6">{error}</p>
      )}

      {loading && (
        <div className="rounded-sm border border-pitch-line bg-[#0d1c14] p-7 text-center">
          <p className="font-mono text-xs text-chalk-dim uppercase tracking-widest animate-pulse">
            Collecte des statistiques · Calcul du modèle…
          </p>
        </div>
      )}

      {prediction && selectedMatch && !loading && (
        <div className="animate-count-up">
          <Scoreboard
            prediction={prediction}
            homeTeam={selectedMatch.home_team}
            awayTeam={selectedMatch.away_team}
          />
        </div>
      )}

      <footer className="mt-12 text-center">
        <p className="font-mono text-[9px] text-chalk-dim/50 uppercase tracking-widest">
          Prédictions statistiques à titre informatif uniquement
        </p>
      </footer>
    </main>
  );
}
