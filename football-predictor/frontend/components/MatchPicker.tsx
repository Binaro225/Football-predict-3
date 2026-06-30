"use client";

import { useEffect, useState } from "react";
import { getCompetitions, getMatches, type Competition, type Match } from "@/lib/api";

export default function MatchPicker({
  onSelect,
  selectedMatch,
}: {
  onSelect: (match: Match | null) => void;
  selectedMatch: Match | null;
}) {
  const [competitions, setCompetitions] = useState<Competition[]>([]);
  const [competition, setCompetition] = useState("PL");
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCompetitions()
      .then(setCompetitions)
      .catch(() => setCompetitions([{ code: "PL", name: "Premier League" }]));
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    onSelect(null);
    getMatches(competition)
      .then((m) => setMatches(m))
      .catch(() => setError("Impossible de charger les matchs pour le moment."))
      .finally(() => setLoading(false));
  }, [competition]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-4">
      <div>
        <label className="font-mono text-[10px] tracking-[0.15em] text-chalk-dim uppercase block mb-2">
          Compétition
        </label>
        <div className="flex gap-2 overflow-x-auto pb-1 -mx-1 px-1 scrollbar-none">
          {competitions.map((c) => (
            <button
              key={c.code}
              onClick={() => setCompetition(c.code)}
              className={`shrink-0 font-body text-sm px-3.5 py-2 rounded-sm border transition-colors ${
                competition === c.code
                  ? "border-amber bg-amber/10 text-amber"
                  : "border-pitch-line text-chalk-dim hover:border-chalk-dim"
              }`}
            >
              {c.name}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="font-mono text-[10px] tracking-[0.15em] text-chalk-dim uppercase block mb-2">
          Match
        </label>

        {loading && (
          <div className="space-y-2">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-14 rounded-sm border border-pitch-line bg-pitch-line/30 animate-pulse" />
            ))}
          </div>
        )}

        {error && (
          <p className="text-sm text-away-red font-body py-3">{error}</p>
        )}

        {!loading && !error && matches.length === 0 && (
          <p className="text-sm text-chalk-dim font-body py-3">
            Aucun match programmé prochainement dans cette compétition.
          </p>
        )}

        {!loading && !error && matches.length > 0 && (
          <div className="space-y-2">
            {matches.map((m) => {
              const isSelected = selectedMatch?.id === m.id;
              const date = new Date(m.utc_date);
              const dateLabel = date.toLocaleDateString("fr-FR", {
                weekday: "short",
                day: "numeric",
                month: "short",
              });
              return (
                <button
                  key={m.id}
                  onClick={() => onSelect(m)}
                  className={`w-full text-left rounded-sm border p-3.5 transition-colors ${
                    isSelected
                      ? "border-amber bg-amber/[0.07]"
                      : "border-pitch-line hover:border-chalk-dim"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="font-display text-base text-chalk truncate pr-3">
                      {m.home_team} <span className="text-chalk-dim font-body text-sm">vs</span> {m.away_team}
                    </div>
                    <span className="font-mono text-[10px] text-chalk-dim shrink-0 uppercase">
                      {dateLabel}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
