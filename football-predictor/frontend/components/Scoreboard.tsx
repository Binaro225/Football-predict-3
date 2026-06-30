"use client";

import { useEffect, useState } from "react";
import type { Prediction } from "@/lib/api";

function AnimatedPercent({ value, delay = 0 }: { value: number; delay?: number }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const target = Math.round(value * 100);
    const timeout = setTimeout(() => {
      const duration = 700;
      const steps = 24;
      const stepTime = duration / steps;
      let current = 0;
      const interval = setInterval(() => {
        current += 1;
        setDisplay(Math.round((target * current) / steps));
        if (current >= steps) clearInterval(interval);
      }, stepTime);
    }, delay);
    return () => clearTimeout(timeout);
  }, [value, delay]);

  return <span>{display}%</span>;
}

export default function Scoreboard({
  prediction,
  homeTeam,
  awayTeam,
}: {
  prediction: Prediction;
  homeTeam: string;
  awayTeam: string;
}) {
  const { probabilities, btts, over_under_2_5, likely_score, confidence } = prediction;

  const bars = [
    { label: homeTeam.toUpperCase(), short: "1", value: probabilities.home_win, color: "bg-home-blue" },
    { label: "NUL", short: "X", value: probabilities.draw, color: "bg-chalk-dim" },
    { label: awayTeam.toUpperCase(), short: "2", value: probabilities.away_win, color: "bg-away-red" },
  ];

  return (
    <div className="scoreboard-glow rounded-sm border border-pitch-line bg-[#0d1c14] p-5 sm:p-7 shadow-[0_0_60px_-15px_rgba(232,163,61,0.15)]">
      {/* En-tête type panneau lumineux */}
      <div className="flex items-center justify-between border-b border-pitch-line pb-4 mb-5">
        <span className="font-mono text-[10px] tracking-[0.2em] text-amber uppercase">
          Pronostic IA
        </span>
        <span className="font-mono text-[10px] tracking-[0.2em] text-chalk-dim uppercase">
          Confiance {Math.round(confidence * 100)}%
        </span>
      </div>

      {/* Score probable géant */}
      <div className="text-center mb-6">
        <p className="font-mono text-[10px] tracking-[0.25em] text-chalk-dim uppercase mb-2">
          Score probable
        </p>
        <p className="font-display text-6xl sm:text-7xl font-semibold tabular-nums text-chalk">
          {likely_score}
        </p>
        <p className="font-display text-lg mt-2 text-amber">{prediction.prediction}</p>
      </div>

      {/* Barres de probabilité 1-X-2 */}
      <div className="space-y-3 mb-6">
        {bars.map((bar, i) => (
          <div key={bar.short} className="flex items-center gap-3">
            <span className="font-mono text-xs w-5 text-chalk-dim shrink-0">{bar.short}</span>
            <div className="flex-1">
              <div className="flex justify-between items-baseline mb-1">
                <span className="font-body text-xs text-chalk-dim truncate max-w-[140px] sm:max-w-none">
                  {bar.label}
                </span>
                <span className="font-mono text-sm font-medium text-chalk tabular-nums">
                  <AnimatedPercent value={bar.value} delay={i * 120} />
                </span>
              </div>
              <div className="h-1.5 w-full bg-pitch-line rounded-full overflow-hidden">
                <div
                  className={`h-full ${bar.color} rounded-full transition-all duration-700 ease-out`}
                  style={{ width: `${Math.round(bar.value * 100)}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* BTTS / Over-Under */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        <div className="border border-pitch-line rounded-sm p-3 text-center">
          <p className="font-mono text-[9px] tracking-[0.15em] text-chalk-dim uppercase mb-1">
            Les 2 marquent
          </p>
          <p className="font-display text-xl text-chalk">{btts.prediction}</p>
          <p className="font-mono text-[10px] text-chalk-dim mt-1">
            <AnimatedPercent value={btts.probability_yes} delay={400} />
          </p>
        </div>
        <div className="border border-pitch-line rounded-sm p-3 text-center">
          <p className="font-mono text-[9px] tracking-[0.15em] text-chalk-dim uppercase mb-1">
            Total buts
          </p>
          <p className="font-display text-xl text-chalk">{over_under_2_5.prediction}</p>
          <p className="font-mono text-[10px] text-chalk-dim mt-1">
            <AnimatedPercent value={over_under_2_5.probability_over} delay={500} />
          </p>
        </div>
      </div>

      {/* Facteurs clés */}
      <div>
        <p className="font-mono text-[9px] tracking-[0.15em] text-chalk-dim uppercase mb-2">
          Facteurs déterminants
        </p>
        <ul className="space-y-1.5">
          {prediction.key_factors.map((factor, i) => (
            <li key={i} className="flex gap-2 text-sm text-chalk-dim font-body">
              <span className="text-amber shrink-0">—</span>
              {factor}
            </li>
          ))}
        </ul>
      </div>

      {prediction.data_quality.api_mode === "demo" && (
        <p className="mt-5 pt-4 border-t border-pitch-line font-mono text-[9px] text-chalk-dim/70 uppercase tracking-wide">
          Mode démonstration — données simulées (clé API non configurée)
        </p>
      )}
    </div>
  );
}
