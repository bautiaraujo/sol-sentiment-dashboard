"use client";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ComposedChart, ReferenceLine, Cell
} from "recharts";

/* ────────────────────────────────────────────────────────────── */
/*  TYPES                                                        */
/* ────────────────────────────────────────────────────────────── */
interface ModelMetricsClf {
  accuracy: number; precision: number; recall: number; f1: number;
  auc?: number; n_test?: number;
  feature_importance?: Record<string, number>;
}
interface ModelMetricsReg {
  mae: number; rmse: number; r2: number;
  n_test?: number;
  feature_importance?: Record<string, number>;
}
interface McNemarResult { b: number; c: number; chi2: number; p: number }
interface StatSource {
  n_days: number; corr_same_day: number; corr_next_day: number;
  corr_lag1?: number; naive_accuracy: number;
  conclusion: string; detail: string;
}

export interface DashboardData {
  last_updated: string;
  today_price: number | null;
  today_date: string | null;
  model_start_date?: string;
  model_end_date?: string;
  test_cutoff?: string;
  model_days?: number;
  total_price_days?: number;
  sentiment_coverage_pct?: number;
  fg_coverage_pct?: number;
  best_sentiment_source?: string;
  statistical_analysis?: Record<string, StatSource>;
  /* Legacy format (baseline vs full) */
  classifier: {
    baseline: ModelMetricsClf;
    full: ModelMetricsClf;
    mcnemar?: McNemarResult;
  };
  regression: {
    baseline: ModelMetricsReg;
    full: ModelMetricsReg;
  };
  /* Detailed multi-model (v4) */
  classifier_detail?: {
    models_own_test: Record<string, ModelMetricsClf>;
    models_fair_test: Record<string, ModelMetricsClf>;
    mcnemar: Record<string, McNemarResult>;
    n_common_days?: number;
  };
  regression_detail?: {
    models_own_test: Record<string, ModelMetricsReg>;
    models_fair_test: Record<string, ModelMetricsReg>;
  };
  price_history:   { date: string; real: number }[];
  price_test:      { date: string; real: number; pred_base: number; pred_full: number }[];
  forecast_7d:     { date: string; pred_base: number; pred_full: number }[];
  sentiment_daily: { date: string; sentiment: number; price: number }[];
  fg_daily?:       { date: string; fg_value: number; price: number }[];
  reddit_posts:    { date: string; title: string; score: number; num_comments: number; sent_score: number | null; url: string }[];
}

/* ────────────────────────────────────────────────────────────── */
/*  CONSTANTS                                                    */
/* ────────────────────────────────────────────────────────────── */
const C = {
  real:     "#E8F4FF",
  baseline: "#F5A623",
  full:     "#4F80FF",
  reddit:   "#FF6B35",
  fg:       "#10CFAA",
  combined: "#9B6BFF",
  positive: "#10CFAA",
  negative: "#FF4D6A",
  muted:    "#6B89B0",
  grid:     "#1E3A5F",
  card:     "#0C1830",
  forecast: "#9B6BFF",
};
const MODEL_COLORS: Record<string, string> = {
  baseline:   C.baseline,
  reddit:     C.reddit,
  fear_greed: C.fg,
  combined:   C.combined,
};
const MODEL_LABELS: Record<string, string> = {
  baseline:   "Baseline",
  reddit:     "+ Reddit",
  fear_greed: "+ Fear & Greed",
  combined:   "+ Combinado",
};

const fmtPct  = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmtUsd  = (v: number) => `$${v.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
const fmtDate = (s: string) => s ? s.slice(5) : "";

/* ────────────────────────────────────────────────────────────── */
/*  PRICE CHART                                                  */
/* ────────────────────────────────────────────────────────────── */
function PriceChart({ data }: { data: DashboardData }) {
  const testMap  = new Map((data.price_test ?? []).map(d => [d.date, d]));
  const history  = data.price_history ?? [];
  const forecast = data.forecast_7d ?? [];
  const bestSent = data.best_sentiment_source ?? "full";
  const sentLabel = MODEL_LABELS[bestSent] ?? "+Sentiment";

  const histPoints = history.map(h => {
    const t = testMap.get(h.date);
    return { date: h.date, real: h.real,
             pred_base: t?.pred_base ?? undefined,
             pred_full: t?.pred_full ?? undefined };
  });
  const fcPoints = forecast.map(f => ({
    date: f.date, real: undefined as number | undefined,
    pred_base: f.pred_base, pred_full: f.pred_full,
  }));

  const chartData = [...histPoints.slice(-180), ...fcPoints];
  const todayDate = data.today_date ?? undefined;
  const testStart = Array.from(testMap.keys()).sort()[0] ?? undefined;

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} />
        <XAxis dataKey="date" tickFormatter={fmtDate}
               tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false}
               interval={Math.floor(chartData.length / 8)} />
        <YAxis tickFormatter={v => `$${v}`} tick={{ fill: C.muted, fontSize: 10 }}
               axisLine={false} tickLine={false} width={70} domain={["auto", "auto"]} />
        <Tooltip contentStyle={{ background: C.card, border: `1px solid ${C.grid}`, borderRadius: 8, fontSize: 12 }}
                 formatter={(v: number | string) => typeof v === "number" ? fmtUsd(v) : "—"}
                 labelStyle={{ color: C.muted }} />
        <Legend wrapperStyle={{ fontSize: 11, color: C.muted }}
                formatter={v => v === "real" ? "Precio Real" :
                                v === "pred_base" ? "Baseline" :
                                v === "pred_full" ? `${sentLabel}` : ""} />
        {testStart && (
          <ReferenceLine x={testStart} stroke={C.muted} strokeDasharray="4 3"
            label={{ value: "Test →", fill: C.muted, fontSize: 9, position: "insideTopLeft" }} />
        )}
        {todayDate && (
          <ReferenceLine x={todayDate} stroke={C.forecast} strokeDasharray="6 3"
            label={{ value: "HOY", fill: C.forecast, fontSize: 10, position: "insideTopRight" }} />
        )}
        <Line dataKey="real"      name="real"      stroke={C.real}     strokeWidth={2}   dot={false} connectNulls={false} />
        <Line dataKey="pred_base" name="pred_base" stroke={C.baseline} strokeWidth={1.5} dot={false} strokeDasharray="5 4" connectNulls />
        <Line dataKey="pred_full" name="pred_full" stroke={C.full}     strokeWidth={2}   dot={false} strokeDasharray="3 2" connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  FORECAST CARDS                                               */
/* ────────────────────────────────────────────────────────────── */
function ForecastCards({ forecast, todayPrice, bestSent }: {
  forecast: DashboardData["forecast_7d"]; todayPrice: number | null; bestSent: string;
}) {
  if (!forecast?.length) return null;
  const label = MODEL_LABELS[bestSent] ?? "+Sentiment";
  return (
    <div className="glass-card p-4 fade-in">
      <p className="text-xs uppercase tracking-widest text-muted mb-3">
        Forecast 7 días — modelo {label}
      </p>
      <div className="grid grid-cols-7 gap-1">
        {forecast.map((f, i) => {
          const prev = i === 0 ? todayPrice : forecast[i - 1].pred_full;
          const up = f.pred_full > (prev ?? f.pred_full);
          return (
            <div key={f.date} className="flex flex-col items-center gap-1 p-2 rounded-lg"
                 style={{ background: "rgba(155,107,255,0.08)" }}>
              <span className="text-[10px] text-muted font-mono">{fmtDate(f.date)}</span>
              <span className="text-[11px] font-mono font-bold" style={{ color: C.forecast }}>
                ${f.pred_full.toLocaleString("en-US", { maximumFractionDigits: 2 })}
              </span>
              <span style={{ color: up ? C.positive : C.negative }} className="text-xs">{up ? "↑" : "↓"}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  MULTI-MODEL COMPARISON TABLE                                 */
/* ────────────────────────────────────────────────────────────── */
function ModelComparisonClf({ detail, legacy }: {
  detail?: DashboardData["classifier_detail"];
  legacy: DashboardData["classifier"];
}) {
  const fair = detail?.models_fair_test;
  const nCommon = detail?.n_common_days;
  const mcnemar = detail?.mcnemar ?? {};

  // If we have fair test data, show multi-model. Otherwise fall back to legacy.
  if (fair && Object.keys(fair).length > 1) {
    const modelNames = ["baseline", "reddit", "fear_greed", "combined"].filter(n => fair[n]);
    const baseAcc = fair["baseline"]?.accuracy ?? 0;
    const metrics: (keyof ModelMetricsClf)[] = ["accuracy", "precision", "recall", "f1", "auc"];

    return (
      <div className="glass-card p-5 fade-in">
        <div className="flex items-center justify-between mb-4">
          <p className="text-xs uppercase tracking-widest text-muted">
            Comparación justa · Clasificador
          </p>
          {nCommon && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                  style={{ background: "rgba(79,128,255,0.12)", color: C.full }}>
              {nCommon} días en común
            </span>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-muted border-b border-border text-left">
                <th className="pb-2 pr-4">Métrica</th>
                {modelNames.map(n => (
                  <th key={n} className="pb-2 px-2 text-right" style={{ color: MODEL_COLORS[n] }}>
                    {MODEL_LABELS[n]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.map(k => (
                <tr key={k} className="border-b border-[#111F38]">
                  <td className="py-1.5 pr-4 text-muted uppercase">{k}</td>
                  {modelNames.map(n => {
                    const val = fair[n]?.[k];
                    if (val === undefined) return <td key={n} className="py-1.5 px-2 text-right text-muted">—</td>;
                    const isBase = n === "baseline";
                    const baseVal = fair["baseline"]?.[k] ?? 0;
                    // For accuracy/precision/recall/f1/auc higher is better
                    const better = (val as number) > (baseVal as number);
                    return (
                      <td key={n} className="py-1.5 px-2 text-right"
                          style={{ color: isBase ? C.muted : better ? C.positive : C.negative }}>
                        {fmtPct(val as number)}
                      </td>
                    );
                  })}
                </tr>
              ))}
              {/* n_test row */}
              <tr className="border-b border-[#111F38]">
                <td className="py-1.5 pr-4 text-muted uppercase">n_test</td>
                {modelNames.map(n => (
                  <td key={n} className="py-1.5 px-2 text-right text-muted">
                    {fair[n]?.n_test ?? "—"}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>

        {/* McNemar results */}
        {Object.keys(mcnemar).length > 0 && (
          <div className="mt-4 pt-3 border-t border-[#111F38]">
            <p className="text-[10px] uppercase tracking-widest text-muted mb-2">Test de McNemar</p>
            <div className="flex flex-wrap gap-3">
              {Object.entries(mcnemar).map(([key, mc]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted">
                    {key.replace("baseline_vs_", "vs ")}:
                  </span>
                  <span className="text-[10px] font-mono">χ²={mc.chi2}</span>
                  <span className="px-1.5 py-0.5 rounded-full text-[10px] font-mono"
                    style={{
                      background: mc.p < 0.05 ? "rgba(16,207,170,0.12)" : "rgba(107,137,176,0.15)",
                      color: mc.p < 0.05 ? C.positive : C.muted,
                    }}>
                    p={mc.p} {mc.p < 0.05 ? "✓" : ""}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Fallback: legacy 2-model view
  return (
    <div className="glass-card p-5 fade-in">
      <p className="text-xs uppercase tracking-widest text-muted mb-3">Clasificador · Baseline vs +Sentiment</p>
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-muted border-b border-border text-left">
            <th className="pb-1 pr-4">Métrica</th>
            <th className="pb-1 pr-4 text-right">Baseline</th>
            <th className="pb-1 text-right">+Sentiment</th>
          </tr>
        </thead>
        <tbody>
          {(["accuracy", "precision", "recall", "f1"] as const).map(k => (
            <tr key={k} className="border-b border-[#111F38]">
              <td className="py-1 pr-4 text-muted capitalize">{k}</td>
              <td className="py-1 pr-4 text-right text-body">{fmtPct(legacy.baseline[k])}</td>
              <td className="py-1 text-right"
                  style={{ color: legacy.full[k] >= legacy.baseline[k] ? C.positive : C.negative }}>
                {fmtPct(legacy.full[k])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  MULTI-MODEL REGRESSION                                       */
/* ────────────────────────────────────────────────────────────── */
function ModelComparisonReg({ detail, legacy }: {
  detail?: DashboardData["regression_detail"];
  legacy: DashboardData["regression"];
}) {
  const fair = detail?.models_fair_test;

  if (fair && Object.keys(fair).length > 1) {
    const modelNames = ["baseline", "reddit", "fear_greed", "combined"].filter(n => fair[n]);
    const baseMae = fair["baseline"]?.mae ?? 999;

    return (
      <div className="glass-card p-5 fade-in">
        <p className="text-xs uppercase tracking-widest text-muted mb-4">
          Comparación justa · Regresor
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-muted border-b border-border text-left">
                <th className="pb-2 pr-4">Métrica</th>
                {modelNames.map(n => (
                  <th key={n} className="pb-2 px-2 text-right" style={{ color: MODEL_COLORS[n] }}>
                    {MODEL_LABELS[n]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(["mae", "rmse"] as const).map(k => (
                <tr key={k} className="border-b border-[#111F38]">
                  <td className="py-1.5 pr-4 text-muted uppercase">{k}</td>
                  {modelNames.map(n => {
                    const val = fair[n]?.[k];
                    if (val === undefined) return <td key={n} className="py-1.5 px-2 text-right text-muted">—</td>;
                    const isBase = n === "baseline";
                    const baseVal = fair["baseline"]?.[k] ?? 999;
                    // Lower is better for MAE/RMSE
                    const better = val < baseVal;
                    return (
                      <td key={n} className="py-1.5 px-2 text-right"
                          style={{ color: isBase ? C.muted : better ? C.positive : C.negative }}>
                        {fmtUsd(val)}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr className="border-b border-[#111F38]">
                <td className="py-1.5 pr-4 text-muted uppercase">R²</td>
                {modelNames.map(n => {
                  const val = fair[n]?.r2;
                  if (val === undefined) return <td key={n} className="py-1.5 px-2 text-right text-muted">—</td>;
                  const isBase = n === "baseline";
                  const baseVal = fair["baseline"]?.r2 ?? 0;
                  const better = val > baseVal;
                  return (
                    <td key={n} className="py-1.5 px-2 text-right"
                        style={{ color: isBase ? C.muted : better ? C.positive : C.negative }}>
                      {val.toFixed(4)}
                    </td>
                  );
                })}
              </tr>
              <tr>
                <td className="py-1.5 pr-4 text-muted uppercase">n_test</td>
                {modelNames.map(n => (
                  <td key={n} className="py-1.5 px-2 text-right text-muted">
                    {fair[n]?.n_test ?? "—"}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Fallback legacy
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      <KpiCard label="MAE"  baseline={legacy.baseline.mae}  full={legacy.full.mae}
               format={fmtUsd} higher={false} delay="0.32" />
      <KpiCard label="RMSE" baseline={legacy.baseline.rmse} full={legacy.full.rmse}
               format={fmtUsd} higher={false} delay="0.34" />
      <KpiCard label="R²"   baseline={legacy.baseline.r2}   full={legacy.full.r2}
               format={v => v.toFixed(4)} delay="0.36" />
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  ACCURACY BAR CHART (multi-model)                             */
/* ────────────────────────────────────────────────────────────── */
function AccuracyBarMulti({ models }: { models: Record<string, ModelMetricsClf> }) {
  const names = ["baseline", "reddit", "fear_greed", "combined"].filter(n => models[n]);
  const barData = names.map(n => ({
    name: MODEL_LABELS[n],
    v: models[n].accuracy,
    color: MODEL_COLORS[n],
  }));

  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={barData} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} vertical={false} />
        <XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 9 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 1]} tickFormatter={fmtPct} tick={{ fill: C.muted, fontSize: 10 }}
               axisLine={false} tickLine={false} width={42} />
        <Tooltip formatter={(v: number | string) => typeof v === "number" ? fmtPct(v) : String(v)}
                 contentStyle={{ background: C.card, border: `1px solid ${C.grid}` }} />
        <Bar dataKey="v" radius={[4, 4, 0, 0]}
             label={{ position: "top", formatter: (v: number) => fmtPct(v), fill: C.muted, fontSize: 10 }}>
          {barData.map((d, i) => <Cell key={i} fill={d.color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  STATISTICAL ANALYSIS PANEL                                   */
/* ────────────────────────────────────────────────────────────── */
function StatAnalysis({ stats }: { stats: Record<string, StatSource> }) {
  const sources = Object.entries(stats);
  if (!sources.length) return null;

  const sourceLabels: Record<string, string> = {
    reddit: "Reddit (RoBERTa)",
    fear_greed: "Fear & Greed Index",
  };

  return (
    <div className="glass-card p-5 fade-in">
      <p className="text-xs uppercase tracking-widest text-muted mb-4">
        Análisis estadístico · ¿Hay señal predictiva?
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sources.map(([key, s]) => (
          <div key={key} className="rounded-lg p-4" style={{ background: "rgba(30,58,95,0.3)" }}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold" style={{ color: key === "reddit" ? C.reddit : C.fg }}>
                {sourceLabels[key] ?? key}
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono"
                style={{
                  background: s.conclusion === "NO_SIGNAL" ? "rgba(255,77,106,0.12)" : "rgba(16,207,170,0.12)",
                  color: s.conclusion === "NO_SIGNAL" ? C.negative : C.positive,
                }}>
                {s.conclusion === "NO_SIGNAL" ? "✗ Sin señal" : "~ Señal débil"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-y-2 text-xs font-mono">
              <span className="text-muted">Corr. mismo día:</span>
              <span className="text-right">{s.corr_same_day.toFixed(4)}</span>
              <span className="text-muted">Corr. día siguiente:</span>
              <span className="text-right" style={{ color: Math.abs(s.corr_next_day) > 0.05 ? C.positive : C.negative }}>
                {s.corr_next_day.toFixed(4)}
              </span>
              {s.corr_lag1 !== undefined && (
                <>
                  <span className="text-muted">Corr. lag1:</span>
                  <span className="text-right">{s.corr_lag1.toFixed(4)}</span>
                </>
              )}
              <span className="text-muted">Naive accuracy:</span>
              <span className="text-right">{fmtPct(s.naive_accuracy)}</span>
              <span className="text-muted">n días:</span>
              <span className="text-right">{s.n_days}</span>
            </div>
            <p className="text-[10px] text-muted mt-3 leading-relaxed">{s.detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  FEAR & GREED CHART                                           */
/* ────────────────────────────────────────────────────────────── */
function FearGreedChart({ data }: { data: DashboardData["fg_daily"] }) {
  if (!data?.length) return <p className="text-muted text-xs">Sin datos de Fear &amp; Greed</p>;
  const sampled = data.length > 200 ? data.filter((_, i) => i % Math.ceil(data.length / 200) === 0) : data;

  const fgColor = (v: number) =>
    v <= 25 ? C.negative : v <= 45 ? C.reddit : v <= 55 ? C.muted : v <= 75 ? C.positive : "#4F80FF";

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={sampled} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} />
        <XAxis dataKey="date" tickFormatter={fmtDate}
               tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false}
               interval="preserveStartEnd" />
        <YAxis yAxisId="fg" domain={[0, 100]} tick={{ fill: C.muted, fontSize: 10 }}
               axisLine={false} tickLine={false} width={30} />
        <Tooltip contentStyle={{ background: C.card, border: `1px solid ${C.grid}`, borderRadius: 8, fontSize: 12 }}
                 formatter={(v: number | string, name: string) =>
                   name === "fg_value" ? [`${v}`, "F&G Index"] :
                   typeof v === "number" ? [fmtUsd(v), "Precio"] : [String(v), name]
                 } />
        <ReferenceLine yAxisId="fg" y={50} stroke={C.grid} strokeDasharray="4 4" />
        <ReferenceLine yAxisId="fg" y={25} stroke={C.negative} strokeOpacity={0.3} strokeDasharray="2 4" />
        <ReferenceLine yAxisId="fg" y={75} stroke={C.positive} strokeOpacity={0.3} strokeDasharray="2 4" />
        <Bar dataKey="fg_value" yAxisId="fg" radius={[1, 1, 0, 0]} opacity={0.8}>
          {sampled.map((d, i) => <Cell key={i} fill={fgColor(d.fg_value)} />)}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  SENTIMENT CHART (Reddit)                                     */
/* ────────────────────────────────────────────────────────────── */
function SentimentChart({ data }: { data: DashboardData["sentiment_daily"] }) {
  if (!data?.length) return <p className="text-muted text-xs">Sin datos de sentimiento</p>;
  const prices = data.map(d => d.price);
  const pMin = Math.min(...prices), pMax = Math.max(...prices);
  const cd = data.map(d => ({
    ...d,
    priceNorm: pMax > pMin ? parseFloat(((d.price - pMin) / (pMax - pMin)).toFixed(4)) : 0.5,
  }));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={cd} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} />
        <XAxis dataKey="date" tickFormatter={fmtDate}
               tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false}
               interval="preserveStartEnd" />
        <YAxis yAxisId="sent" domain={[-1, 1]} tick={{ fill: C.muted, fontSize: 10 }}
               axisLine={false} tickLine={false} width={36} />
        <YAxis yAxisId="price" orientation="right" domain={[0, 1]} hide />
        <Tooltip contentStyle={{ background: C.card, border: `1px solid ${C.grid}`, borderRadius: 8, fontSize: 12 }} />
        <ReferenceLine yAxisId="sent" y={0} stroke={C.grid} strokeDasharray="4 4" />
        <Bar dataKey="sentiment" yAxisId="sent" radius={[2, 2, 0, 0]} fill={C.reddit} opacity={0.75} />
        <Line dataKey="priceNorm" yAxisId="price" stroke={C.baseline}
              strokeWidth={1.5} dot={false} strokeDasharray="4 3" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  KPI CARD (legacy fallback)                                   */
/* ────────────────────────────────────────────────────────────── */
function KpiCard({ label, baseline, full, format, higher = true, delay = "0" }: {
  label: string; baseline: number; full: number;
  format: (v: number) => string; higher?: boolean; delay?: string;
}) {
  const better = higher ? full > baseline : full < baseline;
  const pct = baseline !== 0 ? Math.abs(((full - baseline) / Math.abs(baseline)) * 100).toFixed(1) : "—";
  return (
    <div className="glass-card glow-on-hover p-4 flex flex-col gap-2 fade-in"
         style={{ animationDelay: `${delay}s` }}>
      <p className="text-xs uppercase tracking-widest text-muted">{label}</p>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[10px] text-muted mb-0.5">+Sentiment</p>
          <p className="font-mono text-xl" style={{ color: better ? C.positive : C.negative }}>{format(full)}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-muted mb-0.5">Baseline</p>
          <p className="font-mono text-base" style={{ color: C.muted }}>{format(baseline)}</p>
        </div>
      </div>
      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono self-start"
        style={{
          background: better ? "rgba(16,207,170,0.12)" : "rgba(255,77,106,0.12)",
          color: better ? C.positive : C.negative,
        }}>
        {better ? "▲" : "▼"} {pct}%
      </span>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  REDDIT TABLE                                                 */
/* ────────────────────────────────────────────────────────────── */
function RedditTable({ posts }: { posts: DashboardData["reddit_posts"] }) {
  const sc = (v: number | null) => v == null ? C.muted : v >= 0.05 ? C.positive : v <= -0.05 ? C.negative : C.muted;
  const sl = (v: number | null) => v == null ? "—" : v >= 0.05 ? "POS" : v <= -0.05 ? "NEG" : "NEU";
  return (
    <div className="overflow-auto max-h-[360px]">
      <table className="w-full text-xs">
        <thead className="sticky top-0" style={{ background: C.card }}>
          <tr className="text-muted uppercase tracking-wider text-left border-b border-border">
            <th className="py-2 pr-2">Fecha</th>
            <th className="py-2 pr-2">Título</th>
            <th className="py-2 pr-2 text-right">Score</th>
            <th className="py-2 text-right">Sent.</th>
          </tr>
        </thead>
        <tbody>
          {posts.map((p, i) => (
            <tr key={i} className="border-b border-[#111F38] hover:bg-[#0f2040] transition-colors">
              <td className="py-1.5 pr-2 font-mono text-muted whitespace-nowrap text-[10px]">{p.date}</td>
              <td className="py-1.5 pr-2 max-w-[260px]">
                <a href={p.url} target="_blank" rel="noreferrer"
                   className="hover:text-heading transition-colors line-clamp-1" title={p.title}>
                  {p.title}
                </a>
              </td>
              <td className="py-1.5 pr-2 font-mono text-right">{p.score.toLocaleString()}</td>
              <td className="py-1.5 text-right">
                <span className="inline-block px-1.5 py-0.5 rounded-full font-mono text-[10px]"
                      style={{ background: `${sc(p.sent_score)}22`, color: sc(p.sent_score) }}>
                  {sl(p.sent_score)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  SECTION TITLE                                                */
/* ────────────────────────────────────────────────────────────── */
function SectionTitle({ icon, label, delay }: { icon: string; label: string; delay: string }) {
  return (
    <div className="flex items-center gap-2 mb-4 fade-in" style={{ animationDelay: `${delay}s` }}>
      <span style={{ color: C.full }} className="text-sm">{icon}</span>
      <h2 className="font-display font-bold text-heading text-base tracking-tight">{label}</h2>
      <div className="flex-1 h-px bg-border ml-2" />
    </div>
  );
}

/* ────────────────────────────────────────────────────────────── */
/*  MAIN DASHBOARD                                               */
/* ────────────────────────────────────────────────────────────── */
export function Dashboard({ data }: { data: DashboardData | null }) {
  if (!data) return (
    <div className="flex min-h-screen items-center justify-center text-muted font-mono text-sm">
      Sin datos — ejecutá <code className="ml-2 text-primary">python export_for_dashboard_v4.py</code>
    </div>
  );

  const { classifier: cls, regression: reg } = data;
  const bestSent = data.best_sentiment_source ?? "full";
  const hasFG = !!(data.fg_daily?.length);
  const hasMultiModel = !!(data.classifier_detail?.models_fair_test);
  const fairClf = data.classifier_detail?.models_fair_test ?? {};

  const updatedAt = new Date(data.last_updated).toLocaleString("es-AR",
    { dateStyle: "medium", timeStyle: "short" });

  return (
    <main className="min-h-screen bg-bg text-body px-4 py-8 max-w-[1400px] mx-auto">

      {/* ── HEADER ── */}
      <header className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8 fade-in">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">◎</span>
            <h1 className="font-display font-extrabold text-3xl text-heading tracking-tight">
              SOL/USD · Sentiment Dashboard
            </h1>
          </div>
          <p className="text-sm text-muted">
            XGBoost + RoBERTa + Fear &amp; Greed Index · Precios 2024+ ·
            Test cutoff: {data.test_cutoff ?? data.model_end_date}
          </p>
        </div>
        <div className="flex gap-4 items-end">
          {data.today_price && (
            <div className="glass-card px-4 py-2 text-right">
              <p className="text-[10px] text-muted uppercase tracking-widest">Precio HOY</p>
              <p className="font-mono text-2xl font-bold" style={{ color: C.positive }}>
                {fmtUsd(data.today_price)}
              </p>
              <p className="text-[10px] text-muted">{data.today_date}</p>
            </div>
          )}
          <div className="flex flex-col items-end gap-1">
            <div className="flex gap-2">
              {data.sentiment_coverage_pct != null && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                      style={{ background: "rgba(255,107,53,0.12)", color: C.reddit }}>
                  Reddit {data.sentiment_coverage_pct}%
                </span>
              )}
              {data.fg_coverage_pct != null && data.fg_coverage_pct > 0 && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                      style={{ background: "rgba(16,207,170,0.12)", color: C.fg }}>
                  F&amp;G {data.fg_coverage_pct}%
                </span>
              )}
            </div>
            <div className="text-right">
              <p className="text-[10px] text-muted uppercase tracking-widest mb-0.5">Actualizado</p>
              <p className="font-mono text-xs text-body">{updatedAt}</p>
            </div>
          </div>
        </div>
      </header>

      {/* ── PRICE CHART + FORECAST ── */}
      <section className="mb-8">
        <SectionTitle icon="◈" label="Precio Real · Test Set · Forecast 7 días" delay="0.05" />
        <div className="glass-card p-5 mb-4 fade-in">
          <p className="text-[11px] text-muted mb-3">
            <span style={{ color: C.real }}>━</span> Precio real &nbsp;
            <span style={{ color: C.baseline }}>╌</span> Baseline &nbsp;
            <span style={{ color: C.full }}>┅</span> {MODEL_LABELS[bestSent] ?? "+Sentiment"} &nbsp;·&nbsp;
            Predicciones en test set y forecast
          </p>
          <PriceChart data={data} />
        </div>
        <ForecastCards forecast={data.forecast_7d ?? []} todayPrice={data.today_price} bestSent={bestSent} />
      </section>

      {/* ── STATISTICAL ANALYSIS ── */}
      {data.statistical_analysis && Object.keys(data.statistical_analysis).length > 0 && (
        <section className="mb-8">
          <SectionTitle icon="⬡" label="Análisis Estadístico — ¿Tiene el sentimiento poder predictivo?" delay="0.15" />
          <StatAnalysis stats={data.statistical_analysis} />
        </section>
      )}

      {/* ── CLASSIFIER ── */}
      <section className="mb-8">
        <SectionTitle icon="⬤" label="Clasificador de Dirección (sube / baja)" delay="0.20" />
        {hasMultiModel ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="glass-card p-4 fade-in">
              <p className="text-xs uppercase tracking-widest text-muted mb-2">Accuracy comparada</p>
              <AccuracyBarMulti models={fairClf} />
            </div>
            <div className="lg:col-span-2">
              <ModelComparisonClf detail={data.classifier_detail} legacy={cls} />
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
              <KpiCard label="Accuracy"  baseline={cls.baseline.accuracy}  full={cls.full.accuracy}  format={fmtPct} delay="0.22" />
              <KpiCard label="Precision" baseline={cls.baseline.precision} full={cls.full.precision} format={fmtPct} delay="0.24" />
              <KpiCard label="Recall"    baseline={cls.baseline.recall}    full={cls.full.recall}    format={fmtPct} delay="0.26" />
              <KpiCard label="F1"        baseline={cls.baseline.f1}        full={cls.full.f1}        format={fmtPct} delay="0.28" />
            </div>
            <ModelComparisonClf detail={data.classifier_detail} legacy={cls} />
          </>
        )}
      </section>

      {/* ── REGRESSOR ── */}
      <section className="mb-8">
        <SectionTitle icon="◆" label="Regresor de Precio — Métricas en Test Set" delay="0.30" />
        <ModelComparisonReg detail={data.regression_detail} legacy={reg} />
      </section>

      {/* ── SENTIMENT SOURCES ── */}
      <section className="mb-8">
        <SectionTitle icon="◎" label="Fuentes de Sentimiento" delay="0.38" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          {/* Reddit */}
          <div className="glass-card p-4 fade-in">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full" style={{ background: C.reddit }} />
              <p className="text-xs uppercase tracking-widest text-muted">Reddit · RoBERTa</p>
            </div>
            <p className="text-[10px] text-muted mb-2">
              {data.sentiment_coverage_pct}% cobertura · ≥5 posts/día
            </p>
            <SentimentChart data={data.sentiment_daily} />
          </div>

          {/* F&G */}
          {hasFG && (
            <div className="glass-card p-4 fade-in">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: C.fg }} />
                <p className="text-xs uppercase tracking-widest text-muted">Fear &amp; Greed Index</p>
              </div>
              <p className="text-[10px] text-muted mb-2">
                {data.fg_coverage_pct}% cobertura · Alternative.me
              </p>
              <FearGreedChart data={data.fg_daily} />
            </div>
          )}
        </div>

        {/* Reddit posts table */}
        <div className="glass-card p-4 fade-in">
          <p className="text-xs uppercase tracking-widest text-muted mb-3">Top posts Reddit por score</p>
          <RedditTable posts={data.reddit_posts} />
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="text-center text-[10px] text-muted font-mono mt-10 pb-4 space-y-0.5">
        <p>Tesina · Licenciatura en Ciencias de Datos</p>
        <p>XGBoost + RoBERTa + Fear &amp; Greed Index (Alternative.me)</p>
        <p>Precios: Yahoo Finance 2024+ · Reddit r/Solana · Cron diario vía GitHub Actions</p>
      </footer>
    </main>
  );
}
