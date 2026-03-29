"use client";

import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ComposedChart, Area,
  ReferenceLine,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DashboardData {
  last_updated: string;
  classifier: {
    baseline: { accuracy: number; precision: number; recall: number; f1: number };
    full:     { accuracy: number; precision: number; recall: number; f1: number };
    mcnemar:  { b: number; c: number; chi2: number; p: number };
  };
  regression: {
    baseline: { mae: number; rmse: number; r2: number };
    full:     { mae: number; rmse: number; r2: number };
  };
  price_predictions: { date: string; real: number; pred_base: number; pred_full: number }[];
  sentiment_daily:   { date: string; sentiment: number; price: number }[];
  reddit_posts: {
    date: string; title: string; score: number;
    num_comments: number; sent_score: number | null; url: string;
  }[];
}

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  real:     "#E8F4FF",
  baseline: "#F5A623",
  full:     "#4F80FF",
  positive: "#10CFAA",
  negative: "#FF4D6A",
  muted:    "#6B89B0",
  grid:     "#1E3A5F",
  card:     "#0C1830",
};

// ── Formatters ────────────────────────────────────────────────────────────────
const fmtPct  = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmtUsd  = (v: number) => `$${v.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
const fmtDate = (s: string) => s.slice(5); // MM-DD

// ── Tooltip genérico ──────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label, usd = false }: {
  active?: boolean; payload?: { name: string; value: number; color: string }[];
  label?: string; usd?: boolean;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card px-3 py-2 text-xs space-y-1">
      <p className="font-mono text-body mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }} className="font-mono">
          {p.name}: {usd ? fmtUsd(p.value) : typeof p.value === "number" && p.value < 2 ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  );
}

// ── KPI Card ──────────────────────────────────────────────────────────────────
function KpiCard({
  label, baseline, full, format, higher = true, delay = "0",
}: {
  label: string; baseline: number; full: number;
  format: (v: number) => string; higher?: boolean; delay?: string;
}) {
  const better = higher ? full > baseline : full < baseline;
  const delta  = full - baseline;
  const pct    = baseline !== 0 ? ((delta / Math.abs(baseline)) * 100).toFixed(1) : "—";

  return (
    <div
      className="glass-card glow-on-hover p-5 flex flex-col gap-3 fade-in"
      style={{ animationDelay: `${delay}s` }}
    >
      <p className="text-xs uppercase tracking-widest text-muted font-display font-600">{label}</p>

      <div className="flex items-end justify-between">
        {/* Full model (big number) */}
        <div>
          <p className="text-[11px] text-muted mb-1">Con Sentiment</p>
          <p className="font-mono text-2xl font-500"
             style={{ color: better ? C.positive : C.negative }}>
            {format(full)}
          </p>
        </div>
        {/* Baseline */}
        <div className="text-right">
          <p className="text-[11px] text-muted mb-1">Baseline</p>
          <p className="font-mono text-lg text-body">{format(baseline)}</p>
        </div>
      </div>

      {/* Delta badge */}
      <div className="flex items-center gap-2">
        <span
          className="px-2 py-0.5 rounded-full text-[11px] font-mono font-500"
          style={{
            background: better ? "rgba(16,207,170,0.12)" : "rgba(255,77,106,0.12)",
            color:       better ? C.positive : C.negative,
          }}
        >
          {better ? "▲" : "▼"} {Math.abs(parseFloat(pct))}%
        </span>
        <span className="text-[11px] text-muted">vs baseline</span>
      </div>
    </div>
  );
}

// ── Price Chart ───────────────────────────────────────────────────────────────
function PriceChart({ data }: { data: DashboardData["price_predictions"] }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} />
        <XAxis dataKey="date" tickFormatter={fmtDate} tick={{ fill: C.muted, fontSize: 11 }}
               axisLine={false} tickLine={false} />
        <YAxis tickFormatter={(v) => `$${v}`} tick={{ fill: C.muted, fontSize: 11 }}
               axisLine={false} tickLine={false} width={64} />
        <Tooltip content={<ChartTooltip usd />} />
        <Legend
          wrapperStyle={{ fontSize: 12, color: C.muted, fontFamily: "var(--font-fira)" }}
          formatter={(v) => v === "real" ? "Precio Real"
                           : v === "pred_base" ? "Baseline (mercado)"
                           : "Full (mercado + Reddit)"}
        />
        <Line dataKey="real"      name="real"      stroke={C.real}     strokeWidth={2} dot={false} />
        <Line dataKey="pred_base" name="pred_base" stroke={C.baseline} strokeWidth={1.5} dot={false} strokeDasharray="5 4" />
        <Line dataKey="pred_full" name="pred_full" stroke={C.full}     strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Sentiment Chart ───────────────────────────────────────────────────────────
function SentimentChart({ data }: { data: DashboardData["sentiment_daily"] }) {
  // Normaliza precio a escala de sentimiento (0-1) para overlay
  const prices = data.map((d) => d.price);
  const pMin   = Math.min(...prices);
  const pMax   = Math.max(...prices);
  const norm   = (p: number) => ((p - pMin) / (pMax - pMin)).toFixed(4);

  const chartData = data.map((d) => ({
    date:      d.date,
    sentiment: d.sentiment,
    priceNorm: parseFloat(norm(d.price)),
    price:     d.price,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} />
        <XAxis dataKey="date" tickFormatter={fmtDate} tick={{ fill: C.muted, fontSize: 11 }}
               axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis yAxisId="sent" domain={[-1, 1]} tick={{ fill: C.muted, fontSize: 11 }}
               axisLine={false} tickLine={false} width={40} />
        <YAxis yAxisId="price" orientation="right" domain={[0, 1]}
               tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} width={10} hide />
        <Tooltip
          content={({ active, payload, label }) => {
            if (!active || !payload?.length) return null;
            const raw = data.find((d) => d.date === label);
            return (
              <div className="glass-card px-3 py-2 text-xs space-y-1">
                <p className="font-mono text-body mb-1">{label}</p>
                <p style={{ color: C.positive }} className="font-mono">
                  Sentiment: {payload[0]?.value?.toFixed(4)}
                </p>
                {raw && (
                  <p style={{ color: C.baseline }} className="font-mono">
                    Precio: {fmtUsd(raw.price)}
                  </p>
                )}
              </div>
            );
          }}
        />
        <ReferenceLine yAxisId="sent" y={0} stroke={C.grid} strokeDasharray="4 4" />
        <Bar dataKey="sentiment" yAxisId="sent" name="Sentiment"
             fill={C.full} opacity={0.7} radius={[2, 2, 0, 0]}
             /* color dinámico por valor */
             shape={(props: Record<string, number>) => {
               const { x, y, width, height, sentiment } = props as Record<string, number>;
               const fill = sentiment >= 0 ? C.positive : C.negative;
               return (
                 <rect x={x} y={y} width={width} height={height}
                       fill={fill} opacity={0.75} rx={2} />
               );
             }}
        />
        <Line dataKey="priceNorm" yAxisId="price" name="Precio (norm.)"
              stroke={C.baseline} strokeWidth={1.5} dot={false} strokeDasharray="4 3" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ── McNemar Badge ─────────────────────────────────────────────────────────────
function McNemarBadge({ m }: { m: DashboardData["classifier"]["mcnemar"] }) {
  const sig = m.p < 0.05;
  return (
    <div className="glass-card p-4 fade-in fade-in-8 flex flex-col gap-2">
      <p className="text-xs uppercase tracking-widest text-muted font-display font-600">
        Test de McNemar
      </p>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="font-mono text-sm text-body">χ² ≈ {m.chi2}</span>
        <span className="font-mono text-sm text-body">p ≈ {m.p}</span>
        <span
          className="px-2.5 py-0.5 rounded-full text-xs font-mono font-500"
          style={{
            background: sig ? "rgba(16,207,170,0.12)" : "rgba(107,137,176,0.15)",
            color:       sig ? C.positive : C.muted,
          }}
        >
          {sig ? "✓ Significativo (p<0.05)" : "No significativo (p≥0.05)"}
        </span>
      </div>
      <p className="text-[11px] text-muted">
        b={m.b} (baseline mal, full bien) · c={m.c} (baseline bien, full mal)
      </p>
    </div>
  );
}

// ── Reddit Table ──────────────────────────────────────────────────────────────
function RedditTable({ posts }: { posts: DashboardData["reddit_posts"] }) {
  const sentColor = (v: number | null) => {
    if (v === null) return C.muted;
    return v >= 0.05 ? C.positive : v <= -0.05 ? C.negative : C.muted;
  };
  const sentLabel = (v: number | null) => {
    if (v === null) return "—";
    if (v >= 0.05)  return "POS";
    if (v <= -0.05) return "NEG";
    return "NEU";
  };

  return (
    <div className="overflow-auto max-h-[420px]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-card">
          <tr className="text-muted uppercase tracking-wider text-left border-b border-border">
            <th className="py-2 pr-3 font-display">Fecha</th>
            <th className="py-2 pr-3 font-display">Título</th>
            <th className="py-2 pr-3 font-display text-right">Score</th>
            <th className="py-2 pr-3 font-display text-right">Coment.</th>
            <th className="py-2 font-display text-right">Sentiment</th>
          </tr>
        </thead>
        <tbody>
          {posts.map((p, i) => (
            <tr
              key={i}
              className="border-b border-[#111F38] hover:bg-[#0f2040] transition-colors"
            >
              <td className="py-2 pr-3 font-mono text-muted whitespace-nowrap">{p.date}</td>
              <td className="py-2 pr-3 text-body max-w-[320px]">
                <a href={p.url} target="_blank" rel="noreferrer"
                   className="hover:text-heading transition-colors line-clamp-1"
                   title={p.title}>
                  {p.title}
                </a>
              </td>
              <td className="py-2 pr-3 font-mono text-right text-body">{p.score.toLocaleString()}</td>
              <td className="py-2 pr-3 font-mono text-right text-muted">{p.num_comments}</td>
              <td className="py-2 text-right">
                <span
                  className="inline-block px-2 py-0.5 rounded-full font-mono font-500"
                  style={{
                    background: `${sentColor(p.sent_score)}22`,
                    color:       sentColor(p.sent_score),
                  }}
                >
                  {sentLabel(p.sent_score)}
                  {p.sent_score !== null && (
                    <span className="ml-1 opacity-70">{p.sent_score.toFixed(3)}</span>
                  )}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Classifier Accuracy Mini-Bar ──────────────────────────────────────────────
function AccuracyBar({ baseline, full }: { baseline: number; full: number }) {
  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart
        data={[
          { name: "Baseline",         accuracy: baseline },
          { name: "Full (+ Sentiment)", accuracy: full },
        ]}
        margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
      >
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} vertical={false} />
        <XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 1]} tickFormatter={fmtPct} tick={{ fill: C.muted, fontSize: 11 }}
               axisLine={false} tickLine={false} width={46} />
        <Tooltip formatter={(v: number) => fmtPct(v)} contentStyle={{ background: C.card, border: `1px solid ${C.grid}` }} />
        <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}
             fill={C.full} label={{ position: "top", formatter: fmtPct, fill: C.muted, fontSize: 11 }}>
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export function Dashboard({ data }: { data: DashboardData | null }) {
  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center text-muted font-mono text-sm">
        Sin datos — ejecutá <code className="ml-2 text-primary">python export_for_dashboard.py</code>
      </div>
    );
  }

  const { classifier: cls, regression: reg } = data;
  const updatedAt = new Date(data.last_updated).toLocaleString("es-AR", {
    dateStyle: "medium", timeStyle: "short",
  });

  return (
    <main className="min-h-screen bg-bg text-body px-4 py-8 max-w-[1400px] mx-auto">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-10 fade-in">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">◎</span>
            <h1 className="font-display font-800 text-3xl text-heading tracking-tight">
              SOL/USD · Sentiment Dashboard
            </h1>
          </div>
          <p className="text-sm text-muted font-display">
            Predicción de precios de Solana usando XGBoost + análisis de sentimiento r/Solana
          </p>
        </div>
        <div className="text-right">
          <p className="text-[11px] text-muted uppercase tracking-widest mb-0.5">Última actualización</p>
          <p className="font-mono text-sm text-body">{updatedAt}</p>
        </div>
      </header>

      {/* ── Sección: Clasificador ───────────────────────────────────────────── */}
      <section className="mb-8">
        <SectionTitle icon="⬤" label="Clasificador de Dirección (sube / baja)" delay="0.05" />

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <KpiCard label="Accuracy"  baseline={cls.baseline.accuracy}  full={cls.full.accuracy}  format={fmtPct} delay="0.08" />
          <KpiCard label="Precision" baseline={cls.baseline.precision} full={cls.full.precision} format={fmtPct} delay="0.12" />
          <KpiCard label="Recall"    baseline={cls.baseline.recall}    full={cls.full.recall}    format={fmtPct} delay="0.16" />
          <KpiCard label="F1 Score"  baseline={cls.baseline.f1}        full={cls.full.f1}        format={fmtPct} delay="0.20" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Accuracy bar */}
          <div className="glass-card p-5 fade-in fade-in-5">
            <p className="text-xs uppercase tracking-widest text-muted font-display mb-3">
              Accuracy comparada
            </p>
            <AccuracyBar baseline={cls.baseline.accuracy} full={cls.full.accuracy} />
          </div>

          {/* McNemar */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <McNemarBadge m={cls.mcnemar} />
            {/* Mini-métricas tabla */}
            <div className="glass-card p-4 fade-in fade-in-6">
              <p className="text-xs uppercase tracking-widest text-muted font-display mb-3">
                Detalle métricas
              </p>
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-muted border-b border-border text-left">
                    <th className="pb-1 pr-4">Métrica</th>
                    <th className="pb-1 pr-4 text-right">Baseline</th>
                    <th className="pb-1 text-right">+ Sentiment</th>
                  </tr>
                </thead>
                <tbody>
                  {(["accuracy","precision","recall","f1"] as const).map((k) => (
                    <tr key={k} className="border-b border-[#111F38]">
                      <td className="py-1.5 pr-4 text-muted capitalize">{k}</td>
                      <td className="py-1.5 pr-4 text-right text-body">{fmtPct(cls.baseline[k])}</td>
                      <td className="py-1.5 text-right font-500"
                          style={{ color: cls.full[k] >= cls.baseline[k] ? C.positive : C.negative }}>
                        {fmtPct(cls.full[k])}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* ── Sección: Regresor ──────────────────────────────────────────────── */}
      <section className="mb-8">
        <SectionTitle icon="◈" label="Regresor de Precio Absoluto (XGBoost)" delay="0.22" />

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
          <KpiCard label="MAE"  baseline={reg.baseline.mae}  full={reg.full.mae}  format={fmtUsd} higher={false} delay="0.24" />
          <KpiCard label="RMSE" baseline={reg.baseline.rmse} full={reg.full.rmse} format={fmtUsd} higher={false} delay="0.28" />
          <KpiCard label="R²"   baseline={reg.baseline.r2}   full={reg.full.r2}   format={(v) => v.toFixed(4)} delay="0.32" />
        </div>

        {/* Gráfico principal: precio real vs predicciones */}
        <div className="glass-card p-5 fade-in fade-in-7">
          <p className="text-xs uppercase tracking-widest text-muted font-display mb-4">
            Precio Real vs Predicciones — Segmento de Test
          </p>
          <PriceChart data={data.price_predictions} />
        </div>
      </section>

      {/* ── Sección: Sentiment + Reddit ─────────────────────────────────────── */}
      <section className="mb-8">
        <SectionTitle icon="◆" label="Sentimiento Reddit r/Solana" delay="0.34" />

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* Sentiment timeline */}
          <div className="lg:col-span-2 glass-card p-5 fade-in fade-in-5">
            <p className="text-xs uppercase tracking-widest text-muted font-display mb-1">
              Sentimiento diario
            </p>
            <p className="text-[11px] text-muted mb-3">
              Barras: sentimiento (RoBERTa) · Línea: precio normalizado
            </p>
            <SentimentChart data={data.sentiment_daily} />
          </div>

          {/* Reddit table */}
          <div className="lg:col-span-3 glass-card p-5 fade-in fade-in-6">
            <p className="text-xs uppercase tracking-widest text-muted font-display mb-3">
              Top posts por score
            </p>
            <RedditTable posts={data.reddit_posts} />
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="text-center text-[11px] text-muted font-mono mt-12 pb-6 space-y-1">
        <p>Tesina · Licenciatura en Ciencias de Datos · XGBoost + cardiffnlp/twitter-roberta-base-sentiment-latest</p>
        <p>Datos: CoinGecko API · Reddit r/Solana · Actualización automática vía GitHub Actions</p>
      </footer>
    </main>
  );
}

// ── Section title helper ──────────────────────────────────────────────────────
function SectionTitle({ icon, label, delay }: { icon: string; label: string; delay: string }) {
  return (
    <div className="flex items-center gap-2 mb-4 fade-in" style={{ animationDelay: `${delay}s` }}>
      <span style={{ color: C.full }} className="text-sm">{icon}</span>
      <h2 className="font-display font-700 text-heading text-base tracking-tight">{label}</h2>
      <div className="flex-1 h-px bg-border ml-2" />
    </div>
  );
}
