"use client";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ComposedChart, ReferenceLine
} from "recharts";

export interface DashboardData {
  last_updated: string;
  today_price: number | null;
  today_date: string | null;
  model_start_date?: string;
  model_end_date?: string;
  model_days?: number;
  total_price_days?: number;
  sentiment_coverage_pct?: number;
  classifier: {
    baseline: { accuracy: number; precision: number; recall: number; f1: number };
    full:     { accuracy: number; precision: number; recall: number; f1: number };
    mcnemar:  { b: number; c: number; chi2: number; p: number };
  };
  regression: {
    baseline: { mae: number; rmse: number; r2: number };
    full:     { mae: number; rmse: number; r2: number };
  };
  price_history:   { date: string; real: number }[];
  price_test:      { date: string; real: number; pred_base: number; pred_full: number }[];
  forecast_7d:     { date: string; pred_base: number; pred_full: number }[];
  sentiment_daily: { date: string; sentiment: number; price: number }[];
  reddit_posts:    { date: string; title: string; score: number; num_comments: number; sent_score: number | null; url: string }[];
  price_predictions?: { date: string; real: number; pred_base: number; pred_full: number }[];
}

const C = {
  real:"#E8F4FF", baseline:"#F5A623", full:"#4F80FF",
  positive:"#10CFAA", negative:"#FF4D6A", muted:"#6B89B0",
  grid:"#1E3A5F", card:"#0C1830", forecast:"#9B6BFF",
};
const fmtPct  = (v: number) => `${(v*100).toFixed(1)}%`;
const fmtUsd  = (v: number) => `$${v.toLocaleString("en-US",{maximumFractionDigits:2})}`;
const fmtDate = (s: string) => s ? s.slice(5) : "";

function PriceChart({ data }: { data: DashboardData }) {
  const testMap  = new Map((data.price_test ?? data.price_predictions ?? []).map(d => [d.date, d]));
  const history  = data.price_history ?? [];
  const forecast = data.forecast_7d ?? [];

  const histPoints = history.map(h => {
    const t = testMap.get(h.date);
    return { date: h.date, real: h.real,
             pred_base: t?.pred_base ?? undefined,
             pred_full: t?.pred_full ?? undefined };
  });
  const fcPoints = forecast.map(f => ({
    date: f.date, real: undefined as number|undefined,
    pred_base: f.pred_base, pred_full: f.pred_full,
  }));

  const chartData = [...histPoints.slice(-180), ...fcPoints];
  const todayDate = data.today_date ?? undefined;
  const testStart = Array.from(testMap.keys()).sort()[0] ?? undefined;

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={chartData} margin={{top:4,right:8,left:0,bottom:0}}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4}/>
        <XAxis dataKey="date" tickFormatter={fmtDate}
               tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}
               interval={Math.floor(chartData.length / 8)}/>
        <YAxis tickFormatter={v => `$${v}`} tick={{fill:C.muted,fontSize:10}}
               axisLine={false} tickLine={false} width={70} domain={["auto","auto"]}/>
        <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.grid}`,borderRadius:8,fontSize:12}}
                 formatter={(v: number|string) => typeof v === "number" ? fmtUsd(v) : "—"}
                 labelStyle={{color:C.muted}}/>
        <Legend wrapperStyle={{fontSize:11,color:C.muted}}
                formatter={v => v==="real"?"Precio Real":
                                v==="pred_base"?"Baseline":
                                v==="pred_full"?"Full +Reddit":""}/>
        {testStart && (
          <ReferenceLine x={testStart} stroke={C.muted} strokeDasharray="4 3"
            label={{value:"Test →",fill:C.muted,fontSize:9,position:"insideTopLeft"}}/>
        )}
        {todayDate && (
          <ReferenceLine x={todayDate} stroke={C.forecast} strokeDasharray="6 3"
            label={{value:"HOY",fill:C.forecast,fontSize:10,position:"insideTopRight"}}/>
        )}
        <Line dataKey="real"      name="real"      stroke={C.real}     strokeWidth={2}   dot={false} connectNulls={false}/>
        <Line dataKey="pred_base" name="pred_base" stroke={C.baseline} strokeWidth={1.5} dot={false} strokeDasharray="5 4" connectNulls/>
        <Line dataKey="pred_full" name="pred_full" stroke={C.full}     strokeWidth={2}   dot={false} strokeDasharray="3 2" connectNulls/>
      </LineChart>
    </ResponsiveContainer>
  );
}

function ForecastCards({ forecast, todayPrice }: {
  forecast: DashboardData["forecast_7d"]; todayPrice: number | null;
}) {
  if (!forecast?.length) return null;
  return (
    <div className="glass-card p-4 fade-in">
      <p className="text-xs uppercase tracking-widest text-muted mb-3">Forecast 7 días — modelo full (+Reddit)</p>
      <div className="grid grid-cols-7 gap-1">
        {forecast.map((f, i) => {
          const prev = i === 0 ? todayPrice : forecast[i-1].pred_full;
          const up   = f.pred_full > (prev ?? f.pred_full);
          return (
            <div key={f.date} className="flex flex-col items-center gap-1 p-2 rounded-lg"
                 style={{background:"rgba(155,107,255,0.08)"}}>
              <span className="text-[10px] text-muted font-mono">{fmtDate(f.date)}</span>
              <span className="text-[11px] font-mono font-bold" style={{color:C.forecast}}>
                ${f.pred_full.toLocaleString("en-US",{maximumFractionDigits:2})}
              </span>
              <span style={{color:up?C.positive:C.negative}} className="text-xs">{up?"↑":"↓"}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SentimentChart({ data }: { data: DashboardData["sentiment_daily"] }) {
  if (!data?.length) return <p className="text-muted text-xs">Sin datos de sentimiento</p>;
  const prices = data.map(d => d.price);
  const pMin = Math.min(...prices), pMax = Math.max(...prices);
  const cd = data.map(d => ({
    ...d,
    priceNorm: pMax > pMin ? parseFloat(((d.price-pMin)/(pMax-pMin)).toFixed(4)) : 0.5,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={cd} margin={{top:4,right:8,left:0,bottom:0}}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4}/>
        <XAxis dataKey="date" tickFormatter={fmtDate}
               tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}
               interval="preserveStartEnd"/>
        <YAxis yAxisId="sent" domain={[-1,1]} tick={{fill:C.muted,fontSize:10}}
               axisLine={false} tickLine={false} width={36}/>
        <YAxis yAxisId="price" orientation="right" domain={[0,1]} hide/>
        <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.grid}`,borderRadius:8,fontSize:12}}/>
        <ReferenceLine yAxisId="sent" y={0} stroke={C.grid} strokeDasharray="4 4"/>
        <Bar dataKey="sentiment" yAxisId="sent" radius={[2,2,0,0]} fill={C.full} opacity={0.75}/>
        <Line dataKey="priceNorm" yAxisId="price" stroke={C.baseline}
              strokeWidth={1.5} dot={false} strokeDasharray="4 3"/>
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function AccuracyBar({ baseline, full }: { baseline: number; full: number }) {
  const barData = [
    { name: "Baseline",   v: baseline },
    { name: "+Sentiment", v: full },
  ];
  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={barData} margin={{top:8,right:4,left:0,bottom:0}}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" strokeOpacity={0.4} vertical={false}/>
        <XAxis dataKey="name" tick={{fill:C.muted,fontSize:10}} axisLine={false} tickLine={false}/>
        <YAxis domain={[0,1]} tickFormatter={fmtPct} tick={{fill:C.muted,fontSize:10}}
               axisLine={false} tickLine={false} width={42}/>
        <Tooltip formatter={(v: number|string) => typeof v === "number" ? fmtPct(v) : String(v)}
                 contentStyle={{background:C.card,border:`1px solid ${C.grid}`}}/>
        <Bar dataKey="v" radius={[4,4,0,0]} fill={C.full}
             label={{position:"top",formatter:(v: number) => fmtPct(v),fill:C.muted,fontSize:10}}/>
      </BarChart>
    </ResponsiveContainer>
  );
}

function KpiCard({label,baseline,full,format,higher=true,delay="0"}:{
  label:string;baseline:number;full:number;
  format:(v:number)=>string;higher?:boolean;delay?:string}) {
  const better = higher ? full > baseline : full < baseline;
  const pct = baseline !== 0 ? Math.abs(((full-baseline)/Math.abs(baseline))*100).toFixed(1) : "—";
  return (
    <div className="glass-card glow-on-hover p-4 flex flex-col gap-2 fade-in"
         style={{animationDelay:`${delay}s`}}>
      <p className="text-xs uppercase tracking-widest text-muted">{label}</p>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[10px] text-muted mb-0.5">+Sentiment</p>
          <p className="font-mono text-xl" style={{color:better?C.positive:C.negative}}>{format(full)}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-muted mb-0.5">Baseline</p>
          <p className="font-mono text-base" style={{color:C.muted}}>{format(baseline)}</p>
        </div>
      </div>
      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono self-start"
        style={{background:better?"rgba(16,207,170,0.12)":"rgba(255,77,106,0.12)",
                color:better?C.positive:C.negative}}>
        {better?"▲":"▼"} {pct}%
      </span>
    </div>
  );
}

function RedditTable({ posts }: { posts: DashboardData["reddit_posts"] }) {
  const sc = (v: number|null) => v==null?C.muted:v>=0.05?C.positive:v<=-0.05?C.negative:C.muted;
  const sl = (v: number|null) => v==null?"—":v>=0.05?"POS":v<=-0.05?"NEG":"NEU";
  return (
    <div className="overflow-auto max-h-[360px]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-card">
          <tr className="text-muted uppercase tracking-wider text-left border-b border-border">
            <th className="py-2 pr-2">Fecha</th>
            <th className="py-2 pr-2">Título</th>
            <th className="py-2 pr-2 text-right">Score</th>
            <th className="py-2 text-right">Sent.</th>
          </tr>
        </thead>
        <tbody>
          {posts.map((p,i) => (
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
                      style={{background:`${sc(p.sent_score)}22`,color:sc(p.sent_score)}}>
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

function SectionTitle({icon,label,delay}:{icon:string;label:string;delay:string}) {
  return (
    <div className="flex items-center gap-2 mb-4 fade-in" style={{animationDelay:`${delay}s`}}>
      <span style={{color:C.full}} className="text-sm">{icon}</span>
      <h2 className="font-display font-bold text-heading text-base tracking-tight">{label}</h2>
      <div className="flex-1 h-px bg-border ml-2"/>
    </div>
  );
}

export function Dashboard({ data }: { data: DashboardData | null }) {
  if (!data) return (
    <div className="flex min-h-screen items-center justify-center text-muted font-mono text-sm">
      Sin datos — ejecutá <code className="ml-2 text-primary">python export_for_dashboard.py</code>
    </div>
  );

  const { classifier: cls, regression: reg } = data;
  const updatedAt = new Date(data.last_updated).toLocaleString("es-AR",
    {dateStyle:"medium",timeStyle:"short"});

  return (
    <main className="min-h-screen bg-bg text-body px-4 py-8 max-w-[1400px] mx-auto">

      <header className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8 fade-in">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">◎</span>
            <h1 className="font-display font-extrabold text-3xl text-heading tracking-tight">
              SOL/USD · Sentiment Dashboard
            </h1>
          </div>
          <p className="text-sm text-muted">
            XGBoost + RoBERTa · Precios 2024+ ·
            Modelo: {data.model_start_date} → {data.model_end_date} ({data.model_days} días)
          </p>
        </div>
        <div className="flex gap-4 items-end">
          {data.today_price && (
            <div className="glass-card px-4 py-2 text-right">
              <p className="text-[10px] text-muted uppercase tracking-widest">Precio HOY</p>
              <p className="font-mono text-2xl font-bold" style={{color:C.positive}}>
                {fmtUsd(data.today_price)}
              </p>
              <p className="text-[10px] text-muted">{data.today_date}</p>
            </div>
          )}
          <div className="text-right">
            <p className="text-[10px] text-muted uppercase tracking-widest mb-0.5">Actualizado</p>
            <p className="font-mono text-xs text-body">{updatedAt}</p>
          </div>
        </div>
      </header>

      <section className="mb-8">
        <SectionTitle icon="◈" label="Precio Real · Test Set · Forecast 7 días" delay="0.05"/>
        <div className="glass-card p-5 mb-4 fade-in">
          <p className="text-[11px] text-muted mb-3">
            <span style={{color:C.real}}>━</span> Precio real (2024→hoy) &nbsp;
            <span style={{color:C.baseline}}>╌</span> Baseline &nbsp;
            <span style={{color:C.full}}>┅</span> Full (+Reddit) &nbsp;·&nbsp;
            Predicciones aparecen en el test set y en el forecast
          </p>
          <PriceChart data={data}/>
        </div>
        <ForecastCards forecast={data.forecast_7d??[]} todayPrice={data.today_price}/>
      </section>

      <section className="mb-8">
        <SectionTitle icon="⬤" label="Clasificador de Dirección (sube / baja)" delay="0.20"/>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <KpiCard label="Accuracy"  baseline={cls.baseline.accuracy}  full={cls.full.accuracy}  format={fmtPct} delay="0.22"/>
          <KpiCard label="Precision" baseline={cls.baseline.precision} full={cls.full.precision} format={fmtPct} delay="0.24"/>
          <KpiCard label="Recall"    baseline={cls.baseline.recall}    full={cls.full.recall}    format={fmtPct} delay="0.26"/>
          <KpiCard label="F1"        baseline={cls.baseline.f1}        full={cls.full.f1}        format={fmtPct} delay="0.28"/>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="glass-card p-4 fade-in">
            <p className="text-xs uppercase tracking-widest text-muted mb-2">Accuracy comparada</p>
            <AccuracyBar baseline={cls.baseline.accuracy} full={cls.full.accuracy}/>
          </div>
          <div className="lg:col-span-2 glass-card p-4 fade-in">
            <p className="text-xs uppercase tracking-widest text-muted mb-2">McNemar · Detalle métricas</p>
            <div className="flex items-center gap-3 mb-3 flex-wrap">
              <span className="font-mono text-xs">χ²≈{cls.mcnemar.chi2}</span>
              <span className="font-mono text-xs">p≈{cls.mcnemar.p}</span>
              <span className="px-2 py-0.5 rounded-full text-[11px] font-mono"
                style={{background:cls.mcnemar.p<0.05?"rgba(16,207,170,0.12)":"rgba(107,137,176,0.15)",
                        color:cls.mcnemar.p<0.05?C.positive:C.muted}}>
                {cls.mcnemar.p<0.05?"✓ Significativo (p<0.05)":"No significativo (p≥0.05)"}
              </span>
            </div>
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-muted border-b border-border text-left">
                  <th className="pb-1 pr-4">Métrica</th>
                  <th className="pb-1 pr-4 text-right">Baseline</th>
                  <th className="pb-1 text-right">+Sentiment</th>
                </tr>
              </thead>
              <tbody>
                {(["accuracy","precision","recall","f1"] as const).map(k => (
                  <tr key={k} className="border-b border-[#111F38]">
                    <td className="py-1 pr-4 text-muted capitalize">{k}</td>
                    <td className="py-1 pr-4 text-right text-body">{fmtPct(cls.baseline[k])}</td>
                    <td className="py-1 text-right"
                        style={{color:cls.full[k]>=cls.baseline[k]?C.positive:C.negative}}>
                      {fmtPct(cls.full[k])}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mb-8">
        <SectionTitle icon="◆" label="Regresor de Precio — Métricas en Test Set" delay="0.30"/>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <KpiCard label="MAE"  baseline={reg.baseline.mae}  full={reg.full.mae}
                   format={fmtUsd} higher={false} delay="0.32"/>
          <KpiCard label="RMSE" baseline={reg.baseline.rmse} full={reg.full.rmse}
                   format={fmtUsd} higher={false} delay="0.34"/>
          <KpiCard label="R²"   baseline={reg.baseline.r2}   full={reg.full.r2}
                   format={v => v.toFixed(4)} delay="0.36"/>
        </div>
      </section>

      <section className="mb-8">
        <SectionTitle icon="◎" label="Sentimiento Reddit r/Solana" delay="0.38"/>
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-2 glass-card p-4 fade-in">
            <p className="text-xs uppercase tracking-widest text-muted mb-1">Sentimiento diario real</p>
            <p className="text-[10px] text-muted mb-2">
              {data.model_days} días con Reddit · {data.sentiment_coverage_pct}% del período 2024+
            </p>
            <SentimentChart data={data.sentiment_daily}/>
          </div>
          <div className="lg:col-span-3 glass-card p-4 fade-in">
            <p className="text-xs uppercase tracking-widest text-muted mb-3">Top posts por score</p>
            <RedditTable posts={data.reddit_posts}/>
          </div>
        </div>
      </section>

      <footer className="text-center text-[10px] text-muted font-mono mt-10 pb-4 space-y-0.5">
        <p>Tesina · Licenciatura en Ciencias de Datos · XGBoost + cardiffnlp/twitter-roberta-base-sentiment-latest</p>
        <p>Precios: Yahoo Finance 2024+ · Reddit r/Solana · Cron diario vía GitHub Actions</p>
      </footer>
    </main>
  );
}
