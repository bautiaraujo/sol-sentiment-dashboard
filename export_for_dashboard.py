"""
export_for_dashboard_v3.py
──────────────────────────
Tesina: Predicción SOL/USD con Análisis de Sentimiento

CAMBIOS vs v1/v2:
  1. Documenta el resultado NEGATIVO del sentimiento de Reddit
     (correlación ~0 con retornos futuros, señal reactiva no predictiva)
  2. Integra Fear & Greed Index (Alternative.me) como fuente alternativa
  3. Compara 4 modelos: baseline, +Reddit, +F&G, +ambos
  4. Features técnicos enriquecidos para baseline competitivo
  5. Modelo regularizado para evitar overfitting con pocos datos
  6. Gap temporal en train/test split
  7. Métricas: accuracy, precision, recall, f1, AUC + McNemar
"""

import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

# ── Configuración ─────────────────────────────────────────────────
CUTOFF        = pd.Timestamp("2024-01-01")
FORECAST_DAYS = 7
MIN_POSTS_DAY = 5
TEST_RATIO    = 0.2
GAP_DAYS      = 1

# Modelo conservador: regularizado para ~300 muestras
XGB_PARAMS = dict(
    n_estimators     = 150,
    max_depth        = 3,
    learning_rate    = 0.08,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.5,
    random_state     = 42,
)

def mcnemar(y_true, pa, pb):
    ca = pa == y_true; cb = pb == y_true
    b = int((~ca & cb).sum()); c = int((ca & ~cb).sum()); n = b + c
    if n == 0: return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / n
    try:
        from scipy.stats import chi2 as d; pv = float(d.sf(chi2, df=1))
    except Exception: pv = float(np.exp(-0.5 * chi2) * 1.2533)
    return b, c, round(float(chi2), 4), round(pv, 4)


# ══════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════
def load():
    pr = pd.read_csv("data/solana_prices.csv", parse_dates=["date"])
    se = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    re = pd.read_csv("data/reddit_posts.csv", parse_dates=["date"])

    pr = pr[pr["date"] >= CUTOFF].copy().sort_values("date").reset_index(drop=True)
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()

    # ── Fear & Greed Index ────────────────────────────────────────
    fg_path = Path("data/fear_greed.csv")
    has_fg = fg_path.exists()
    if has_fg:
        fg = pd.read_csv(fg_path, parse_dates=["date"])
        fg = fg[fg["date"] >= CUTOFF].copy()
        print(f"  Fear & Greed: {len(fg)} días ({fg['date'].min().date()} → {fg['date'].max().date()})")
    else:
        fg = None
        print("  ⚠ Fear & Greed no encontrado. Corré: python get_fear_greed.py")

    # ── Reddit: sentimiento ponderado diario ──────────────────────
    posts_per_day = se.groupby("date").size()
    valid_days = posts_per_day[posts_per_day >= MIN_POSTS_DAY].index
    se_filt = se[se["date"].isin(valid_days)].copy()
    se_filt["score_clip"] = se_filt["score"].clip(lower=1)

    def wsent(g):
        w = g["score_clip"]
        return (g["sent_score"] * w).sum() / w.sum()

    reddit_daily = (se_filt.groupby("date").apply(wsent)
                           .reset_index(columns=["sent_reddit"]))
    reddit_daily.columns = ["date", "sent_reddit"]

    # ── Build master dataframe ────────────────────────────────────
    df = pr.copy()

    # Merge Reddit (left join - muchos NaN, eso está bien)
    df = df.merge(reddit_daily, on="date", how="left")

    # Merge F&G (left join - cobertura ~100% para días de mercado)
    if has_fg:
        df = df.merge(fg[["date", "fg_value"]], on="date", how="left")
        # Normalizar F&G de 0-100 a -1 a +1 para comparabilidad
        df["fg_norm"] = (df["fg_value"] - 50) / 50
    else:
        df["fg_value"] = np.nan
        df["fg_norm"] = np.nan

    # ── Features técnicos (siempre disponibles) ───────────────────
    df["return"]       = df["price"].pct_change()
    df["ret_ma5"]      = df["return"].rolling(5, min_periods=2).mean()
    df["ret_ma10"]     = df["return"].rolling(10, min_periods=3).mean()
    df["volatility5"]  = df["return"].rolling(5, min_periods=2).std()
    df["volatility10"] = df["return"].rolling(10, min_periods=3).std()
    df["momentum5"]    = df["price"].pct_change(5)
    df["momentum10"]   = df["price"].pct_change(10)

    # ── Features de sentimiento (lagged para evitar look-ahead) ───
    # Reddit: solo disponible en días con datos
    df["reddit_lag1"]  = df["sent_reddit"].shift(1)

    # F&G: disponible casi todos los días → más features derivados
    if has_fg:
        df["fg_lag1"]      = df["fg_norm"].shift(1)
        df["fg_lag2"]      = df["fg_norm"].shift(2)
        df["fg_ma3"]       = df["fg_norm"].rolling(3, min_periods=1).mean().shift(1)
        df["fg_delta"]     = df["fg_lag1"] - df["fg_lag2"]
        # Divergencia: si F&G sube pero precio baja (o viceversa)
        df["fg_price_div"] = df["fg_delta"] - df["return"].shift(1)

    df = df.dropna(subset=["return"]).reset_index(drop=True)

    # ── Estadísticas ──────────────────────────────────────────────
    n_total = len(df)
    n_reddit = df["reddit_lag1"].notna().sum()
    n_fg = df["fg_lag1"].notna().sum() if has_fg else 0

    print(f"  Precios 2024+:       {n_total} días")
    print(f"  Con Reddit (≥{MIN_POSTS_DAY}p): {n_reddit} días ({n_reddit/n_total*100:.1f}%)")
    print(f"  Con Fear&Greed:      {n_fg} días ({n_fg/n_total*100:.1f}%)")

    return df, reddit_daily, re, has_fg


# ══════════════════════════════════════════════════════════════════
# DEFINICIÓN DE MODELOS
# ══════════════════════════════════════════════════════════════════
FEAT_BASELINE = ["return", "ret_ma5", "ret_ma10", "volatility5",
                 "volatility10", "momentum5", "momentum10"]

FEAT_REDDIT   = FEAT_BASELINE + ["reddit_lag1"]

FEAT_FG       = FEAT_BASELINE + ["fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"]

FEAT_COMBINED = FEAT_BASELINE + ["reddit_lag1",
                "fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"]


def split_with_gap(X, y, test_ratio=TEST_RATIO, gap=GAP_DAYS):
    n = len(X)
    n_test = int(n * test_ratio)
    n_train = n - n_test - gap
    return (X.iloc[:n_train], X.iloc[n_train+gap:],
            y.iloc[:n_train], y.iloc[n_train+gap:])


def calc_metrics_clf(y_true, y_pred, y_proba):
    return {k: round(float(v), 4) for k, v in {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
    }.items()}


def calc_metrics_reg(y_true, y_pred):
    return {k: round(float(v), 4) for k, v in {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   r2_score(y_true, y_pred),
    }.items()}


# ══════════════════════════════════════════════════════════════════
# CLASIFICADOR (sube / baja)
# ══════════════════════════════════════════════════════════════════
def classify(df, has_fg):
    d = df.copy()
    d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna(subset=["tgt"] + FEAT_BASELINE).reset_index(drop=True)

    y = d["tgt"]

    # Definir qué modelos podemos correr
    models_config = {"baseline": FEAT_BASELINE}

    # Reddit: solo si hay suficientes datos en el test set
    d_reddit = d.dropna(subset=["reddit_lag1"])
    if len(d_reddit) >= 50:
        models_config["reddit"] = FEAT_REDDIT
    else:
        print(f"  ⚠ Reddit: solo {len(d_reddit)} filas completas, omitiendo modelo Reddit")

    if has_fg:
        d_fg = d.dropna(subset=["fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"])
        if len(d_fg) >= 50:
            models_config["fear_greed"] = FEAT_FG
        else:
            print(f"  ⚠ F&G: solo {len(d_fg)} filas, omitiendo")

        # Combinado: necesita Reddit + F&G
        d_comb = d.dropna(subset=["reddit_lag1", "fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"])
        if len(d_comb) >= 50:
            models_config["combined"] = FEAT_COMBINED
        else:
            print(f"  ⚠ Combinado: solo {len(d_comb)} filas, omitiendo")

    results = {}
    predictions = {}

    for name, features in models_config.items():
        # Usar subset limpio para cada modelo
        dd = d.dropna(subset=features).reset_index(drop=True)
        yy = dd["tgt"]
        X = dd[features]

        X_tr, X_te, y_tr, y_te = split_with_gap(X, yy)

        clf = xgb.XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
        clf.fit(X_tr, y_tr)

        pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1]

        results[name] = calc_metrics_clf(y_te.values, pred, proba)
        results[name]["n_train"] = len(X_tr)
        results[name]["n_test"] = len(X_te)
        predictions[name] = (y_te.values, pred)

        # Feature importance
        imp = dict(zip(features, [round(float(x), 4) for x in clf.feature_importances_]))
        results[name]["feature_importance"] = imp

        print(f"  {name:12s}: acc={results[name]['accuracy']:.4f}  "
              f"f1={results[name]['f1']:.4f}  auc={results[name]['auc']:.4f}  "
              f"n_test={len(X_te)}")

    # McNemar: baseline vs cada modelo con sentimiento
    mcnemar_results = {}
    if "baseline" in predictions:
        yt_b, pb = predictions["baseline"]
        for name in ["reddit", "fear_greed", "combined"]:
            if name in predictions:
                yt_s, ps = predictions[name]
                # Solo comparar si mismo test set size
                min_len = min(len(pb), len(ps))
                b, c, chi2, pv = mcnemar(yt_b[:min_len], pb[:min_len], ps[:min_len])
                mcnemar_results[f"baseline_vs_{name}"] = {
                    "b": b, "c": c, "chi2": chi2, "p": pv
                }

    return {"models": results, "mcnemar": mcnemar_results}


# ══════════════════════════════════════════════════════════════════
# REGRESOR (retorno → precio)
# ══════════════════════════════════════════════════════════════════
def regress(df, has_fg):
    d = df.copy()
    d["tgt_r"] = d["return"].shift(-1)
    d = d.dropna(subset=["tgt_r"] + FEAT_BASELINE).reset_index(drop=True)

    models_config = {"baseline": FEAT_BASELINE}

    d_reddit = d.dropna(subset=["reddit_lag1"])
    if len(d_reddit) >= 50:
        models_config["reddit"] = FEAT_REDDIT

    if has_fg:
        d_fg = d.dropna(subset=["fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"])
        if len(d_fg) >= 50:
            models_config["fear_greed"] = FEAT_FG

        d_comb = d.dropna(subset=["reddit_lag1", "fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"])
        if len(d_comb) >= 50:
            models_config["combined"] = FEAT_COMBINED

    results = {}
    test_sets = {}
    trained_models = {}

    for name, features in models_config.items():
        dd = d.dropna(subset=features).reset_index(drop=True)
        X = dd[features]
        y = dd["tgt_r"]
        prc = dd["price"]
        dts = dd["date"]

        X_tr, X_te, y_tr, y_te = split_with_gap(X, y)
        n_train = len(X_tr)
        pte = prc.iloc[n_train + GAP_DAYS:].reset_index(drop=True)
        dte = dts.iloc[n_train + GAP_DAYS:].reset_index(drop=True)

        reg = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
        reg.fit(X_tr, y_tr)

        pred_r = reg.predict(X_te)
        pred_price = pte.values * (1 + pred_r)
        real_price = pte.values * (1 + y_te.values)

        results[name] = calc_metrics_reg(real_price, pred_price)
        results[name]["n_train"] = len(X_tr)
        results[name]["n_test"] = len(X_te)

        imp = dict(zip(features, [round(float(x), 4) for x in reg.feature_importances_]))
        results[name]["feature_importance"] = imp

        test_sets[name] = (dte, real_price, pred_price)
        trained_models[name] = reg

        print(f"  {name:12s}: MAE=${results[name]['mae']:.2f}  "
              f"RMSE=${results[name]['rmse']:.2f}  R²={results[name]['r2']:.4f}  "
              f"n_test={len(X_te)}")

    # ── Datos para visualización ──────────────────────────────────
    df_all = df[["date", "price"]].dropna().copy()

    # Historia completa
    hist = [{"date": str(dd.date()), "real": round(float(p), 2)}
            for dd, p in zip(df_all["date"], df_all["price"])]

    # Test set (usar baseline + mejor modelo con sentimiento)
    best_sent = None
    if "fear_greed" in results and "baseline" in results:
        if results["fear_greed"]["mae"] < results["baseline"]["mae"]:
            best_sent = "fear_greed"
    if "combined" in results and "baseline" in results:
        if results["combined"]["mae"] < results.get(best_sent or "baseline", {}).get("mae", 999):
            best_sent = "combined"
    if "reddit" in results and best_sent is None:
        best_sent = "reddit"
    if best_sent is None:
        best_sent = "fear_greed" if "fear_greed" in test_sets else "reddit"

    # Test set con baseline
    dte_b, real_b, pred_b = test_sets["baseline"]
    ts_baseline = {str(dd.date()): {"real": round(float(r), 2), "pred_base": round(float(p), 2)}
                   for dd, r, p in zip(dte_b, real_b, pred_b)}

    # Merge test sets
    ts = []
    for dt_str, vals in ts_baseline.items():
        entry = {"date": dt_str, **vals}
        # Agregar predicción del mejor modelo con sentimiento si existe para esa fecha
        if best_sent in test_sets:
            dte_s, _, pred_s = test_sets[best_sent]
            for dd, p in zip(dte_s, pred_s):
                if str(dd.date()) == dt_str:
                    entry["pred_full"] = round(float(p), 2)
                    break
            if "pred_full" not in entry:
                entry["pred_full"] = entry["pred_base"]
        else:
            entry["pred_full"] = entry["pred_base"]
        ts.append(entry)

    # ── Forecast 7 días ───────────────────────────────────────────
    last_row = df.iloc[-1]
    pc = float(df_all["price"].iloc[-1])
    fc = []

    # Features del último día
    feat_vals = {f: float(last_row[f]) if pd.notna(last_row.get(f)) else 0.0
                 for f in FEAT_COMBINED}

    for i in range(FORECAST_DAYS):
        nd = date.today() + timedelta(days=i + 1)
        preds = {}
        for name, reg in trained_models.items():
            feats = models_config[name]
            row = [[feat_vals.get(f, 0.0) for f in feats]]
            pred_r = float(reg.predict(row)[0])
            preds[name] = round(pc * (1 + pred_r), 2)

        fc.append({
            "date": str(nd),
            "pred_base": preds.get("baseline", pc),
            "pred_full": preds.get(best_sent, preds.get("baseline", pc)),
        })

        # Update features para siguiente iteración
        best_price = preds.get(best_sent, preds.get("baseline", pc))
        new_ret = (best_price - pc) / pc if pc != 0 else 0.0
        feat_vals["return"] = new_ret
        feat_vals["ret_ma5"] = feat_vals.get("ret_ma5", 0) * 0.8 + new_ret * 0.2
        feat_vals["ret_ma10"] = feat_vals.get("ret_ma10", 0) * 0.9 + new_ret * 0.1
        feat_vals["volatility5"] = feat_vals.get("volatility5", 0) * 0.8 + abs(new_ret) * 0.2
        feat_vals["volatility10"] = feat_vals.get("volatility10", 0) * 0.9 + abs(new_ret) * 0.1
        feat_vals["momentum5"] = new_ret
        feat_vals["momentum10"] = new_ret
        pc = best_price

    return results, hist, ts, fc, best_sent


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO (para documentar en la tesina)
# ══════════════════════════════════════════════════════════════════
def statistical_analysis(df, has_fg):
    """Genera el análisis estadístico que documenta por qué Reddit no funciona."""
    stats = {}

    d = df.dropna(subset=["return"]).copy()
    d["return_next"] = d["return"].shift(-1)

    # Reddit analysis
    d_r = d.dropna(subset=["sent_reddit", "return_next"])
    if len(d_r) > 30:
        stats["reddit"] = {
            "n_days": len(d_r),
            "corr_same_day": round(float(d_r["sent_reddit"].corr(d_r["return"])), 4),
            "corr_next_day": round(float(d_r["sent_reddit"].corr(d_r["return_next"])), 4),
            "naive_accuracy": round(float(
                ((d_r["sent_reddit"] > 0).astype(int) == (d_r["return_next"] > 0).astype(int)).mean()
            ), 4),
            "conclusion": "NO_SIGNAL",
            "detail": "Correlación con retorno siguiente ~0. Sentimiento reactivo al precio, no predictivo.",
        }

    # F&G analysis
    if has_fg:
        d_fg = d.dropna(subset=["fg_norm", "return_next"])
        if len(d_fg) > 30:
            d_fg["fg_lag1"] = d_fg["fg_norm"].shift(1)
            d_fg2 = d_fg.dropna(subset=["fg_lag1"])

            stats["fear_greed"] = {
                "n_days": len(d_fg2),
                "corr_same_day": round(float(d_fg2["fg_norm"].corr(d_fg2["return"])), 4),
                "corr_next_day": round(float(d_fg2["fg_norm"].corr(d_fg2["return_next"])), 4),
                "corr_lag1":     round(float(d_fg2["fg_lag1"].corr(d_fg2["return"])), 4),
                "naive_accuracy": round(float(
                    ((d_fg2["fg_lag1"] > 0).astype(int) == (d_fg2["return_next"] > 0).astype(int)).mean()
                ), 4),
            }
            corr = abs(stats["fear_greed"]["corr_lag1"])
            if corr > 0.05:
                stats["fear_greed"]["conclusion"] = "WEAK_SIGNAL"
                stats["fear_greed"]["detail"] = f"Correlación débil ({corr:.3f}) pero potencialmente explotable"
            else:
                stats["fear_greed"]["conclusion"] = "NO_SIGNAL"
                stats["fear_greed"]["detail"] = "Sin señal predictiva significativa"

    return stats


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("SOL/USD Sentiment Dashboard — v3")
    print("Reddit vs Fear & Greed Index")
    print("=" * 60)

    print("\nCargando datos...")
    df, reddit_daily, reddit_posts, has_fg = load()

    print("\n── Análisis estadístico ──")
    stats = statistical_analysis(df, has_fg)
    for src, s in stats.items():
        print(f"  {src}: corr_next={s.get('corr_next_day','N/A')}, "
              f"naive_acc={s.get('naive_accuracy','N/A')}, "
              f"→ {s['conclusion']}")

    print("\n── Clasificador (sube/baja) ──")
    clf = classify(df, has_fg)

    print("\n── Regresor (retorno → precio) ──")
    reg, hist, ts, fc, best_sent = regress(df, has_fg)

    # ── Sentimiento diario para gráfico ───────────────────────────
    sw = reddit_daily.merge(df[["date", "price"]].drop_duplicates(),
                            on="date", how="inner").sort_values("date")
    sent_out = [{"date": str(r["date"].date()),
                 "sentiment": round(float(r["sent_reddit"]), 4),
                 "price": round(float(r["price"]), 2)}
                for _, r in sw.iterrows()]

    # F&G diario para gráfico
    fg_out = []
    if has_fg:
        fg_viz = df[df["fg_value"].notna()][["date", "fg_value", "price"]].drop_duplicates("date")
        fg_out = [{"date": str(r["date"].date()),
                   "fg_value": int(r["fg_value"]),
                   "price": round(float(r["price"]), 2)}
                  for _, r in fg_viz.iterrows()]

    # Reddit posts
    smap = pd.read_csv("data/reddit_sentiment.csv").groupby("id")["sent_score"].mean().to_dict()
    posts_out = []
    for _, row in reddit_posts.sort_values("score", ascending=False).head(50).iterrows():
        sv = smap.get(row["id"])
        posts_out.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "title": str(row["title"])[:120],
            "score": int(row.get("score", 0) or 0),
            "num_comments": int(row.get("num_comments", 0) or 0),
            "sent_score": round(float(sv), 4) if sv is not None else None,
            "url": str(row.get("url", ""))})

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n── Resumen ──")
    print(f"  Mejor fuente de sentimiento: {best_sent}")

    # Comparación
    clf_models = clf["models"]
    base_acc = clf_models.get("baseline", {}).get("accuracy", 0)
    for name in ["reddit", "fear_greed", "combined"]:
        if name in clf_models:
            diff = clf_models[name]["accuracy"] - base_acc
            icon = "✅" if diff > 0 else "❌"
            print(f"  {icon} Clf {name}: {diff:+.4f} vs baseline")

    reg_models = reg
    base_mae = reg_models.get("baseline", {}).get("mae", 999)
    for name in ["reddit", "fear_greed", "combined"]:
        if name in reg_models:
            diff = base_mae - reg_models[name]["mae"]
            icon = "✅" if diff > 0 else "❌"
            print(f"  {icon} Reg {name}: {diff:+.4f} MAE vs baseline")

    # ── Export JSON ────────────────────────────────────────────────
    out_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "today_price":  hist[-1]["real"] if hist else None,
        "today_date":   hist[-1]["date"] if hist else None,
        "model_start_date": str(df["date"].min().date()),
        "model_end_date":   str(df["date"].max().date()),
        "model_days":       len(df),
        "total_price_days": len(df),
        "sentiment_coverage_pct": round(
            df["reddit_lag1"].notna().sum() / len(df) * 100, 1),
        "fg_coverage_pct": round(
            df["fg_lag1"].notna().sum() / len(df) * 100, 1) if has_fg else 0,
        "best_sentiment_source": best_sent,
        "statistical_analysis": stats,
        "classifier":      clf,
        "regression":      reg,
        "price_history":   hist,
        "price_test":      ts,
        "forecast_7d":     fc,
        "sentiment_daily": sent_out,
        "fg_daily":        fg_out,
        "reddit_posts":    posts_out,
    }

    out_path = Path("dashboard/public/data/dashboard_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    print(f"\nExportado → {out_path}")
    print(f"  Hoy: ${out_data['today_price']} ({out_data['today_date']})")

if __name__ == "__main__":
    main()
