"""
export_for_dashboard_v2.py
──────────────────────────
Mejoras vs v1:
  1. Feature engineering: 8+ features derivados del sentimiento
  2. Rolling calculado sobre calendario real (reindex diario)
  3. Modelo simplificado (menos overfitting con pocos datos)
  4. Gap temporal de 1 día entre train/test
  5. Validación cruzada temporal (TimeSeriesSplit) para tuning
  6. Métricas extendidas: accuracy, precision, recall, f1, AUC
"""

import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

CUTOFF        = pd.Timestamp("2024-01-01")
FORECAST_DAYS = 7
MIN_POSTS_DAY = 3      # bajado de 5 → 3 para más cobertura (más datos > pureza extrema)
ROLLING_SENT  = 5       # ventana rolling del sentimiento (calendario, no obs)
ROLLING_RET   = 5       # ventana rolling de retornos (features técnicos)
TEST_RATIO    = 0.2
GAP_DAYS      = 1       # gap entre train y test para evitar leakage

# ── Modelo MÁS SIMPLE: menos overfitting con pocos datos ─────────
CLF_P = dict(
    eval_metric    = "logloss",
    n_estimators   = 150,       # ← bajado de 400
    max_depth      = 3,         # ← bajado de 4
    learning_rate  = 0.08,      # ← subido de 0.05 (converge antes)
    subsample      = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,       # ← nuevo: evita splits con pocas muestras
    reg_alpha      = 0.1,       # ← nuevo: L1 regularization
    reg_lambda     = 1.0,       # ← nuevo: L2 regularization
    random_state   = 42,
)
REG_P = dict(
    objective      = "reg:squarederror",
    n_estimators   = 150,
    max_depth      = 3,
    learning_rate  = 0.08,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha      = 0.1,
    reg_lambda     = 1.0,
    random_state   = 42,
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


def load():
    pr = pd.read_csv("data/solana_prices.csv",    parse_dates=["date"])
    se = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    re = pd.read_csv("data/reddit_posts.csv",     parse_dates=["date"])

    pr = pr[pr["date"] >= CUTOFF].copy()
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()

    # ── Sentimiento ponderado por upvotes, filtro calidad ─────────
    posts_per_day = se.groupby("date").size()
    valid_days    = posts_per_day[posts_per_day >= MIN_POSTS_DAY].index
    se_filtered   = se[se["date"].isin(valid_days)].copy()
    se_filtered["score_clip"] = se_filtered["score"].clip(lower=1)

    def weighted_sent(g):
        return pd.Series({
            "sent_score": (g["sent_score"] * g["score_clip"]).sum() / g["score_clip"].sum(),
            "sent_std":    g["sent_score"].std() if len(g) > 1 else 0.0,
            "n_posts":     len(g),
            "total_score": g["score_clip"].sum(),
        })

    sd_daily = (se_filtered.groupby("date")
                           .apply(weighted_sent)
                           .reset_index())

    # ── CAMBIO CLAVE: reindex a calendario diario antes del rolling ──
    # Así rolling(5) = 5 días CALENDARIO reales, no 5 observaciones dispersas
    date_range = pd.date_range(pr["date"].min(), pr["date"].max(), freq="D")
    sd_calendar = (sd_daily.set_index("date")
                           .reindex(date_range)
                           .rename_axis("date")
                           .reset_index())

    # Rolling sobre calendario → forward fill limitado a 3 días
    sd_calendar["sent_rolling"] = (sd_calendar["sent_score"]
                                   .rolling(window=ROLLING_SENT, min_periods=1)
                                   .mean())
    sd_calendar["sent_std_rolling"] = (sd_calendar["sent_std"]
                                        .rolling(window=ROLLING_SENT, min_periods=1)
                                        .mean())
    sd_calendar["n_posts_rolling"] = (sd_calendar["n_posts"]
                                       .rolling(window=ROLLING_SENT, min_periods=1)
                                       .sum())

    # Forward fill limitado para cubrir gaps de 1-2 días sin posts
    for col in ["sent_rolling", "sent_std_rolling", "n_posts_rolling"]:
        sd_calendar[col] = sd_calendar[col].ffill(limit=2)

    # ── Merge con precios ──────────────────────────────────────────
    df = pr.merge(sd_calendar[["date", "sent_rolling", "sent_std_rolling", "n_posts_rolling"]],
                  on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # ── Features técnicos de precio ────────────────────────────────
    df["return"]      = df["price"].pct_change()
    df["ret_ma5"]     = df["return"].rolling(ROLLING_RET, min_periods=1).mean()
    df["volatility5"] = df["return"].rolling(ROLLING_RET, min_periods=2).std()
    df["momentum5"]   = df["price"].pct_change(ROLLING_RET)

    # ── Features derivados del sentimiento ──────────────────────────
    # SHIFT(1): usar sentimiento de AYER para predecir HOY (evitar look-ahead)
    df["sent_lag1"]       = df["sent_rolling"].shift(1)
    df["sent_std_lag1"]   = df["sent_std_rolling"].shift(1)
    df["n_posts_lag1"]    = df["n_posts_rolling"].shift(1)

    # Cambio de sentimiento (delta): ¿está subiendo o bajando la opinión?
    df["sent_delta"]      = df["sent_lag1"] - df["sent_rolling"].shift(2)

    # Divergencia precio-sentimiento: si el precio sube pero el sentimiento baja → señal
    df["sent_price_div"]  = df["sent_delta"] - df["return"].shift(1)

    # Interacción: sentimiento × volumen de posts (sentimiento importa más con más posts)
    df["sent_x_volume"]   = df["sent_lag1"] * np.log1p(df["n_posts_lag1"].fillna(0))

    # ── Separar: df_model (con sentimiento) y df_all (historia completa) ──
    # df_model: solo filas con sentimiento válido
    df_model = df.dropna(subset=["sent_lag1", "return"]).reset_index(drop=True)

    # df_all: historia completa para visualización
    df_all = pr.copy()
    df_all["return"] = df_all["price"].pct_change()
    df_all = df_all.dropna().reset_index(drop=True)

    n = len(df_model)
    print(f"  Precios 2024+:          {len(df_all)} dias")
    print(f"  Dias con sent>={MIN_POSTS_DAY} posts: {len(valid_days)} dias")
    print(f"  Dias en dataset modelo: {n} dias  ({n/len(df_all)*100:.1f}% cobertura)")
    print(f"  Rango: {df_model['date'].min().date()} -> {df_model['date'].max().date()}")
    print(f"  sent_lag1 std: {df_model['sent_lag1'].std():.4f}")
    print(f"  sent_delta std: {df_model['sent_delta'].std():.4f}")

    return df_model, df_all, sd_daily, re


# ── Features para cada modelo ─────────────────────────────────────
BASELINE_FEATURES = ["return", "ret_ma5", "volatility5", "momentum5"]

FULL_FEATURES = BASELINE_FEATURES + [
    "sent_lag1",        # sentimiento lagged 1 día
    "sent_std_lag1",    # incertidumbre del sentimiento
    "sent_delta",       # cambio de sentimiento
    "sent_price_div",   # divergencia precio vs sentimiento
    "sent_x_volume",    # sentimiento × log(volumen posts)
]


def split_with_gap(X, y, test_ratio=TEST_RATIO, gap=GAP_DAYS):
    """Train/test split temporal con gap para evitar leakage."""
    n = len(X)
    n_test = int(n * test_ratio)
    n_train = n - n_test - gap
    X_tr = X.iloc[:n_train]
    y_tr = y.iloc[:n_train]
    X_te = X.iloc[n_train + gap:]
    y_te = y.iloc[n_train + gap:]
    return X_tr, X_te, y_tr, y_te


def classify(df_model):
    d = df_model.copy()
    d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    y = d["tgt"].values

    Xb = d[BASELINE_FEATURES]
    Xf = d[FULL_FEATURES]

    Xb_tr, Xb_te, y_tr, y_te = split_with_gap(Xb, pd.Series(y))
    Xf_tr, Xf_te, _,   _     = split_with_gap(Xf, pd.Series(y))

    # ── Cross-validation temporal para verificar estabilidad ──────
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores_b, cv_scores_f = [], []
    for tr_idx, te_idx in tscv.split(Xb_tr):
        mb_cv = xgb.XGBClassifier(**CLF_P)
        mf_cv = xgb.XGBClassifier(**CLF_P)
        mb_cv.fit(Xb_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        mf_cv.fit(Xf_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        cv_scores_b.append(accuracy_score(y_tr.iloc[te_idx], mb_cv.predict(Xb_tr.iloc[te_idx])))
        cv_scores_f.append(accuracy_score(y_tr.iloc[te_idx], mf_cv.predict(Xf_tr.iloc[te_idx])))

    print(f"  CV baseline acc: {np.mean(cv_scores_b):.4f} ± {np.std(cv_scores_b):.4f}")
    print(f"  CV full acc:     {np.mean(cv_scores_f):.4f} ± {np.std(cv_scores_f):.4f}")

    # ── Entrenar modelos finales ──────────────────────────────────
    mb = xgb.XGBClassifier(**CLF_P); mb.fit(Xb_tr, y_tr)
    mf = xgb.XGBClassifier(**CLF_P); mf.fit(Xf_tr, y_tr)
    pb = mb.predict(Xb_te); pf = mf.predict(Xf_te)

    # Probabilidades para AUC
    pb_proba = mb.predict_proba(Xb_te)[:, 1]
    pf_proba = mf.predict_proba(Xf_te)[:, 1]

    def m(yt, yh, yp):
        return {k: round(float(v), 4) for k, v in {
            "accuracy":  accuracy_score(yt, yh),
            "precision": precision_score(yt, yh, zero_division=0),
            "recall":    recall_score(yt, yh, zero_division=0),
            "f1":        f1_score(yt, yh, zero_division=0),
            "auc":       roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.0,
        }.items()}

    b, c, chi2, pv = mcnemar(y_te.values, pb, pf)
    n_test = len(y_te)
    print(f"  Test set clasificador: {n_test} dias")

    # Feature importance del modelo full
    imp = dict(zip(FULL_FEATURES, [round(float(x), 4) for x in mf.feature_importances_]))
    print(f"  Feature importance: {imp}")

    return {
        "baseline": m(y_te, pb, pb_proba),
        "full":     m(y_te, pf, pf_proba),
        "mcnemar":  {"b": b, "c": c, "chi2": chi2, "p": pv},
        "cv_baseline_acc": round(float(np.mean(cv_scores_b)), 4),
        "cv_full_acc":     round(float(np.mean(cv_scores_f)), 4),
        "feature_importance": imp,
    }


def regress(df_model, df_all):
    d = df_model.copy()
    d["tgt_r"] = d["return"].shift(-1)
    d = d.dropna().reset_index(drop=True)

    Xb = d[BASELINE_FEATURES]
    Xf = d[FULL_FEATURES]
    y  = d["tgt_r"]
    dts = d["date"]
    prc = d["price"]

    Xb_tr, Xb_te, ytr, yte = split_with_gap(Xb, y)
    Xf_tr, Xf_te, _,   _   = split_with_gap(Xf, y)

    # Índices del test set alineados con precio y fecha
    n_train = len(Xb_tr)
    gap = GAP_DAYS
    pte = prc.iloc[n_train + gap:].reset_index(drop=True)
    dte = dts.iloc[n_train + gap:].reset_index(drop=True)

    rb = xgb.XGBRegressor(**REG_P); rb.fit(Xb_tr, ytr)
    rf = xgb.XGBRegressor(**REG_P); rf.fit(Xf_tr, ytr)

    rb_r = rb.predict(Xb_te); rf_r = rf.predict(Xf_te)
    ypb    = pte.values * (1 + rb_r)
    ypf    = pte.values * (1 + rf_r)
    real_p = pte.values * (1 + yte.values)

    def rm(yt, yh):
        return {k: round(float(v), 4) for k, v in {
            "mae":  mean_absolute_error(yt, yh),
            "rmse": float(np.sqrt(mean_squared_error(yt, yh))),
            "r2":   r2_score(yt, yh),
        }.items()}

    met = {"baseline": rm(real_p, ypb), "full": rm(real_p, ypf)}

    # Feature importance regresor
    imp = dict(zip(FULL_FEATURES, [round(float(x), 4) for x in rf.feature_importances_]))
    met["feature_importance"] = imp
    print(f"  Reg feature importance: {imp}")

    # Historia completa 2024+
    hist = [{"date": str(dd.date()), "real": round(float(p), 2)}
            for dd, p in zip(df_all["date"], df_all["price"])]

    # Test set
    ts = [{"date": str(dd.date()), "real": round(float(r), 2),
           "pred_base": round(float(pb2), 2), "pred_full": round(float(pf2), 2)}
          for dd, r, pb2, pf2 in zip(dte, real_p, ypb, ypf)]

    # ── Forecast recursivo 7 días ─────────────────────────────────
    last_row = df_model.iloc[-1]
    pc = float(df_all["price"].iloc[-1])

    # Construir features iniciales del último día conocido
    ret      = float(last_row["return"])
    ret_ma5  = float(last_row["ret_ma5"])
    vol5     = float(last_row["volatility5"])
    mom5     = float(last_row["momentum5"])
    sent     = float(last_row.get("sent_lag1", 0))
    sent_std = float(last_row.get("sent_std_lag1", 0))
    sent_d   = float(last_row.get("sent_delta", 0))
    sent_pd  = float(last_row.get("sent_price_div", 0))
    sent_xv  = float(last_row.get("sent_x_volume", 0))

    fc = []
    for i in range(FORECAST_DAYS):
        nd = date.today() + timedelta(days=i + 1)
        base_row = [[ret, ret_ma5, vol5, mom5]]
        full_row = [[ret, ret_ma5, vol5, mom5, sent, sent_std, sent_d, sent_pd, sent_xv]]

        rb_i = float(rb.predict(base_row)[0])
        rf_i = float(rf.predict(full_row)[0])
        pb2  = round(pc * (1 + rb_i), 2)
        pf2  = round(pc * (1 + rf_i), 2)
        fc.append({"date": str(nd), "pred_base": pb2, "pred_full": pf2})

        # Actualizar features para siguiente iteración
        new_ret = (pf2 - pc) / pc if pc != 0 else 0.0
        ret_ma5 = ret_ma5 * 0.8 + new_ret * 0.2   # aproximación del rolling
        vol5    = vol5 * 0.8 + abs(new_ret) * 0.2  # aproximación
        mom5    = (pf2 / (pc / (1 + mom5)) - 1) if mom5 != -1 else new_ret
        ret     = new_ret
        pc      = pf2
        # Sentimiento se mantiene constante en forecast (no hay datos futuros)

    return met, hist, ts, fc, rb, rf


def main():
    print("=" * 60)
    print("SOL/USD Sentiment Dashboard — v2 (feature engineering)")
    print("=" * 60)

    print("\nCargando datos con filtros de calidad...")
    df_model, df_all, sd, reddit = load()

    print("\nClasificador...")
    clf = classify(df_model)
    print(f"  → baseline acc={clf['baseline']['accuracy']}  full={clf['full']['accuracy']}")
    print(f"  → baseline f1={clf['baseline']['f1']}   full f1={clf['full']['f1']}")
    print(f"  → baseline auc={clf['baseline']['auc']}  full auc={clf['full']['auc']}")

    print("\nRegresor retornos + Forecast 7d...")
    reg, hist, ts, fc, _, _ = regress(df_model, df_all)
    print(f"  → baseline MAE={reg['baseline']['mae']}  full MAE={reg['full']['mae']}")
    print(f"  → baseline R²={reg['baseline']['r2']}   full R²={reg['full']['r2']}")

    # ── Sentimiento diario para gráfico ───────────────────────────
    sw = sd.merge(df_all[["date", "price"]], on="date", how="inner").sort_values("date")
    sent_out = [{"date": str(r["date"].date()),
                 "sentiment": round(float(r["sent_score"]), 4),
                 "price": round(float(r["price"]), 2)}
                for _, r in sw.iterrows()]

    smap = pd.read_csv("data/reddit_sentiment.csv").groupby("id")["sent_score"].mean().to_dict()
    posts_out = []
    for _, row in reddit.sort_values("score", ascending=False).head(50).iterrows():
        sv = smap.get(row["id"])
        posts_out.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "title": str(row["title"])[:120],
            "score": int(row.get("score", 0) or 0),
            "num_comments": int(row.get("num_comments", 0) or 0),
            "sent_score": round(float(sv), 4) if sv is not None else None,
            "url": str(row.get("url", ""))})

    # ── Improvement summary ───────────────────────────────────────
    clf_diff = clf["full"]["accuracy"] - clf["baseline"]["accuracy"]
    reg_diff = reg["baseline"]["mae"]  - reg["full"]["mae"]
    print(f"\n  {'✅' if clf_diff > 0 else '❌'} Clasificador: full {'>' if clf_diff > 0 else '<'} baseline por {abs(clf_diff):.4f} acc")
    print(f"  {'✅' if reg_diff > 0 else '❌'} Regresor: full {'>' if reg_diff > 0 else '<'} baseline por {abs(reg_diff):.4f} MAE")

    out_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "today_price":  hist[-1]["real"] if hist else None,
        "today_date":   hist[-1]["date"] if hist else None,
        "model_start_date": str(df_model["date"].min().date()),
        "model_end_date":   str(df_model["date"].max().date()),
        "model_days":       len(df_model),
        "total_price_days": len(df_all),
        "sentiment_coverage_pct": round(len(df_model) / len(df_all) * 100, 1),
        "classifier":      clf,
        "regression":      reg,
        "price_history":   hist,
        "price_test":      ts,
        "forecast_7d":     fc,
        "sentiment_daily": sent_out,
        "reddit_posts":    posts_out,
    }
    out_path = Path("dashboard/public/data/dashboard_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    print(f"\nExportado → {out_path}")
    print(f"  Hoy: ${out_data['today_price']} ({out_data['today_date']})")
    print(f"  Cobertura: {out_data['sentiment_coverage_pct']}% ({len(df_model)} dias)")
    print(f"  Forecast: {[x['pred_full'] for x in fc]}")

if __name__ == "__main__":
    main()
