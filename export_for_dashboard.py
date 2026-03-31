import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score)

CUTOFF        = pd.Timestamp("2024-01-01")
FORECAST_DAYS = 7
MIN_POSTS_DAY = 5    # minimo de posts para considerar el sentimiento confiable
ROLLING_DAYS  = 3    # media movil del sentimiento

def mcnemar(y_true, pa, pb):
    ca = pa == y_true; cb = pb == y_true
    b = int((~ca & cb).sum()); c = int((ca & ~cb).sum()); n = b + c
    if n == 0: return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / n
    try:
        from scipy.stats import chi2 as d; pv = float(d.sf(chi2, df=1))
    except Exception: pv = float(np.exp(-0.5 * chi2) * 1.2533)
    return b, c, round(float(chi2), 4), round(pv, 4)

CLF_P = dict(eval_metric="logloss", n_estimators=400, max_depth=4,
             learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
REG_P = dict(objective="reg:squarederror", n_estimators=400, max_depth=4,
             learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)

def load():
    pr = pd.read_csv("data/solana_prices.csv",    parse_dates=["date"])
    se = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    re = pd.read_csv("data/reddit_posts.csv",     parse_dates=["date"])

    pr = pr[pr["date"] >= CUTOFF].copy()
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()

    # ── Cambio 1: filtro de calidad mínima ────────────────────────────
    # Solo usar días con >= MIN_POSTS_DAY posts (sentimiento confiable)
    posts_per_day = se.groupby("date").size()
    valid_days    = posts_per_day[posts_per_day >= MIN_POSTS_DAY].index
    se_filtered   = se[se["date"].isin(valid_days)].copy()

    # ── Cambio 2: sentimiento ponderado por upvotes ────────────────────
    # Un post con 5000 upvotes vale más que uno con 1 upvote
    se_filtered["score_clip"] = se_filtered["score"].clip(lower=1)

    def weighted_sent(g):
        return (g["sent_score"] * g["score_clip"]).sum() / g["score_clip"].sum()

    sd_weighted = (se_filtered.groupby("date")
                              .apply(weighted_sent)
                              .reset_index())
    sd_weighted.columns = ["date", "sent_score"]

    # Merge: solo dias con sentimiento de calidad (inner join efectivo)
    df_model = pr.merge(sd_weighted, on="date", how="inner")
    df_model["return"] = df_model["price"].pct_change()
    df_model = df_model.dropna().reset_index(drop=True)

    # ── Cambio 3: rolling mean de sentimiento ─────────────────────────
    df_model = df_model.sort_values("date").reset_index(drop=True)
    df_model["sent_score"] = (df_model["sent_score"]
                              .rolling(window=ROLLING_DAYS, min_periods=1)
                              .mean().shift(1))

    # Historia completa de precios para visualizacion
    df_all = pr.copy()
    df_all["return"] = df_all["price"].pct_change()
    df_all = df_all.dropna().reset_index(drop=True)

    n, na = len(df_model), len(df_all)
    print(f"  Precios 2024+:        {na} dias")
    print(f"  Dias con sent>=5 posts: {len(valid_days)} dias")
    print(f"  Dias en dataset modelo: {n} dias")
    print(f"  Rango modelo: {df_model['date'].min().date()} -> {df_model['date'].max().date()}")
    print(f"  sent_score std: {df_model['sent_score'].std():.4f}  var: {df_model['sent_score'].var():.4f}")

    return df_model, df_all, sd_weighted, re

def classify(df_model):
    d = df_model.copy()
    d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    y  = d["tgt"].values
    Xb = d[["return"]]
    Xf = d[["return", "sent_score"]]
    Xb_tr, Xb_te, y_tr, y_te = train_test_split(Xb, y, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _,   _     = train_test_split(Xf, y, shuffle=False, test_size=0.2)
    mb = xgb.XGBClassifier(**CLF_P); mb.fit(Xb_tr, y_tr)
    mf = xgb.XGBClassifier(**CLF_P); mf.fit(Xf_tr, y_tr)
    pb = mb.predict(Xb_te); pf = mf.predict(Xf_te)
    def m(yt, yh): return {k: round(float(v), 4) for k, v in {
        "accuracy":  accuracy_score(yt, yh),
        "precision": precision_score(yt, yh, zero_division=0),
        "recall":    recall_score(yt, yh, zero_division=0),
        "f1":        f1_score(yt, yh, zero_division=0)}.items()}
    b, c, chi2, pv = mcnemar(y_te, pb, pf)
    n_test = len(y_te)
    print(f"  Test set clasificador: {n_test} dias")
    return {"baseline": m(y_te, pb), "full": m(y_te, pf),
            "mcnemar": {"b": b, "c": c, "chi2": chi2, "p": pv}}

def regress(df_model, df_all):
    d = df_model.copy()
    d["tgt_r"] = d["return"].shift(-1)
    d = d.dropna().reset_index(drop=True)
    Xb = d[["return"]]; Xf = d[["return", "sent_score"]]
    y = d["tgt_r"]; dts = d["date"]; prc = d["price"]
    Xb_tr, Xb_te, ytr, yte = train_test_split(Xb, y, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _,   _   = train_test_split(Xf, y, shuffle=False, test_size=0.2)
    pte = prc.iloc[len(Xb_tr):].reset_index(drop=True)
    dte = dts.iloc[len(Xb_tr):].reset_index(drop=True)
    rb = xgb.XGBRegressor(**REG_P); rb.fit(Xb_tr, ytr)
    rf = xgb.XGBRegressor(**REG_P); rf.fit(Xf_tr, ytr)
    rb_r = rb.predict(Xb_te); rf_r = rf.predict(Xf_te)
    ypb    = pte.values * (1 + rb_r)
    ypf    = pte.values * (1 + rf_r)
    real_p = pte.values * (1 + yte.values)
    def rm(yt, yh): return {k: round(float(v), 4) for k, v in {
        "mae":  mean_absolute_error(yt, yh),
        "rmse": float(np.sqrt(mean_squared_error(yt, yh))),
        "r2":   r2_score(yt, yh)}.items()}
    met = {"baseline": rm(real_p, ypb), "full": rm(real_p, ypf)}
    # Historia completa 2024+
    hist = [{"date": str(dd.date()), "real": round(float(p), 2)}
            for dd, p in zip(df_all["date"], df_all["price"])]
    # Test set
    ts = [{"date": str(dd.date()), "real": round(float(r), 2),
           "pred_base": round(float(pb2), 2), "pred_full": round(float(pf2), 2)}
          for dd, r, pb2, pf2 in zip(dte, real_p, ypb, ypf)]
    # Forecast recursivo 7 dias
    pc = float(df_all["price"].iloc[-1]); pp = float(df_all["price"].iloc[-2])
    rc = (pc - pp) / pp if pp != 0 else 0.0
    sc = float(df_model["sent_score"].iloc[-1])
    fc = []
    for i in range(FORECAST_DAYS):
        nd   = date.today() + timedelta(days=i + 1)
        rb_i = float(rb.predict([[rc]])[0])
        rf_i = float(rf.predict([[rc, sc]])[0])
        pb2  = round(pc * (1 + rb_i), 2)
        pf2  = round(pc * (1 + rf_i), 2)
        fc.append({"date": str(nd), "pred_base": pb2, "pred_full": pf2})
        pp = pc; pc = pf2; rc = (pc - pp) / pp if pp != 0 else 0.0
    return met, hist, ts, fc

def main():
    print("Cargando datos con filtros de calidad...")
    df_model, df_all, sd, reddit = load()

    print("Clasificador...")
    clf = classify(df_model)
    print(f"  baseline acc={clf['baseline']['accuracy']}  full={clf['full']['accuracy']}")

    print("Regresor retornos + Forecast 7d...")
    reg, hist, ts, fc = regress(df_model, df_all)
    print(f"  baseline MAE={reg['baseline']['mae']}  full MAE={reg['full']['mae']}")

    # Sentimiento diario para el grafico (usando sent ponderado, no rolling)
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
    print("Exportado -> " + str(out_path))
    print("  Hoy: $" + str(out_data["today_price"]) + " (" + str(out_data["today_date"]) + ")")
    print("  Cobertura: " + str(out_data["sentiment_coverage_pct"]) + "% (" + str(len(df_model)) + " dias)")
    print("  Forecast: " + str([x["pred_full"] for x in fc]))

if __name__ == "__main__":
    main()
