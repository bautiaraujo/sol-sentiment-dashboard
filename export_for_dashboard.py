import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score)

CUTOFF        = pd.Timestamp("2024-01-01")
FORECAST_DAYS = 7

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
    """Replica exacta de model.py original: left merge + dropna = inner join efectivo."""
    pr = pd.read_csv("data/solana_prices.csv",    parse_dates=["date"])
    se = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    re = pd.read_csv("data/reddit_posts.csv",     parse_dates=["date"])
    pr = pr[pr["date"] >= CUTOFF].copy()
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()
    sd = se.groupby("date")["sent_score"].mean().reset_index()
    df = pr.merge(sd, on="date", how="left")
    df["return"] = df["price"].pct_change()
    df = df.dropna().reset_index(drop=True)
    df_all = pr.copy()
    df_all["return"] = df_all["price"].pct_change()
    df_all = df_all.dropna().reset_index(drop=True)
    n, na = len(df), len(df_all)
    print(f"  Precios 2024+: {na} | Con sentimiento real: {n} ({round(n/na*100,1)}%)")
    print(f"  Rango modelo: {df['date'].min().date()} -> {df['date'].max().date()}")
    return df, df_all, sd, re

def classify(df):
    """Clasificador: replica exacta de model.py."""
    d = df.copy()
    d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    y = d["tgt"].values
    Xb = d[["return"]]; Xf = d[["return", "sent_score"]]
    Xb_tr, Xb_te, y_tr, y_te = train_test_split(Xb, y, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _, _       = train_test_split(Xf, y, shuffle=False, test_size=0.2)
    mb = xgb.XGBClassifier(**CLF_P); mb.fit(Xb_tr, y_tr)
    mf = xgb.XGBClassifier(**CLF_P); mf.fit(Xf_tr, y_tr)
    pb = mb.predict(Xb_te); pf = mf.predict(Xf_te)
    def m(yt, yh): return {k: round(float(v), 4) for k, v in {
        "accuracy":  accuracy_score(yt, yh),
        "precision": precision_score(yt, yh, zero_division=0),
        "recall":    recall_score(yt, yh, zero_division=0),
        "f1":        f1_score(yt, yh, zero_division=0)}.items()}
    b, c, chi2, pv = mcnemar(y_te, pb, pf)
    return {"baseline": m(y_te, pb), "full": m(y_te, pf),
            "mcnemar": {"b": b, "c": c, "chi2": chi2, "p": pv}}

def regress(df, df_all):
    """
    Regresor de RETORNOS (no precio absoluto).
    Evita el problema de nivel: si train=Aug2025 ($160) y test=Mar2026 ($130),
    predecir precio absoluto da MAE alto por el drift. Prediciendo retornos
    (cambios %) el modelo aprende la dinamica sin importar el nivel.
    Reconstruccion: precio_predicho = precio_hoy * (1 + retorno_predicho)
    """
    d = df.copy()
    d["tgt_r"] = d["return"].shift(-1)
    d = d.dropna().reset_index(drop=True)
    Xb = d[["return"]]; Xf = d[["return", "sent_score"]]
    y = d["tgt_r"]; dts = d["date"]; prc = d["price"]
    Xb_tr, Xb_te, ytr, yte = train_test_split(Xb, y, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _, _     = train_test_split(Xf, y, shuffle=False, test_size=0.2)
    pte = prc.iloc[len(Xb_tr):].reset_index(drop=True)
    dte = dts.iloc[len(Xb_tr):].reset_index(drop=True)
    rb = xgb.XGBRegressor(**REG_P); rb.fit(Xb_tr, ytr)
    rf = xgb.XGBRegressor(**REG_P); rf.fit(Xf_tr, ytr)
    rb_r = rb.predict(Xb_te); rf_r = rf.predict(Xf_te)
    ypb = pte.values * (1 + rb_r)
    ypf = pte.values * (1 + rf_r)
    real_p = pte.values * (1 + yte.values)
    def rm(yt, yh): return {k: round(float(v), 4) for k, v in {
        "mae":  mean_absolute_error(yt, yh),
        "rmse": float(np.sqrt(mean_squared_error(yt, yh))),
        "r2":   r2_score(yt, yh)}.items()}
    met = {"baseline": rm(real_p, ypb), "full": rm(real_p, ypf)}
    hist = [{"date": str(dd.date()), "real": round(float(p), 2)}
            for dd, p in zip(df_all["date"], df_all["price"])]
    ts = [{"date": str(dd.date()), "real": round(float(r), 2),
           "pred_base": round(float(pb2), 2), "pred_full": round(float(pf2), 2)}
          for dd, r, pb2, pf2 in zip(dte, real_p, ypb, ypf)]
    pc = float(df_all["price"].iloc[-1]); pp = float(df_all["price"].iloc[-2])
    rc = (pc - pp) / pp if pp != 0 else 0.0
    sc = float(df["sent_score"].iloc[-1])
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
    print("Cargando datos...")
    df, df_all, sd, reddit = load()
    print("Clasificador...")
    clf = classify(df)
    print("  baseline=" + str(clf["baseline"]["accuracy"]) +
          "  full=" + str(clf["full"]["accuracy"]))
    print("Regresor retornos + Forecast...")
    reg, hist, ts, fc = regress(df, df_all)
    print("  baseline MAE=" + str(reg["baseline"]["mae"]) +
          "  full MAE=" + str(reg["full"]["mae"]))
    sw = sd.merge(df_all[["date", "price"]], on="date", how="inner").sort_values("date")
    sent = [{"date": str(r["date"].date()),
             "sentiment": round(float(r["sent_score"]), 4),
             "price": round(float(r["price"]), 2)} for _, r in sw.iterrows()]
    smap = pd.read_csv("data/reddit_sentiment.csv").groupby("id")["sent_score"].mean().to_dict()
    posts = []
    for _, row in reddit.sort_values("score", ascending=False).head(50).iterrows():
        sv = smap.get(row["id"])
        posts.append({
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
        "model_start_date": str(df["date"].min().date()),
        "model_end_date":   str(df["date"].max().date()),
        "model_days":       len(df),
        "total_price_days": len(df_all),
        "sentiment_coverage_pct": round(len(df) / len(df_all) * 100, 1),
        "classifier":      clf,
        "regression":      reg,
        "price_history":   hist,
        "price_test":      ts,
        "forecast_7d":     fc,
        "sentiment_daily": sent,
        "reddit_posts":    posts,
    }
    out_path = Path("dashboard/public/data/dashboard_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)
    print("Exportado -> " + str(out_path))
    print("  Hoy: " + str(out_data["today_price"]) + " (" + str(out_data["today_date"]) + ")")
    print("  Cobertura: " + str(out_data["sentiment_coverage_pct"]) + "% (" + str(len(df)) + " dias)")
    print("  Forecast: " + str([x["pred_full"] for x in fc]))

if __name__ == "__main__":
    main()
