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
    """
    Replica exacta del approach de model.py original que funciono:
    - Merge LEFT precios con sentimiento diario
    - dropna() elimina filas sin sentimiento -> efectivamente inner join
    - Features simples: solo 'return' para baseline, 'return+sent_score' para full
    """
    pr = pd.read_csv("data/solana_prices.csv",    parse_dates=["date"])
    se = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    re = pd.read_csv("data/reddit_posts.csv",     parse_dates=["date"])

    pr = pr[pr["date"] >= CUTOFF].copy()
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()

    # Sentimiento diario promedio
    sent_daily = se.groupby("date")["sent_score"].mean().reset_index()

    # Merge LEFT + dropna = solo dias con sentimiento real (igual que model.py)
    df = pr.merge(sent_daily, on="date", how="left")
    df["return"] = df["price"].pct_change()
    df = df.dropna().reset_index(drop=True)   # elimina filas sin sent_score o return

    # Historia completa de precios para el grafico (sin filtrar por sentimiento)
    df_all = pr.copy()
    df_all["return"] = df_all["price"].pct_change()
    df_all = df_all.dropna().reset_index(drop=True)

    n = len(df)
    n_all = len(df_all)
    print(f"  Precios totales 2024+: {n_all} dias")
    print(f"  Dias con sentimiento real: {n} dias ({round(n/n_all*100,1)}%)")
    print(f"  Rango modelo: {df['date'].min().date()} -> {df['date'].max().date()}")

    return df, df_all, sent_daily, re


def classify(df):
    """Clasificador: replica exacta de model.py original."""
    d = df.copy()
    d["target"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna().reset_index(drop=True)

    y  = d["target"].values
    Xb = d[["return"]]                      # baseline: solo mercado
    Xf = d[["return", "sent_score"]]        # full: mercado + sentimiento

    Xb_tr, Xb_te, y_tr, y_te = train_test_split(Xb, y, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _,   _     = train_test_split(Xf, y, shuffle=False, test_size=0.2)

    mb = xgb.XGBClassifier(**CLF_P); mb.fit(Xb_tr, y_tr)
    mf = xgb.XGBClassifier(**CLF_P); mf.fit(Xf_tr, y_tr)

    pb = mb.predict(Xb_te)
    pf = mf.predict(Xf_te)

    def m(yt, yh): return {k: round(float(v), 4) for k, v in {
        "accuracy":  accuracy_score(yt, yh),
        "precision": precision_score(yt, yh, zero_division=0),
        "recall":    recall_score(yt, yh, zero_division=0),
        "f1":        f1_score(yt, yh, zero_division=0)}.items()}

    b, c, chi2, pv = mcnemar(y_te, pb, pf)
    return {"baseline": m(y_te, pb), "full": m(y_te, pf),
            "mcnemar": {"b": b, "c": c, "chi2": chi2, "p": pv}}


def regress_and_forecast(df, df_all):
    """
    Regresor sobre dias con sentimiento real.
    Target: precio siguiente (igual que model_regression.py original).
    Forecast: reconstruye precio via retorno predicho para evitar drift.
    """
    d = df.copy()
    d["target"] = d["price"].shift(-1)
    d = d.dropna().reset_index(drop=True)

    X_base = d[["price", "return"]]
    X_full = d[["price", "return", "sent_score"]]
    y      = d["target"]
    dates  = d["date"]

    Xb_tr, Xb_te, ytr, yte = train_test_split(X_base, y, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _,   _   = train_test_split(X_full, y, shuffle=False, test_size=0.2)
    dates_te = dates.iloc[len(Xb_tr):].reset_index(drop=True)

    rb = xgb.XGBRegressor(**REG_P); rb.fit(Xb_tr, ytr)
    rf = xgb.XGBRegressor(**REG_P); rf.fit(Xf_tr, ytr)

    ypb = rb.predict(Xb_te)
    ypf = rf.predict(Xf_te)

    def rm(yt, yh): return {k: round(float(v), 4) for k, v in {
        "mae":  mean_absolute_error(yt, yh),
        "rmse": float(np.sqrt(mean_squared_error(yt, yh))),
        "r2":   r2_score(yt, yh)}.items()}

    metrics = {"baseline": rm(yte, ypb), "full": rm(yte, ypf)}

    # Historia completa 2024+ para el grafico
    history = [{"date": str(dd.date()), "real": round(float(p), 2)}
               for dd, p in zip(df_all["date"], df_all["price"])]

    # Test set: predicciones vs real
    test_s = [{"date": str(dd.date()), "real": round(float(r), 2),
               "pred_base": round(float(pb2), 2), "pred_full": round(float(pf2), 2)}
              for dd, r, pb2, pf2 in zip(dates_te, yte.values, ypb, ypf)]

    # Forecast recursivo 7 dias desde hoy
    # Usa retornos para evitar drift de nivel de precio
    pc = float(df_all["price"].iloc[-1])
    pp = float(df_all["price"].iloc[-2])
    rc = (pc - pp) / pp if pp != 0 else 0.0
    sc = float(df["sent_score"].iloc[-1])  # ultimo sentimiento real

    fc = []
    for i in range(FORECAST_DAYS):
        nd    = date.today() + timedelta(days=i + 1)
        # Prediccion del precio siguiente
        pb2   = float(rb.predict([[pc, rc]])[0])
        pf2   = float(rf.predict([[pc, rc, sc]])[0])
        # Convertir a retorno y reconstruir para mantener ancla en precio actual
        ret_b = (pb2 - pc) / pc if pc != 0 else 0.0
        ret_f = (pf2 - pc) / pc if pc != 0 else 0.0
        pred_b = round(pc * (1 + ret_b), 2)
        pred_f = round(pc * (1 + ret_f), 2)
        fc.append({"date": str(nd), "pred_base": pred_b, "pred_full": pred_f})
        pp = pc; pc = pred_f
        rc = (pc - pp) / pp if pp != 0 else 0.0

    return metrics, history, test_s, fc


def main():
    print("Cargando datos (enfoque model.py original)...")
    df, df_all, sent_daily, reddit = load()

    print(f"Clasificador (n={len(df)} dias con sentimiento real)...")
    clf_res = classify(df)
    print(f"  baseline acc={clf_res['baseline']['accuracy']}  "
          f"full acc={clf_res['full']['accuracy']}")

    print("Regresor + Forecast 7d...")
    reg_res, hist, test_s, fc = regress_and_forecast(df, df_all)
    print(f"  baseline MAE={reg_res['baseline']['mae']}  "
          f"full MAE={reg_res['full']['mae']}")

    # Sentimiento diario para el grafico
    sw = sent_daily.merge(df_all[["date", "price"]], on="date", how="inner").sort_values("date")
    sent_out = [{"date": str(r["date"].date()),
                 "sentiment": round(float(r["sent_score"]), 4),
                 "price": round(float(r["price"]), 2)}
                for _, r in sw.iterrows()]

    # Top posts Reddit
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

    pkg = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "today_price":  hist[-1]["real"] if hist else None,
        "today_date":   hist[-1]["date"] if hist else None,
        "model_start_date": str(df["date"].min().date()),
        "model_end_date":   str(df["date"].max().date()),
        "model_days":       len(df),
        "total_price_days": len(df_all),
        "sentiment_coverage_pct": round(len(df) / len(df_all) * 100, 1),
        "classifier":      clf_res,
        "regression":      reg_res,
        "price_history":   hist,
        "price_test":      test_s,
        "forecast_7d":     fc,
        "sentiment_daily": sent_out,
        "reddit_posts":    posts_out,
    }

    out_path = Path("dashboard/public/data/dashboard_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pkg, fh, ensure_ascii=False, indent=2)

    print(f"\nExportado -> {out_path}")
    print(f"  Precio hoy:    ${pkg['today_price']} ({pkg['today_date']})")
    print(f"  Dias modelo:   {pkg['model_days']} (cobertura {pkg['sentiment_coverage_pct']}%)")
    print(f"  Clf baseline:  acc={clf_res['baseline']['accuracy']} | f1={clf_res['baseline']['f1']}")
    print(f"  Clf full:      acc={clf_res['full']['accuracy']} | f1={clf_res['full']['f1']}")
    print(f"  Reg baseline:  MAE={reg_res['baseline']['mae']} | R2={reg_res['baseline']['r2']}")
    print(f"  Reg full:      MAE={reg_res['full']['mae']} | R2={reg_res['full']['r2']}")
    print(f"  Forecast:      {[x['pred_full'] for x in fc]}")

if __name__ == "__main__":
    main()
