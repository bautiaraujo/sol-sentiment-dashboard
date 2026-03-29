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
    pr = pd.read_csv("data/solana_prices.csv",    parse_dates=["date"])
    se = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    re = pd.read_csv("data/reddit_posts.csv",     parse_dates=["date"])

    pr = pr[pr["date"] >= CUTOFF].sort_values("date").reset_index(drop=True)
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()

    # Sentimiento diario: promedio real de posts ese dia
    sd = se.groupby("date")["sent_score"].mean().reset_index()
    sd.columns = ["date", "sent_score"]

    # Merge: precio completo (2024+) para visualizacion
    df_all = pr.copy()
    df_all["return"] = df_all["price"].pct_change()
    df_all = df_all.dropna(subset=["return"]).reset_index(drop=True)

    # Dataset de entrenamiento: SOLO dias con sentimiento REAL
    # Sin interpolacion, sin relleno — solo donde Reddit tiene datos
    df_model = df_all.merge(sd, on="date", how="inner")
    df_model = df_model.sort_values("date").reset_index(drop=True)

    return df_all, df_model, sd, re


def classify(df_model):
    d = df_model.copy()
    d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna(subset=["tgt"]).reset_index(drop=True)
    y  = d["tgt"].values
    Xb = d[["return"]]
    Xf = d[["return", "sent_score"]]
    Xb_tr,Xb_te,y_tr,y_te = train_test_split(Xb,y,shuffle=False,test_size=0.2)
    Xf_tr,Xf_te,_,_       = train_test_split(Xf,y,shuffle=False,test_size=0.2)
    mb = xgb.XGBClassifier(**CLF_P); mb.fit(Xb_tr, y_tr)
    mf = xgb.XGBClassifier(**CLF_P); mf.fit(Xf_tr, y_tr)
    pb = mb.predict(Xb_te); pf = mf.predict(Xf_te)
    def m(yt,yh): return {k:round(float(v),4) for k,v in {
        "accuracy":accuracy_score(yt,yh),
        "precision":precision_score(yt,yh,zero_division=0),
        "recall":recall_score(yt,yh,zero_division=0),
        "f1":f1_score(yt,yh,zero_division=0)}.items()}
    b,c,chi2,pv = mcnemar(y_te,pb,pf)
    return {"baseline":m(y_te,pb),"full":m(y_te,pf),
            "mcnemar":{"b":b,"c":c,"chi2":chi2,"p":pv}}


def regress_and_forecast(df_all, df_model):
    """
    Entrena sobre retornos (no precio absoluto) solo en dias con sentimiento real.
    Reconstruye precio: p_t+1 = p_t * (1 + ret_predicho).
    Forecast usa el ultimo sentimiento real conocido.
    """
    d = df_model.copy()
    d["tgt_ret"] = d["return"].shift(-1)
    d = d.dropna(subset=["tgt_ret"]).reset_index(drop=True)

    y      = d["tgt_ret"]
    dates  = d["date"]
    prices = d["price"]

    Xb_tr,Xb_te,ytr,yte = train_test_split(d[["return"]],y,shuffle=False,test_size=0.2)
    Xf_tr,Xf_te,_,_     = train_test_split(d[["return","sent_score"]],y,shuffle=False,test_size=0.2)
    prices_te = prices.iloc[len(Xb_tr):].reset_index(drop=True)
    dates_te  = dates.iloc[len(Xb_tr):].reset_index(drop=True)

    rb = xgb.XGBRegressor(**REG_P); rb.fit(Xb_tr, ytr)
    rf = xgb.XGBRegressor(**REG_P); rf.fit(Xf_tr, ytr)

    ret_b = rb.predict(Xb_te); ret_f = rf.predict(Xf_te)
    ypb = prices_te.values * (1 + ret_b)
    ypf = prices_te.values * (1 + ret_f)
    real_next = prices_te.values * (1 + yte.values)

    def rm(yt,yh): return {k:round(float(v),4) for k,v in {
        "mae":mean_absolute_error(yt,yh),
        "rmse":float(np.sqrt(mean_squared_error(yt,yh))),
        "r2":r2_score(yt,yh)}.items()}
    metrics = {"baseline":rm(real_next,ypb),"full":rm(real_next,ypf)}

    # Historia completa 2024+ para el grafico
    history = [{"date":str(dd.date()),"real":round(float(p),2)}
               for dd,p in zip(df_all["date"],df_all["price"])]

    # Test set: precio reconstruido vs real
    test_s = [{"date":str(dd.date()),
               "real":round(float(r),2),
               "pred_base":round(float(pb2),2),
               "pred_full":round(float(pf2),2)}
              for dd,r,pb2,pf2 in zip(dates_te, real_next, ypb, ypf)]

    # Forecast 7 dias desde hoy usando precio actual y ultimo sentimiento real
    pc = float(df_all["price"].iloc[-1])
    pp = float(df_all["price"].iloc[-2])
    rc = (pc - pp) / pp if pp != 0 else 0.0
    # Ultimo sentimiento real conocido (del ultimo dia con datos Reddit)
    sc = float(df_model["sent_score"].iloc[-1])

    fc = []
    for i in range(FORECAST_DAYS):
        nd    = date.today() + timedelta(days=i + 1)
        rb_r  = float(rb.predict([[rc]])[0])
        rf_r  = float(rf.predict([[rc, sc]])[0])
        pb2   = round(pc * (1 + rb_r), 2)
        pf2   = round(pc * (1 + rf_r), 2)
        fc.append({"date":str(nd),"pred_base":pb2,"pred_full":pf2})
        pp = pc; pc = pf2
        rc = (pc - pp) / pp if pp != 0 else 0.0
        # Sentimiento se mantiene constante (no hay datos futuros de Reddit)

    return metrics, history, test_s, fc


def main():
    print("Cargando datos...")
    df_all, df_model, sent_daily, reddit = load()

    n_all   = len(df_all)
    n_model = len(df_model)
    print(f"  Precios 2024+:         {n_all} dias ({df_all['date'].min().date()} -> {df_all['date'].max().date()})")
    print(f"  Dias con Reddit real:  {n_model} dias ({df_model['date'].min().date()} -> {df_model['date'].max().date()})")
    print(f"  Cobertura sentimiento: {round(n_model/n_all*100,1)}% del total de precios")

    print(f"Clasificador (entrenado en {n_model} dias con sentimiento real)...")
    clf_res = classify(df_model)

    print("Regresor de retornos + Forecast 7d...")
    reg_res, hist, test_s, fc = regress_and_forecast(df_all, df_model)

    # Sentimiento diario para el grafico (solo dias reales)
    sw = sent_daily.merge(df_all[["date","price"]], on="date", how="inner").sort_values("date")
    sent_out = [{"date":str(r["date"].date()),
                 "sentiment":round(float(r["sent_score"]),4),
                 "price":round(float(r["price"]),2)}
                for _,r in sw.iterrows()]

    # Top posts Reddit
    smap = pd.read_csv("data/reddit_sentiment.csv").groupby("id")["sent_score"].mean().to_dict()
    posts_out = []
    for _,row in reddit.sort_values("score",ascending=False).head(50).iterrows():
        sv = smap.get(row["id"])
        posts_out.append({
            "date":str(row["date"].date()) if hasattr(row["date"],"date") else str(row["date"]),
            "title":str(row["title"])[:120],
            "score":int(row.get("score",0) or 0),
            "num_comments":int(row.get("num_comments",0) or 0),
            "sent_score":round(float(sv),4) if sv is not None else None,
            "url":str(row.get("url",""))})

    pkg = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "today_price":  hist[-1]["real"] if hist else None,
        "today_date":   hist[-1]["date"] if hist else None,
        "model_start_date": str(df_model["date"].min().date()),
        "model_end_date":   str(df_model["date"].max().date()),
        "model_days":       n_model,
        "total_price_days": n_all,
        "sentiment_coverage_pct": round(n_model/n_all*100,1),
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
    print(f"  Precio hoy:  ${pkg['today_price']} ({pkg['today_date']})")
    print(f"  Modelo:      {pkg['model_start_date']} -> {pkg['model_end_date']} ({n_model} dias)")
    print(f"  Clf:  baseline acc={clf_res['baseline']['accuracy']}  full={clf_res['full']['accuracy']}")
    print(f"  Reg:  baseline MAE=${reg_res['baseline']['mae']}  full MAE=${reg_res['full']['mae']}")
    print(f"  Forecast 7d: {[x['pred_full'] for x in fc]}")

if __name__ == "__main__":
    main()
