import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score)

CUTOFF        = pd.Timestamp("2024-01-01")
FORECAST_DAYS = 7
SENT_WINDOW   = 7    # rolling window para suavizar sentimiento
MODEL_MIN_DAYS = 120  # minimo de dias para entrenar el modelo

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
    pr = pr[pr["date"] >= CUTOFF].copy().sort_values("date").reset_index(drop=True)
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()

    # Sentimiento diario real
    sd_real = se.groupby("date")["sent_score"].mean().reset_index()
    sd_real.columns = ["date", "sent_raw"]

    df = pr.merge(sd_real, on="date", how="left")
    df["has_real_sent"] = df["sent_raw"].notna()
    df["return"] = df["price"].pct_change()
    df["sent_score"] = df["sent_raw"].rolling(window=SENT_WINDOW, min_periods=1).mean()
    df["sent_score"] = df["sent_score"].fillna(0.0)
    df = df.dropna(subset=["price", "return"]).reset_index(drop=True)
    return df, se, sd_real, re


def get_model_df(df):
    """
    Determina el periodo de entrenamiento:
    - Preferencia: el rango donde hay sentimiento real de Reddit
    - Garantia: siempre al menos MODEL_MIN_DAYS dias de datos
    Esto asegura que el modelo tenga suficiente historia de precios
    sin importar cuan corta sea la cobertura de Reddit.
    """
    real_dates = df[df["has_real_sent"]]["date"]

    if real_dates.empty:
        # Sin sentimiento real: usar ultimos MODEL_MIN_DAYS dias
        start = df["date"].max() - pd.Timedelta(days=MODEL_MIN_DAYS)
        print(f"  WARN: sin sentimiento real. Usando ultimos {MODEL_MIN_DAYS} dias.")
        df_model = df[df["date"] >= start].copy().reset_index(drop=True)
        return df_model, 0.0

    first_real = real_dates.min()
    last_real  = real_dates.max()
    reddit_days = (last_real - first_real).days + 1

    # Si el periodo con Reddit es muy corto (< MODEL_MIN_DAYS),
    # extender hacia atras para dar mas contexto al modelo
    if reddit_days < MODEL_MIN_DAYS:
        extended_start = last_real - pd.Timedelta(days=MODEL_MIN_DAYS - 1)
        model_start = min(first_real, extended_start)
        print(f"  Reddit cubre solo {reddit_days} dias -> extendiendo a {MODEL_MIN_DAYS} dias")
    else:
        model_start = first_real

    df_model = df[df["date"] >= model_start].copy().reset_index(drop=True)
    coverage  = round(df_model["has_real_sent"].mean() * 100, 1)

    print(f"  Periodo Reddit: {first_real.date()} -> {last_real.date()} ({reddit_days} dias)")
    print(f"  Periodo modelo: {model_start.date()} -> {df_model['date'].max().date()} ({len(df_model)} dias)")
    print(f"  Cobertura sentimiento real: {coverage}%")
    return df_model, coverage


def classify(df_model):
    d = df_model.copy()
    d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna(subset=["tgt"]).reset_index(drop=True)
    y = d["tgt"].values
    Xb = d[["return"]]
    Xf = d[["return", "sent_score"]]
    Xb_tr,Xb_te,y_tr,y_te = train_test_split(Xb,y,shuffle=False,test_size=0.2)
    Xf_tr,Xf_te,_,_       = train_test_split(Xf,y,shuffle=False,test_size=0.2)
    mb = xgb.XGBClassifier(**CLF_P); mb.fit(Xb_tr, y_tr)
    mf = xgb.XGBClassifier(**CLF_P); mf.fit(Xf_tr, y_tr)
    pb = mb.predict(Xb_te); pf = mf.predict(Xf_te)
    def m(yt,yh): return {k:round(float(v),4) for k,v in {
        "accuracy":accuracy_score(yt,yh),"precision":precision_score(yt,yh,zero_division=0),
        "recall":recall_score(yt,yh,zero_division=0),"f1":f1_score(yt,yh,zero_division=0)}.items()}
    b,c,chi2,pv = mcnemar(y_te,pb,pf)
    return {"baseline":m(y_te,pb),"full":m(y_te,pf),
            "mcnemar":{"b":b,"c":c,"chi2":chi2,"p":pv}}


def regress_and_forecast(df_all, df_model):
    d = df_model.copy(); d["tgt"] = d["price"].shift(-1)
    d = d.dropna(subset=["tgt"]).reset_index(drop=True)
    X = d[["price","return","sent_score"]]; y = d["tgt"]; dates = d["date"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,shuffle=False,test_size=0.2)
    dte = dates.iloc[len(Xtr):].reset_index(drop=True)
    rb = xgb.XGBRegressor(**REG_P); rb.fit(Xtr[["price","return"]],ytr)
    rf = xgb.XGBRegressor(**REG_P); rf.fit(Xtr,ytr)
    ypb = rb.predict(Xte[["price","return"]]); ypf = rf.predict(Xte)
    def rm(yt,yh): return {k:round(float(v),4) for k,v in {
        "mae":mean_absolute_error(yt,yh),
        "rmse":float(np.sqrt(mean_squared_error(yt,yh))),
        "r2":r2_score(yt,yh)}.items()}
    metrics = {"baseline":rm(yte,ypb),"full":rm(yte,ypf)}

    # Historia completa 2024+ para visualizacion
    history = [{"date":str(dd.date()),"real":round(float(p),2)}
               for dd,p in zip(df_all["date"],df_all["price"])]

    # Test set con predicciones (ultimo 20% del periodo del modelo)
    test_s = [{"date":str(dd.date()),"real":round(float(r),2),
               "pred_base":round(float(pb2),2),"pred_full":round(float(pf2),2)}
              for dd,r,pb2,pf2 in zip(dte,yte.values,ypb,ypf)]

    # Forecast recursivo 7 dias
    pc = float(df_all["price"].iloc[-1]); pp = float(df_all["price"].iloc[-2])
    rc = (pc-pp)/pp if pp!=0 else 0.0
    sc = float(df_all["sent_score"].tail(SENT_WINDOW).mean())
    fc = []
    for i in range(FORECAST_DAYS):
        nd = date.today() + timedelta(days=i+1)
        prd_b = float(rb.predict([[pc,rc]])[0])
        prd_f = float(rf.predict([[pc,rc,sc]])[0])
        fc.append({"date":str(nd),"pred_base":round(prd_b,2),"pred_full":round(prd_f,2)})
        pp=pc; pc=prd_f; rc=(pc-pp)/pp if pp!=0 else 0.0
    return metrics, history, test_s, fc


def main():
    print("Cargando datos 2024+...")
    df_all, sentiment, sent_daily, reddit = load()
    n_real = int(df_all["has_real_sent"].sum())
    total  = len(df_all)
    print(f"  Precios: {total} dias | Dias con Reddit real: {n_real} ({round(n_real/total*100,1)}%)")

    df_model, coverage = get_model_df(df_all)

    print(f"Entrenando clasificador ({len(df_model)} dias)...")
    clf_res = classify(df_model)

    print("Entrenando regresor + forecast 7d...")
    reg_res, hist, test_s, fc = regress_and_forecast(df_all, df_model)

    # Sentimiento diario para el grafico
    sw = sent_daily.merge(df_all[["date","price"]], on="date", how="inner").sort_values("date")
    sent_out = [{"date":str(r["date"].date()),
                 "sentiment":round(float(r["sent_raw"]),4),
                 "price":round(float(r["price"]),2)}
                for _,r in sw.iterrows()]

    # Top posts Reddit
    smap = sentiment.groupby("id")["sent_score"].mean().to_dict()
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
        "model_days":       len(df_model),
        "sentiment_coverage_pct": coverage,
        "classifier":  clf_res,
        "regression":  reg_res,
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
    print(f"  Hoy:     ${pkg['today_price']} ({pkg['today_date']})")
    print(f"  Modelo:  {pkg['model_start_date']} -> {pkg['model_end_date']} ({pkg['model_days']} dias)")
    print(f"  Cobertura sentimiento: {coverage}%")
    print(f"  Clf: baseline={clf_res['baseline']['accuracy']} | full={clf_res['full']['accuracy']}")
    print(f"  Reg: baseline MAE={reg_res['baseline']['mae']} | full MAE={reg_res['full']['mae']}")
    print(f"  Forecast 7d: {[x['pred_full'] for x in fc]}")

if __name__ == "__main__":
    main()
