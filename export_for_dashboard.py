"""
export_for_dashboard.py  —  Pipeline con forecast 7 dias
"""
import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score)

CUTOFF = pd.Timestamp("2024-01-01")
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
    pr = pr[pr["date"] >= CUTOFF].copy()
    se = se[se["date"] >= CUTOFF].copy()
    re = re[re["date"] >= CUTOFF].copy()
    sd = se.groupby("date")["sent_score"].mean().reset_index()
    df = pr.merge(sd, on="date", how="left")
    df["sent_score"] = df["sent_score"].interpolate().ffill().bfill()
    df["return"] = df["price"].pct_change()
    return df.dropna().reset_index(drop=True), se, sd, re

def classify(df):
    d = df.copy(); d["tgt"] = (d["return"].shift(-1) > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    y = d["tgt"].values
    Xb = d[["return"]]; Xf = d[["return","sent_score"]]
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

def regress_and_forecast(df):
    d = df.copy(); d["tgt"] = d["price"].shift(-1)
    d = d.dropna().reset_index(drop=True)
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
    history = [{"date":str(dd.date()),"real":round(float(p),2)}
               for dd,p in zip(df["date"],df["price"])]
    test_s = [{"date":str(dd.date()),"real":round(float(r),2),
               "pred_base":round(float(pb2),2),"pred_full":round(float(pf2),2)}
              for dd,r,pb2,pf2 in zip(dte,yte.values,ypb,ypf)]
    # Forecast recursivo 7 dias
    pc = float(df["price"].iloc[-1]); pp = float(df["price"].iloc[-2])
    rc = (pc-pp)/pp if pp!=0 else 0.0
    sc = float(df["sent_score"].tail(7).mean())
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
    df, sentiment, sent_daily, reddit = load()
    print(f"  {len(df)} dias | {df['date'].min().date()} -> {df['date'].max().date()}")
    print("Clasificador..."); clf_res = classify(df)
    print("Regresor + Forecast..."); reg_res, hist, test_s, fc = regress_and_forecast(df)
    sw = sent_daily.merge(df[["date","price"]], on="date", how="inner").sort_values("date")
    sent_out = [{"date":str(r["date"].date()),"sentiment":round(float(r["sent_score"]),4),
                 "price":round(float(r["price"]),2)} for _,r in sw.iterrows()]
    smap = sentiment.groupby("id")["sent_score"].mean().to_dict()
    posts_out = []
    for _,row in reddit.sort_values("score",ascending=False).head(50).iterrows():
        sv = smap.get(row["id"])
        posts_out.append({"date":str(row["date"].date()) if hasattr(row["date"],"date") else str(row["date"]),
                          "title":str(row["title"])[:120],"score":int(row.get("score",0) or 0),
                          "num_comments":int(row.get("num_comments",0) or 0),
                          "sent_score":round(float(sv),4) if sv is not None else None,
                          "url":str(row.get("url",""))})
    pkg = {"last_updated":datetime.now(timezone.utc).isoformat(),
           "today_price":hist[-1]["real"] if hist else None,
           "today_date":hist[-1]["date"] if hist else None,
           "classifier":clf_res,"regression":reg_res,
           "price_history":hist,"price_test":test_s,
           "forecast_7d":fc,"sentiment_daily":sent_out,"reddit_posts":posts_out}
    out_path = Path("dashboard/public/data/dashboard_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pkg, fh, ensure_ascii=False, indent=2)
    print(f"\nExportado -> {out_path}")
    print(f"  Hoy: ${pkg['today_price']} ({pkg['today_date']})")
    print(f"  Forecast full: {[x['pred_full'] for x in fc]}")
    print(f"  Clf acc: {clf_res['baseline']['accuracy']} -> {clf_res['full']['accuracy']}")
    print(f"  Reg MAE: {reg_res['baseline']['mae']} -> {reg_res['full']['mae']}")

if __name__ == "__main__":
    main()
