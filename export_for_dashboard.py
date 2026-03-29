"""
export_for_dashboard.py
Corre el pipeline completo (clasificador + regresor) y exporta
un único JSON que el dashboard Next.js consume.

Uso:
    python export_for_dashboard.py

Requiere que existan:
    data/solana_prices.csv
    data/reddit_sentiment.csv
    data/reddit_posts.csv
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int((~correct_a & correct_b).sum())
    c = int((correct_a & ~correct_b).sum())
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / n
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, df=1))
    except Exception:
        p_value = float(np.exp(-0.5 * chi2) * 1.2533141373155)
    return b, c, float(chi2), p_value


def xgb_clf_params():
    return dict(
        eval_metric="logloss", n_estimators=400, max_depth=4,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42,
    )


def xgb_reg_params():
    return dict(
        objective="reg:squarederror", n_estimators=400, max_depth=4,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── carga ────────────────────────────────────────────────────────────────
    prices    = pd.read_csv("data/solana_prices.csv",    parse_dates=["date"])
    sentiment = pd.read_csv("data/reddit_sentiment.csv", parse_dates=["date"])
    reddit    = pd.read_csv("data/reddit_posts.csv",     parse_dates=["date"])

    sent_daily = sentiment.groupby("date")["sent_score"].mean().reset_index()
    df = prices.merge(sent_daily, on="date", how="left")
    df["return"] = df["price"].pct_change()
    df = df.dropna().reset_index(drop=True)

    # ── clasificador ─────────────────────────────────────────────────────────
    df_cls = df.copy()
    df_cls["target"] = (df_cls["return"].shift(-1) > 0).astype(int)
    df_cls = df_cls.dropna().reset_index(drop=True)

    y_cls     = df_cls["target"].values
    X_base_c  = df_cls[["return"]]
    X_full_c  = df_cls[["return", "sent_score"]]

    Xb_tr, Xb_te, y_tr, y_te = train_test_split(X_base_c, y_cls, shuffle=False, test_size=0.2)
    Xf_tr, Xf_te, _,   yf_te = train_test_split(X_full_c, y_cls, shuffle=False, test_size=0.2)

    base_clf = xgb.XGBClassifier(**xgb_clf_params()); base_clf.fit(Xb_tr, y_tr)
    full_clf = xgb.XGBClassifier(**xgb_clf_params()); full_clf.fit(Xf_tr, y_tr)

    base_pred_c = base_clf.predict(Xb_te)
    full_pred_c = full_clf.predict(Xf_te)

    b, c, chi2, p = mcnemar_test(y_te, base_pred_c, full_pred_c)

    def cls_metrics(y_true, y_hat):
        return {
            "accuracy":  round(float(accuracy_score(y_true, y_hat)), 4),
            "precision": round(float(precision_score(y_true, y_hat, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_hat, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, y_hat, zero_division=0)), 4),
        }

    # ── regresor ─────────────────────────────────────────────────────────────
    df_reg = df.copy()
    df_reg["target"] = df_reg["price"].shift(-1)
    df_reg = df_reg.dropna().reset_index(drop=True)

    X_reg    = df_reg[["price", "return", "sent_score"]]
    y_reg    = df_reg["target"]
    dates_df = df_reg["date"]

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, shuffle=False, test_size=0.2)
    dates_te = dates_df.iloc[len(X_tr_r):].reset_index(drop=True)

    base_reg = xgb.XGBRegressor(**xgb_reg_params())
    full_reg = xgb.XGBRegressor(**xgb_reg_params())
    base_reg.fit(X_tr_r[["price", "return"]], y_tr_r)
    full_reg.fit(X_tr_r, y_tr_r)

    yp_base = base_reg.predict(X_te_r[["price", "return"]])
    yp_full = full_reg.predict(X_te_r)

    def reg_metrics(y_true, y_hat):
        return {
            "mae":  round(float(mean_absolute_error(y_true, y_hat)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_hat))), 4),
            "r2":   round(float(r2_score(y_true, y_hat)), 4),
        }

    # ── price predictions (serie test) ───────────────────────────────────────
    price_preds = [
        {
            "date":      str(d.date()),
            "real":      round(float(r), 2),
            "pred_base": round(float(pb), 2),
            "pred_full": round(float(pf), 2),
        }
        for d, r, pb, pf in zip(dates_te, y_te_r.values, yp_base, yp_full)
    ]

    # ── sentiment diario (todo el período) ───────────────────────────────────
    sent_with_price = sent_daily.merge(prices, on="date", how="inner").sort_values("date")
    sentiment_series = [
        {
            "date":      str(row["date"].date()),
            "sentiment": round(float(row["sent_score"]), 4),
            "price":     round(float(row["price"]), 2),
        }
        for _, row in sent_with_price.iterrows()
    ]

    # ── posts de Reddit (top 50 por score) ───────────────────────────────────
    sent_map = sentiment.groupby("id")["sent_score"].mean().to_dict()
    top_posts = reddit.sort_values("score", ascending=False).head(50)
    reddit_rows = []
    for _, row in top_posts.iterrows():
        sv = sent_map.get(row["id"])
        reddit_rows.append({
            "date":         str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "title":        str(row["title"])[:120],
            "score":        int(row["score"]),
            "num_comments": int(row["num_comments"]),
            "sent_score":   round(float(sv), 4) if sv is not None else None,
            "url":          str(row["url"]),
        })

    # ── ensamblado final ─────────────────────────────────────────────────────
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "classifier": {
            "baseline": cls_metrics(y_te, base_pred_c),
            "full":     cls_metrics(y_te, full_pred_c),
            "mcnemar":  {"b": b, "c": c, "chi2": round(chi2, 4), "p": round(p, 4)},
        },
        "regression": {
            "baseline": reg_metrics(y_te_r, yp_base),
            "full":     reg_metrics(y_te_r, yp_full),
        },
        "price_predictions": price_preds,
        "sentiment_daily":   sentiment_series,
        "reddit_posts":      reddit_rows,
    }

    out_path = Path("dashboard/public/data/dashboard_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅  dashboard_data.json exportado en {out_path}")
    print(f"    Clasificador  → baseline acc={output['classifier']['baseline']['accuracy']}"
          f"  full acc={output['classifier']['full']['accuracy']}")
    print(f"    Regresor      → baseline MAE={output['regression']['baseline']['mae']}"
          f"  full MAE={output['regression']['full']['mae']}")


if __name__ == "__main__":
    main()
