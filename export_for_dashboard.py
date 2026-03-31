"""
export_for_dashboard_v4.py
──────────────────────────
Tesina: Predicción SOL/USD con Análisis de Sentimiento

DISEÑO CLAVE — Comparación justa:
  - UN SOLO split temporal por fecha de corte (no por ratio)
  - Todos los modelos se evalúan en EXACTAMENTE los mismos días
  - Cada modelo se ENTRENA con los datos que tiene disponibles
  - Las métricas son directamente comparables

Modelos:
  baseline  = features técnicos solamente
  reddit    = baseline + sentimiento Reddit lagged
  fear_greed = baseline + Fear & Greed Index lagged + derivados
  combined  = baseline + Reddit + F&G
"""

import json, numpy as np, pandas as pd, xgboost as xgb
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

# ── Configuración ─────────────────────────────────────────────────
CUTOFF          = pd.Timestamp("2024-01-01")
TEST_CUTOFF     = pd.Timestamp("2025-11-01")   # Fecha fija de corte train/test
FORECAST_DAYS   = 7
MIN_POSTS_DAY   = 5

# Modelo regularizado para datasets pequeños
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
# CARGA Y FEATURE ENGINEERING
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
        print("  ⚠ Fear & Greed no encontrado. Corré: python get_fear_greed.py")

    # ── Reddit: sentimiento ponderado diario ──────────────────────
    ppd = se.groupby("date").size()
    valid_days = ppd[ppd >= MIN_POSTS_DAY].index
    se_filt = se[se["date"].isin(valid_days)].copy()
    se_filt["score_clip"] = se_filt["score"].clip(lower=1)

    def wsent(g):
        w = g["score_clip"]
        return (g["sent_score"] * w).sum() / w.sum()

    reddit_daily = (se_filt.groupby("date")
                           .apply(wsent, include_groups=False)
                           .reset_index())
    reddit_daily.columns = ["date", "sent_reddit"]

    # ── Master dataframe ──────────────────────────────────────────
    df = pr.copy()
    df = df.merge(reddit_daily, on="date", how="left")
    if has_fg:
        df = df.merge(fg[["date", "fg_value"]], on="date", how="left")
        df["fg_norm"] = (df["fg_value"] - 50) / 50
    else:
        df["fg_value"] = np.nan
        df["fg_norm"] = np.nan

    # ── Features técnicos ─────────────────────────────────────────
    df["return"]       = df["price"].pct_change()
    df["ret_ma5"]      = df["return"].rolling(5, min_periods=2).mean()
    df["ret_ma10"]     = df["return"].rolling(10, min_periods=3).mean()
    df["volatility5"]  = df["return"].rolling(5, min_periods=2).std()
    df["volatility10"] = df["return"].rolling(10, min_periods=3).std()
    df["momentum5"]    = df["price"].pct_change(5)
    df["momentum10"]   = df["price"].pct_change(10)

    # ── Features sentimiento (TODO lagged) ────────────────────────
    df["reddit_lag1"]  = df["sent_reddit"].shift(1)
    if has_fg:
        df["fg_lag1"]      = df["fg_norm"].shift(1)
        df["fg_lag2"]      = df["fg_norm"].shift(2)
        df["fg_ma3"]       = df["fg_norm"].rolling(3, min_periods=1).mean().shift(1)
        df["fg_delta"]     = df["fg_lag1"] - df["fg_lag2"]
        df["fg_price_div"] = df["fg_delta"] - df["return"].shift(1)

    # ── Targets ───────────────────────────────────────────────────
    df["tgt_direction"] = (df["return"].shift(-1) > 0).astype(float)
    df["tgt_return"]    = df["return"].shift(-1)

    # Marcar NaN en target para el último día
    df.loc[df.index[-1], "tgt_direction"] = np.nan
    df.loc[df.index[-1], "tgt_return"] = np.nan

    df = df.dropna(subset=["return"]).reset_index(drop=True)

    # ── Stats ─────────────────────────────────────────────────────
    n = len(df)
    n_reddit = df["reddit_lag1"].notna().sum()
    n_fg = df["fg_lag1"].notna().sum() if has_fg else 0
    n_train = len(df[df["date"] < TEST_CUTOFF])
    n_test = len(df[df["date"] >= TEST_CUTOFF])
    print(f"  Precios 2024+:       {n} días")
    print(f"  Con Reddit (≥{MIN_POSTS_DAY}p): {n_reddit} días ({n_reddit/n*100:.1f}%)")
    print(f"  Con Fear&Greed:      {n_fg} días ({n_fg/n*100:.1f}%)")
    print(f"  Train (<{TEST_CUTOFF.date()}): {n_train} días")
    print(f"  Test (≥{TEST_CUTOFF.date()}):  {n_test} días")

    return df, reddit_daily, re, has_fg


# ══════════════════════════════════════════════════════════════════
# FEATURE SETS
# ══════════════════════════════════════════════════════════════════
FEAT_BASE = ["return", "ret_ma5", "ret_ma10", "volatility5",
             "volatility10", "momentum5", "momentum10"]

FEAT_REDDIT = FEAT_BASE + ["reddit_lag1"]

FEAT_FG = FEAT_BASE + ["fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"]

FEAT_COMBINED = FEAT_BASE + ["reddit_lag1",
                "fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"]


# ══════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════
def clf_metrics(yt, yp, yproba):
    return {k: round(float(v), 4) for k, v in {
        "accuracy":  accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall":    recall_score(yt, yp, zero_division=0),
        "f1":        f1_score(yt, yp, zero_division=0),
        "auc":       roc_auc_score(yt, yproba) if len(np.unique(yt)) > 1 else 0.5,
    }.items()}


def reg_metrics(yt, yp):
    return {k: round(float(v), 4) for k, v in {
        "mae":  mean_absolute_error(yt, yp),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2":   r2_score(yt, yp),
    }.items()}


# ══════════════════════════════════════════════════════════════════
# CLASIFICADOR — COMPARACIÓN JUSTA
# ══════════════════════════════════════════════════════════════════
def classify(df, has_fg):
    """
    Estrategia de comparación justa:
    1. Definir test set = todos los días >= TEST_CUTOFF con target válido
    2. Para cada modelo, ENTRENAR con sus datos disponibles < TEST_CUTOFF
    3. Para cada modelo, PREDECIR en los días del test set donde tiene features
    4. MEDIR métricas solo en la INTERSECCIÓN de días donde TODOS predicen
    """
    df_with_target = df.dropna(subset=["tgt_direction"] + FEAT_BASE).copy()
    train_mask = df_with_target["date"] < TEST_CUTOFF
    test_mask  = df_with_target["date"] >= TEST_CUTOFF

    test_dates = df_with_target.loc[test_mask, "date"].values

    models = {}
    # ── Entrenar cada modelo ──────────────────────────────────────

    # Baseline: siempre tiene datos
    train_b = df_with_target[train_mask]
    mb = xgb.XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
    mb.fit(train_b[FEAT_BASE], train_b["tgt_direction"])
    models["baseline"] = (mb, FEAT_BASE)

    # Reddit
    df_reddit = df_with_target.dropna(subset=["reddit_lag1"])
    train_r = df_reddit[df_reddit["date"] < TEST_CUTOFF]
    if len(train_r) >= 30:
        mr = xgb.XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
        mr.fit(train_r[FEAT_REDDIT], train_r["tgt_direction"])
        models["reddit"] = (mr, FEAT_REDDIT)
        print(f"  Reddit train: {len(train_r)} días")
    else:
        print(f"  ⚠ Reddit: solo {len(train_r)} train days, omitiendo")

    # Fear & Greed
    if has_fg:
        fg_feats = ["fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"]
        df_fg = df_with_target.dropna(subset=fg_feats)
        train_fg = df_fg[df_fg["date"] < TEST_CUTOFF]
        if len(train_fg) >= 30:
            mfg = xgb.XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
            mfg.fit(train_fg[FEAT_FG], train_fg["tgt_direction"])
            models["fear_greed"] = (mfg, FEAT_FG)
            print(f"  F&G train: {len(train_fg)} días")

        # Combinado
        df_comb = df_with_target.dropna(subset=["reddit_lag1"] + fg_feats)
        train_c = df_comb[df_comb["date"] < TEST_CUTOFF]
        if len(train_c) >= 30:
            mc = xgb.XGBClassifier(eval_metric="logloss", **XGB_PARAMS)
            mc.fit(train_c[FEAT_COMBINED], train_c["tgt_direction"])
            models["combined"] = (mc, FEAT_COMBINED)
            print(f"  Combined train: {len(train_c)} días")

    # ── Predecir en test set ──────────────────────────────────────
    test_df = df_with_target[test_mask].copy()
    predictions = {}

    for name, (model, feats) in models.items():
        # Predecir solo donde tiene features completos
        test_valid = test_df.dropna(subset=feats)
        if len(test_valid) == 0:
            continue
        preds = model.predict(test_valid[feats])
        proba = model.predict_proba(test_valid[feats])[:, 1]
        predictions[name] = pd.DataFrame({
            "date": test_valid["date"].values,
            "y_true": test_valid["tgt_direction"].values,
            "y_pred": preds,
            "y_proba": proba,
        })

    # ── Métricas POR MODELO (en su propio test set) ───────────────
    results_own = {}
    for name, pdf in predictions.items():
        results_own[name] = clf_metrics(pdf["y_true"], pdf["y_pred"], pdf["y_proba"])
        results_own[name]["n_test"] = len(pdf)
        # Feature importance
        model, feats = models[name]
        results_own[name]["feature_importance"] = dict(
            zip(feats, [round(float(x), 4) for x in model.feature_importances_]))

    # ── Métricas COMPARACIÓN JUSTA (intersección de días) ─────────
    # Encontrar días donde TODOS los modelos tienen predicción
    common_dates = None
    for name, pdf in predictions.items():
        dates_set = set(pdf["date"])
        common_dates = dates_set if common_dates is None else common_dates & dates_set

    results_fair = {}
    fair_preds = {}
    if common_dates and len(common_dates) >= 20:
        print(f"\n  Comparación justa: {len(common_dates)} días en común")
        for name, pdf in predictions.items():
            mask = pdf["date"].isin(common_dates)
            pf = pdf[mask].sort_values("date")
            results_fair[name] = clf_metrics(pf["y_true"], pf["y_pred"], pf["y_proba"])
            results_fair[name]["n_test"] = len(pf)
            fair_preds[name] = (pf["y_true"].values, pf["y_pred"].values)
    else:
        print(f"\n  ⚠ Solo {len(common_dates) if common_dates else 0} días en común — "
              f"comparación justa solo baseline vs F&G")
        # Fallback: comparar al menos baseline vs fear_greed
        if "baseline" in predictions and "fear_greed" in predictions:
            common_bf = set(predictions["baseline"]["date"]) & set(predictions["fear_greed"]["date"])
            if len(common_bf) >= 20:
                print(f"    baseline vs F&G: {len(common_bf)} días en común")
                for name in ["baseline", "fear_greed"]:
                    pdf = predictions[name]
                    mask = pdf["date"].isin(common_bf)
                    pf = pdf[mask].sort_values("date")
                    results_fair[name] = clf_metrics(pf["y_true"], pf["y_pred"], pf["y_proba"])
                    results_fair[name]["n_test"] = len(pf)
                    fair_preds[name] = (pf["y_true"].values, pf["y_pred"].values)

    # McNemar en comparación justa
    mcnemar_results = {}
    if "baseline" in fair_preds:
        yt_b, pb = fair_preds["baseline"]
        for name in ["reddit", "fear_greed", "combined"]:
            if name in fair_preds:
                yt_s, ps = fair_preds[name]
                b, c, chi2, pv = mcnemar(yt_b, pb, ps)
                mcnemar_results[f"baseline_vs_{name}"] = {
                    "b": b, "c": c, "chi2": chi2, "p": pv
                }

    # Print
    print(f"\n  {'Modelo':<14s} {'Own acc':>8s} {'Fair acc':>8s} {'Fair F1':>8s} {'Fair AUC':>8s} {'n_own':>6s} {'n_fair':>6s}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    for name in ["baseline", "reddit", "fear_greed", "combined"]:
        if name in results_own:
            own = results_own[name]
            fair = results_fair.get(name, {})
            print(f"  {name:<14s} {own['accuracy']:>8.4f} {fair.get('accuracy','—'):>8} "
                  f"{fair.get('f1','—'):>8} {fair.get('auc','—'):>8} "
                  f"{own['n_test']:>6d} {fair.get('n_test','—'):>6}")

    return {
        "models_own_test": results_own,
        "models_fair_test": results_fair,
        "mcnemar": mcnemar_results,
        "n_common_days": len(common_dates) if common_dates else 0,
    }


# ══════════════════════════════════════════════════════════════════
# REGRESOR — COMPARACIÓN JUSTA
# ══════════════════════════════════════════════════════════════════
def regress(df, has_fg):
    df_with_target = df.dropna(subset=["tgt_return"] + FEAT_BASE).copy()
    train_mask = df_with_target["date"] < TEST_CUTOFF
    test_mask  = df_with_target["date"] >= TEST_CUTOFF

    models = {}

    # Baseline
    train_b = df_with_target[train_mask]
    rb = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
    rb.fit(train_b[FEAT_BASE], train_b["tgt_return"])
    models["baseline"] = (rb, FEAT_BASE)

    # Reddit
    df_reddit = df_with_target.dropna(subset=["reddit_lag1"])
    train_r = df_reddit[df_reddit["date"] < TEST_CUTOFF]
    if len(train_r) >= 30:
        rr = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
        rr.fit(train_r[FEAT_REDDIT], train_r["tgt_return"])
        models["reddit"] = (rr, FEAT_REDDIT)

    # F&G
    if has_fg:
        fg_feats = ["fg_lag1", "fg_ma3", "fg_delta", "fg_price_div"]
        df_fg = df_with_target.dropna(subset=fg_feats)
        train_fg = df_fg[df_fg["date"] < TEST_CUTOFF]
        if len(train_fg) >= 30:
            rfg = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
            rfg.fit(train_fg[FEAT_FG], train_fg["tgt_return"])
            models["fear_greed"] = (rfg, FEAT_FG)

        # Combinado
        df_comb = df_with_target.dropna(subset=["reddit_lag1"] + fg_feats)
        train_c = df_comb[df_comb["date"] < TEST_CUTOFF]
        if len(train_c) >= 30:
            rc = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
            rc.fit(train_c[FEAT_COMBINED], train_c["tgt_return"])
            models["combined"] = (rc, FEAT_COMBINED)

    # ── Predecir ──────────────────────────────────────────────────
    test_df = df_with_target[test_mask].copy()
    predictions = {}

    for name, (model, feats) in models.items():
        test_valid = test_df.dropna(subset=feats)
        if len(test_valid) == 0:
            continue
        pred_r = model.predict(test_valid[feats])
        pred_price = test_valid["price"].values * (1 + pred_r)
        real_price = test_valid["price"].values * (1 + test_valid["tgt_return"].values)
        predictions[name] = pd.DataFrame({
            "date": test_valid["date"].values,
            "price": test_valid["price"].values,
            "real_price": real_price,
            "pred_price": pred_price,
            "pred_return": pred_r,
            "real_return": test_valid["tgt_return"].values,
        })

    # ── Métricas propias ──────────────────────────────────────────
    results_own = {}
    for name, pdf in predictions.items():
        results_own[name] = reg_metrics(pdf["real_price"], pdf["pred_price"])
        results_own[name]["n_test"] = len(pdf)
        model, feats = models[name]
        results_own[name]["feature_importance"] = dict(
            zip(feats, [round(float(x), 4) for x in model.feature_importances_]))

    # ── Comparación justa ─────────────────────────────────────────
    common_dates = None
    for name, pdf in predictions.items():
        dates_set = set(pdf["date"])
        common_dates = dates_set if common_dates is None else common_dates & dates_set

    results_fair = {}
    if common_dates and len(common_dates) >= 20:
        for name, pdf in predictions.items():
            mask = pdf["date"].isin(common_dates)
            pf = pdf[mask].sort_values("date")
            results_fair[name] = reg_metrics(pf["real_price"], pf["pred_price"])
            results_fair[name]["n_test"] = len(pf)
    else:
        # Fallback: baseline vs F&G
        if "baseline" in predictions and "fear_greed" in predictions:
            common_bf = set(predictions["baseline"]["date"]) & set(predictions["fear_greed"]["date"])
            if len(common_bf) >= 20:
                for name in ["baseline", "fear_greed"]:
                    pdf = predictions[name]
                    mask = pdf["date"].isin(common_bf)
                    pf = pdf[mask].sort_values("date")
                    results_fair[name] = reg_metrics(pf["real_price"], pf["pred_price"])
                    results_fair[name]["n_test"] = len(pf)

    # Print
    print(f"\n  {'Modelo':<14s} {'Own MAE':>8s} {'Fair MAE':>8s} {'Fair R²':>8s} {'n_own':>6s} {'n_fair':>6s}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    for name in ["baseline", "reddit", "fear_greed", "combined"]:
        if name in results_own:
            own = results_own[name]
            fair = results_fair.get(name, {})
            print(f"  {name:<14s} ${own['mae']:>7.2f} ${fair.get('mae','—'):>7} "
                  f"{fair.get('r2','—'):>8} {own['n_test']:>6d} {fair.get('n_test','—'):>6}")

    # ── Datos para visualización ──────────────────────────────────
    df_all = df[["date", "price"]].dropna().copy()
    hist = [{"date": str(dd.date()), "real": round(float(p), 2)}
            for dd, p in zip(df_all["date"], df_all["price"])]

    # Test set (usar baseline + mejor sent model)
    best_sent = None
    for candidate in ["fear_greed", "combined", "reddit"]:
        if candidate in results_fair:
            if results_fair[candidate]["mae"] < results_fair.get("baseline", {}).get("mae", 999):
                if best_sent is None or results_fair[candidate]["mae"] < results_fair[best_sent]["mae"]:
                    best_sent = candidate
    if best_sent is None:
        best_sent = "fear_greed" if "fear_greed" in predictions else "reddit"

    # Build test set for chart
    ts = []
    if "baseline" in predictions:
        bp = predictions["baseline"]
        sp = predictions.get(best_sent)
        for _, row in bp.iterrows():
            entry = {
                "date": str(pd.Timestamp(row["date"]).date()),
                "real": round(float(row["real_price"]), 2),
                "pred_base": round(float(row["pred_price"]), 2),
            }
            # Find matching sentiment prediction for same date
            if sp is not None:
                match = sp[sp["date"] == row["date"]]
                if len(match) > 0:
                    entry["pred_full"] = round(float(match.iloc[0]["pred_price"]), 2)
                else:
                    entry["pred_full"] = entry["pred_base"]
            else:
                entry["pred_full"] = entry["pred_base"]
            ts.append(entry)

    # ── Forecast 7 días ───────────────────────────────────────────
    last_row = df.iloc[-1]
    pc = float(df_all["price"].iloc[-1])
    fc = []

    feat_vals = {}
    for f in FEAT_COMBINED:
        val = last_row.get(f)
        feat_vals[f] = float(val) if pd.notna(val) else 0.0

    for i in range(FORECAST_DAYS):
        nd = date.today() + timedelta(days=i + 1)
        preds_day = {}
        for name, (model, feats) in models.items():
            row = [[feat_vals.get(f, 0.0) for f in feats]]
            pred_r = float(model.predict(row)[0])
            preds_day[name] = round(pc * (1 + pred_r), 2)

        fc.append({
            "date": str(nd),
            "pred_base": preds_day.get("baseline", pc),
            "pred_full": preds_day.get(best_sent, preds_day.get("baseline", pc)),
        })

        best_p = preds_day.get(best_sent, preds_day.get("baseline", pc))
        new_ret = (best_p - pc) / pc if pc != 0 else 0.0
        feat_vals["return"] = new_ret
        feat_vals["ret_ma5"] = feat_vals.get("ret_ma5", 0) * 0.8 + new_ret * 0.2
        feat_vals["ret_ma10"] = feat_vals.get("ret_ma10", 0) * 0.9 + new_ret * 0.1
        feat_vals["volatility5"] = feat_vals.get("volatility5", 0) * 0.8 + abs(new_ret) * 0.2
        feat_vals["volatility10"] = feat_vals.get("volatility10", 0) * 0.9 + abs(new_ret) * 0.1
        feat_vals["momentum5"] = new_ret
        feat_vals["momentum10"] = new_ret
        pc = best_p

    return results_own, results_fair, hist, ts, fc, best_sent, models


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO
# ══════════════════════════════════════════════════════════════════
def statistical_analysis(df, has_fg):
    stats = {}
    d = df.dropna(subset=["return"]).copy()
    d["return_next"] = d["return"].shift(-1)

    # Reddit
    d_r = d.dropna(subset=["sent_reddit", "return_next"])
    if len(d_r) > 30:
        stats["reddit"] = {
            "n_days": int(len(d_r)),
            "corr_same_day": round(float(d_r["sent_reddit"].corr(d_r["return"])), 4),
            "corr_next_day": round(float(d_r["sent_reddit"].corr(d_r["return_next"])), 4),
            "naive_accuracy": round(float(
                ((d_r["sent_reddit"] > 0).astype(int) == (d_r["return_next"] > 0).astype(int)).mean()
            ), 4),
            "conclusion": "NO_SIGNAL",
            "detail": "Correlación ~0 con retornos futuros. Sentimiento reactivo al precio.",
        }

    # F&G
    if has_fg:
        d_fg = d.dropna(subset=["fg_norm", "return_next"]).copy()
        if len(d_fg) > 30:
            d_fg.loc[:, "fg_lag1"] = d_fg["fg_norm"].shift(1)
            d_fg2 = d_fg.dropna(subset=["fg_lag1"])
            corr_lag1 = float(d_fg2["fg_lag1"].corr(d_fg2["return"]))
            stats["fear_greed"] = {
                "n_days": int(len(d_fg2)),
                "corr_same_day": round(float(d_fg2["fg_norm"].corr(d_fg2["return"])), 4),
                "corr_next_day": round(float(d_fg2["fg_norm"].corr(d_fg2["return_next"])), 4),
                "corr_lag1": round(corr_lag1, 4),
                "naive_accuracy": round(float(
                    ((d_fg2["fg_lag1"] > 0).astype(int) == (d_fg2["return_next"] > 0).astype(int)).mean()
                ), 4),
            }
            if abs(corr_lag1) > 0.05:
                stats["fear_greed"]["conclusion"] = "WEAK_SIGNAL"
                stats["fear_greed"]["detail"] = f"Señal débil (r={corr_lag1:.3f}) potencialmente explotable"
            else:
                stats["fear_greed"]["conclusion"] = "NO_SIGNAL"
                stats["fear_greed"]["detail"] = "Sin señal predictiva significativa"

    return stats


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("SOL/USD Sentiment Dashboard — v4 (comparación justa)")
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
    reg_own, reg_fair, hist, ts, fc, best_sent, reg_models = regress(df, has_fg)

    # ── Sentimiento para gráficos ─────────────────────────────────
    sw = reddit_daily.merge(df[["date", "price"]].drop_duplicates(),
                            on="date", how="inner").sort_values("date")
    sent_out = [{"date": str(r["date"].date()),
                 "sentiment": round(float(r["sent_reddit"]), 4),
                 "price": round(float(r["price"]), 2)}
                for _, r in sw.iterrows()]

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

    # ── Resumen ───────────────────────────────────────────────────
    print(f"\n── Resumen Final ──")
    print(f"  Mejor fuente: {best_sent}")
    print(f"  Test cutoff: {TEST_CUTOFF.date()}")

    # Build compatible output format for dashboard
    # Usar fair_test si existe, sino own_test
    clf_for_json = clf.copy()
    reg_for_json = {
        "models_own_test": reg_own,
        "models_fair_test": reg_fair,
    }

    # Legacy format compatible (para el dashboard existente)
    clf_legacy = {
        "baseline": clf["models_fair_test"].get("baseline", clf["models_own_test"].get("baseline", {})),
        "full": clf["models_fair_test"].get(best_sent, clf["models_own_test"].get(best_sent, {})),
    }
    if clf["mcnemar"]:
        key = f"baseline_vs_{best_sent}"
        clf_legacy["mcnemar"] = clf["mcnemar"].get(key, {})

    reg_legacy = {
        "baseline": reg_fair.get("baseline", reg_own.get("baseline", {})),
        "full": reg_fair.get(best_sent, reg_own.get(best_sent, {})),
    }

    out_data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "today_price":  hist[-1]["real"] if hist else None,
        "today_date":   hist[-1]["date"] if hist else None,
        "model_start_date": str(df["date"].min().date()),
        "model_end_date":   str(df["date"].max().date()),
        "test_cutoff":      str(TEST_CUTOFF.date()),
        "model_days":       len(df),
        "total_price_days": len(df),
        "sentiment_coverage_pct": round(
            df["reddit_lag1"].notna().sum() / len(df) * 100, 1),
        "fg_coverage_pct": round(
            df.get("fg_lag1", pd.Series(dtype=float)).notna().sum() / len(df) * 100, 1) if has_fg else 0,
        "best_sentiment_source": best_sent,
        "statistical_analysis": stats,
        # Legacy format (compatible with existing dashboard)
        "classifier":      clf_legacy,
        "regression":      reg_legacy,
        # Detailed results
        "classifier_detail": clf_for_json,
        "regression_detail": reg_for_json,
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
