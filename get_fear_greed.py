"""
get_fear_greed.py
─────────────────
Descarga el Crypto Fear & Greed Index histórico desde Alternative.me.
API 100% gratuita, sin API key, datos diarios desde 2018.

Output: data/fear_greed.csv con columnas [date, fg_value, fg_class]
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

OUT = Path("data/fear_greed.csv")

def fetch_fear_greed():
    """Fetch all historical F&G data from Alternative.me."""
    url = "https://api.alternative.me/fng/"
    params = {"limit": 0, "format": "json"}  # limit=0 → all data
    
    print("Descargando Fear & Greed Index desde Alternative.me...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    
    rows = []
    for d in data:
        ts = int(d["timestamp"])
        dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        rows.append({
            "date": dt,
            "fg_value": int(d["value"]),
            "fg_class": d["value_classification"],
        })
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    
    print(f"  Rango: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total días: {len(df)}")
    print(f"  Distribución:")
    print(df["fg_class"].value_counts().to_string(header=False))
    
    # Guardar
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nGuardado → {OUT}")
    return df


def update_incremental():
    """Si ya existe el CSV, solo agrega días nuevos."""
    if OUT.exists():
        existing = pd.read_csv(OUT, parse_dates=["date"])
        last_date = existing["date"].max()
        print(f"Datos existentes hasta {last_date.date()}, actualizando...")
        
        new = fetch_fear_greed()
        combined = pd.concat([existing, new]).drop_duplicates(subset="date", keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_csv(OUT, index=False)
        added = len(combined) - len(existing)
        print(f"  Agregados {added} días nuevos. Total: {len(combined)}")
        return combined
    else:
        return fetch_fear_greed()


if __name__ == "__main__":
    update_incremental()
