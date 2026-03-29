import requests
import pandas as pd
from datetime import date, datetime
from pathlib import Path

OUTPUT = Path("data/solana_prices.csv")
START_DATE = date(2024, 1, 1)


def fetch_days(days: int) -> pd.DataFrame:
    """Descarga precios diarios usando el endpoint gratuito de CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    daily = df.groupby("date")["price"].mean().reset_index()
    return daily


def get_solana_prices(start: date = START_DATE) -> pd.DataFrame:
    """
    Descarga precios desde 'start' hasta hoy usando la API gratuita.
    Si ya existe el CSV, solo descarga los ultimos 30 dias y los combina.
    """
    today = date.today()
    days_total = (today - start).days + 1

    if OUTPUT.exists():
        # Actualizar: solo bajar los ultimos 30 dias y combinar
        print(f"CSV existente encontrado. Actualizando ultimos 30 dias...")
        new_data = fetch_days(30)
        existing = pd.read_csv(OUTPUT, parse_dates=["date"])
        existing["date"] = existing["date"].dt.date
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        # Primera vez: bajar todo desde START_DATE
        print(f"Descargando {days_total} dias de precios SOL/USD (2024 → hoy)...")
        new_data = fetch_days(days_total)
        combined = new_data

    combined = combined.drop_duplicates(subset="date", keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    # Filtrar solo desde START_DATE
    combined = combined[combined["date"] >= START_DATE].copy()
    combined["date"] = combined["date"].astype(str)
    return combined


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = get_solana_prices()
    df.to_csv(OUTPUT, index=False)
    print(f"OK: {len(df)} dias de precios -> {OUTPUT}")
    print(f"   Rango: {df['date'].min()} -> {df['date'].max()}")
