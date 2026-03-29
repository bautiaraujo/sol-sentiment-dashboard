import requests
import pandas as pd
from datetime import date, datetime, timezone
from pathlib import Path

OUTPUT = Path("data/solana_prices.csv")
START_DATE = date(2024, 1, 1)   # base histórica desde 2024


def fetch_range(from_ts: int, to_ts: int) -> pd.DataFrame:
    """Descarga precios diarios usando el endpoint /market_chart/range."""
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart/range"
    params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True).dt.date
    daily = prices.groupby("date")["price"].mean().reset_index()
    return daily


def get_solana_prices(start: date = START_DATE) -> pd.DataFrame:
    """
    Descarga precios diarios de SOL/USD desde 'start' hasta hoy.
    Si ya existe el CSV, solo descarga los días faltantes y los agrega.
    """
    today = date.today()

    # Si el CSV existe, cargarlo y calcular desde dónde continuar
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT, parse_dates=["date"])
        existing["date"] = existing["date"].dt.date
        last_date = existing["date"].max()
        fetch_from = last_date  # solapamos el último día por si estaba incompleto
    else:
        existing = pd.DataFrame(columns=["date", "price"])
        fetch_from = start

    if fetch_from >= today:
        print(f"Datos ya actualizados hasta {fetch_from}. Sin cambios.")
        return existing

    from_ts = int(datetime(fetch_from.year, fetch_from.month, fetch_from.day,
                           tzinfo=timezone.utc).timestamp())
    to_ts   = int(datetime(today.year, today.month, today.day,
                           23, 59, 59, tzinfo=timezone.utc).timestamp())

    print(f"Descargando precios SOL/USD del {fetch_from} al {today}...")
    new_data = fetch_range(from_ts, to_ts)

    # Combinar y deduplicar
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["date"] = combined["date"].astype(str)

    return combined


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = get_solana_prices()
    df.to_csv(OUTPUT, index=False)
    print(f"OK: {len(df)} dias de precios -> {OUTPUT}")
    print(f"   Rango: {df['date'].min()} → {df['date'].max()}")
