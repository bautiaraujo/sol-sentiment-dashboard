import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

OUTPUT     = Path("data/solana_prices.csv")
START_DATE = date(2024, 1, 1)
TICKER     = "SOL-USD"


def get_solana_prices(start: date = START_DATE) -> pd.DataFrame:
    """
    Descarga precios diarios de SOL/USD desde Yahoo Finance (gratis, sin API key).
    Nota: yfinance tiene end exclusivo, se le suma 1 dia para incluir hoy.
    Si el precio de hoy aun no esta disponible (mercado abierto), usa el de ayer.
    """
    today    = date.today()
    end_date = today + timedelta(days=1)   # end es exclusivo en yfinance

    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT, parse_dates=["date"])
        existing["date"] = existing["date"].dt.date
        last_date = existing["date"].max()
        if last_date >= today:
            print(f"Datos ya actualizados hasta {last_date}.")
            existing["date"] = existing["date"].astype(str)
            return existing
        fetch_from = last_date  # solapamos 1 dia por seguridad
    else:
        existing   = pd.DataFrame(columns=["date", "price"])
        fetch_from = start

    print(f"Descargando SOL/USD: {fetch_from} -> {today} (Yahoo Finance)...")
    tkr = yf.Ticker(TICKER)
    raw = tkr.history(start=str(fetch_from), end=str(end_date), interval="1d")

    if raw.empty:
        raise RuntimeError("Yahoo Finance no devolvio datos. Verificar conexion.")

    raw          = raw.reset_index()
    raw["date"]  = pd.to_datetime(raw["Date"]).dt.date
    raw["price"] = raw["Close"].astype(float)
    new_data     = raw[["date", "price"]].copy()

    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last")
    combined = combined[combined["date"] >= START_DATE]
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["date"] = combined["date"].astype(str)
    return combined


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = get_solana_prices()
    df.to_csv(OUTPUT, index=False)
    print(f"OK: {len(df)} dias de precios -> {OUTPUT}")
    print(f"   Rango: {df['date'].min()} -> {df['date'].max()}")
    print(f"   Ultimo precio: ${df['price'].iloc[-1]:.2f} ({df['date'].iloc[-1]})")
