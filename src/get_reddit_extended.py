"""
get_reddit_extended.py
======================
Estrategia agresiva para conseguir la mayor cantidad de posts historicos
posible con la API gratuita de Reddit (sin Pushshift).

Tecnicas usadas:
  1. Multiples subreddits relacionados a Solana/crypto
  2. top() con todos los filtros de tiempo disponibles
  3. search() con multiples keywords y periodos de tiempo
  4. hot() y controversial() para diversidad

Ejecutar UNA SOLA VEZ para enriquecer el dataset historico.
Los posts nuevos se agregan al CSV existente sin duplicar.

Uso:
    python src/get_reddit_extended.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import praw

OUTPUT    = Path("data/reddit_posts.csv")
SUBREDDITS = ["Solana", "CryptoCurrency", "solana_crypto", "SolanaBeach"]

# Keywords que cubren distintos angulos del precio y sentimiento SOL
SEARCH_KEYWORDS = [
    "SOL price", "Solana price", "Solana prediction",
    "Solana bullish", "Solana bearish", "SOL moon",
    "Solana dump", "Solana pump", "SOL ATH",
    "Solana news", "Solana ecosystem", "Solana DeFi",
    "Solana NFT", "Solana staking", "SOL sell",
    "Solana crash", "Solana rally", "SOL analysis",
    "buy Solana", "sell SOL", "Solana outlook",
]

TIME_FILTERS = ["day", "week", "month", "year", "all"]


def make_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "sol-extended-scraper/1.0"),
        check_for_async=False,
    )


def post_to_row(post, subreddit_name: str) -> dict:
    created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
    title    = post.title or ""
    selftext = post.selftext or ""
    return {
        "id":           post.id,
        "date":         created.date().isoformat(),
        "created_utc":  created.isoformat(),
        "title":        title,
        "selftext":     selftext,
        "text":         (title + " " + selftext).strip(),
        "score":        post.score,
        "num_comments": post.num_comments,
        "subreddit":    subreddit_name,
        "url":          f"https://reddit.com{post.permalink}",
    }


def collect_posts(reddit: praw.Reddit) -> dict:
    """Recolecta posts con todas las estrategias disponibles."""
    rows = {}  # dedup por id

    for sub_name in SUBREDDITS:
        print(f"\n  Subreddit: r/{sub_name}")
        try:
            sr = reddit.subreddit(sub_name)

            # new + hot + rising
            for method_name in ["new", "hot", "controversial"]:
                try:
                    method = getattr(sr, method_name)
                    for post in method(limit=1000):
                        if post.id not in rows:
                            rows[post.id] = post_to_row(post, sub_name)
                    print(f"    [{method_name}] OK — acumulado: {len(rows)}")
                except Exception as e:
                    print(f"    [{method_name}] WARN: {e}")

            # top con todos los filtros de tiempo
            for tf in TIME_FILTERS:
                try:
                    for post in sr.top(time_filter=tf, limit=1000):
                        if post.id not in rows:
                            rows[post.id] = post_to_row(post, sub_name)
                    print(f"    [top/{tf}] OK — acumulado: {len(rows)}")
                except Exception as e:
                    print(f"    [top/{tf}] WARN: {e}")

            # search con keywords y filtros de tiempo
            for kw in SEARCH_KEYWORDS:
                for tf in ["year", "all"]:
                    try:
                        for post in sr.search(kw, sort="relevance",
                                              time_filter=tf, limit=100):
                            if post.id not in rows:
                                rows[post.id] = post_to_row(post, sub_name)
                    except Exception:
                        pass
            print(f"    [search] OK — acumulado: {len(rows)}")

        except Exception as e:
            print(f"  r/{sub_name} ERROR: {e}")

    return rows


def merge_and_save(new_rows: dict) -> pd.DataFrame:
    new_df = pd.DataFrame(list(new_rows.values()))
    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        existing = pd.read_csv(OUTPUT, dtype=str)
    else:
        existing = pd.DataFrame(columns=new_df.columns)
    combined = pd.concat([existing, new_df.astype(str)], ignore_index=True)
    combined = combined.drop_duplicates(subset="id", keep="last")
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("Iniciando recoleccion extendida de posts Reddit...")
    print(f"Subreddits: {SUBREDDITS}")
    print(f"Keywords: {len(SEARCH_KEYWORDS)} terminos de busqueda")

    reddit  = make_reddit()
    rows    = collect_posts(reddit)
    combined = merge_and_save(rows)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Total posts guardados: {len(combined)}")
    if not combined.empty:
        print(f"Rango de fechas: {combined['date'].min()} -> {combined['date'].max()}")
        # Distribucion por anio
        combined['year'] = combined['date'].str[:4]
        dist = combined.groupby('year').size()
        print("Distribucion por anio:")
        for yr, cnt in dist.items():
            print(f"  {yr}: {cnt} posts")
    print(f"\nArchivo: {OUTPUT}")
    print("\nProximo paso: python src/sentiment.py")
