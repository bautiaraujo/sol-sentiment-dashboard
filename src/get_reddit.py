import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import praw

OUTPUT    = Path("data/reddit_posts.csv")
SUBREDDIT = "Solana"


def make_reddit() -> praw.Reddit:
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "sol-sentiment-bot/1.0"),
        check_for_async=False,
    )
    reddit.read_only = True
    return reddit


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


def fetch_posts(reddit: praw.Reddit) -> pd.DataFrame:
    """
    Combina varias estrategias para obtener posts historicos y recientes:
      1. new(limit=1000)              -> ultimos posts cronologicos
      2. top(time_filter=year, 1000) -> los mas votados del ultimo anio
      3. top(time_filter=all,  500)  -> los mas votados de la historia
      4. search reciente (3 meses)   -> posts relevantes recientes
    Esto da cobertura desde 2024 sin necesitar Pushshift.
    """
    sr   = reddit.subreddit(SUBREDDIT)
    rows = {}  # dedup por id

    strategies = [
        ("new",          lambda: sr.new(limit=1000)),
        ("top_year",     lambda: sr.top(time_filter="year",  limit=1000)),
        ("top_all",      lambda: sr.top(time_filter="all",   limit=500)),
        ("hot",          lambda: sr.hot(limit=500)),
        ("search_year",  lambda: sr.search("solana",  sort="new",
                                           time_filter="year", limit=1000)),
        ("search_price", lambda: sr.search("price sol", sort="new",
                                           time_filter="year", limit=500)),
    ]

    for name, fn in strategies:
        try:
            for post in fn():
                if post.id not in rows:
                    rows[post.id] = post_to_row(post, SUBREDDIT)
            print(f"  [{name}] OK — total acumulado: {len(rows)}")
        except Exception as e:
            print(f"  [{name}] WARN: {e}")

    return pd.DataFrame(list(rows.values()))


def merge_and_save(new_df: pd.DataFrame) -> pd.DataFrame:
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
    reddit = make_reddit()

    print(f"Descargando posts de r/{SUBREDDIT} con multiples estrategias...")
    new_df = fetch_posts(reddit)
    print(f"Posts nuevos descargados: {len(new_df)}")

    combined = merge_and_save(new_df)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"OK: {len(combined)} posts en total -> {OUTPUT}")
    if not combined.empty:
        print(f"   Rango: {combined['date'].min()} -> {combined['date'].max()}")
