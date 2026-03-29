import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import praw

OUTPUT = Path("data/reddit_posts.csv")
SUBREDDIT = "Solana"
FETCH_DAYS = 30    # días recientes a buscar en cada ejecución
LIMIT = 500        # máx posts por llamada


def make_reddit() -> praw.Reddit:
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "sol-sentiment-bot/1.0"),
        check_for_async=False,
    )
    reddit.read_only = True
    return reddit


def fetch_new_posts(reddit: praw.Reddit, days: int = FETCH_DAYS) -> pd.DataFrame:
    """Descarga los últimos posts del subreddit."""
    sr = reddit.subreddit(SUBREDDIT)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for post in sr.new(limit=LIMIT):
        created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        if created < cutoff:
            break
        title = post.title or ""
        selftext = post.selftext or ""
        rows.append({
            "id": post.id,
            "date": created.date().isoformat(),
            "created_utc": created.isoformat(),
            "title": title,
            "selftext": selftext,
            "text": (title + " " + selftext).strip(),
            "score": post.score,
            "num_comments": post.num_comments,
            "subreddit": SUBREDDIT,
            "url": f"https://reddit.com{post.permalink}",
        })
    return pd.DataFrame(rows)


def merge_and_save(new_df: pd.DataFrame) -> pd.DataFrame:
    """Combina nuevos posts con los existentes, deduplicando por ID."""
    if OUTPUT.exists() and os.path.getsize(OUTPUT) > 0:
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
    new_df = fetch_new_posts(reddit)
    print(f"Descargados {len(new_df)} posts nuevos.")
    combined = merge_and_save(new_df)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"OK: {len(combined)} posts en total -> {OUTPUT}")
    if not combined.empty:
        print(f"   Rango: {combined['date'].min()} → {combined['date'].max()}")
