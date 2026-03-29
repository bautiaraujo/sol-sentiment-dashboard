import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

INPUT  = Path("data/reddit_posts.csv")
OUTPUT = Path("data/reddit_sentiment.csv")
MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN    = 512
BATCH_SIZE = 32


def load_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return pipeline(
        "text-classification", model=mdl, tokenizer=tok,
        framework="pt", truncation=True, max_length=MAX_LEN,
        return_all_scores=True,
    )


def score_texts(nlp, texts: list) -> list:
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = [str(t)[:4000] for t in texts[i:i+BATCH_SIZE]]
        for res in nlp(chunk, truncation=True, max_length=MAX_LEN, batch_size=BATCH_SIZE):
            d = {r["label"].lower(): r["score"] for r in res}
            scores.append(round(d.get("positive", 0.0) - d.get("negative", 0.0), 6))
        print(f"  Sentiment: {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")
    return scores


if __name__ == "__main__":
    if not INPUT.exists():
        raise FileNotFoundError(f"No existe {INPUT}. Corre src/get_reddit.py primero.")

    posts = pd.read_csv(INPUT)
    posts["text"] = posts["text"].fillna("").astype(str)

    # Solo procesar posts sin sent_score (incremental)
    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT)
        done_ids = set(existing["id"].astype(str))
        new_posts = posts[~posts["id"].astype(str).isin(done_ids)].copy()
        print(f"Posts nuevos a procesar: {len(new_posts)} (ya existentes: {len(existing)})")
    else:
        existing  = pd.DataFrame()
        new_posts = posts.copy()
        print(f"Procesando {len(new_posts)} posts...")

    if len(new_posts) == 0:
        print("Nada nuevo que procesar.")
    else:
        nlp = load_pipeline()
        new_posts["sent_score"] = score_texts(nlp, new_posts["text"].tolist())
        combined = pd.concat([existing, new_posts], ignore_index=True)
        combined = combined.drop_duplicates(subset="id", keep="last")
        combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT, index=False, encoding="utf-8")
        print(f"OK: {len(combined)} posts con sentiment -> {OUTPUT}")
