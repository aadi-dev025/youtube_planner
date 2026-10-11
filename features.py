"""Turn video rows into features the model can use, and define the thing
we're predicting: did this video beat the channel's median views.

Everything here works on a title + a few numbers, so it can score a brand
new idea (which has no views yet) the exact same way it scores a real
video. That's what lets us rank ideas later.
"""

import pandas as pd

TOPICS = ["tips", "challenge", "reaction", "highlight", "news",
          "community", "vlog", "tier", "other"]


def title_features(title):
    t = str(title)
    low = t.lower()
    return {
        "title_len": len(t),
        "title_words": len(t.split()),
        "is_question": int("?" in t or low.startswith(("how", "why", "can", "what"))),
        "has_number": int(any(c.isdigit() for c in t)),
        "says_you": int(" you" in low or low.startswith("you")),
    }


def row_features(row):
    """Features available before a video exists: title + planned timing +
    topic. Deliberately excludes views/likes so we never leak the answer."""
    f = title_features(row["title"])
    f["duration_sec"] = int(row.get("duration_sec", 0))
    f["published_hour"] = int(row.get("published_hour", 12))
    f["is_weekend"] = int(int(row.get("published_dow", 0)) >= 5)
    topic = str(row.get("topic", "other"))
    for t in TOPICS:
        f["topic_" + t] = int(topic == t)
    return f


def build(df):
    """df of videos -> (X features frame, y labels, median_views)."""
    feats = pd.DataFrame([row_features(r) for _, r in df.iterrows()])
    median = df["views"].median()
    y = (df["views"] >= median).astype(int)
    return feats, y, median


def features_for_idea(title, duration_sec, published_hour, published_dow, topic):
    row = {"title": title, "duration_sec": duration_sec,
           "published_hour": published_hour, "published_dow": published_dow,
           "topic": topic}
    return pd.DataFrame([row_features(row)])
