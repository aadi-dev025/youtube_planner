"""What's getting views in your niche right now. Uses the YouTube search
API, ordered by view count, over the last few weeks. Falls back to the
sample list when there's no key so the tool still runs offline."""

import datetime as dt
import requests

API = "https://www.googleapis.com/youtube/v3"

SAMPLE_TRENDS = [
    "new Free Fire update breakdown",
    "fastest way to rank up this season",
    "pro player sensitivity settings",
    "1v4 clutch compilation",
    "best budget phone for gaming",
]


def niche_trends(keywords, key, days=30, per_kw=5):
    if not key:
        return list(SAMPLE_TRENDS)
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat("T") + "Z"
    titles = []
    for kw in keywords:
        try:
            r = requests.get(API + "/search", params={
                "part": "snippet", "q": kw, "type": "video",
                "order": "viewCount", "publishedAfter": since,
                "maxResults": per_kw, "key": key}, timeout=20)
            r.raise_for_status()
            for it in r.json().get("items", []):
                titles.append(it["snippet"]["title"])
        except Exception as e:
            print("  ! trend lookup failed for {}: {}".format(kw, e))
    return titles or list(SAMPLE_TRENDS)
