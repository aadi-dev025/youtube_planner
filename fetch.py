"""Pull a channel's videos + stats from the YouTube Data API v3 and write
them to a csv in the same shape as sample/videos.csv.

Only public data (views, likes, comments), so an API key is enough -- no
OAuth. Get a key at console.cloud.google.com -> enable "YouTube Data API
v3" -> Credentials -> API key.
"""

import csv
import datetime as dt
import requests

API = "https://www.googleapis.com/youtube/v3"


def _iso8601_to_seconds(d):
    # youtube gives durations like "PT4M30S". small parser, no deps.
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0
    h, mn, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mn * 60 + s


def _uploads_playlist(channel_id, key):
    r = requests.get(API + "/channels", params={
        "part": "contentDetails", "id": channel_id, "key": key}, timeout=20)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise SystemExit("channel not found -- check the channel_id (starts with UC)")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _video_ids(playlist_id, key, cap=200):
    ids, token = [], None
    while len(ids) < cap:
        r = requests.get(API + "/playlistItems", params={
            "part": "contentDetails", "playlistId": playlist_id,
            "maxResults": 50, "pageToken": token, "key": key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        ids += [it["contentDetails"]["videoId"] for it in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            break
    return ids[:cap]


def _topic_from_title(title):
    # cheap keyword bucketing so there's a topic column to learn from.
    t = title.lower()
    buckets = [
        ("tips", ["how to", "tips", "settings", "mistake", "better", "guide"]),
        ("challenge", ["challenge", "only", "24 hour", "can i"]),
        ("reaction", ["react", "reacting", "worst", "old"]),
        ("highlight", ["clutch", "insane", "1v4", "montage"]),
        ("news", ["update", "new", "first look", "patch"]),
        ("community", ["subscriber", "giveaway", "giving away", "with you"]),
        ("vlog", ["day in", "life", "stopped", "vlog"]),
    ]
    for name, words in buckets:
        if any(w in t for w in words):
            return name
    return "other"


def fetch(channel_id, key, out_path):
    uploads = _uploads_playlist(channel_id, key)
    vids = _video_ids(uploads, key)
    rows = []
    # stats endpoint takes up to 50 ids at a time
    for i in range(0, len(vids), 50):
        batch = vids[i:i + 50]
        r = requests.get(API + "/videos", params={
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(batch), "key": key}, timeout=20)
        r.raise_for_status()
        for it in r.json().get("items", []):
            sn, st, cd = it["snippet"], it.get("statistics", {}), it["contentDetails"]
            pub = dt.datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00"))
            title = sn["title"]
            rows.append({
                "video_id": it["id"],
                "title": title,
                "views": int(st.get("viewCount", 0)),
                "likes": int(st.get("likeCount", 0)),
                "comments": int(st.get("commentCount", 0)),
                "duration_sec": _iso8601_to_seconds(cd.get("duration")),
                "published_hour": pub.hour,
                "published_dow": pub.weekday(),
                "topic": _topic_from_title(title),
            })

    cols = ["video_id", "title", "views", "likes", "comments",
            "duration_sec", "published_hour", "published_dow", "topic"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
