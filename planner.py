#!/usr/bin/env python3
"""yt-planner: figure out what works on my channel, then rank video ideas.

I made this because planning videos was guesswork. Now it learns from my
own uploads what tends to beat my median views, pulls what's trending in
the niche, drafts ideas, and ranks them with that model.

Two commands:
  fetch  -- pull my channel from the youtube api into a csv
  plan   -- train on the csv, generate ideas, rank them

Runs offline on sample/videos.csv if you don't have api keys yet.
"""

import argparse
import json
import os
import sys

import pandas as pd

from ytplanner import features as F
from ytplanner import model as M
from ytplanner import trends as T
from ytplanner import ideas as I


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def signals_to_words(signals):
    nice = {
        "is_question": "question-style titles",
        "has_number": "a number in the title",
        "says_you": 'the word "you" in the title',
        "title_len": "longer titles",
        "title_words": "more words in the title",
        "duration_sec": "longer videos",
        "published_hour": "posting later in the day",
        "is_weekend": "posting on the weekend",
    }
    def label(name):
        if name.startswith("topic_"):
            return name[6:] + " videos"
        return nice.get(name, name)
    helps = ", ".join(label(n) for n, c in signals["helps"][:4])
    hurts = ", ".join(label(n) for n, c in signals["hurts"][:4])
    return "helps: {}\nhurts: {}".format(helps, hurts)


def cmd_fetch(args, cfg):
    from ytplanner import fetch as FE
    key = cfg.get("youtube_api_key")
    chan = cfg.get("channel_id")
    if not key or not chan:
        sys.exit("need youtube_api_key and channel_id in your config to fetch")
    n = FE.fetch(chan, key, args.out)
    print("wrote {} videos to {}".format(n, args.out))


def cmd_plan(args, cfg):
    df = pd.read_csv(args.data)
    print("[*] training on {} videos from {}".format(len(df), args.data))

    model, report = M.train(df)
    print("    median views: {:.0f}".format(report["median_views"]))
    if "cv_accuracy_mean" in report:
        print("    cross-val accuracy: {:.2f} +/- {:.2f} (baseline {:.2f}, {}-fold)".format(
            report["cv_accuracy_mean"], report["cv_accuracy_std"],
            report["baseline"], report["cv_folds"]))
    else:
        print("    note: {}".format(report.get("note", "")))

    signals = M.top_signals(model, report["features"])
    words = signals_to_words(signals)
    print("\n[*] what works on this channel")
    for line in words.splitlines():
        print("    " + line)

    trend_list = T.niche_trends(cfg.get("niche_keywords", []),
                                cfg.get("youtube_api_key", ""))
    idea_list, how = I.generate(words, trend_list, n=args.n,
                                llm_key=cfg.get("llm_api_key", ""))
    print("\n[*] {} ideas (via {}), ranked by predicted performance\n".format(
        len(idea_list), how))

    # score each idea with the channel's own model. assume a decent slot:
    # weekend, evening, the channel's best topic guess from the title.
    ranked = []
    for title in idea_list:
        topic = _guess_topic(title)
        feats = F.features_for_idea(title, duration_sec=480,
                                    published_hour=19, published_dow=6,
                                    topic=topic)
        p = M.score_idea(model, feats)
        ranked.append((p, title))
    ranked.sort(reverse=True)

    for p, title in ranked:
        print("    {:>4.0%}  {}".format(p, title))


def _guess_topic(title):
    from ytplanner.fetch import _topic_from_title
    return _topic_from_title(title)


def main():
    ap = argparse.ArgumentParser(description="learn what works, rank video ideas")
    ap.add_argument("-c", "--config", default="config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="pull channel data from youtube api")
    f.add_argument("--out", default="videos.csv")

    p = sub.add_parser("plan", help="train and rank ideas")
    p.add_argument("--data", default="sample/videos.csv")
    p.add_argument("-n", type=int, default=8, help="how many ideas")

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "fetch":
        cmd_fetch(args, cfg)
    else:
        cmd_plan(args, cfg)


if __name__ == "__main__":
    main()
