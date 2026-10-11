"""Generate video ideas. If an LLM key is set, ask the model to write
titles using what works on the channel + current niche trends. If not,
fall back to a template generator so the tool still produces ideas.

Either way the ideas are just candidate titles -- the model in model.py
is what actually ranks them. The LLM proposes, the channel's own data
decides.
"""

import json
import requests


def _prompt(signals_words, trends, n):
    return (
        "You write YouTube video titles for a gaming channel.\n\n"
        "What tends to do well on this channel:\n" + signals_words + "\n\n"
        "Trending in the niche right now:\n- " + "\n- ".join(trends) + "\n\n"
        "Write {} new video title ideas that fit this channel's winning "
        "pattern and connect to a trend. Return a JSON list of strings, "
        "titles only, nothing else.".format(n)
    )


def _from_llm(key, signals_words, trends, n):
    # Anthropic messages API. Kept to one simple call.
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": "claude-3-5-haiku-20241022", "max_tokens": 500,
              "messages": [{"role": "user",
                            "content": _prompt(signals_words, trends, n)}]},
        timeout=40)
    r.raise_for_status()
    text = r.json()["content"][0]["text"]
    start, end = text.find("["), text.rfind("]")
    return [str(x) for x in json.loads(text[start:end + 1])]


def _from_templates(trends, n):
    stems = [
        "How to {}",
        "5 mistakes everyone makes with {}",
        "I tried {} for a week",
        "Why your {} is wrong",
        "{} but on hard mode",
        "Pro vs noob: {}",
    ]
    out = []
    for i in range(n):
        trend = trends[i % len(trends)].lower()
        out.append(stems[i % len(stems)].format(trend))
    return out


def generate(signals_words, trends, n=8, llm_key=""):
    if llm_key:
        try:
            return _from_llm(llm_key, signals_words, trends, n), "llm"
        except Exception as e:
            print("  ! llm call failed, using templates: {}".format(e))
    return _from_templates(trends, n), "templates"
