# yt-planner

A tool that learns what works on my YouTube channel and ranks video ideas
before I make them.

## why

I run a small gaming channel and planning videos was guesswork. I'd have
an idea, make it, and then find out it flopped. So I wanted something that
looks at what already worked on my channel, checks what's trending in the
niche, and tells me which of my next ideas is most likely to do well.

## what it does

1. pulls my channel's videos and stats from the YouTube API
2. trains a small model on my own uploads to learn what beats my median
   views (title style, length, topic, when I post)
3. pulls what's trending in the niche right now
4. drafts new video ideas (an LLM writes them if I give it a key, else a
   template generator)
5. ranks those ideas with the model from step 2

So the LLM proposes ideas and my own channel's data decides which are
worth making.

## the model

It's a logistic regression (scikit-learn) that predicts whether a video
will land above the channel's median views, trained only on things you
know *before* posting: the title, planned length, topic, and timing. It
never uses views/likes to predict views, which would be cheating -- there's
a test that checks for exactly that.

Logistic regression on purpose, not something fancier: a channel only has
a few dozen videos, so a big model would just memorise them. A linear model
stays honest, and its coefficients double as the "what works on my channel"
report.

It's evaluated with cross-validation and printed next to the baseline, so
you can see whether it actually beats guessing. On a small channel the
number won't be amazing, and that's fine -- it's honest, not inflated.

## run it (no keys needed)

```
pip install -r requirements.txt
python3 planner.py plan --data sample/videos.csv
```

That trains on the bundled sample channel and prints ranked ideas, so you
can see the whole thing work without any setup.

Example output:

```
[*] training on 20 videos from sample/videos.csv
    median views: 4450
    cross-val accuracy: 0.85 +/- 0.20 (baseline 0.50, 5-fold)

[*] what works on this channel
    helps: posting later in the day, challenge videos, a number in the title
    hurts: longer titles, community videos, longer videos

[*] ideas, ranked by predicted performance
     76%  Why your 1v4 clutch is wrong
     62%  Pro vs noob: new update breakdown
     ...
```

## run it on your own channel

1. copy `config.example.json` to `config.json`
2. get a YouTube Data API key: console.cloud.google.com -> new project ->
   enable "YouTube Data API v3" -> Credentials -> API key
3. put the key and your channel id (starts with `UC`) in `config.json`
4. pull your data, then plan:

```
python3 planner.py fetch --out videos.csv
python3 planner.py plan --data videos.csv
```

For nicer ideas, add an Anthropic API key to `config.json` under
`llm_api_key`. Without it the tool uses templates and still runs fine.

## tests

```
python3 -m pytest
```

The important test is the one that makes sure the features never leak
views into the prediction.

## honest limits

- it predicts *relative* performance on your own channel, not real view
  counts, and definitely not "will this go viral"
- a small channel means a small training set, so treat the ranking as a
  nudge, not a verdict
- topic buckets are keyword-based and rough; good enough to learn from,
  not perfect

## what I'd do next

- better topic detection (embeddings instead of keywords)
- pull thumbnails and use whether a face is in them as a feature
- track predictions vs what actually happened and show the hit rate
