"""Tests for the parts that must stay correct: feature extraction never
leaks the target, and the model trains + ranks without blowing up."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from ytplanner import features as F
from ytplanner import model as M


def _df():
    path = os.path.join(os.path.dirname(__file__), "..", "sample", "videos.csv")
    return pd.read_csv(path)


def test_features_do_not_leak_views():
    X, y, median = F.build(_df())
    # views/likes/comments must NOT be in the feature set, or we'd be
    # cheating by predicting views from views.
    for banned in ("views", "likes", "comments"):
        assert banned not in X.columns


def test_question_feature():
    f = F.title_features("Can I win using only melee?")
    assert f["is_question"] == 1
    f2 = F.title_features("My new settings")
    assert f2["is_question"] == 0


def test_idea_features_match_training_columns():
    X, _, _ = F.build(_df())
    idea = F.features_for_idea("How to rank up fast", 480, 19, 6, "tips")
    # same columns in the same order, so the model can score an idea
    assert list(idea.columns) == list(X.columns)


def test_model_trains_and_scores():
    df = _df()
    model, report = M.train(df)
    assert report["n_videos"] == len(df)
    idea = F.features_for_idea("5 tips to aim better", 420, 18, 5, "tips")
    p = M.score_idea(model, idea)
    assert 0.0 <= p <= 1.0


def test_signals_have_both_directions():
    df = _df()
    model, report = M.train(df)
    sig = M.top_signals(model, report["features"])
    assert sig["helps"] and sig["hurts"]
