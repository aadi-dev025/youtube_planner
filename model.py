"""The ML bit. A logistic regression that, trained on your own channel,
guesses whether a video will land above your median views.

Logistic regression on purpose: with a small channel (a few dozen videos)
a big model would just memorise. A linear model stays honest and, nicely,
its coefficients tell you which features push a video up or down -- which
is the actually-useful output.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F


def _make():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def train(df):
    X, y, median = F.build(df)
    model = _make()

    # honest evaluation: cross-validated, because with this little data a
    # single train/test split is basically noise. if there's too little
    # data to split, say so instead of faking a score.
    report = {"n_videos": len(df), "median_views": float(median)}
    if len(df) >= 10 and y.nunique() == 2:
        k = min(5, int(y.value_counts().min()))
        if k >= 2:
            scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
            report["cv_folds"] = k
            report["cv_accuracy_mean"] = float(scores.mean())
            report["cv_accuracy_std"] = float(scores.std())
            report["baseline"] = float(max(y.mean(), 1 - y.mean()))
        else:
            report["note"] = "one class too small to cross-validate"
    else:
        report["note"] = "not enough data to evaluate; model still trained"

    model.fit(X, y)
    report["features"] = list(X.columns)
    return model, report


def top_signals(model, feature_names, k=6):
    """Pull the strongest + and - coefficients so we can tell the user, in
    words, what tends to work on their channel."""
    clf = model.named_steps["logisticregression"]
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    low = [(feature_names[i], float(coefs[i])) for i in order[:k]]
    high = [(feature_names[i], float(coefs[i])) for i in order[::-1][:k]]
    return {"helps": high, "hurts": low}


def score_idea(model, idea_feats):
    """Probability this idea beats the median. Single row in -> float out."""
    return float(model.predict_proba(idea_feats)[0][1])


def save(model, path):
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)
