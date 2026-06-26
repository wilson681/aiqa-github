"""
evaluation.py
=============
Offline evaluation utilities — how we *prove* the recommenders work and compare
them fairly (assignment requirement (d): "test and evaluate ... using
appropriate evaluation metrics").

Two families of metrics are implemented, because a recommender has two jobs:

1.  RATING PREDICTION  ->  "how close is the predicted score to the truth?"
        * RMSE  (Root Mean Squared Error) — penalises large mistakes more.
        * MAE   (Mean Absolute Error)      — average size of the error.
        Lower is better for both.

2.  TOP-N RANKING       ->  "are the recommended items actually relevant?"
        * Precision@K — of the K we recommended, what fraction were relevant?
        * Recall@K    — of all relevant items, what fraction did we surface?
        * F1@K        — harmonic mean of precision and recall.
        Higher is better for all three.

Evaluation protocol
--------------------
We use a per-user hold-out split (see data_loader.train_test_split_ratings):
models are trained on the TRAIN ratings only, then asked to predict / rank the
held-out TEST ratings they have never seen.  An item in the test set is judged
"relevant" if the user actually rated it >= `like_threshold` (default 4.0).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .data_loader import MovieLensData


# --------------------------------------------------------------------------- #
#  Build a "training-only" data object (so models never see the test ratings)
# --------------------------------------------------------------------------- #
def build_train_data(full: MovieLensData, train_df: pd.DataFrame) -> MovieLensData:
    """Clone `full` but replace ratings/rating_matrix with the TRAIN split.

    The user/movie id maps and movie metadata stay identical to the full
    dataset, so every user and movie is still known (important for fair
    cold-start handling) — only the *observed ratings* are restricted to train.
    """
    td = copy.copy(full)                       # shallow copy: shares movies, maps
    td.ratings = train_df.reset_index(drop=True)

    rm = np.zeros((full.n_users, full.n_movies), dtype=np.float32)
    rows = train_df.userId.map(full.u_index).to_numpy()
    cols = train_df.movieId.map(full.i_index)
    valid = cols.notna().to_numpy()
    rm[rows[valid], cols[valid].astype(int)] = train_df.rating.to_numpy()[valid]
    td.rating_matrix = rm
    return td


# --------------------------------------------------------------------------- #
#  1. Rating-prediction metrics
# --------------------------------------------------------------------------- #
def rating_metrics(model, test_df: pd.DataFrame, max_samples: int | None = None,
                   seed: int = 0) -> dict:
    """Compute RMSE and MAE of `model.predict_rating` over the test ratings.

    Predictions that come back as None (model cannot answer) are skipped and
    reported as `coverage` — the fraction of test pairs the model could score.
    """
    df = test_df
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed)

    errs, n_total, n_scored = [], 0, 0
    for uid, mid, true_r in zip(df.userId, df.movieId, df.rating):
        n_total += 1
        pred = model.predict_rating(int(uid), int(mid))
        if pred is None:
            continue
        n_scored += 1
        errs.append(pred - true_r)

    errs = np.asarray(errs, dtype=np.float64)
    if errs.size == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "coverage": 0.0}
    return {
        "RMSE": float(np.sqrt(np.mean(errs ** 2))),
        "MAE": float(np.mean(np.abs(errs))),
        "coverage": n_scored / n_total,
    }


# --------------------------------------------------------------------------- #
#  2. Top-N ranking metrics
# --------------------------------------------------------------------------- #
def ranking_metrics(model, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    k: int = 10, like_threshold: float = 4.0,
                    max_users: int | None = None, seed: int = 0,
                    recommend_kwargs: dict | None = None) -> dict:
    """Average Precision@K, Recall@K and F1@K over users.

    For each evaluated user we build their set of *relevant* items (test movies
    they rated >= like_threshold), ask the model for its Top-K list, and measure
    the overlap.  Users with no relevant test items are skipped (precision is
    undefined for them).
    """
    recommend_kwargs = recommend_kwargs or {}

    # relevant items per user = highly-rated held-out movies
    liked = test_df[test_df.rating >= like_threshold]
    relevant_by_user = liked.groupby("userId").movieId.apply(set).to_dict()

    users = list(relevant_by_user.keys())
    rng = np.random.default_rng(seed)
    if max_users is not None and len(users) > max_users:
        users = list(rng.choice(users, size=max_users, replace=False))

    precisions, recalls, f1s, hit_users = [], [], [], 0
    for uid in users:
        relevant = relevant_by_user[uid]
        if not relevant:
            continue
        recs = model.recommend(int(uid), top_n=k, **recommend_kwargs)
        rec_ids = {mid for mid, _ in recs}
        if not rec_ids:
            continue
        hits = len(rec_ids & relevant)
        precision = hits / k
        recall = hits / len(relevant)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        if hits > 0:
            hit_users += 1

    n = max(len(precisions), 1)
    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"F1@{k}": float(np.mean(f1s)) if f1s else 0.0,
        "HitRate": hit_users / n,
        "n_users": len(precisions),
    }


def format_table(rows: list[dict], columns: list[str], title: str = "") -> str:
    """Render a list of metric dicts as a fixed-width text table for the report."""
    name_w = max([len("Method")] + [len(r["Method"]) for r in rows])
    header = "Method".ljust(name_w) + "".join(c.rjust(14) for c in columns)
    line = "-" * len(header)
    out = []
    if title:
        out += [title, "=" * len(title)]
    out += [header, line]
    for r in rows:
        cells = "".join(
            (f"{r[c]:.4f}".rjust(14) if isinstance(r.get(c), (int, float)) else str(r.get(c, "")).rjust(14))
            for c in columns
        )
        out.append(r["Method"].ljust(name_w) + cells)
    return "\n".join(out)
