"""
hybrid.py  —  JOINT SOLUTION (extra effort for an "Excellent" grade)
====================================================================
A Hybrid recommender that combines Member 1's Content-Based model with
Member 2's Collaborative-Filtering model to get the best of both worlds.

Why hybridise?
--------------
The two base methods fail in opposite situations:

    Content-Based (CBF)            Collaborative (CF / FunkSVD)
    --------------------           ----------------------------
    + works with no other users    + discovers cross-genre surprises
    + solves item cold-start       + learns hidden taste factors
    - over-specialises to a genre  - fails on new users/items (cold-start)
    - ignores quality/popularity   - needs a dense-enough ratings matrix

A hybrid lets the strength of one cover the weakness of the other.

Strategy used here — WEIGHTED hybrid with a cold-start SWITCH
------------------------------------------------------------
1.  Get a predicted rating from BOTH base models.
2.  If collaborative filtering cannot produce a prediction (a brand-new user or
    item — the cold-start case) we *switch* to using the content-based score
    alone.  Otherwise we *blend*:

        score_hybrid =  α · score_CF  +  (1 − α) · score_CBF        (0 ≤ α ≤ 1)

    α (alpha) is the trust we place in collaborative filtering.  α = 0.7 means
    "weight CF 70 %, content 30 %", a good default once a user has some history.

For Top-N we normalise each base model's scores to a common 0–1 range before
blending so that neither model's raw scale dominates the mix.
"""

from __future__ import annotations

import numpy as np

from .collaborative import FunkSVD, ItemBasedCF
from .content_based import ContentBasedRecommender
from .data_loader import MovieLensData


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale a score vector to [0, 1] (ignoring -inf padding)."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    lo, hi = finite.min(), finite.max()
    if hi - lo < 1e-12:
        out = np.where(np.isfinite(x), 0.5, x)
        return out
    out = (x - lo) / (hi - lo)
    out[~np.isfinite(x)] = -np.inf
    return out


class HybridRecommender:
    """Weighted hybrid of a content-based and a collaborative recommender.

    Parameters
    ----------
    cf_model : 'svd' | 'knn'
        Which collaborative model to pair with the content model.
    alpha : float
        Weight on the collaborative score (1-alpha goes to content).
    """

    def __init__(self, data: MovieLensData, cf_model: str = "svd", alpha: float = 0.7):
        self.data = data
        self.alpha = alpha
        self.cf_kind = cf_model
        self.content = ContentBasedRecommender(data)
        self.cf = FunkSVD(data) if cf_model == "svd" else ItemBasedCF(data)
        self._fitted = False

    def fit(self) -> "HybridRecommender":
        """Train both base models (they are independent, so just fit each)."""
        self.content.fit()
        self.cf.fit()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Rating prediction (weighted blend + cold-start switch)
    # ------------------------------------------------------------------ #
    def predict_rating(self, user_id: int, movie_id: int) -> float | None:
        self._check_fitted()
        cb = self.content.predict_rating(user_id, movie_id)
        cf = self.cf.predict_rating(user_id, movie_id)

        # cold-start switching: lean on whichever model can answer
        if cf is None and cb is None:
            return None
        if cf is None:
            return cb
        if cb is None:
            return cf
        return self.alpha * cf + (1 - self.alpha) * cb

    # ------------------------------------------------------------------ #
    #  Top-N (normalise both score vectors, then blend)
    # ------------------------------------------------------------------ #
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True):
        """Blend normalised content and CF rankings into one Top-N list."""
        self._check_fitted()
        n = self.data.n_movies
        cb_scores = np.full(n, -np.inf, dtype=np.float32)
        cf_scores = np.full(n, -np.inf, dtype=np.float32)

        # gather a generous candidate list from each base model, then map the
        # scores back onto the full item vector so we can blend element-wise
        for mid, s in self.content.recommend(user_id, top_n=200, exclude_seen=exclude_seen):
            cb_scores[self.data.i_index[mid]] = s
        cf_list = (self.cf.recommend(user_id, top_n=200, exclude_seen=exclude_seen)
                   if self.cf_kind == "svd"
                   else self.cf.recommend(user_id, top_n=200, exclude_seen=exclude_seen,
                                          refine=False))
        for mid, s in cf_list:
            cf_scores[self.data.i_index[mid]] = s

        cb_n, cf_n = _minmax(cb_scores), _minmax(cf_scores)

        # combine: where only one model has a candidate, use it directly
        blended = np.full(n, -np.inf, dtype=np.float32)
        both = np.isfinite(cb_n) & np.isfinite(cf_n)
        only_cb = np.isfinite(cb_n) & ~np.isfinite(cf_n)
        only_cf = np.isfinite(cf_n) & ~np.isfinite(cb_n)
        blended[both] = self.alpha * cf_n[both] + (1 - self.alpha) * cb_n[both]
        blended[only_cf] = self.alpha * cf_n[only_cf]
        blended[only_cb] = (1 - self.alpha) * cb_n[only_cb]

        if exclude_seen:
            u = self.data.u_index.get(user_id)
            if u is not None:
                seen = np.where(self.data.rating_matrix[u] > 0)[0]
                blended[seen] = -np.inf

        order = np.argsort(blended)[::-1][:top_n]
        return [(self.data.movie_ids[k], float(blended[k])) for k in order
                if np.isfinite(blended[k])]

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before using HybridRecommender.")


if __name__ == "__main__":  # demo: `python -m src.hybrid`
    from .data_loader import load_data

    data = load_data()
    hy = HybridRecommender(data, cf_model="svd", alpha=0.7).fit()
    print("Top-5 hybrid recommendations for user 1:")
    for mid, score in hy.recommend(1, top_n=5):
        print(f"  {score:.3f}  {data.title(mid)}")
