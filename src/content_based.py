"""
content_based.py  —  MEMBER 1's SOLUTION
========================================
Content-Based Filtering (CBF) recommender.

Core idea
---------
"Recommend more of what you already like."  A content-based system ignores
other users completely and instead looks at the *attributes of the items*.
If you rated several Animation/Adventure films highly, it will surface other
Animation/Adventure films, because they are similar *in content*.

Algorithm (three stages)
-------------------------
1.  ITEM PROFILES — TF-IDF
        Every movie has a text "content soup" (genres + tags, see data_loader).
        We turn each soup into a numeric vector with TF-IDF
        (Term Frequency × Inverse Document Frequency):

            tfidf(term, movie) = tf(term, movie) * log( N / df(term) )

        * tf  rewards a term that appears in this movie's soup,
        * idf  down-weights terms that appear in almost every movie
               (e.g. "Drama"), because common terms are poor discriminators.
        The result is an  (n_movies x n_terms)  sparse matrix.

2.  SIMILARITY — Cosine
        The closeness of two movies is the cosine of the angle between their
        TF-IDF vectors:

            cos(a, b) = (a · b) / (||a|| * ||b||)        in [0, 1] for TF-IDF

        1.0 = identical genre/tag fingerprint, 0.0 = nothing in common.

3.  USER PROFILE + SCORING
        A user is represented by the movies they have rated.  To predict how
        much user u would like an unseen movie i we take a *similarity-weighted
        average of the user's own ratings*:

            pred(u, i) =  Σ_j  sim(i, j) * r(u, j)
                          ---------------------------      (j = movies u rated)
                          Σ_j  |sim(i, j)|

        For a Top-N list we compute this score for every unseen movie and
        return the highest-scoring ones.

Why content-based?
------------------
+ Works for brand-new users' niche tastes and needs no other users' data.
+ Naturally explainable ("recommended because it is also Animation|Fantasy").
+ Handles the item cold-start problem: a freshly added movie can be
  recommended immediately from its genres/tags alone.
-  Tends to over-specialise (keeps recommending the same genre) and cannot
   discover cross-genre surprises — that is what collaborative filtering adds.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import MovieLensData


class ContentBasedRecommender:
    """TF-IDF + cosine-similarity content-based filtering."""

    def __init__(self, data: MovieLensData):
        self.data = data
        self.tfidf_matrix = None          # (n_movies x n_terms) sparse TF-IDF
        self.vectorizer: TfidfVectorizer | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  TRAINING  (build the item profiles)
    # ------------------------------------------------------------------ #
    def fit(self) -> "ContentBasedRecommender":
        """Vectorise every movie's content soup into a TF-IDF matrix.

        `token_pattern` keeps multi-word tags meaningful by treating each
        whitespace-separated token as a term; `min_df=1` keeps even rare tags
        because they are highly descriptive.
        """
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"[^\s]+",   # any run of non-space chars is one token
            min_df=1,
            sublinear_tf=True,         # use 1+log(tf) — dampens repeated terms
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data.movies["content"])
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  SIMILARITY helpers
    # ------------------------------------------------------------------ #
    def similar_movies(self, movie_id: int, top_n: int = 10):
        """Return the `top_n` movies most similar in content to `movie_id`.

        Used directly by the UI ("more like this") and also lets a brand-new
        user get recommendations from a single seed movie (cold-start).
        """
        self._check_fitted()
        if movie_id not in self.data.i_index:
            return []
        j = self.data.i_index[movie_id]
        # cosine between this one movie and all movies -> shape (1, n_movies)
        sims = cosine_similarity(self.tfidf_matrix[j], self.tfidf_matrix).ravel()
        sims[j] = -1.0                       # never return the movie itself
        order = np.argsort(sims)[::-1][:top_n]
        return [(self.data.movie_ids[k], float(sims[k])) for k in order]

    # ------------------------------------------------------------------ #
    #  PREDICTION  (rating + Top-N)
    # ------------------------------------------------------------------ #
    def _user_rated(self, user_id: int):
        """(column indices, ratings) of the movies a user has rated."""
        sub = self.data.ratings[self.data.ratings.userId == user_id]
        cols = sub.movieId.map(self.data.i_index).dropna().astype(int).to_numpy()
        rates = sub.set_index("movieId").loc[
            self.data.movie_ids[cols], "rating"
        ].to_numpy()
        return cols, rates

    def predict_rating(self, user_id: int, movie_id: int) -> float | None:
        """Predict the rating user `user_id` would give `movie_id`.

        Implements the similarity-weighted average described in the header.
        Returns None when the prediction is undefined (unknown movie, or the
        user has no content overlap with the target).
        """
        self._check_fitted()
        if movie_id not in self.data.i_index:
            return None
        target = self.data.i_index[movie_id]
        cols, rates = self._user_rated(user_id)
        if len(cols) == 0:
            return None

        # cosine similarity between the target movie and each rated movie
        sims = cosine_similarity(
            self.tfidf_matrix[target], self.tfidf_matrix[cols]
        ).ravel()

        denom = np.abs(sims).sum()
        if denom == 0:
            return None                        # no content overlap at all
        return float(np.dot(sims, rates) / denom)

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True):
        """Top-N content-based recommendations for an existing user.

        We score *all* movies in one vectorised pass:

            scores = TFIDF_all  ·  ( TFIDF_rated.T  ·  ratings )

        The bracketed term is the user's TF-IDF "taste vector" (a rating-weighted
        sum of the profiles of movies they liked); the outer product then scores
        every movie by its alignment with that taste vector.
        """
        self._check_fitted()
        cols, rates = self._user_rated(user_id)
        if len(cols) == 0:
            return []

        # Build the weighted user profile in TF-IDF space, then score all movies.
        # (rates @ tfidf[cols]) is a 1 x n_terms dense taste vector.
        taste = rates @ self.tfidf_matrix[cols]            # np.matrix (1 x terms)
        taste = np.asarray(taste)                          # -> ndarray
        scores = (self.tfidf_matrix @ taste.T).ravel()     # (n_movies,)

        # normalise by total rating mass so scores are on the rating scale-ish
        scores = scores / (np.abs(rates).sum() + 1e-9)

        if exclude_seen:
            scores[cols] = -np.inf                         # hide already-seen
        order = np.argsort(scores)[::-1][:top_n]
        return [(self.data.movie_ids[k], float(scores[k])) for k in order
                if np.isfinite(scores[k])]

    def recommend_from_movie(self, movie_id: int, top_n: int = 10):
        """Cold-start helper: recommend from a single liked movie (no history)."""
        return self.similar_movies(movie_id, top_n=top_n)

    def recommend_for_profile(self, profile: dict[int, float], top_n: int = 10,
                              exclude_ids=None):
        """Recommend for an AD-HOC taste profile built in the GUI.

        `profile` maps movieId -> rating (e.g. a 'Like' adds {id: 5.0}).  This
        powers the "Build my own taste" mode: a brand-new user with no account
        gets recommendations purely from the movies they just liked — exactly
        the item cold-start scenario content-based filtering excels at.
        """
        self._check_fitted()
        cols, rates = [], []
        for mid, r in profile.items():
            if mid in self.data.i_index:
                cols.append(self.data.i_index[mid])
                rates.append(float(r))
        if not cols:
            return []
        cols = np.array(cols)
        rates = np.array(rates, dtype=np.float32)

        taste = np.asarray(rates @ self.tfidf_matrix[cols])         # taste vector
        scores = (self.tfidf_matrix @ taste.T).ravel() / (np.abs(rates).sum() + 1e-9)

        scores[cols] = -np.inf                                       # hide liked
        for mid in (exclude_ids or []):                              # hide 'not interested'
            if mid in self.data.i_index:
                scores[self.data.i_index[mid]] = -np.inf
        order = np.argsort(scores)[::-1][:top_n]
        return [(self.data.movie_ids[k], float(scores[k])) for k in order
                if np.isfinite(scores[k])]

    # ------------------------------------------------------------------ #
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before using the recommender.")


if __name__ == "__main__":  # demo: `python -m src.content_based`
    from .data_loader import load_data

    data = load_data()
    cb = ContentBasedRecommender(data).fit()

    print("Movies similar to 'Toy Story (1995)':")
    for mid, sim in cb.similar_movies(1, top_n=5):
        print(f"  {sim:.3f}  {data.title(mid)}")

    print("\nTop-5 content-based recommendations for user 1:")
    for mid, score in cb.recommend(1, top_n=5):
        print(f"  {score:.3f}  {data.title(mid)}")
