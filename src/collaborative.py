"""
collaborative.py  —  MEMBER 2's SOLUTION
========================================
Collaborative Filtering (CF).  Two complementary implementations are provided:

    A.  ItemBasedCF   — memory-based, item-item k-Nearest-Neighbours
    B.  FunkSVD       — model-based, latent-factor matrix factorization

Core idea
---------
"People who agreed in the past will agree in the future."  Unlike content-based
filtering, CF ignores what an item *is* and looks only at the *ratings matrix*:
it finds patterns of agreement between users / items.  This lets it discover
cross-genre surprises that a content model never could.

------------------------------------------------------------------------------
A. ITEM-BASED k-NEAREST-NEIGHBOURS  (memory based)
------------------------------------------------------------------------------
1.  Represent every movie as a vector of the ratings it received from all users
    (a column of the user-item matrix).
2.  Mean-centre each item's ratings (subtract the item's average) so that a
    generous user who rates everything 5 does not dominate — this is the
    "adjusted cosine"/centred-cosine similarity.
3.  Similarity between two movies = cosine of their centred rating vectors.
4.  Predict the rating user u would give item i from the k most similar items
    that u has already rated:

        pred(u, i) = mean_i +  Σ_j  sim(i, j) * ( r(u, j) - mean_j )
                               --------------------------------------
                               Σ_j  | sim(i, j) |
                               (j = the k neighbours of i that u rated)

------------------------------------------------------------------------------
B. FUNK-SVD  (model based — the algorithm that won the Netflix Prize)
------------------------------------------------------------------------------
Approximate the giant, mostly-empty rating matrix R as the product of two
small dense matrices of *latent factors* plus bias terms:

        r_hat(u, i) = μ + b_u + b_i + p_u · q_i

    μ      global average rating
    b_u    how generous user u is vs. average
    b_i    how loved item i is vs. average
    p_u    user u's taste in a k-dimensional latent space
    q_i    item i's profile in that same latent space

The latent dimensions are learned, not given — they end up capturing hidden
themes such as "amount of action" or "arthouse vs. blockbuster".  We learn the
parameters by Stochastic Gradient Descent over the *observed* ratings only,
minimising the regularised squared error:

        min  Σ ( r_ui - r_hat_ui )²  +  λ ( b_u² + b_i² + ||p_u||² + ||q_i||² )

Because it factors the matrix, FunkSVD generalises better than memory-based CF
and is far more memory-efficient at prediction time.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .data_loader import MovieLensData


# =========================================================================== #
#  A.  ITEM-BASED k-NEAREST-NEIGHBOURS
# =========================================================================== #
class ItemBasedCF:
    """Memory-based item-item collaborative filtering (centred cosine)."""

    def __init__(self, data: MovieLensData, k: int = 30):
        self.data = data
        self.k = k                       # neighbourhood size
        self.item_means = None           # per-item average rating
        self.item_support = None         # number of ratings each item received
        self.norm_items = None           # L2-normalised centred item vectors
        self._fitted = False

    def fit(self, rating_matrix: np.ndarray | None = None) -> "ItemBasedCF":
        """Pre-compute item means and the normalised centred item matrix.

        We store items as ROWS of a sparse matrix so that a cosine similarity
        becomes a single sparse dot product, and we never materialise the full
        (n_movies x n_movies) similarity matrix (which would be ~95M entries).
        """
        R = self.data.rating_matrix if rating_matrix is None else rating_matrix
        R = R.astype(np.float32)
        mask = R > 0                                   # observed entries

        # per-item mean over OBSERVED ratings only
        sums = R.sum(axis=0)
        counts = mask.sum(axis=0)
        self.item_means = np.divide(
            sums, counts, out=np.zeros_like(sums), where=counts > 0
        )
        self.item_support = counts.astype(np.int32)    # ratings per item

        # centre observed ratings, leave unobserved as 0, then L2-normalise rows
        centred = np.where(mask, R - self.item_means, 0.0).T   # (n_movies x n_users)
        self.norm_items = normalize(csr_matrix(centred), norm="l2", axis=1)
        self._fitted = True
        return self

    def _similar_to(self, item_idx: int) -> np.ndarray:
        """Cosine similarity of one item against all items (dense vector)."""
        return (self.norm_items @ self.norm_items[item_idx].T).toarray().ravel()

    def predict_rating(self, user_id: int, movie_id: int) -> float | None:
        """Predict a single rating using the k most similar rated items."""
        self._check_fitted()
        if movie_id not in self.data.i_index or user_id not in self.data.u_index:
            return None
        i = self.data.i_index[movie_id]
        u = self.data.u_index[user_id]

        user_row = self.data.rating_matrix[u]
        rated = np.where(user_row > 0)[0]
        rated = rated[rated != i]
        if len(rated) == 0:
            return float(self.item_means[i])           # fall back to item mean

        sims = self._similar_to(i)[rated]
        # keep only the k most similar neighbours with positive similarity
        order = np.argsort(sims)[::-1]
        order = order[sims[order] > 0][: self.k]
        if len(order) == 0:
            return float(self.item_means[i])

        nb = rated[order]
        nb_sims = sims[order]
        nb_dev = user_row[nb] - self.item_means[nb]    # neighbour deviations
        pred = self.item_means[i] + np.dot(nb_sims, nb_dev) / np.abs(nb_sims).sum()
        # clamp into the valid 0.5–5.0 MovieLens range
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True,
                  refine: bool = True, candidate_pool: int = 60, min_support: int = 10):
        """Top-N recommendations: score every unseen movie, return the best.

        Two-stage design (fast candidate generation + accurate re-ranking):

        Stage 1 (fast, vectorised) — score ALL items at once with the closed
        form  scores = NormItems · ( NormItems.T · centred_user_ratings ),
        which is the un-normalised numerator of the prediction formula.  This is
        a great *ranking* signal but its magnitude is not on the rating scale.

        Stage 2 (accurate) — for the top `candidate_pool` items only, recompute
        the fully normalised `predict_rating` so the returned scores are genuine
        predicted ratings in 0.5–5.0 (used for display and for consistency with
        evaluation).  Set refine=False to skip stage 2 (pure ranking, faster).
        """
        self._check_fitted()
        if user_id not in self.data.u_index:
            return []
        u = self.data.u_index[user_id]
        user_row = self.data.rating_matrix[u]
        rated = np.where(user_row > 0)[0]
        if len(rated) == 0:
            return []

        # centred deviations of the user's ratings, as a sparse column vector
        dev = np.zeros(self.data.n_movies, dtype=np.float32)
        dev[rated] = user_row[rated] - self.item_means[rated]

        # aggregate over users then back over items  (see formula in header)
        user_space = self.norm_items.T @ dev           # length n_users
        scores = self.norm_items @ user_space          # length n_movies
        scores = self.item_means + scores              # add item baseline

        if exclude_seen:
            scores[rated] = -np.inf
        # Avoid promoting obscure items rated by only a handful of users
        # (memory-based CF inflates their scores) — keep items with enough support.
        if min_support > 1:
            scores[self.item_support < min_support] = -np.inf

        if not refine:
            order = np.argsort(scores)[::-1][:top_n]
            return [(self.data.movie_ids[k], float(scores[k])) for k in order
                    if np.isfinite(scores[k])]

        # Stage 2: re-rank a shortlist with the exact normalised prediction.
        pool = np.argsort(scores)[::-1][: max(candidate_pool, top_n)]
        pool = [k for k in pool if np.isfinite(scores[k])]
        refined = []
        for k in pool:
            pred = self.predict_rating(user_id, self.data.movie_ids[k])
            refined.append((self.data.movie_ids[k], pred if pred is not None else 0.0))
        refined.sort(key=lambda t: t[1], reverse=True)
        return refined[:top_n]

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before using ItemBasedCF.")


# =========================================================================== #
#  B.  FUNK-SVD  (latent-factor matrix factorization, SGD-trained)
# =========================================================================== #
class FunkSVD:
    """Biased matrix factorization trained with Stochastic Gradient Descent."""

    def __init__(
        self,
        data: MovieLensData,
        n_factors: int = 50,
        n_epochs: int = 30,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.data = data
        self.n_factors = n_factors      # k = size of the latent space
        self.n_epochs = n_epochs        # SGD passes over the data
        self.lr = lr                    # learning rate
        self.reg = reg                  # L2 regularisation strength (λ)
        self.seed = seed
        self.verbose = verbose
        # learned parameters
        self.mu = 0.0
        self.bu = self.bi = None
        self.P = self.Q = None
        self._fitted = False

    def fit(self, train_df=None) -> "FunkSVD":
        """Learn μ, biases and latent factors by SGD over observed ratings.

        Parameters
        ----------
        train_df : pd.DataFrame or None
            If given (userId, movieId, rating), train on exactly those rows —
            this is how the evaluator trains on the training split only.
            If None, train on the full rating log.
        """
        df = self.data.ratings if train_df is None else train_df
        rng = np.random.default_rng(self.seed)

        n_users, n_items = self.data.n_users, self.data.n_movies
        self.mu = float(df.rating.mean())
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        # small random init so latent dimensions start out distinct
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)

        # pre-map ids to matrix indices once (big speed-up over per-row lookups)
        u_arr = df.userId.map(self.data.u_index).to_numpy()
        i_arr = df.movieId.map(self.data.i_index).to_numpy()
        r_arr = df.rating.to_numpy(dtype=np.float32)
        # drop ratings whose movie is not in the catalogue (tag-only ids)
        good = ~np.isnan(i_arr.astype(float))
        u_arr, i_arr, r_arr = u_arr[good].astype(int), i_arr[good].astype(int), r_arr[good]

        order = np.arange(len(r_arr))
        for epoch in range(self.n_epochs):
            rng.shuffle(order)                          # SGD = shuffle each epoch
            sq_err = 0.0
            for idx in order:
                u, i, r = u_arr[idx], i_arr[idx], r_arr[idx]
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = r - pred
                sq_err += err * err

                # gradient-descent updates (move params against the gradient)
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu, qi = self.P[u].copy(), self.Q[i]
                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)

            if self.verbose:
                rmse = np.sqrt(sq_err / len(r_arr))
                print(f"  epoch {epoch + 1:2d}/{self.n_epochs}  train RMSE={rmse:.4f}")

        self._fitted = True
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float | None:
        """Predict r_hat(u, i) = μ + b_u + b_i + p_u · q_i (clamped to 0.5–5)."""
        self._check_fitted()
        if user_id not in self.data.u_index or movie_id not in self.data.i_index:
            return self.mu                               # global-average fallback
        u, i = self.data.u_index[user_id], self.data.i_index[movie_id]
        pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True):
        """Top-N: score all items for the user in one matrix multiply."""
        self._check_fitted()
        if user_id not in self.data.u_index:
            return []
        u = self.data.u_index[user_id]
        scores = self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]   # (n_movies,)
        if exclude_seen:
            seen = np.where(self.data.rating_matrix[u] > 0)[0]
            scores[seen] = -np.inf
        order = np.argsort(scores)[::-1][:top_n]
        return [(self.data.movie_ids[k], float(scores[k])) for k in order
                if np.isfinite(scores[k])]

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before using FunkSVD.")


if __name__ == "__main__":  # demo: `python -m src.collaborative`
    from .data_loader import load_data

    data = load_data()

    print("=== Item-based KNN ===")
    knn = ItemBasedCF(data, k=30).fit()
    print("Top-5 for user 1:")
    for mid, s in knn.recommend(1, top_n=5):
        print(f"  {s:.3f}  {data.title(mid)}")

    print("\n=== FunkSVD ===")
    svd = FunkSVD(data, n_factors=50, n_epochs=20, verbose=True).fit()
    print("Top-5 for user 1:")
    for mid, s in svd.recommend(1, top_n=5):
        print(f"  {s:.3f}  {data.title(mid)}")
