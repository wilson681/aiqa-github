"""
data_loader.py
==============
Data ingestion and pre-processing for the MovieLens (ml-latest-small) dataset.

This is STEP 1 of the recommender pipeline. Every model in this project consumes
the objects produced here, so the data is loaded and cleaned exactly once.

What this module does
---------------------
1.  Load the three raw CSV files (movies, ratings, tags).
2.  Clean the movie metadata:
        * split the "Adventure|Animation|Children" genre string into tokens,
        * pull the release year out of the title,
        * aggregate the free-text tags belonging to each movie.
3.  Build a single text field per movie (the "content soup") that the
    content-based model turns into a TF-IDF vector.
4.  Build the dense user-item rating matrix used by the collaborative models.
5.  Provide a reproducible per-user train/test split for offline evaluation.

The class `MovieLensData` is the single object passed around the whole project.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Absolute path to the bundled dataset (…/<repo>/data) so the code works no
# matter which directory the user launches Python from.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class MovieLensData:
    """Container holding every cleaned data structure the models need.

    Attributes
    ----------
    movies : pd.DataFrame
        movieId, title, year, genres (list), tags (str), content (str = "soup").
    ratings : pd.DataFrame
        userId, movieId, rating, timestamp (the full rating log).
    user_ids / movie_ids : np.ndarray
        Sorted unique ids — define the row/column order of the rating matrix.
    u_index / i_index : dict
        Maps a raw userId / movieId to its 0-based row / column position.
    rating_matrix : np.ndarray  (n_users x n_movies)
        Dense matrix of ratings; 0.0 marks an *unobserved* (user, movie) pair.
    """

    movies: pd.DataFrame
    ratings: pd.DataFrame
    user_ids: np.ndarray
    movie_ids: np.ndarray
    u_index: dict
    i_index: dict
    rating_matrix: np.ndarray = field(repr=False)

    # ---- small convenience helpers used everywhere in the project ----------
    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)

    def title(self, movie_id: int) -> str:
        """Human-readable title for a raw movieId (used by the UI)."""
        row = self.movies.loc[self.movies.movieId == movie_id, "title"]
        return row.iloc[0] if len(row) else f"<unknown movieId {movie_id}>"

    def movies_rated_by(self, user_id: int) -> pd.DataFrame:
        """All (movieId, rating) pairs for one user, highest rating first."""
        sub = self.ratings[self.ratings.userId == user_id]
        return sub.sort_values("rating", ascending=False)

    def search(self, text: str, limit: int = 10) -> pd.DataFrame:
        """Case-insensitive substring search over movie titles (for the UI)."""
        mask = self.movies.title.str.contains(re.escape(text), case=False, na=False)
        return self.movies.loc[mask, ["movieId", "title", "genres"]].head(limit)


# --------------------------------------------------------------------------- #
#  Loading + cleaning
# --------------------------------------------------------------------------- #
def _extract_year(title: str) -> str:
    """Pull a 4-digit release year out of a title like 'Toy Story (1995)'."""
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return m.group(1) if m else ""


def load_data(data_dir: str = DATA_DIR, min_movie_ratings: int = 0) -> MovieLensData:
    """Read the CSVs, clean them, and assemble a `MovieLensData` object.

    Parameters
    ----------
    data_dir : str
        Folder containing movies.csv / ratings.csv / tags.csv.
    min_movie_ratings : int
        Optionally drop movies with fewer than this many ratings.  Kept at 0 by
        default so the demo shows the full catalogue; evaluation may raise it to
        reduce noise from movies that were rated only once or twice.
    """
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    tags = pd.read_csv(os.path.join(data_dir, "tags.csv"))

    # ---- 1. optional popularity filter -----------------------------------
    if min_movie_ratings > 0:
        counts = ratings.movieId.value_counts()
        keep = counts[counts >= min_movie_ratings].index
        ratings = ratings[ratings.movieId.isin(keep)]
        movies = movies[movies.movieId.isin(keep)]

    # ---- 2. clean the movie metadata -------------------------------------
    movies = movies.copy()
    movies["year"] = movies.title.map(_extract_year)
    # "(no genres listed)" is MovieLens' placeholder for missing genres.
    movies["genres"] = movies.genres.replace("(no genres listed)", "")
    movies["genres_list"] = movies.genres.map(lambda g: g.split("|") if g else [])

    # Aggregate every free-text tag belonging to a movie into one lowercase
    # string, e.g. "funny highly quotable will ferrell".
    tag_agg = (
        tags.assign(tag=tags.tag.astype(str).str.lower())
        .groupby("movieId")["tag"]
        .apply(lambda s: " ".join(s))
        .rename("tags")
    )
    movies = movies.merge(tag_agg, on="movieId", how="left")
    movies["tags"] = movies["tags"].fillna("")

    # ---- 3. build the "content soup" used by the content-based model -----
    # Genres are repeated twice so they carry more weight than sparse tags.
    def soup(row) -> str:
        genres = " ".join(row.genres_list)
        return f"{genres} {genres} {row.tags}".strip().lower()

    movies["content"] = movies.apply(soup, axis=1)
    movies = movies.reset_index(drop=True)

    # ---- 4. build id<->index maps and the dense rating matrix ------------
    user_ids = np.sort(ratings.userId.unique())
    movie_ids = np.sort(movies.movieId.unique())
    u_index = {uid: i for i, uid in enumerate(user_ids)}
    i_index = {mid: j for j, mid in enumerate(movie_ids)}

    rating_matrix = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)
    # Vectorised fill: map every rating row to (row, col) then scatter it in.
    rows = ratings.userId.map(u_index).to_numpy()
    cols = ratings.movieId.map(i_index)
    valid = cols.notna().to_numpy()                       # ignore tag-only movies
    rating_matrix[rows[valid], cols[valid].astype(int)] = ratings.rating.to_numpy()[valid]

    return MovieLensData(
        movies=movies,
        ratings=ratings.reset_index(drop=True),
        user_ids=user_ids,
        movie_ids=movie_ids,
        u_index=u_index,
        i_index=i_index,
        rating_matrix=rating_matrix,
    )


def train_test_split_ratings(
    ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42, min_train: int = 5
):
    """Per-user hold-out split for offline evaluation.

    Standard random splitting can place *all* of a user's ratings in the test
    set, leaving the models with nothing to learn that user's taste from.  To
    avoid that, we split *within each user*: a fraction of every user's ratings
    is held out for testing while keeping at least `min_train` in the training
    set.  This mirrors how a deployed system always has some history per user.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for _, group in ratings.groupby("userId"):
        idx = group.index.to_numpy().copy()        # .copy() -> writable for shuffle
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        # never hold out so much that the user has < min_train training ratings
        n_test = min(n_test, max(0, len(idx) - min_train))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_df = ratings.loc[train_idx].reset_index(drop=True)
    test_df = ratings.loc[test_idx].reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":  # quick smoke test: `python -m src.data_loader`
    data = load_data()
    print(f"Users:  {data.n_users}")
    print(f"Movies: {data.n_movies}")
    print(f"Ratings:{len(data.ratings)}")
    density = (data.rating_matrix > 0).mean() * 100
    print(f"Matrix density: {density:.2f}%  (rest are unobserved)")
    print("\nExample 'content soup' for Toy Story:")
    print(" ", data.movies.loc[data.movies.movieId == 1, "content"].iloc[0])
