"""
app_cli.py  —  Interactive command-line demo (the "prototype" UI)
=================================================================
A menu-driven console application that lets a tutor try every recommender live
during the demo.  It is intentionally dependency-free (only the project modules)
so it always runs, satisfying the "no bugs during demonstration" criterion.

Run with:
    python app_cli.py

Features
--------
  1. Recommend for an existing user   (compare all 4 methods side by side)
  2. "More movies like this"          (content-based item-to-item)
  3. Cold-start: recommend from a movie you like (no account needed)
  4. Search the movie catalogue
  5. Predict the rating a user would give a movie (all methods)
  6. Show a user's rating history
"""

from __future__ import annotations

import sys

from src.collaborative import FunkSVD, ItemBasedCF
from src.content_based import ContentBasedRecommender
from src.data_loader import load_data
from src.hybrid import HybridRecommender


BANNER = r"""
============================================================
   MOVIE RECOMMENDER SYSTEM  —  AI Assignment (Topic 3)
   Methods: Content-Based | Collaborative (KNN + SVD) | Hybrid
============================================================
"""


class RecommenderApp:
    def __init__(self):
        print(BANNER)
        print("Loading dataset and training models (a few seconds) ...\n")
        self.data = load_data()
        self.content = ContentBasedRecommender(self.data).fit()
        self.knn = ItemBasedCF(self.data, k=30).fit()
        self.svd = FunkSVD(self.data, n_factors=50, n_epochs=20).fit()
        self.hybrid = HybridRecommender(self.data, cf_model="svd", alpha=0.7)
        # reuse already-trained base models inside the hybrid
        self.hybrid.content, self.hybrid.cf, self.hybrid._fitted = self.content, self.svd, True
        print(f"Ready!  {self.data.n_users} users, {self.data.n_movies} movies, "
              f"{len(self.data.ratings)} ratings loaded.\n")

    # ------------------------------------------------------------------ #
    #  small input helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ask_int(prompt: str, default: int | None = None) -> int | None:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  (!) Please enter a number.")
            return None

    def _show(self, recs, header: str):
        print(f"\n  {header}")
        print("  " + "-" * len(header))
        if not recs:
            print("  (no recommendations — user/movie unknown or no history)")
            return
        for rank, (mid, score) in enumerate(recs, 1):
            genres = self.data.movies.loc[
                self.data.movies.movieId == mid, "genres"].iloc[0]
            print(f"   {rank:2d}. {self.data.title(mid):<48} "
                  f"[{score:5.2f}]  {genres}")

    # ------------------------------------------------------------------ #
    #  menu actions
    # ------------------------------------------------------------------ #
    def recommend_for_user(self):
        uid = self._ask_int(f"\nEnter user id (1-{self.data.n_users}): ")
        if uid is None:
            return
        if uid not in self.data.u_index:
            print("  (!) Unknown user.")
            return
        n = self._ask_int("How many recommendations? [10]: ", default=10)

        print(f"\n>>> Comparing all methods for user {uid} <<<")
        self._show(self.content.recommend(uid, top_n=n), "1) CONTENT-BASED (genres/tags)")
        self._show(self.knn.recommend(uid, top_n=n), "2) COLLABORATIVE — Item-KNN")
        self._show(self.svd.recommend(uid, top_n=n), "3) COLLABORATIVE — FunkSVD")
        self._show(self.hybrid.recommend(uid, top_n=n), "4) HYBRID (CBF + SVD)")

    def more_like_this(self):
        self._search_then(lambda mid: self._show(
            self.content.similar_movies(mid, top_n=10),
            f"Movies most similar to '{self.data.title(mid)}'"))

    def cold_start(self):
        print("\nCold-start: get recommendations from ONE movie you like "
              "(no account needed).")
        self._search_then(lambda mid: self._show(
            self.content.recommend_from_movie(mid, top_n=10),
            f"Because you like '{self.data.title(mid)}', you may enjoy"))

    def search_catalogue(self):
        term = input("\nSearch title contains: ").strip()
        if not term:
            return
        hits = self.data.search(term, limit=15)
        if hits.empty:
            print("  (no matches)")
            return
        print()
        for _, r in hits.iterrows():
            print(f"   {r.movieId:>6}  {r.title:<48} {r.genres}")

    def predict_rating(self):
        uid = self._ask_int(f"\nEnter user id (1-{self.data.n_users}): ")
        if uid is None or uid not in self.data.u_index:
            print("  (!) Unknown user.")
            return
        self._search_then(lambda mid: self._print_predictions(uid, mid))

    def _print_predictions(self, uid: int, mid: int):
        print(f"\n  Predicted rating of user {uid} for "
              f"'{self.data.title(mid)}':")
        for name, model in [("Content-Based", self.content), ("Item-KNN", self.knn),
                            ("FunkSVD", self.svd), ("Hybrid", self.hybrid)]:
            pred = model.predict_rating(uid, mid)
            shown = f"{pred:.2f} / 5.00" if pred is not None else "n/a (cannot score)"
            print(f"    {name:<14}: {shown}")

    def show_history(self):
        uid = self._ask_int(f"\nEnter user id (1-{self.data.n_users}): ")
        if uid is None or uid not in self.data.u_index:
            print("  (!) Unknown user.")
            return
        hist = self.data.movies_rated_by(uid).head(15)
        print(f"\n  Top-rated movies of user {uid}:")
        for _, r in hist.iterrows():
            print(f"    {r.rating:>3} *  {self.data.title(int(r.movieId))}")

    # ------------------------------------------------------------------ #
    def _search_then(self, action):
        """Search for a movie, let the user pick one, then run `action(mid)`."""
        term = input("Search the movie by title: ").strip()
        if not term:
            return
        hits = self.data.search(term, limit=10)
        if hits.empty:
            print("  (no matches)")
            return
        rows = list(hits.itertuples(index=False))
        for i, r in enumerate(rows, 1):
            print(f"   {i}. {r.title}  [{r.genres}]")
        choice = self._ask_int("Pick a number (or Enter for #1): ", default=1)
        if choice is None or not (1 <= choice <= len(rows)):
            print("  (!) Invalid choice.")
            return
        action(int(rows[choice - 1].movieId))

    # ------------------------------------------------------------------ #
    def run(self):
        menu = {
            "1": ("Recommend for an existing user (compare all methods)", self.recommend_for_user),
            "2": ("More movies like this (content-based)", self.more_like_this),
            "3": ("Cold-start: recommend from a movie I like", self.cold_start),
            "4": ("Search the movie catalogue", self.search_catalogue),
            "5": ("Predict a user's rating for a movie", self.predict_rating),
            "6": ("Show a user's rating history", self.show_history),
            "0": ("Quit", None),
        }
        while True:
            print("\n" + "=" * 60)
            for key, (label, _) in menu.items():
                print(f"  {key}. {label}")
            choice = input("Choose an option: ").strip()
            if choice == "0":
                print("Goodbye!")
                return
            action = menu.get(choice, (None, None))[1]
            if action is None:
                print("  (!) Invalid option.")
                continue
            try:
                action()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return
            except Exception as exc:                 # never crash during a demo
                print(f"  (!) Something went wrong: {exc}")


if __name__ == "__main__":
    try:
        RecommenderApp().run()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)
