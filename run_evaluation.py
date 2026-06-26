"""
run_evaluation.py
=================
End-to-end experiment that trains every recommender on the SAME train split and
compares them on the SAME test split — this is the evidence behind the
"Results & Discussion" section of the documentation.

Run it with:

    python run_evaluation.py                # full evaluation
    python run_evaluation.py --quick        # smaller sample, runs in seconds

It prints two tables:
    1. Rating-prediction accuracy (RMSE / MAE / coverage)
    2. Top-N ranking quality      (Precision@K / Recall@K / F1@K / HitRate)
"""

from __future__ import annotations

import argparse
import time

from src.collaborative import FunkSVD, ItemBasedCF
from src.content_based import ContentBasedRecommender
from src.data_loader import load_data, train_test_split_ratings
from src.evaluation import (
    build_train_data,
    format_table,
    rating_metrics,
    ranking_metrics,
)
from src.hybrid import HybridRecommender


def main():
    parser = argparse.ArgumentParser(description="Compare recommender methods.")
    parser.add_argument("--quick", action="store_true",
                        help="Sample fewer test pairs/users for a fast run.")
    parser.add_argument("--k", type=int, default=10, help="Top-N cut-off K.")
    parser.add_argument("--min-movie-ratings", type=int, default=5,
                        help="Drop movies with fewer than this many ratings "
                             "to reduce noise (default 5).")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading + preprocessing data ...")
    data = load_data(min_movie_ratings=args.min_movie_ratings)
    print(f"  users={data.n_users}  movies={data.n_movies}  ratings={len(data.ratings)}")

    print("Creating per-user train/test split (80/20) ...")
    train_df, test_df = train_test_split_ratings(data.ratings, test_size=0.2, seed=42)
    train_data = build_train_data(data, train_df)
    print(f"  train={len(train_df)}  test={len(test_df)}")

    # sampling caps keep the --quick run fast without changing the conclusions
    max_samples = 3000 if args.quick else None
    max_users = 120 if args.quick else None

    # ------------------------------------------------------------------ #
    #  Train every model on the TRAIN split only
    # ------------------------------------------------------------------ #
    print("\nTraining models on the training split ...")
    models = {}

    print("  - Content-Based (TF-IDF + cosine)")
    models["Content-Based"] = ContentBasedRecommender(train_data).fit()

    print("  - Item-based KNN")
    models["Item-KNN"] = ItemBasedCF(train_data, k=30).fit()

    print("  - FunkSVD (matrix factorization)")
    models["FunkSVD"] = FunkSVD(train_data, n_factors=50, n_epochs=30).fit(train_df)

    print("  - Hybrid (CBF + FunkSVD, alpha=0.7)")
    hybrid = HybridRecommender(train_data, cf_model="svd", alpha=0.7)
    hybrid.content = models["Content-Based"]      # reuse already-trained bases
    hybrid.cf = models["FunkSVD"]
    hybrid._fitted = True
    models["Hybrid"] = hybrid

    # ------------------------------------------------------------------ #
    #  1. Rating-prediction accuracy
    # ------------------------------------------------------------------ #
    print("\nEvaluating rating prediction (RMSE / MAE) ...")
    rating_rows = []
    for name, model in models.items():
        m = rating_metrics(model, test_df, max_samples=max_samples)
        rating_rows.append({"Method": name, **m})
        print(f"  {name:<14} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"coverage={m['coverage']:.2%}")

    # ------------------------------------------------------------------ #
    #  2. Top-N ranking quality
    # ------------------------------------------------------------------ #
    print(f"\nEvaluating Top-{args.k} ranking (Precision/Recall/F1) ...")
    # ranking only needs ordering, so use the faster non-refined KNN ranking
    rec_kwargs = {
        "Item-KNN": {"refine": False},
    }
    ranking_rows = []
    for name, model in models.items():
        m = ranking_metrics(
            model, train_df, test_df, k=args.k, like_threshold=4.0,
            max_users=max_users, recommend_kwargs=rec_kwargs.get(name, {}),
        )
        ranking_rows.append({"Method": name, **m})
        print(f"  {name:<14} P@{args.k}={m[f'Precision@{args.k}']:.4f}  "
              f"R@{args.k}={m[f'Recall@{args.k}']:.4f}  "
              f"F1@{args.k}={m[f'F1@{args.k}']:.4f}  HitRate={m['HitRate']:.2%}")

    # ------------------------------------------------------------------ #
    #  Final formatted tables (copy these into the report)
    # ------------------------------------------------------------------ #
    print("\n\n" + format_table(
        rating_rows, ["RMSE", "MAE", "coverage"],
        title="TABLE 1 — Rating Prediction Accuracy (lower is better)",
    ))
    print("\n\n" + format_table(
        ranking_rows,
        [f"Precision@{args.k}", f"Recall@{args.k}", f"F1@{args.k}", "HitRate"],
        title=f"TABLE 2 — Top-{args.k} Ranking Quality (higher is better)",
    ))

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
