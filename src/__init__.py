"""
Movie Recommender System — AI Assignment (Topic 3: Recommender System).

This package implements and compares TWO recommender-system methods
(one per group member) plus a hybrid that combines them:

    Member 1  ->  Content-Based Filtering   (src.content_based)
    Member 2  ->  Collaborative Filtering    (src.collaborative)
    Joint     ->  Hybrid Recommender         (src.hybrid)

Supporting modules:
    src.data_loader  -> load + preprocess the MovieLens dataset
    src.evaluation   -> offline evaluation metrics (RMSE/MAE, Precision/Recall/F1@K)
"""

__all__ = [
    "data_loader",
    "content_based",
    "collaborative",
    "hybrid",
    "evaluation",
]
