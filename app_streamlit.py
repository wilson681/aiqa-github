"""
app_streamlit.py  —  Optional web UI (nicer-looking prototype for the demo)
===========================================================================
A small Streamlit dashboard wrapping the same recommender models as the CLI.
It is OPTIONAL: the CLI (app_cli.py) is the primary prototype and needs no extra
libraries.  Use this for a more visual demonstration.

Install + run:
    pip install streamlit
    streamlit run app_streamlit.py

The models are cached with @st.cache_resource so they train only once per
session, keeping the UI responsive.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.collaborative import FunkSVD, ItemBasedCF
from src.content_based import ContentBasedRecommender
from src.data_loader import load_data
from src.hybrid import HybridRecommender


@st.cache_resource(show_spinner="Loading data and training models ...")
def load_models():
    """Load data + train all models once, cached for the whole session."""
    data = load_data()
    content = ContentBasedRecommender(data).fit()
    knn = ItemBasedCF(data, k=30).fit()
    svd = FunkSVD(data, n_factors=50, n_epochs=20).fit()
    hybrid = HybridRecommender(data, cf_model="svd", alpha=0.7)
    hybrid.content, hybrid.cf, hybrid._fitted = content, svd, True
    return data, content, knn, svd, hybrid


def recs_to_df(data, recs, score_label="Score"):
    """Turn a list of (movieId, score) into a display DataFrame."""
    rows = []
    for rank, (mid, score) in enumerate(recs, 1):
        genres = data.movies.loc[data.movies.movieId == mid, "genres"].iloc[0]
        rows.append({"#": rank, "Title": data.title(mid),
                     "Genres": genres, score_label: round(score, 3)})
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Movie Recommender System", page_icon="🎬",
                       layout="wide")
    st.title("🎬 Movie Recommender System")
    st.caption("AI Assignment — Topic 3 · Content-Based vs Collaborative vs Hybrid")

    data, content, knn, svd, hybrid = load_models()

    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Mode", ["Recommend for a user",
                                 "More like a movie (cold-start)"])
        top_n = st.slider("Number of recommendations", 5, 20, 10)
        st.divider()
        st.metric("Users", data.n_users)
        st.metric("Movies", data.n_movies)
        st.metric("Ratings", len(data.ratings))

    if mode == "Recommend for a user":
        uid = st.number_input("User id", min_value=int(data.user_ids.min()),
                              max_value=int(data.user_ids.max()), value=1, step=1)
        with st.expander("This user's top-rated movies (their taste profile)"):
            hist = data.movies_rated_by(int(uid)).head(10)
            hist = hist.assign(title=hist.movieId.map(data.title))
            st.dataframe(hist[["title", "rating"]], hide_index=True,
                         use_container_width=True)

        if st.button("Recommend", type="primary"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Content-Based")
                st.dataframe(recs_to_df(data, content.recommend(int(uid), top_n),
                             "Affinity"), hide_index=True, use_container_width=True)
                st.subheader("Collaborative — FunkSVD")
                st.dataframe(recs_to_df(data, svd.recommend(int(uid), top_n),
                             "Pred."), hide_index=True, use_container_width=True)
            with c2:
                st.subheader("Collaborative — Item-KNN")
                st.dataframe(recs_to_df(data, knn.recommend(int(uid), top_n),
                             "Pred."), hide_index=True, use_container_width=True)
                st.subheader("Hybrid (CBF + SVD)")
                st.dataframe(recs_to_df(data, hybrid.recommend(int(uid), top_n),
                             "Blend"), hide_index=True, use_container_width=True)

    else:
        term = st.text_input("Search a movie you like", "Toy Story")
        if term:
            hits = data.search(term, limit=20)
            if hits.empty:
                st.warning("No matching movies.")
            else:
                choice = st.selectbox("Pick the movie", hits.title.tolist())
                mid = int(hits.loc[hits.title == choice, "movieId"].iloc[0])
                st.subheader(f"Because you like “{choice}”")
                st.dataframe(recs_to_df(data, content.similar_movies(mid, top_n),
                             "Similarity"), hide_index=True,
                             use_container_width=True)


if __name__ == "__main__":
    main()
