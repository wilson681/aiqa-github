"""
app_streamlit.py  —  MovieMind: interactive movie-recommender GUI
=================================================================
A Netflix-style web app for the recommender system. It is a real, usable
interface (not just a results dump):

  * 🔍  Search the movie catalogue
  * 👍  Like a movie  /  👎  mark "Not interested"  (recommendations update live)
  * ✨  "Build my own taste" — a brand-new user gets personalised picks from the
        movies they just liked (item cold-start, handled by Content-Based + KNN)
  * 👤  "Existing profile" — pick a user and compare all 4 algorithms side by side
  * 🎭  Filter recommendations by genre

Run it:
    pip install streamlit
    streamlit run app_streamlit.py

The models are trained once and cached with @st.cache_resource, so the UI stays
responsive. Everything runs offline from the bundled dataset.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.collaborative import FunkSVD, ItemBasedCF
from src.content_based import ContentBasedRecommender
from src.data_loader import load_data
from src.hybrid import HybridRecommender

LIKE_RATING = 5.0          # a 👍 is treated as a 5-star rating in the taste profile


# --------------------------------------------------------------------------- #
#  Model loading (cached once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="🍿 Loading movies and training recommenders ...")
def load_engine():
    data = load_data()
    content = ContentBasedRecommender(data).fit()
    knn = ItemBasedCF(data, k=30).fit()
    svd = FunkSVD(data, n_factors=50, n_epochs=20).fit()
    hybrid = HybridRecommender(data, cf_model="svd", alpha=0.7)
    hybrid.content, hybrid.cf, hybrid._fitted = content, svd, True
    # popularity (rating count) — used to seed the "Build my own taste" page
    pop = data.ratings.movieId.value_counts()
    return data, content, knn, svd, hybrid, pop


@st.cache_data
def all_genres(_data):
    g = set()
    for lst in _data.movies.genres_list:
        g.update(lst)
    return sorted(x for x in g if x)


# --------------------------------------------------------------------------- #
#  Session state helpers (the user's likes / not-interested set)
# --------------------------------------------------------------------------- #
def _init_state():
    st.session_state.setdefault("liked", {})        # {movieId: rating}
    st.session_state.setdefault("nope", set())      # {movieId} not interested


def like_movie(mid):
    mid = int(mid)
    st.session_state.liked[mid] = LIKE_RATING
    st.session_state.nope.discard(mid)


def not_interested(mid):
    mid = int(mid)
    st.session_state.nope.add(mid)
    st.session_state.liked.pop(mid, None)


def forget(mid):
    mid = int(mid)
    st.session_state.liked.pop(mid, None)
    st.session_state.nope.discard(mid)


# --------------------------------------------------------------------------- #
#  Recommendation routing
# --------------------------------------------------------------------------- #
def _genre_ok(data, mid, wanted):
    if not wanted:
        return True
    row = data.movies.loc[data.movies.movieId == mid, "genres_list"]
    return bool(row.iloc[0]) and any(g in row.iloc[0] for g in wanted)


def _hybrid_profile(content, knn, data, profile, exclude, n):
    """Blend normalised Content-Based + Item-KNN scores for an ad-hoc profile."""
    cb = dict(content.recommend_for_profile(profile, top_n=300, exclude_ids=exclude))
    ck = dict(knn.recommend_for_profile(profile, top_n=300, exclude_ids=exclude))
    ids = set(cb) | set(ck)
    if not ids:
        return []

    def norm(d):
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        return {k: (0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo)) for k, v in d.items()}

    cbn, ckn = norm(cb), norm(ck)
    blended = {i: 0.7 * ckn.get(i, 0.0) + 0.3 * cbn.get(i, 0.0) for i in ids}
    return sorted(blended.items(), key=lambda t: t[1], reverse=True)[:n]


def get_recommendations(engine, *, models, mode, profile, user_id, exclude, n, genres):
    """Return a list of (movieId, score) for the chosen engine/mode."""
    data, content, knn, svd, hybrid, _ = models
    buf = n + len(exclude) + 60          # over-fetch so genre/exclude filters still fill N

    if mode == "build":
        if engine == "Content-Based":
            recs = content.recommend_for_profile(profile, top_n=buf, exclude_ids=exclude)
        elif engine == "Item-KNN":
            recs = knn.recommend_for_profile(profile, top_n=buf, exclude_ids=exclude)
        else:  # Hybrid
            recs = _hybrid_profile(content, knn, data, profile, exclude, buf)
    else:  # existing user
        model = {"Content-Based": content, "Item-KNN": knn,
                 "FunkSVD": svd, "Hybrid": hybrid}[engine]
        recs = model.recommend(int(user_id), top_n=buf)

    out = []
    for mid, score in recs:
        if mid in exclude or mid in profile:
            continue
        if not _genre_ok(data, mid, genres):
            continue
        out.append((mid, score))
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
#  UI building blocks
# --------------------------------------------------------------------------- #
def movie_meta(data, mid):
    row = data.movies.loc[data.movies.movieId == mid].iloc[0]
    return row.title, (row.genres or "—"), row.year


def render_card(data, mid, *, score=None, score_label="match", key_ns=""):
    """One movie 'card' with Like / Not-interested actions."""
    title, genres, year = movie_meta(data, mid)
    with st.container(border=True):
        top = st.columns([6, 1, 1])
        with top[0]:
            st.markdown(f"**{title}**")
            st.caption(f"🎭 {genres}")
            if score is not None:
                st.caption(f"⭐ {score_label}: {score:.2f}")
        with top[1]:
            st.button("👍", key=f"like_{key_ns}_{mid}", help="I like this",
                      on_click=like_movie, args=(mid,))
        with top[2]:
            st.button("🚫", key=f"nope_{key_ns}_{mid}", help="Not interested",
                      on_click=not_interested, args=(mid,))


# --------------------------------------------------------------------------- #
#  Main app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="MovieMind", page_icon="🎬", layout="wide")
    _init_state()
    models = load_engine()
    data, content, knn, svd, hybrid, pop = models
    genres_list = all_genres(data)

    st.title("🎬 MovieMind")
    st.caption("A personal movie recommender · Content-Based · Collaborative (KNN + SVD) · Hybrid")

    # ---- sidebar: controls + the user's taste profile ---------------------
    with st.sidebar:
        st.header("⚙️ Settings")
        mode_label = st.radio(
            "Mode",
            ["✨ Build my own taste", "👤 Use an existing profile"],
        )
        mode = "build" if mode_label.startswith("✨") else "existing"

        if mode == "build":
            engine_options = ["Hybrid", "Content-Based", "Item-KNN"]
        else:
            engine_options = ["Hybrid", "FunkSVD", "Item-KNN", "Content-Based",
                              "🆚 Compare all"]
        engine = st.selectbox("Recommendation engine", engine_options)

        user_id = 1
        if mode == "existing":
            user_id = st.number_input("User id", min_value=int(data.user_ids.min()),
                                      max_value=int(data.user_ids.max()), value=1, step=1)

        n = st.slider("How many recommendations", 5, 20, 8)
        genre_filter = st.multiselect("🎭 Filter by genre", genres_list)

        st.divider()
        st.subheader("👍 My list")
        if st.session_state.liked:
            for mid in list(st.session_state.liked):
                c = st.columns([5, 1])
                c[0].write(data.title(mid))
                c[1].button("✕", key=f"un_{mid}", on_click=forget, args=(mid,))
        else:
            st.caption("Nothing yet — like some movies to build your taste.")

        st.subheader("🚫 Not interested")
        if st.session_state.nope:
            for mid in list(st.session_state.nope):
                c = st.columns([5, 1])
                c[0].caption(data.title(mid))
                c[1].button("↩", key=f"unn_{mid}", help="Undo",
                            on_click=forget, args=(mid,))
        else:
            st.caption("None.")

        st.divider()
        st.caption(f"📚 {data.n_users} users · {data.n_movies} movies · "
                   f"{len(data.ratings):,} ratings")

    # ---- main column: search + recommendations ----------------------------
    search_col, rec_col = st.columns([1, 1.3], gap="large")

    with search_col:
        st.subheader("🔍 Find movies")
        term = st.text_input("Search by title", placeholder="e.g. Toy Story, Matrix, Batman")
        if term:
            hits = data.search(term, limit=12)
            if hits.empty:
                st.info("No matches — try another title.")
            for mid in hits.movieId:
                render_card(data, int(mid), key_ns="search")
        elif mode == "build":
            st.caption("Or start from a popular movie:")
            for mid in pop.head(8).index:
                render_card(data, int(mid), key_ns="pop")

    with rec_col:
        profile = st.session_state.liked
        exclude = st.session_state.nope

        if mode == "build" and not profile:
            st.subheader("🍿 Recommended for you")
            st.info("👈 Search for movies you enjoy and press **👍 Like**. "
                    "Your personalised recommendations will appear here instantly.")
            return

        if mode == "existing" and engine == "🆚 Compare all":
            st.subheader(f"🆚 All methods for user {int(user_id)}")
            with st.expander("This user's favourite movies (their taste)"):
                hist = data.movies_rated_by(int(user_id)).head(8)
                for _, r in hist.iterrows():
                    st.write(f"⭐ {r.rating:.1f} — {data.title(int(r.movieId))}")
            cols = st.columns(2)
            for idx, eng in enumerate(["Content-Based", "Item-KNN", "FunkSVD", "Hybrid"]):
                with cols[idx % 2]:
                    st.markdown(f"**{eng}**")
                    recs = get_recommendations(eng, models=models, mode="existing",
                                               profile={}, user_id=user_id,
                                               exclude=exclude, n=n, genres=genre_filter)
                    for mid, sc in recs:
                        render_card(data, mid, score=sc, score_label="score", key_ns=eng)
            return

        st.subheader("🍿 Recommended for you")
        st.caption(f"Engine: **{engine}**"
                   + (f" · for user {int(user_id)}" if mode == "existing" else ""))
        recs = get_recommendations(engine, models=models, mode=mode, profile=profile,
                                   user_id=user_id, exclude=exclude, n=n, genres=genre_filter)
        if not recs:
            st.warning("No recommendations match the current filters. "
                       "Try removing the genre filter or liking more movies.")
        for mid, sc in recs:
            render_card(data, mid, score=sc, key_ns="rec")


if __name__ == "__main__":
    main()
