# 🎬 Movie Recommender System — AI Assignment (Topic 3)

A movie recommender system that implements and **compares two recommender
methods (one per group member)** plus a hybrid that combines them, evaluated on
the **MovieLens (ml-latest-small)** dataset.

| Member | Method | File |
|--------|--------|------|
| **Member 1** | **Content-Based Filtering** (TF-IDF + cosine similarity) | `src/content_based.py` |
| **Member 2** | **Collaborative Filtering** (Item-based KNN + FunkSVD matrix factorization) | `src/collaborative.py` |
| *Joint (extra)* | **Hybrid** (weighted blend + cold-start switch) | `src/hybrid.py` |

---

## 1. Quick start

```bash
# 1. install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# 2. run the interactive demo (the prototype)
python app_cli.py

# 3. reproduce the evaluation / comparison tables
python run_evaluation.py            # full run
python run_evaluation.py --quick    # fast sampled run (seconds)

# 4. (optional) nicer web UI
pip install streamlit
streamlit run app_streamlit.py
```

The dataset is bundled in `data/`, so everything runs **offline** — nothing is
downloaded at runtime.

---

## 2. Project structure

```
.
├── data/                     # MovieLens ml-latest-small (bundled, see citation)
│   ├── movies.csv  ratings.csv  tags.csv  links.csv
│   └── MOVIELENS_README.txt  # original dataset readme + license + citation
├── src/
│   ├── data_loader.py        # STEP 1: load + preprocess + train/test split
│   ├── content_based.py      # Member 1: TF-IDF + cosine
│   ├── collaborative.py      # Member 2: Item-KNN + FunkSVD
│   ├── hybrid.py             # weighted hybrid (extra effort)
│   └── evaluation.py         # RMSE/MAE + Precision/Recall/F1@K
├── app_cli.py                # interactive console prototype  <- main UI
├── app_streamlit.py          # optional web UI
├── run_evaluation.py         # trains all models + prints comparison tables
├── docs/
│   ├── Documentation.md      # Part 1 report (rubric-aligned)
│   └── evaluation_results.txt# saved output of run_evaluation.py
├── requirements.txt
└── README.md
```

---

## 3. The pipeline at a glance

```
 raw CSVs ─▶ data_loader ─▶  ┌─ Content-Based (TF-IDF + cosine) ─┐
 (movies,     (clean, build  ├─ Item-KNN (centred cosine)        ├─▶ evaluation
  ratings,     user-item     ├─ FunkSVD (latent factors, SGD)    │   (RMSE/MAE,
  tags)        matrix)       └─ Hybrid (α·CF + (1-α)·CBF)        ┘    P/R/F1@K)
```

Each module's docstring explains its algorithm in detail. Run any module
standalone to see a mini-demo, e.g. `python -m src.content_based`.

---

## 4. Methods in one paragraph each

- **Content-Based Filtering** — Builds a TF-IDF vector from each movie's
  genres + user tags, measures movie-to-movie closeness with cosine similarity,
  and recommends movies similar to the ones a user already rated highly.
  *Strength:* explainable, no cold-start for new items. *Weakness:*
  over-specialises to known genres.

- **Collaborative Filtering** — Learns from the **ratings matrix** only.
  *Item-KNN* predicts a rating from the user's ratings of the most similar items
  (centred-cosine similarity). *FunkSVD* factorises the matrix into user/item
  latent factors with biases, trained by SGD (the Netflix-Prize approach).
  *Strength:* finds cross-genre patterns. *Weakness:* cold-start for new
  users/items.

- **Hybrid** — Blends the normalised scores of both
  (`α·CF + (1-α)·CBF`, α = 0.7) and falls back to content-based when
  collaborative filtering cannot score an item (cold-start switch).

---

## 5. Results summary

See `docs/Documentation.md` (Section: Results & Discussion) and
`docs/evaluation_results.txt` for the full tables produced by
`run_evaluation.py`. Headline finding: collaborative methods predict ratings
more accurately than content-based filtering, FunkSVD gives the best Top-N
precision, and the hybrid is the most robust across both tasks.

---

## 6. Dataset & citation

This project uses the **MovieLens ml-latest-small** dataset (100 836 ratings,
9 742 movies, 610 users) by GroupLens Research, redistributed here for
educational use under its license (see `data/MOVIELENS_README.txt`).

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets:
> History and Context.* ACM Transactions on Interactive Intelligent Systems
> (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

---

## 7. AI-use disclosure

AI assistance (an AI coding assistant) was used to help scaffold and document
this codebase. All algorithms were reviewed, executed, and verified by the group
members; the evaluation numbers are reproducible via `run_evaluation.py`.
See Appendix B in `docs/Documentation.md`.
