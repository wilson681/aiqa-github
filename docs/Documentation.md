# A Comparative Movie Recommender System: Content-Based Filtering vs. Collaborative Filtering with a Hybrid Extension

**Course:** Artificial Intelligence (Session 202605)
**Assignment Topic:** Topic 3 — Recommender System
**Group size:** 2 members

| Role | Name | Student ID | Method implemented |
|------|------|-----------|--------------------|
| Member 1 (Leader) | _________________ | __________ | Content-Based Filtering (TF-IDF + cosine) |
| Member 2 | _________________ | __________ | Collaborative Filtering (Item-KNN + FunkSVD) |

> _Fill in the names and IDs above before submission. The hybrid model (Section 3.6) was built jointly as extra effort toward an "Excellent" grade._

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Methodology](#3-methodology)
4. [Results & Discussion](#4-results--discussion)
5. [Conclusion](#5-conclusion)
6. [References](#6-references)
7. [Appendix A — Plagiarism Statement Form](#appendix-a--plagiarism-statement-form)
8. [Appendix B — AI Disclosure Statement](#appendix-b--ai-disclosure-statement)
9. [Appendix C — Student Free-Rider Report Form](#appendix-c--student-free-rider-report-form)

---

## 1. Introduction

### 1.1 Background
The amount of content available on digital platforms has grown far faster than
any individual's capacity to browse it. A streaming catalogue can hold tens of
thousands of titles, yet a user is willing to scroll through only a handful
before losing interest — a problem widely described as *information overload*
(Ricci, Rokach, & Shapira, 2015). **Recommender systems** address this by
learning each user's preferences and surfacing a short, personalised list of
items they are most likely to enjoy. They are now a core component of services
such as Netflix, Amazon, YouTube and Spotify, where a large share of consumption
is driven by recommendations rather than active search (Gomez-Uribe & Hunt,
2016).

Two families of techniques dominate the field. **Content-based filtering**
recommends items whose *attributes* resemble those the user has liked before,
while **collaborative filtering** recommends items that *like-minded users* have
enjoyed, using only the matrix of past ratings (Aggarwal, 2016). Each family has
well-documented strengths and weaknesses, which motivates the comparative study
carried out in this project.

### 1.2 Problem Statement
Given a historical log of how users have rated movies, the system must answer two
related questions for any target user:

1. **Rating prediction:** *How many stars would this user give to a movie they
   have not yet seen?*
2. **Top-N recommendation:** *Which N unseen movies should we put in front of
   this user?*

No single algorithm is uniformly best. Content-based methods over-specialise and
cannot surprise the user; collaborative methods are powerful but fail when data
is sparse or when a user/item is new (the *cold-start* problem) (Schein et al.,
2002). The problem this project tackles is therefore not only to *build* a
recommender, but to **fairly compare** distinct techniques on the same data and
to investigate whether a **hybrid** can combine their strengths.

### 1.3 Objectives
1. To pre-process a real-world movie-rating dataset (MovieLens) into a form
   suitable for both content-based and collaborative recommendation.
2. To design and implement **two independent recommender methods**, one per group
   member:
   - Member 1 — Content-Based Filtering (TF-IDF + cosine similarity).
   - Member 2 — Collaborative Filtering (memory-based Item-KNN and model-based
     FunkSVD matrix factorization).
3. To implement a **hybrid** model that blends both, including cold-start
   handling (extra effort).
4. To evaluate and **compare** the methods using standard offline metrics —
   RMSE/MAE for rating prediction and Precision/Recall/F1@K for Top-N ranking.
5. To deliver a **working prototype** (interactive CLI + optional web UI) that
   demonstrates the system live.

### 1.4 Significance
A clear, reproducible comparison helps practitioners choose the right technique
for their data conditions: content-based filtering when item metadata is rich
but rating history is thin, collaborative filtering when ratings are plentiful,
and a hybrid when robustness across both situations matters. Educationally, the
project demonstrates the full data-science workflow — preprocessing, modelling,
evaluation, and deployment of a usable prototype — applied to an AI problem of
real commercial value.

### 1.5 Research Gap
Many student and tutorial implementations present a *single* algorithm in
isolation, report a single accuracy figure, and never test the cold-start
behaviour that breaks recommenders in practice. They also frequently conflate the
two distinct tasks (rating prediction vs. ranking), reporting RMSE while claiming
to improve recommendations. This project addresses that gap by (i) implementing
**three contrasting approaches under one evaluation harness**, (ii) reporting
**both** rating-accuracy and ranking metrics, and (iii) explicitly handling
cold-start through the hybrid's switching logic.

---

## 2. Related Work

### 2.1 Content-Based Filtering
Content-based recommenders model a user from the features of items they have
consumed. The classic representation borrows from information retrieval: items
are described by weighted term vectors and compared with the cosine measure
(Salton & Buckley, 1988). Pazzani and Billsus (2007) and Lops, de Gemmis, and
Semeraro (2011) survey the field and identify two persistent weaknesses —
**over-specialisation** (the system keeps recommending the same kind of item)
and **limited content analysis** (recommendations are only as good as the
available metadata). Their strength, conversely, is graceful handling of the
*item* cold-start: a new movie can be recommended from its genres alone, before
anyone has rated it. **Our work** adopts the TF-IDF + cosine formulation over a
combined genre-and-tag representation, and confirms the over-specialisation
weakness empirically (Section 4).

### 2.2 Collaborative Filtering — Memory-Based
Collaborative filtering originated with the GroupLens project (Resnick et al.,
1994) and the Tapestry system (Goldberg et al., 1992). Sarwar, Karypis, Konstan,
and Riedl (2001) introduced **item-based** collaborative filtering, showing that
computing similarities between *items* (rather than users) is both more scalable
and more stable, because item-item relationships change slowly. They recommend
*adjusted/centred cosine* similarity to remove the effect of differing user and
item rating scales. **Our work** implements exactly this centred-cosine item-KNN
and reproduces its strong rating-prediction accuracy, while also documenting its
tendency to inflate the scores of obscure, sparsely-rated items — which we
mitigate with a minimum-support filter.

### 2.3 Collaborative Filtering — Model-Based (Matrix Factorization)
The Netflix Prize (2006–2009) established **latent-factor matrix factorization**
as the state of the art for rating prediction. Simon Funk's blog-published
"FunkSVD" (Funk, 2006) trains user and item factor vectors by stochastic gradient
descent over the *observed* entries only, avoiding the cost of imputing the
millions of missing values. Koren, Bell, and Volinsky (2009) formalised and
extended this with **bias terms** and regularisation, the formulation we adopt.
Matrix factorization generalises better than memory-based CF on sparse data and
is far cheaper at prediction time, but — like all pure CF — it cannot place a
brand-new user or item in the latent space (user/item cold-start). **Our work**
implements biased, regularised FunkSVD from scratch in NumPy and confirms its
strong Top-N precision.

### 2.4 Hybrid Recommenders
Burke (2002) provides the canonical taxonomy of **hybrid** recommenders —
weighted, switching, mixed, feature-combination, cascade, and others — arguing
that combining methods can cancel out their individual weaknesses, particularly
cold-start. **Our work** implements a *weighted* hybrid (a linear blend of the
normalised content and collaborative scores) augmented with a *switching*
fallback to content-based filtering whenever collaborative filtering cannot score
an item, directly targeting the cold-start gap identified in Section 2.1–2.3.

### 2.5 Summary of the Gap Addressed
| Prior work | Contribution | Limitation we address |
|-----------|--------------|-----------------------|
| Pazzani & Billsus (2007); Lops et al. (2011) | Content-based methods | Over-specialise; we quantify this and offset it with a hybrid |
| Sarwar et al. (2001) | Item-based CF | Sparse-item inflation; we add a support filter |
| Funk (2006); Koren et al. (2009) | Matrix factorization | User/item cold-start; we add a content fallback |
| Burke (2002) | Hybrid taxonomy | Mostly conceptual; we give a concrete, evaluated implementation |

The novelty of this project is **not** a new algorithm but a **unified,
reproducible comparison** of three contrasting approaches on identical data with
both rating- and ranking-based metrics, plus an explicit cold-start treatment.

---

## 3. Methodology

### 3.1 System Flow
The system follows a classic offline recommender pipeline:

```
 ┌────────────┐   ┌──────────────────┐   ┌─────────────────────────────┐   ┌──────────────┐
 │ Raw CSVs   │──▶│ Pre-processing    │──▶│ Model training              │──▶│ Recommend /  │
 │ movies/    │   │ • clean metadata  │   │ • Content-Based (TF-IDF)    │   │ predict      │
 │ ratings/   │   │ • genre+tag soup  │   │ • Item-KNN (centred cosine) │   │              │
 │ tags       │   │ • user-item matrix│   │ • FunkSVD (SGD factors)     │   └──────┬───────┘
 └────────────┘   │ • train/test split│   │ • Hybrid (blend + switch)   │          │
                  └──────────────────┘   └─────────────────────────────┘          ▼
                                                                          ┌──────────────────┐
                                                                          │ Evaluation        │
                                                                          │ RMSE/MAE,         │
                                                                          │ Precision/Recall/ │
                                                                          │ F1@K              │
                                                                          └──────────────────┘
```

Implementation mapping: `src/data_loader.py` (pre-processing),
`src/content_based.py`, `src/collaborative.py`, `src/hybrid.py` (models),
`src/evaluation.py` + `run_evaluation.py` (evaluation), and `app_cli.py` /
`app_streamlit.py` (prototype UI).

### 3.2 Dataset
We use the **MovieLens `ml-latest-small`** dataset (Harper & Konstan, 2015),
bundled in `data/` so the prototype runs fully offline.

| Property | Value |
|----------|-------|
| Users | 610 |
| Movies | 9,742 |
| Ratings | 100,836 (scale 0.5–5.0, half-star steps) |
| Tags | 3,683 free-text tags |
| Matrix density | ~1.7% (i.e. ~98.3% of user-movie pairs are unrated) |
| Source | GroupLens Research, University of Minnesota |

The extreme sparsity (1.7%) is exactly what makes recommendation hard and is the
reason collaborative methods need careful similarity normalisation.

### 3.3 Data Pre-processing (`src/data_loader.py`)
1. **Metadata cleaning.** The release year is parsed out of each title; the
   `(no genres listed)` placeholder is treated as empty.
2. **Content "soup" construction.** For each movie we concatenate its genres
   (weighted ×2 so they outweigh sparse tags) with the lowercased free-text tags
   contributed by users, producing one text document per movie — e.g. Toy Story
   becomes `"adventure animation children comedy fantasy … pixar fun"`.
3. **User–item rating matrix.** A dense `610 × 9742` matrix is built where entry
   `(u, i)` is the rating, or `0` for "unobserved". Two dictionaries map raw
   `userId`/`movieId` values to row/column indices.
4. **Train/test split.** For evaluation we use a **per-user 80/20 hold-out**
   (`train_test_split_ratings`): 20% of *each* user's ratings are held out for
   testing while keeping at least five in training. This mirrors a deployed
   system that always has *some* history per user, and prevents the pathological
   case where a random split leaves a user with no training data.

### 3.4 Method 1 — Content-Based Filtering (Member 1, `src/content_based.py`)
**Step 1 — Item profiles via TF-IDF.** Each movie's content soup is converted to
a numeric vector using Term-Frequency–Inverse-Document-Frequency:

  `tfidf(t, m) = tf(t, m) × log(N / df(t))`

where `tf` rewards a term that appears in movie *m* and the `idf` factor
down-weights terms (like the genre "Drama") that appear in almost every movie and
therefore carry little discriminating power. We enable `sublinear_tf` (using
`1 + log(tf)`) to dampen repeated terms. The result is a sparse
`9742 × n_terms` matrix.

**Step 2 — Similarity via cosine.** The closeness of two movies is the cosine of
the angle between their TF-IDF vectors:

  `cos(a, b) = (a · b) / (‖a‖ · ‖b‖)`,  which lies in [0, 1] for TF-IDF vectors.

**Step 3 — User profile and scoring.** A user is represented by the movies they
rated. The predicted rating for an unseen movie *i* is a similarity-weighted
average of the user's own ratings:

  `pred(u, i) = Σⱼ sim(i, j)·r(u, j) / Σⱼ |sim(i, j)|`  (j ranges over movies u rated)

For Top-N, the user's ratings are folded into a single TF-IDF "taste vector" and
every movie is scored against it in one sparse matrix multiplication, then the
highest-scoring unseen movies are returned.

**Cold-start use.** Because the model needs only item content, it can recommend
"more movies like X" from a *single* seed movie — used by the prototype's
cold-start mode for brand-new users.

### 3.5 Method 2 — Collaborative Filtering (Member 2, `src/collaborative.py`)
Two complementary CF implementations are provided.

**(A) Memory-based Item-KNN (centred cosine).**
1. Each movie is represented by the column of ratings it received from all users.
2. Ratings are **mean-centred per item** (subtract the item's average) so that a
   generous user who rates everything highly does not dominate — this is the
   *adjusted/centred cosine* of Sarwar et al. (2001).
3. The predicted rating uses the *k* most similar items the user has already
   rated:

   `pred(u, i) = mean_i + Σⱼ sim(i, j)·(r(u, j) − mean_j) / Σⱼ |sim(i, j)|`

   over the k neighbours j of i that u rated. To avoid materialising the
   95-million-entry item-item matrix, similarities are computed on demand from an
   L2-normalised sparse item matrix (a single sparse dot product). For Top-N, a
   fast vectorised score generates candidates, which are then re-ranked with the
   exact normalised prediction; a **minimum-support filter** prevents obscure
   items rated by only a handful of users from being over-promoted.

**(B) Model-based FunkSVD (latent-factor matrix factorization).**
The sparse rating matrix is approximated by

  `r̂(u, i) = μ + b_u + b_i + pᵤ · qᵢ`

where μ is the global mean, `b_u`/`b_i` are user/item biases, and `pᵤ`, `qᵢ` are
*k*-dimensional latent factor vectors that the model learns. Parameters are
trained by **stochastic gradient descent** over the observed ratings only,
minimising the regularised squared error

  `Σ (r_ui − r̂_ui)² + λ(b_u² + b_i² + ‖pᵤ‖² + ‖qᵢ‖²)`.

The latent dimensions are not given in advance — they emerge during training and
tend to capture hidden themes (e.g. "blockbuster vs. arthouse"). Hyper-parameters
used: `n_factors = 50`, `n_epochs = 30`, learning rate `0.005`, regularisation
`λ = 0.02`. At prediction time scoring all items for a user is a single
matrix-vector product, making FunkSVD very fast to serve.

### 3.6 Method 3 — Hybrid (Joint, extra effort, `src/hybrid.py`)
A **weighted** hybrid with a **switching** cold-start fallback:

  `score_hybrid = α · score_CF + (1 − α) · score_CBF`,  with α = 0.7.

If collaborative filtering cannot produce a score (cold-start user/item), the
system *switches* to the content-based score alone. For Top-N, each base model's
scores are min-max normalised to [0, 1] before blending so neither scale
dominates. This directly targets the cold-start gap of pure CF (Burke, 2002).

### 3.7 Evaluation Metrics (`src/evaluation.py`)
Because a recommender has two distinct jobs, two metric families are used:

- **Rating prediction** (lower is better):
  - **RMSE** `= √(mean((pred − true)²))` — penalises large errors heavily.
  - **MAE** `= mean(|pred − true|)` — average error magnitude.
- **Top-N ranking** at K = 10 (higher is better). A held-out movie is "relevant"
  if the user rated it ≥ 4.0.
  - **Precision@K** = (relevant items in the Top-K) / K.
  - **Recall@K** = (relevant items in the Top-K) / (all relevant items).
  - **F1@K** = harmonic mean of Precision@K and Recall@K.
  - **HitRate** = fraction of users with at least one relevant item in their
    Top-K (a practical "did we help this user at all?" measure).

All models are trained on the **training split only** and evaluated on the unseen
test split, so the comparison is fair and leakage-free. The full experiment is
reproducible with `python run_evaluation.py`.

---

## 4. Results & Discussion

All figures below were produced by `python run_evaluation.py` on the per-user
80/20 split (movies with fewer than 5 ratings filtered out to reduce noise:
610 users, 3,650 movies, 90,274 ratings; 72,236 train / 18,038 test). The raw
console output is saved in `docs/evaluation_results.txt`.

### 4.1 Table 1 — Rating-Prediction Accuracy (lower is better)

| Method | RMSE | MAE | Coverage |
|--------|------|-----|----------|
| Content-Based | 0.9058 | 0.7066 | 99.93% |
| **Item-KNN** | **0.8395** | **0.6378** | 100% |
| FunkSVD | 0.8472 | 0.6479 | 100% |
| Hybrid | 0.8401 | 0.6458 | 100% |

### 4.2 Table 2 — Top-10 Ranking Quality (higher is better)

| Method | Precision@10 | Recall@10 | F1@10 | HitRate |
|--------|-------------|-----------|-------|---------|
| Content-Based | 0.0057 | 0.0057 | 0.0039 | 4.84% |
| Item-KNN | 0.0404 | **0.0445** | **0.0352** | 28.05% |
| **FunkSVD** | **0.0491** | 0.0356 | 0.0328 | 30.55% |
| Hybrid | 0.0489 | 0.0361 | 0.0330 | **31.55%** |

### 4.3 Discussion
**Collaborative filtering clearly beats content-based filtering.** On rating
prediction the two CF methods reduce RMSE by ~0.06–0.07 stars versus
content-based, and on ranking they are roughly **7–9× higher** on Precision@10
and reach 28–31% HitRate versus under 5% for content-based. This is the expected
result: genres and tags are a coarse description of taste, so a content model
keeps recommending same-genre movies that the user did not necessarily choose to
watch, whereas CF exploits the actual agreement patterns between users.

**Item-KNN vs. FunkSVD — different strengths.** Item-KNN gives the **best rating
accuracy** (RMSE 0.8395) and the best **Recall@10/F1@10**, because its
neighbourhood prediction is well-calibrated to the rating scale. FunkSVD gives
the best **Precision@10** (0.0491), reflecting matrix factorization's known
strength at ranking the *most* relevant items to the top. The two are
complementary rather than strictly ordered — exactly the situation a hybrid is
designed to exploit.

**The hybrid is the most robust.** It is within 0.0006 RMSE of the best predictor,
essentially ties FunkSVD on precision, and achieves the **highest HitRate
(31.55%)** — meaning it helped the largest share of users. It never "wins" a
single column outright, but it is never far from the best on *any* metric and, by
construction, it is the only method that still produces sensible recommendations
in cold-start conditions. For a production system that must serve new and
established users alike, this robustness is the most valuable property.

**Why the absolute ranking numbers look small.** Precision@10 of ~0.05 is normal
for MovieLens offline evaluation: each user has only a few held-out "relevant"
movies (often <10), so even a perfect system is capped at a low precision because
there simply are not 10 known-relevant items to find. The *relative* ordering
between methods is therefore the meaningful signal, and it is consistent across
both tables.

**Limitations of the evaluation.** Offline metrics measure agreement with *past*
ratings, not genuine user satisfaction; a movie the system recommends that the
user never rated is counted as "wrong" even if they would have loved it. A live
A/B test or a user-satisfaction questionnaire (assignment metric d-iii) would
complement these offline numbers.

---

## 5. Conclusion

### 5.1 Achievements
This project delivered a complete, working recommender system that fulfils the
assignment requirements: a cleaned real-world dataset, **two independent methods
(one per member)** — content-based filtering and collaborative filtering — plus a
**hybrid** built as extra effort, all compared under one leakage-free evaluation
harness with both rating- and ranking-based metrics, and an interactive prototype
(CLI and optional web UI) for live demonstration. The empirical study confirms
the textbook expectations: collaborative filtering outperforms content-based
filtering on this data, Item-KNN and FunkSVD trade off recall vs. precision, and
the hybrid is the most robust overall (highest HitRate, near-best everywhere).

### 5.2 Limitations
- Offline metrics approximate but do not equal real user satisfaction.
- Content features are limited to genres and tags; richer signals (cast,
  director, plot synopsis, posters) were not used.
- FunkSVD hyper-parameters were set sensibly but not exhaustively tuned.
- The dataset (610 users) is small; results may shift at larger scale.

### 5.3 Future Work
- **Richer content** via text embeddings (e.g. sentence-transformers) over plot
  summaries, and image features from posters.
- **Implicit feedback** (clicks, watch-time) and time-aware models that capture
  changing taste.
- **Hyper-parameter search** (grid/Bayesian) and modern factorization libraries
  (implicit/LightFM) or neural collaborative filtering.
- **Online evaluation** through an A/B test and a user-satisfaction
  questionnaire, and adding *diversity*/*novelty* metrics to counter
  over-specialisation.

---

## 6. References
*(APA 7th edition)*

Aggarwal, C. C. (2016). *Recommender systems: The textbook*. Springer.

Burke, R. (2002). Hybrid recommender systems: Survey and experiments. *User
Modeling and User-Adapted Interaction, 12*(4), 331–370.
https://doi.org/10.1023/A:1021240730564

Funk, S. (2006). *Netflix update: Try this at home* [Blog post]. Retrieved from
https://sifter.org/~simon/journal/20061211.html

Goldberg, D., Nichols, D., Oki, B. M., & Terry, D. (1992). Using collaborative
filtering to weave an information tapestry. *Communications of the ACM, 35*(12),
61–70. https://doi.org/10.1145/138859.138867

Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix recommender system:
Algorithms, business value, and innovation. *ACM Transactions on Management
Information Systems, 6*(4), Article 13. https://doi.org/10.1145/2843948

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and
context. *ACM Transactions on Interactive Intelligent Systems, 5*(4),
Article 19. https://doi.org/10.1145/2827872

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for
recommender systems. *Computer, 42*(8), 30–37.
https://doi.org/10.1109/MC.2009.263

Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender
systems: State of the art and trends. In F. Ricci, L. Rokach, B. Shapira, &
P. B. Kantor (Eds.), *Recommender systems handbook* (pp. 73–105). Springer.

Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. In
P. Brusilovsky, A. Kobsa, & W. Nejdl (Eds.), *The adaptive web* (pp. 325–341).
Springer.

Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994).
GroupLens: An open architecture for collaborative filtering of netnews. In
*Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work*
(pp. 175–186). https://doi.org/10.1145/192844.192905

Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems: Introduction
and challenges. In *Recommender systems handbook* (2nd ed., pp. 1–34). Springer.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text
retrieval. *Information Processing & Management, 24*(5), 513–523.
https://doi.org/10.1016/0306-4573(88)90021-0

Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based
collaborative filtering recommendation algorithms. In *Proceedings of the 10th
International Conference on World Wide Web* (pp. 285–295).
https://doi.org/10.1145/371920.372071

Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). Methods and
metrics for cold-start recommendations. In *Proceedings of the 25th Annual
International ACM SIGIR Conference* (pp. 253–260).
https://doi.org/10.1145/564376.564421

### Tools & Libraries
- Python 3.11; NumPy; pandas; scikit-learn (TF-IDF, cosine similarity); SciPy
  (sparse matrices); Streamlit (optional web UI).
- Dataset: MovieLens `ml-latest-small`, GroupLens Research
  (https://grouplens.org/datasets/movielens/), used under its educational
  license (see `data/MOVIELENS_README.txt`).

---

## Appendix A — Plagiarism Statement Form

> *Duplicate this section for each group member and sign before submission.*

We declare that this assignment is our own work, except where due acknowledgement
is made, and has not been previously submitted for assessment elsewhere. We have
read and understood the TARUMT Plagiarism Policy. We have not copied from nor
shared our work (including code) with any person other than our team members and
tutor.

| Name | Student ID | Signature | Date |
|------|-----------|-----------|------|
| _________________ | __________ | __________ | ________ |
| _________________ | __________ | __________ | ________ |

---

## Appendix B — AI Disclosure Statement

In accordance with the course AI-use guideline, we disclose the following use of
AI tools in producing this assignment.

| Item | Detail |
|------|--------|
| AI tool(s) used | An AI coding/writing assistant |
| Tasks assisted | Scaffolding the Python modules, drafting code comments and this documentation, and explaining the algorithms |
| Example prompts | "Implement item-based collaborative filtering with centred cosine similarity", "Implement FunkSVD with biases and SGD", "Draft a rubric-aligned methodology section for a recommender-system report" |
| Verification steps | Every module was executed and its output inspected; the evaluation results in Section 4 were independently reproduced by running `python run_evaluation.py`; all cited references were checked for accuracy; the team reviewed and understands all submitted code |
| Responsibility | The group members remain fully responsible for the accuracy, logic, and integrity of the final work |

---

## Appendix C — Student Free-Rider Report Form

> *Complete only if needed; attach supporting evidence.*

| Field | Detail |
|-------|--------|
| Reporting student(s) | _______________________________ |
| Student reported | _______________________________ |
| Description of the issue | _______________________________ |
| Evidence attached (meeting notes, commits, messages) | _______________________________ |
| Attempted in-team resolution | ☐ Yes ☐ No — details: ____________ |
| Date | ____________ |
| Signature(s) | ____________ |
