# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Rayan (GitHub: KhanBuilds)
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/KhanBuilds/Rayanflyrank
- **Date:** 2026-07-30

> **Status of this draft.** Every section is filled from work that has been run and verified:
> `w01_research_question.ipynb` (ML-02), `w02_ml_task_framing.ipynb` (ML-03),
> `w03_data_contract.ipynb` (ML-04), `w03_feature_leakage_check.ipynb` (ML-05),
> `w04_signal_audit.ipynb` (ML-06), `w04_baseline_score.ipynb` (ML-07), `w05_model.ipynb` (ML-08),
> `w06_validation_audit.ipynb` (ML-09), `w07_action_playbook.ipynb` (ML-10) and `capstone.ipynb`
> (ML-11/12). Every notebook was executed top to bottom through a Jupyter kernel and ships with its
> outputs stored. Where a number comes from the repo's bundled reference pipeline
> (`outputs/model_report.md`) rather than my own run, it is labelled as such: a bar to clear, never a
> result I claim.

## 0. Abstract

Among mature indexed content items with established search demand, which pages are undergoing measured
organic decline and should be prioritised for editorial review in the coming sprint? The work uses the
FlyRank internship starter dataset — 30,000 pseudonymized content items across 32 clients, one trailing
90-day window. I framed the task as capacity-constrained **ranking** (not classification), committed to
precision@50 against the 54.2% base rate before building anything, and built a transparent five-condition
rule baseline with reason codes on every row. That baseline reaches **precision@50 = 0.740** in-sample and
carries a specific, self-diagnosed defect: its precision *rises* with K, meaning it finds a good band of
candidates but orders that band backwards at the top, where the queue is actually read.

Four models were then trained and compared on a **client-held-out split** (GroupKFold, 32 clients, zero
overlap), with the baseline scored on identical rows. The shipped system is a **hybrid**: the rule selects
the band and supplies the reason codes, a logistic regression orders pages within it. It reaches
**precision@50 = 0.900 against a 0.542 base rate** — 45 of 50 review slots landing on pages measured as
declining, versus 27 for random triage — and it is the only candidate whose precision falls monotonically
from **1.000 at K=10**, i.e. the only one that fixes the ordering defect it was built to fix.

Three results run against the headline and are reported with it. A random row split inflates AUC by
**0.152** over the grouped split — pure client memorisation, and the easiest way to publish a wrong number
on this data. The model's top feature by a factor of 2.4 is **impressions volume**, so it is substantially
ranking by measurable exposure rather than by decay. And its remaining errors are the *same* errors the
rule made: four of the five misses in its top 50 are pages that grew. The output is a ranked review queue —
a decision-support aid telling an editor which pages to open first, not a prediction that refreshing them
recovers traffic.

## 1. Problem framing

**Decision supported.** A content strategist chooses which pages, out of tens of thousands published,
receive a limited pool of optimisation budget and writer hours in a sprint. Editorial capacity is the
binding constraint — 50 slots against 30,000 pages is **0.17%** of the portfolio — so the real question is
ordering, not classification in the abstract.

- **Unit of analysis:** one pseudonymized content item (`content_id`); `client_id` for grouping only.
  Grain verified: 30,000 rows, 30,000 distinct IDs, zero duplicates.
- **Output:** a score per content item, delivered as a **ranked review queue** with reason codes.
- **Action:** the editor opens the top K pages (K ≈ 20–50 per sprint) and either refreshes — updating
  out-of-date facts, realigning headers with current intent, expanding thin sections, refreshing
  metadata — or, having looked, deliberately skips.
- **Cost of a wrong call:** a false positive burns a slot worth roughly $300–$1,000 of editorial time on a
  stable page; a false negative leaves a decaying revenue page unreviewed while competitors take the
  position. Costs are asymmetric and capacity is fixed, which is why **precision at the top of the
  ranking** is the metric and accuracy is not.

**Task type: ranking/scoring.** A binary classifier sits underneath, but its probability is used as a
ranking score and evaluated with precision@K, never thresholded at 0.5 and reported as accuracy. Recall is
near-meaningless here: with 16,262 positives and 50 slots, maximum achievable recall is **0.31%**.

**Why data/ML helps at all.** The honest answer, measured rather than assumed (ML-03 §5, ML-06):

- The naive heuristic has almost nothing to fire on — only **174** pages are more than 180 days stale, and
  only **17** are both ≥180 days stale and ≥500 impressions. A staleness-gated rule fills 17 of 50 slots
  and then tie-breaks arbitrarily.
- The real signals are **bands and interactions**, not thresholds. `days_with_impressions` rises from
  0.149 to 0.649 declining and then falls back; position is an **inverted U** (top-3 pages decline least
  at 0.241, positions 11–20 most at 0.610); CTR carries signal **only conditional on volume**.
- Hand-tuning those bands is fitting the data with worse bookkeeping and no held-out check. So the
  argument is not "ML is stronger" but **"ML is the honest way to fit what I would otherwise hand-tune"** —
  and ML-07 builds the hand-tuned rule anyway so ML-08 has a real bar.

## 2. Data safety

**Data used.** `data/raw/content_refresh_anonymized.csv` — 30,000 rows × 44 columns, 32 pseudonymized
clients, trailing-90-day metrics. The gated warehouse release was **not** used; where it would change a
conclusion, that is stated as a limit rather than worked around.

**Time window — and what the file does not have.** There is **no date column at all** (asserted in code,
ML-04 §1). Every time field is a relative offset from an unstated snapshot date. Three verified
consequences:

- The two 30-day columns cover days 1–60; days 61–90 sit in the 90-day totals and in neither.
  `last_30d + prev_30d == impressions_90d` holds for only **8.7%** of rows.
- `days_with_impressions` caps at **88** while `days_with_sessions` reaches **90** — the GSC reporting lag,
  visible in the data. Search and analytics windows are not the same length.
- `content_age_days` has a minimum of exactly **90**: the file was pre-filtered upstream to mature pages.
  This is why the "mature pages" filter in my ML-02 notebook dropped nothing — **the filter was already
  applied before I received the file**, so my 30,000-row count is the file as shipped, not my selection.

**Columns deliberately excluded — 12 fields.**

| Excluded | Why |
|---|---|
| `trend_direction`, `trend_pct` | The label source (`is_declining_label = trend_direction == "down"`). `trend_pct` is the strongest column in the file (AUC 0.247, i.e. 0.753 inverted). |
| `impressions_last_30d`, `impressions_prev_30d` | **The headline leakage finding — see below.** |
| `clicks_last_30d/prev_30d`, `sessions_last_30d/prev_30d` | Same last-vs-prev ratio shape as the label. Measured agreement 0.5364 / 0.5383 — near the 0.542 base rate, so *not* leaks; dropped anyway as weak and hard to defend in review. |
| `provider_used`, `model_used` | Generation-provenance / product-decision flags (71.5% and 19.1% missing) describing which internal pipeline wrote the page, not whether it is decaying. |
| `content_id`, `client_id` | Pseudonymous IDs — grouping, joining, splitting, audit only. `client_id` is the split key. |

That leaves **32 feature fields**, and ML-04 asserts in code that all 44 file columns plus the engineered
label land in exactly one bucket, so the contract cannot silently drift from the file.

**The leakage finding that changed the exclusion list.** The data dictionary names two forbidden fields.
The true set is four. The label is a −20% threshold on the ratio of `impressions_last_30d` to
`impressions_prev_30d`, so that pair reconstructs the label **exactly — 1.0000 agreement**. What makes it
worth reporting is that **no single-feature screen would catch it**: `impressions_last_30d` scores ROC-AUC
**0.486** on its own, i.e. it looks like the *least* informative column in the file, while
`impressions_prev_30d` scores 0.621 and looks like an ordinary decent feature. A pipeline that ranks
features univariately and drops the suspicious ones would have kept the weakest-looking column and
produced a perfect, meaningless model. **Leakage lives in how the label was defined, not in column names
or univariate statistics.**

**Other leakage and validity risks handled.**

- **`avg_position = 0` means "no data", not rank 1** — 1,205 rows. Recoded to missing plus a
  `has_position_data` flag; left as zero those pages would look like the best-ranked content in the
  portfolio.
- **Rate columns are ×100 percentages** (`ctr = 0.76` is 0.76%), and `scroll_rate` / `ai_traffic_pct`
  legitimately exceed 100 (119 and 23 rows) because numerator and denominator come from different systems.
  Not clipped.
- **Missingness tracks `content_type`, and content type correlates with the label.** `feedly article` is
  **100%** missing `search_volume`/`competition`/`cpc`; `keyword article` is **28.3%** missing
  `word_count`; label rates are **0.287 / 0.561 / 0.572** by type. A blind `fillna(0)` would hand the model
  a proxy for "this is a feedly article" dressed up as a demand feature, so every gap gets an explicit
  `has_*` indicator flag. The underlying cause is visible too: **46.6%** of feedly articles are labelled
  `new` (zero prior-window impressions), so they are structurally unlabelable as declining.
- **Split leakage:** pages within a client share templates, seasonality and tracking, and per-client label
  rates span **0.000 to 0.937** while the three largest clients hold **43.3%** of rows. The split must be
  grouped by `client_id`.

**Client-identifying data.** Confirmed: no client names, domains, URLs or raw search queries anywhere in
`work/`. IDs are pseudonyms and never enter any model — they are used for grouping, splitting and audit
only. **No free text of any kind reaches the feature matrix**: the nine non-numeric columns ML-08 uses are
closed-vocabulary categorical bands (`position_tier`, `content_type`, `age_tier` …) that are one-hot
encoded, not titles, slugs or query strings. The ML-07 top-20 review and the ML-08 error analysis both
withhold identifiers — the miss tables print performance columns only. Both ranked-queue CSVs stay
gitignored (`work/**/*.csv`); only metrics JSONs are committed.

*Consistency, checked mechanically rather than assumed:* a regex sweep for `content_*` / `client_*`
patterns across the stored cell outputs of **all ten** notebooks returns nothing. `w02_ml_task_framing.ipynb`
originally printed six pseudonymous IDs in its unit-of-analysis preview; that cell now withholds them
(`<withheld>`) while still showing the same columns and proving the same grain, so the notebook makes its
point without breaking the convention the rest of the repo follows.

## 3. Baseline

**Built and measured (ML-07).** A transparent additive score — five readable conditions, no fitted
weights, reason codes on every row:

`3×established_coverage + 2×has_demand + 2×mid_position + 1×stale_90d + 1×mature_page`

| Condition | Points | Plain reading |
|---|---|---|
| `20 ≤ days_with_impressions ≤ 87` | 3 | Real sustained coverage, but not showing every day. |
| `impressions_90d ≥ 40` | 2 | Enough demand for a refresh to be worth anything. |
| `3 < avg_position ≤ 50` | 2 | Findable but not already winning. Excludes `avg_position == 0`. |
| `days_since_last_update ≥ 90` | 1 | Not touched in a quarter. |
| `90 ≤ content_age_days < 365` | 1 | Settled, but still current. |

Ties break on `min(log1p(impressions_90d), 10)`.

**Results, with the base rate beside them** (`work/outputs/baseline_metrics.json` is the committed
receipt):

| Metric | Value | vs base rate 0.542 |
|---|---|---|
| precision@20 | **0.750** | 1.38× — 15/20 slots |
| **precision@50** | **0.740** | **1.37× — 37/50 slots** (random triage: 27) |
| precision@100 | 0.800 | 1.48× |
| precision@200 | 0.835 | 1.54× |
| ROC-AUC | 0.649 | — |

It clears the ≥0.70 "useful" threshold I committed to in ML-03 *before* building anything.

**Why it is a fair comparison — and where it is not.** Same rows, same label, same metric as the model
will use. But its thresholds were **read off the ML-03 label-rate crosstabs**, so it is mildly tuned
in-sample and these numbers are **optimistic**; the receipt's `evaluation` field says so. The consequence
for ML-08: model and baseline must both be scored on the **same client-held-out split**. Comparing a
tuned-on-everything rule against a properly held-out model would flatter the rule.

**Two defects I found in my own baseline** (this is the substantive result so far):

1. **Precision rises with K** — 0.750 @20, 0.740 @50, 0.800 @100, 0.835 @200 — which can only happen if
   the highest-ranked rows are worse than those below. The three top-ranked pages in my queue all
   measured **`up`** (+54.4%, +68.5%, +35.2%): the tie-break sorts by traffic size, and the biggest pages
   here are the most stable. **The rule finds a good band (2,990 pages at score 9, 71.4% declining) but
   does not rank within it.** Reversing the tie-break is no better at the top (0.750 @20) and worse deeper
   (0.720 @50, 0.680 @100), so I keep it and report the flaw.
2. **The score is not monotone:** score 7 (0.467, n=2,606) sits below score 6 (0.593, n=5,145), and score
   1 (0.088, n=1,809) below score 0 (0.293, n=75). Five added opinions are not a calibrated ordering.

**Context for the reference numbers.** The repo's bundled *rule* baseline scores precision@50 = 0.240 —
*less than half* the base rate, i.e. it actively mis-prioritises — because it keys on staleness, which
barely exists in this slice. The naive "stale ≥180d AND ≥500 impressions" variant appears to score 0.740,
but that is a tie-break artifact: the rule fires on **17 rows**, so 33 of its top 50 are arbitrary file
order. Beating 0.240 is not the achievement; beating **0.542** is.

## 4. Model / analysis

**Built and measured (ML-08, `work/notebooks/w05_model.ipynb`).** Receipt:
`work/outputs/model_metrics.json`.

- **Target:** `is_declining_label` = 1 when `trend_direction == "down"` — impressions fell more than 20%
  between the most recent 30 days and the 30 before. Base rate 0.5421 (16,262 of 30,000).
- **Feature matrix:** 38 columns from the 32 contract-approved fields — 29 numeric (including six `has_*`
  missing indicators) and 9 categorical, one-hot encoded downstream inside the pipeline.
  `avg_position == 0` recoded to missing before anything else; the heavy-tailed counts `log1p`'d
  (skew 11–18, per ML-05). No free text, no IDs, no label-derived fields — asserted in code, not claimed
  in prose.
  > **Why this says 38 where ML-05 says 68.** Same contract, same fields, different stage of the same
  > matrix. ML-05 one-hot encoded the categoricals *up front* and reported the post-encoding width (68
  > all-numeric columns). ML-08 hands the 9 categoricals to a `ColumnTransformer` and lets it encode
  > **inside the cross-validation loop**, so the pre-encoding width is 38. Encoding inside the fold is the
  > safer construction — it stops a category that appears only in the test fold from influencing the
  > encoding — which is why the number changed rather than the contract.
- **Method:** four candidates, listed in order of readability so that a simple winner would be visible if
  it happened — logistic regression, a depth-3 decision tree, a random forest, and histogram gradient
  boosting. Each is used as a **ranking score** (`predict_proba`), never thresholded at 0.5.
- **No hyperparameter search.** Library defaults plus two conservative variance guards. An untuned model
  that wins is a cleaner claim than a tuned one that wins by tuning.

**Results, all out-of-fold on the client-held-out split, base rate 0.542 beside every figure:**

| System | p@20 | p@50 | p@100 | p@200 | ROC-AUC |
|---|---|---|---|---|---|
| Flag everything (floor) | 0.542 | 0.542 | 0.542 | 0.542 | 0.500 |
| My rule baseline (ML-07, in-sample) | 0.750 | 0.740 | 0.800 | **0.835** | 0.6485 |
| Decision tree (depth 3) | 0.600 | 0.600 | 0.570 | 0.600 | 0.6244 |
| Random forest | 0.550 | 0.680 | 0.740 | 0.780 | 0.6808 |
| Histogram gradient boosting | 0.900 | 0.820 | 0.790 | 0.800 | **0.6910** |
| Logistic regression | 0.800 | 0.880 | 0.810 | 0.805 | 0.6774 |
| **Hybrid — rule band, logistic orders within it (shipped)** | **0.950** | **0.900** | **0.850** | 0.805 | 0.6606 |

**The prediction I committed to in advance was tested and held.** ML-07's defect was specific — *good
band, bad ordering inside it* — and the falsifiable test recorded before training was that a model earns
its place only if precision stops rising with K. Measured:

| K | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| ML-07 rule | 0.700 | 0.750 | 0.740 | 0.800 | 0.835 | 0.808 |
| **Hybrid** | **1.000** | **0.950** | **0.900** | **0.850** | 0.805 | 0.770 |

The rule's curve climbs; the hybrid's falls monotonically. **Only the hybrid passes.** Plain logistic
regression does not (0.700 at K=10, peaking at K=50), and neither does boosting cleanly.

**Two things I did not expect, reported because they are true:**

1. **The readable model won, and only because of a feature decision.** Logistic regression beat both tree
   ensembles at precision@50 — but *only* once the count columns were log-transformed. In an earlier run
   without that step it scored 0.660 and finished last. The honest lesson is not that linear models are
   underrated; it is that the **ML-05 feature-engineering decision mattered more than the choice of
   model**, and I would have mis-attributed the result had I skipped it because trees do not need it.
2. **The rule still wins at K=200** (0.835 vs 0.805) and boosting wins at K=500. If editorial capacity
   were 200 pages a sprint rather than 50, my recommendation would change. The metric is only correct
   because the capacity is what it is.

**Known weakness of the proxy, carried into the paper:** label and features are drawn from the same 90-day
window, so this is **concurrent** decline detection, not forecasting. The honest sentence is "this page
resembles the pages measured as declining", never "this page will decline".

## 5. Evaluation

**Run and measured (ML-08).** Split: **GroupKFold(5) on `client_id`**, zero client overlap between train
and test — asserted in code for all five folds, not assumed. Seed 42 everywhere.

**Why that split, in one measured number.** The same model on the same features, two splits:

| Split | ROC-AUC | p@50 |
|---|---|---|
| Random 80/20 row split | **0.777** | 0.920 |
| Grouped, clients held out | **0.624** | 0.780 |

**A 0.152 AUC gap of pure client memorisation.** Per-client label rates span 0.000–0.937 across 32
clients and the three largest hold 43.3% of rows, so a random split mostly measures which client a page
belongs to. This is the single easiest way to publish an inflated number on this dataset, and it is
measured here rather than asserted. A **time-aware** split remains impossible from this file (no date
column) and needs the warehouse release.

**One asymmetry disclosed against my own result.** The rule's thresholds were read off crosstabs computed
on all 30,000 rows, so it is mildly in-sample everywhere while every model number is strictly out-of-fold.
**The comparison is tilted toward the baseline, not the model.**

**Uncertainty, because precision@50 is a statement about 50 rows** (one row moves it by 0.02).
Bootstrapped 95% intervals on the selected set:

| System | p@50 | 95% interval |
|---|---|---|
| ML-07 rule | 0.740 | [0.620, 0.860] |
| Logistic regression | 0.880 | [0.780, 0.960] |
| Hybrid | 0.900 | [0.820, 0.980] |

The hybrid's interval and the rule's overlap only slightly. On one dataset and one 90-day window I report
the improvement as **measured and directional, not established**.

**The spread across folds argues the other way, and that is reported too.** precision@50 measured
*separately inside* each held-out fold:

| System | mean | worst fold |
|---|---|---|
| Logistic regression | 0.768 | **0.560** |
| Random forest | 0.772 | 0.660 |
| ML-07 rule | 0.784 | 0.680 |
| **Hybrid (shipped)** | 0.820 | 0.740 |
| Histogram gradient boosting | **0.828** | **0.780** |

**Two questions, two winners.** Pooled out-of-fold asks *"rank the whole portfolio at once"* — the hybrid
wins. Per-fold asks *"rank a group of clients this model has never seen"* — boosting wins on both mean and
floor, and plain logistic regression collapses to 0.560 on one fold, barely above the base rate. Part of
that gap is a calibration artifact: pooling requires scores comparable *across* fold models, and logistic
probabilities stack more cleanly than a tree ensemble's. The hybrid is the compromise: second on both
measures, with the band structure preventing any single fold model's mis-calibration from dragging a page
into the top 50. **If a future release shows the fold floor mattering more than the pooled top-50, the
choice should flip to boosting — written down now rather than defended later.**

**My stated prior was wrong in an instructive way.** I predicted false positives would concentrate in
**high-impression pages** and in **low-volume pages**. Neither is what happened. Four of the hybrid's five
top-50 misses are pages that *grew* (+23.6% to +93.7%) — the direction I predicted — but they span 84 to
1,593 impressions, so volume is not the shared trait. **What they share is position: every miss sits at
`avg_position` 4.1–10.5, i.e. page one.** The model cannot separate a healthy page-1 page from a decaying
one. That is the same blind spot ML-06 measured from the other direction — `top_3` pages decline *least*
of any position band (0.241 against a 0.542 base rate) — so strong position is genuinely ambiguous
evidence in this data, and my prior pointed at the wrong axis.

**Sealed-evaluation claim: still none.** Cross-validation is an honest held-out estimate, not a sealed
test — every fold's data was available to me while I worked, and I chose the shipped configuration after
seeing fold results.

**The audit found something I did not plan for (ML-09).** The skill's verification step says to inject a
known leak and confirm the harness catches it. Adding the two forbidden columns back moves out-of-fold AUC
from 0.677 to **0.920 as raw counts** — but to **0.9996 in log space**. The same two columns, the same
model, an 0.08 AUC difference. The reason is that the label is a threshold on a **ratio**, which is linear
in logs and not in raw counts, so a linear model can only fully exploit the leak when the representation
matches the label's functional form.

**The general lesson, which I did not know before running it: leakage is a property of *(column,
representation, model)*, not of a column.** A screen that evaluates columns before deciding how they will
be transformed can under-state a leak by exactly the margin that gets waved through as "the model is just
a bit good". It also retrospectively justifies ML-05's method — that audit caught the pair by testing the
*reconstruction formula* rather than the columns one at a time.

**One check this project cannot fully pass, stated plainly.** The label lives in days 1–60 of the window;
several features are 90-day aggregates *containing* those days. By the strict rule, they are contaminated.
Three things stop that being fatal — the overlap is of magnitude not identity (`impressions_90d` alone
scores AUC 0.585, and a 90-day sum cannot express a direction), the task is framed as concurrent detection
rather than forecasting, and a clean version needs day-61–90 features that this file cannot produce
(the 30-day columns tile the 90-day total in only 8.7% of rows). It is disclosed here rather than passed.

## 6. Interpretation

**What the signal audit actually found (ML-06).** Five intuitions tested with n beside every rate; two
survived:

| Signal | Verdict | Effect |
|---|---|---|
| Thin measurable coverage ⇒ decay | **CONFIRMED** | 0.149 (1–4 active days) → 0.649 (47–69 days); all buckets n > 2,900 |
| Low CTR on a **visible** page ⇒ decay | **CONFIRMED** | 0.677 (CTR ≤ 0.10) → 0.460 (CTR > 1.0), monotone, n = 13,512 |
| Stale pages decline more | **MIXED** | 0.511 (0–30d) → 0.611 (91–180d), reverses to 0.471 at 181+ (n=174) |
| Worse position ⇒ more decline | **OPPOSITE** | Inverted U: `top_3` 0.241, `striking` 0.610, `deep` 0.344 |
| Low engagement on a visible page ⇒ decay | **FALSE (untestable)** | 99.0% of visible pages in one band; smallest bucket n=3 |

**Surprises and negative results, which are the most useful part of this work:**

1. **Position points the opposite way from intuition.** Top-3 pages are the *most stable* in the portfolio
   (0.241 declining against a 0.542 base rate) and very deep pages also decline less — they have little
   traffic left to lose. Decline peaks in the middle of the pack. This is why my rule uses a band rather
   than a threshold.
2. **A finding of mine was overturned by my own audit.** In ML-03 I recorded CTR as a negative result
   (0.507–0.578 across quintiles, essentially flat). That was measured on the wrong population: **13,212
   rows have `ctr == 0`**, overwhelmingly tiny pages where the ratio has no content. Among visible pages
   the signal is clean and monotone with a 0.217 gradient. I corrected the ML-03 notebook rather than leave
   two contradictory statements in the repo. The lesson is the one that matters for the model: **CTR
   matters only in interaction with volume** — a threshold cannot express that, which is part of the case
   for fitting rather than hand-tuning.
3. **Engagement-rate flags are unproven on this data, and that is worth publishing.** 21,629 of 30,000
   pages have `engagement_rate == 0` and 99% of visible pages fall into a single band. Any rule keyed on
   engagement bands here fires on noise. This is an **instrumentation gap**, not a content problem — it
   tells the team where not to spend the next measurement effort.
4. **The staleness story barely exists in this slice.** 174 pages beyond 180 days. The industry-standard
   refresh heuristic has nothing to fire on here.

**Feature importances (ML-08) — the advance warning was correct, and it lands against my own result.**
Before training anything I wrote: *"the reference pipeline's top features form a volume-and-exposure
profile rather than a decay profile; if my own run reproduces that, the honest reading is that the model
partly learns how much measurable traffic this page has."* Permutation importance, averaged over the five
held-out folds (ROC-AUC drop when the column is shuffled):

| Feature | AUC drop |
|---|---|
| `impressions_90d` (log) | **0.119** |
| `clicks_90d` (log) | 0.050 |
| `sessions_90d` / `users_90d` (log) | 0.033 / 0.032 |
| `avg_position` | 0.028 |

**It reproduced.** Volume outranks the next feature by 2.4×, and the only non-volume feature near the top
is position. The model is substantially ranking pages by how much measurable search exposure they carry,
and only secondarily by anything resembling decay — stated here rather than dressed up as decline
detection. It also sets up the error pattern in Section 5: a system whose two strongest signals are
*exposure* and *position* has no way to tell a well-performing page-1 page from a decaying one, and that
is exactly where its top-50 misses land.

**Why the coefficients are not interpreted one at a time.** The fitted logistic model carries +1.18 on
`impressions_90d` while carrying −0.62 on `clicks_90d` and −0.59 on `users_90d` (standardized inputs).
Those columns are strongly collinear — ML-05 named this risk in advance — so the fit has split one signal
across several columns and **the sign of any single coefficient is not a statement about the world**. Read
as a group they suggest *high impressions relative to clicks and sessions*, an exposed page not converting
its exposure. That reading is a hypothesis about the fit, not a measured finding, and it is labelled as
one rather than promoted to an insight.

**Nothing looks like leakage.** A top feature carrying 0.119 AUC inside a 0.677-AUC model is
strong-but-ordinary. The alarm would have been a single feature approaching 1.0 — the shape ML-05 measured
for `trend_pct` (0.753 inverted) before excluding it.

### Applying the same standard to the published research (ML-09)

I audited two findings from *The State of AI-Driven SEO* (March 2026) with the questions I ask of my own
work. The paper's Methodology page is more careful than most published SEO research — it names its
confounders, flags its own unstable buckets, and states plainly that the study is observational. Both
issues below are places where **the paper contradicts either itself or the dataset it shipped with**, so
both are checkable in code rather than matters of taste.

**1. "Refreshing mature pages produces 3.2× health and 57× impressions" (Finding #4) is causal language on
an observational comparison — and the paper's own Methodology page forbids it** ("correlations do not
prove causation"). The mechanism is selection: nobody refreshes pages at random, so part of that 57× is
*which pages were chosen*. There is also a definitional loop — Health Score is 30 points impressions + 30
position + 20 CTR + 20 scroll, so **60 of its 100 points are search performance**, and reporting a health
lift as the effect of refreshing partly restates the input. The fix needs no new data, only different
words: *"mature pages refreshed in the last 30 days show 57× the impressions of those that were not."*

**2. "Logistic regression (71% holdout accuracy)" (ML Appendix) is reported with no base rate, on an 80/20
split across 57 brands.** A majority-class classifier on the paper's own counts scores **62.1%** for free,
so 71% is roughly **9 points of skill**. And the split is ungrouped — which is the objection I can price
precisely, because I ran the identical experiment: on this data, a random split versus a client-grouped
split is worth **0.152 AUC**. My genuine skill above chance is 0.124. **The memorisation available from an
ungrouped split is larger than the signal.**

**3. The paper and its own dataset disagree on the definition of the variable both findings rest on.** Page
5 defines Trend Direction as ±10%; the shipped data dictionary says ±20%. **The data follows ±20%** —
verified by reading the actual `trend_pct` boundaries of each class (`up` starts at exactly +20.0, `down`
ends at exactly −20.0). A reader rebuilding the cohorts at ±10% would reclassify **1,935 rows in this
30,000-row slice alone** and would not know why their reproduction failed. That is a one-line errata, not
an analysis error, but it sits on the paper's central variable.

**What I could not fault.** The paper flags its own 283:1 outlier as unstable ("283 growing pages versus
only 1 declining") against its own headline, labels the ML appendix exploratory and secondary, and states
that no p-values or confidence intervals are reported. Where it polices itself, it does so voluntarily.

**And the reciprocal test.** The two objections I raised are the two I am most at risk of committing.
Section 5 is my answer to the split objection, measured on my own model at my own cost; the base rate
appears beside every precision figure in this report, which is my answer to the other.

## 7. Recommendation

**The ranked actions the shipped output supports.** The queue is the ML-08 hybrid
(`work/outputs/model_action_score.csv`): the ML-07 rule selects the band and supplies the reason codes, the
logistic model orders pages within it. Every row is a **review** recommendation — the system never
recommends publishing a change, only opening a page.

**All 50 rows of the shipped queue sit at baseline score 9**, meaning all five conditions fired on every
one of them. So each recommendation still arrives with the same auditable explanation an editor could read
in ML-07; the model changed only the order, which is precisely the part the rule was measured to get wrong.

| Reason-code pattern | Action | Confidence | Limit |
|---|---|---|---|
| All five codes, model-ranked top 50 | **Refresh review** | Medium-high — 45 of 50 measured declining (0.900 vs 0.542 base rate) | Off-season intent looks identical; no seasonality field exists to rule it out |
| All five codes, **page-1 position** (`avg_position` < 11) | **Refresh review, but verify direction first** | Medium — this is where every measured error sits | 4 of the 5 misses in the top 50 are page-1 pages that *grew* (+23.6% to +93.7%) |
| Any `comparison article` | **Do not trust the rank** | None — measured AUC 0.524, i.e. chance | 697 pages the model cannot order at all |
| `established_coverage` + `has_demand`, no `stale_90d` | Monitor | Low-medium | Recently updated; a second refresh is unlikely to be the lever |
| Missing `mid_position` (top-3, or no position data) | Deprioritise | Medium | `top_3` declines least (0.241); `avg_position == 0` means no reading at all |
| `no_signal` | Leave alone | — | 75 pages, declining rate 0.293 — below the base rate |

**The operational playbook (ML-10) turns each measured failure mode into a guardrail.** Applied to the top
50, the rules produce a sendable set of **27 pages**: 23 deferred by the per-client cap, and every page-1
row flagged for a direction check first. The cap cuts the largest client's share of the sprint from
**58% to 30%**. Full working and the monitoring plan: `work/notebooks/w07_action_playbook.ipynb`; receipt:
`work/outputs/playbook_metrics.json`.

**The filtered set measures 0.926 declining — and that number is not model performance.** It is what a
human decision layered on the ranking produced, and the playbook reports it beside the raw 0.900 precisely
so the two can never be conflated. **The rules remove slots; they do not remove errors.**

**How an editor would use it tomorrow:**

1. Take the top 50 rows of `work/outputs/refresh_queue.csv` (ranked by `hybrid_score`, `action` column
   already applied).
2. **Sanity-check the page-1 rows before assigning them.** Every measured false positive in the top 50
   sits at `avg_position` 4.1–10.5 and is *growing*, not shrinking. Strong position is ambiguous evidence
   in this data (ML-06: `top_3` declines least of any band), and this is the failure mode that survived
   from ML-07 into ML-08.
3. **Cap at ~8 pages per client.** This matters *more* for the shipped queue than for the rule: the
   hybrid's top 50 spans **7 of 32 clients with one supplying 58%** (29 of 50 slots), against the rule's
   44%. Better ranking, worse portfolio coverage. A product decision the metric cannot see.
4. **Drop `comparison article` rows from the queue entirely** until there is a model that can rank them.
5. Review the remainder by hand — the reason codes say what to look at first.
6. **Log every decision (refreshed / skipped / monitored) with a date.** That log is the observed outcome
   this project lacks, and it is what would make a genuine past→future label possible next quarter.

**What would tell us this has gone stale (ML-10 monitoring plan).** Each trigger carries a reference value
measured from this build, so drift is a comparison rather than a feeling. Tier 1, every sprint: base rate
outside 0.45–0.65; realised precision@50 below **0.70** (the usefulness floor committed to in ML-03 before
anything was built) for two consecutive sprints; the largest client exceeding 40% of the sendable set,
meaning the cap has stopped working. Tier 3, immediate refit: **any change to the ±20% label threshold** —
not hypothetical, since ML-09 found FlyRank's own research paper documenting that same field as ±10% — and
**any new client onboarded**, whose queue should be withheld until there is history, because the measured
cost of an unseen client is 0.152 AUC.

**The whole plan is blocked on one thing that does not exist.** Every Tier-1 trigger except the base rate
needs to know what the editor actually did. The cheapest high-value change to this project is a
three-column log: `content_id`, `decision`, `date`. It is also the only route to the causal question this
work cannot currently answer.

**Confidence and limits, stated plainly.** Directional and decision-support. The queue is measurably
better than random triage on the metric that matches the decision (**0.900 vs 0.542 at K=50**, evaluated
out-of-fold on held-out clients) — on one 90-day snapshot of 32 pseudonymized clients, with a bootstrap
interval of [0.820, 0.980] that only just clears the baseline's. It is **not** a sealed evaluation, **not**
a forecast, and it does **not** establish that refreshing anything recovers traffic. Nothing here models
search-engine behaviour, and nothing here reads editorial quality.

## 8. Reproducibility

**Environment and seeds.** `RANDOM_STATE = 42` in every notebook (and `scripts/03_train_model.py:38`) —
split, every model, and the bootstrap. Dependencies are `requirements.txt`; the ML-08 run behind Sections
4–6 used **scikit-learn 1.9.0, pandas 3.0.1, numpy 2.4.0** (printed by the notebook's setup cell). Every
estimator is `clone()`d per fold, so no fitted state is shared between folds — worth stating because
`sklearn.pipeline.Pipeline` does *not* clone its final estimator, and reusing one silently produces a
model fitted on the wrong fold.

**A version caveat that belongs next to the headline.** Tree-ensemble precision@K can move a point or two
between library versions. **The direction of the precision curve is the finding, not the third decimal.**

**From a fresh clone:**

```bash
git clone https://github.com/KhanBuilds/Rayanflyrank
cd Rayanflyrank
pip install -r requirements.txt
# then run, in order:
#   work/notebooks/w01_research_question.ipynb      (ML-02)
#   work/notebooks/w02_ml_task_framing.ipynb        (ML-03)
#   work/notebooks/w03_data_contract.ipynb          (ML-04)
#   work/notebooks/w03_feature_leakage_check.ipynb  (ML-05, optional stretch)
#   work/notebooks/w04_signal_audit.ipynb           (ML-06, optional stretch)
#   work/notebooks/w04_baseline_score.ipynb         (ML-07)  -> writes work/outputs/
#   work/notebooks/w05_model.ipynb                  (ML-08)  -> writes work/outputs/model_*
#   work/notebooks/w06_validation_audit.ipynb       (ML-09)
#   work/notebooks/w07_action_playbook.ipynb        (ML-10)  -> writes refresh_queue + playbook_metrics
#   work/notebooks/capstone.ipynb                   (ML-11/12) -> writes work/figures/
```

`w05_model.ipynb` takes roughly **8–10 minutes** to run end to end (five folds × four models, plus
permutation importance over all five folds). It **asserts the rebuilt ML-07 rule matches the committed
`baseline_metrics.json`** before comparing anything, so a stale baseline fails loudly instead of quietly
flattering the model.

Every notebook is **self-contained and order-independent**: each has a setup cell that resolves the repo
root (on Colab it shallow-clones the repo and installs requirements; locally it walks up from the kernel's
start directory until it finds `data/raw`), then loads the starter CSV from
`data/raw/content_refresh_anonymized.csv` — which ships tracked in the repo. `capstone.ipynb` rebuilds the
baseline inline rather than depending on another notebook having been run, and then **asserts its numbers
match the committed receipt**, so a stale artifact fails loudly instead of silently.

**Verification status (2026-08-16).** All ten notebooks above were executed top to bottom with no errors,
and every number quoted in this report is reproduced by a stored cell output. **`w05_model.ipynb` (ML-08),
`w06_validation_audit.ipynb` (ML-09), `w07_action_playbook.ipynb` (ML-10) and `capstone.ipynb` were
executed through a real Jupyter kernel via `nbconvert --execute`, so they ship with their outputs stored in
the file** — the comparison table, the fold spread, the leak-injection test and the error analysis are all
readable without running anything.

**All ten notebooks have since been executed the same way**, so every one of them carries stored outputs
and an `execution_count` sequence of exactly 1..N — which is how a reader can tell each was a single clean
top-to-bottom pass rather than a patchwork of cells run out of order. Verified mechanically across the set,
along with: zero cell errors, zero `TODO`/`PENDING` markers, and zero pseudonymous IDs in any stored
output.

**Receipts.** Three committed metrics files, and every number in this report traces to one of them:

- `work/outputs/baseline_metrics.json` — every precision@K in Section 3, plus its own honesty note in the
  `evaluation` field.
- `work/outputs/model_metrics.json` — every model number in Sections 4–6: the full comparison table, the
  per-fold spread, the permutation importances, the random-vs-grouped split inflation, the top-50 client
  concentration, and a `known_failure_mode` field recording that 4 of 5 top-50 misses are growing pages.
- `work/outputs/playbook_metrics.json` — the operational layer in Section 7: the rules and their
  parameters, the resulting sprint composition, and every monitoring reference value with its trigger.

Each downstream notebook **asserts its numbers against the receipt upstream of it** rather than quoting
them — `capstone.ipynb` and `w07` both refit the shipped ranking and fail loudly if they disagree with
`model_metrics.json`. A stale artifact breaks the run instead of quietly flattering the report.

`work/figures/*.svg` holds the four figures the paper embeds. All three ranked-queue CSVs
(`baseline_action_score.csv`, `model_action_score.csv`, `refresh_queue.csv`, 30,000 rows each) are
**deliberately gitignored** — `git check-ignore` confirms `work/**/*.csv` catches them — because datasets
never enter git; the metrics JSONs are the committed trace instead.

**Sealed/holdout claim: none made.** Cross-validation on held-out clients is an honest out-of-sample
estimate, **not** a sealed test: every fold's data was visible to me while working, and I selected the
shipped configuration after seeing fold results. No cell in this repo builds a sealed frame and no metrics
file claims a blind evaluation. When ML-09 produces one, both the building cell and the metrics file it
writes get committed, so "evaluated once, blind" stays checkable from the repo rather than taken on faith.

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** — [flyrank.ai](https://flyrank.ai). Thanks to the FlyRank
team for the pseudonymized data release and the lane framing.

---

> **Claims checklist status.** All statements are *observed* / *measured* / *directional* /
> *decision-support*. No causal claims — the dataset records no intervention. No "predicted Google's
> algorithm" — I modelled observable performance metrics in one pseudonymized portfolio. No
> client-identifying details. The base rate (0.5421) sits beside every precision figure. Small buckets are
> reported with their n (n=17 stale-visible pages, n=174 at 181+ days freshness, n=697 comparison articles
> the model cannot rank, n=3 in the engagement band I refuse to quote as a rate). Uncertainty is quoted
> where the sample is small: precision@50 rests on 50 rows and carries a bootstrap interval. The
> reference-pipeline numbers (0.240 / 0.740 / AUC 0.750) are labelled as the repo's bundled results, not
> mine. **No sealed-evaluation claim is made** — cross-validation is not a sealed test, and Section 5 says
> so. Three results that run against my own headline are reported *with* it rather than after it: the
> 0.152 AUC of client memorisation a random split would have handed me, the volume-not-decay importance
> profile I flagged as a risk before training and then reproduced, and the fact that the shipped queue has
> **worse** portfolio coverage than the baseline it replaces.
