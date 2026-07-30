# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Rayan (GitHub: KhanBuilds)
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/KhanBuilds/Rayanflyrank
- **Date:** 2026-07-30

> **Status of this draft.** Sections 0–3 and 6–8 are filled from work that has been run and verified:
> `w01_research_question.ipynb` (ML-02), `w02_ml_task_framing.ipynb` (ML-03),
> `w03_data_contract.ipynb` (ML-04), `w03_feature_leakage_check.ipynb` (ML-05),
> `w04_signal_audit.ipynb` (ML-06), `w04_baseline_score.ipynb` (ML-07) and `capstone.ipynb` (ML-11/12).
> **Sections 4 and 5 are PENDING ML-08/ML-09** — no model has been trained. Where a number comes from
> the repo's bundled reference pipeline (`outputs/model_report.md`) rather than my own run, it is labelled
> as such: a bar to clear, never a result I claim.

## 0. Abstract

**Draft — to be rewritten once Sections 4 and 5 hold real numbers.**

Among mature indexed content items with established search demand, which pages are undergoing measured
organic decline and should be prioritised for editorial review in the coming sprint? The work uses the
FlyRank internship starter dataset — 30,000 pseudonymized content items across 32 clients, one trailing
90-day window. I framed the task as capacity-constrained **ranking** (not classification), committed to
precision@50 against the 54.2% base rate before building anything, and built a transparent five-condition
rule baseline with reason codes on every row. That baseline reaches **precision@50 = 0.740 against a
0.542 base rate** in-sample — 37 of 50 review slots landing on pages measured as declining, versus 27 for
random triage — and its two documented defects (a label exactly reconstructible from two impression
columns, and precision that *rises* with K because the top of the ranking is mis-ordered) are the
substantive findings so far; the learned model that should fix the ordering is **not yet trained**. The
output is a ranked review queue: a decision-support aid telling an editor which pages to open first, not
a prediction that refreshing them recovers traffic.

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
`work/`. IDs are pseudonyms; the feature matrix contains no text columns at all (asserted: dtypes are
numeric only); the ML-07 top-20 review withholds identifiers. The ranked-queue CSV stays gitignored
(`work/**/*.csv`); only metrics JSONs are committed.

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

**PENDING — ML-08 (`work/notebooks/w05_model.ipynb`).** No model has been trained, and no model number
appears anywhere in this report. `scikit-learn` is not installed in the environment these notebooks were
last executed in, which is the blocker.

What is already fixed and does not need re-deciding:

- **Target (one sentence):** `is_declining_label` = 1 when `trend_direction == "down"` — impressions fell
  more than 20% between the most recent 30 days and the 30 before. Base rate 0.5421 (16,262 of 30,000).
- **Feature matrix:** built and verified in ML-05 — **68 columns** from the 32 contract-approved fields
  (log1p versions of the heavy-tailed counts, skew 11–18; `avg_position` recoded with a flag; `has_*`
  indicators on every gap; one-hot categoricals with missing as its own level). No text columns, no IDs,
  no label-derived fields — all asserted in code.
- **Method:** a tree-based classifier whose probability is used as a ranking score, because the decision is
  "which K first" and the signal lives in bands and interactions rather than linear thresholds.
- **The falsifiable prediction to test:** the baseline's defect is specific — *good band, bad ordering
  inside it*. A model earns its place here if and only if it improves the ordering **within** the top band,
  which shows up as precision@20 ≥ precision@50 ≥ precision@100 rather than the inverted curve the rule
  produces.

**Known weakness of the proxy, to carry into the paper:** label and features are drawn from the same
90-day window, so this is **concurrent** decline detection, not forecasting. The honest sentence is "this
page resembles the pages measured as declining", never "this page will decline".

## 5. Evaluation

**PENDING — ML-09 (`work/notebooks/w06_validation_audit.ipynb`).** No held-out evaluation has been run,
and **no sealed-evaluation claim is made anywhere in this report.**

The split design is already decided and justified by measurement: **grouped by `client_id`**, no client in
both train and test, because per-client label rates span **0.000–0.937** across 32 clients and the three
largest hold **43.3%** of rows — a random row split would measure client memorisation. A **time-aware**
split is impossible from this file (no date column) and needs the warehouse release. Seed 42.

Still required: model vs baseline in one table on that same split with the 0.542 base rate beside every
figure; and a short error analysis. My stated prior, from the baseline's error pattern: false positives
will concentrate in **high-impression pages** (where a rising page looks structurally like a declining one
to a volume-driven score) and in **low-volume pages** (where a −20% threshold on 5 → 3 impressions is
noise). 19.1% of declining pages have fewer than 100 impressions in 90 days.

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

**Feature importances:** PENDING ML-08. Advance note for honesty: the reference pipeline's top features
(`days_with_impressions`, `log_impressions_90d`, `avg_position`, `content_age_days`) form a
*volume-and-exposure* profile rather than a decay profile. If my own run reproduces that, the honest
reading is that the model partly learns "how much measurable traffic this page has" — and that must be
stated, not dressed up as decline detection.

## 7. Recommendation

**The ranked actions the current output supports.** This is the ML-07 baseline queue; the model-ranked
version is PENDING. Every row is a **review** recommendation — the system never recommends publishing a
change, only opening a page.

| Reason-code pattern | Action | Confidence | Limit |
|---|---|---|---|
| All five codes, mid-size page | **Refresh review** | Medium-high (~71% of this band measured declining) | Off-season intent looks identical; no seasonality field exists to rule it out |
| All five codes, **largest pages in the band** | **Monitor only** | Low — measured counter-signal | The three biggest pages in my top band all grew (+35% to +69%) |
| `established_coverage` + `has_demand`, no `stale_90d` | Monitor | Low-medium | Recently updated; a second refresh is unlikely to be the lever |
| Missing `mid_position` (top-3, or no position data) | Deprioritise | Medium | `top_3` declines least (0.241); `avg_position == 0` means no reading at all |
| `no_signal` | Leave alone | — | 75 pages, declining rate 0.293 — below the base rate |

**How an editor would use it tomorrow:**

1. Take the top 50 rows of `work/outputs/baseline_action_score.csv`.
2. **Move ranks 1–3 (top decile of impressions within the band) to a monitor list** — that is where the
   measured false positives concentrate.
3. **Cap at ~8 pages per client.** Unfiltered, the top 50 spans only **8 of 32 clients** with one client
   supplying **44%** of the queue; an editor serving the whole book would ignore three quarters of it.
   This is a product decision the metric cannot see.
4. Review the remainder by hand — the reason codes say what to look at first.
5. **Log every decision (refreshed / skipped / monitored) with a date.** That log is the observed outcome
   this project currently lacks, and it is what would make a genuine past→future label possible next
   quarter.

**Confidence and limits, stated plainly.** Directional and decision-support. The queue is measurably
better than random triage on the metric that matches the decision (0.740 vs 0.542 at K=50) — in-sample, on
one 90-day snapshot of 32 pseudonymized clients. It is **not** validated out-of-sample, it is **not** a
forecast, and it does **not** establish that refreshing anything recovers traffic. Nothing here models
search-engine behaviour, and nothing here reads editorial quality.

## 8. Reproducibility

**Environment and seeds.** `RANDOM_STATE = 42` in every notebook (and `scripts/03_train_model.py:38`).
Dependencies are `requirements.txt`; the runs behind this report used pandas 2.x, numpy 2.x and matplotlib
3.10.8. **`scikit-learn` is listed in `requirements.txt` but was not installed in the execution
environment**, which is exactly why Sections 4 and 5 are PENDING rather than filled.

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
#   work/notebooks/capstone.ipynb                   (ML-11/12) -> writes work/figures/
```

Every notebook is **self-contained and order-independent**: each has a setup cell that resolves the repo
root (on Colab it shallow-clones the repo and installs requirements; locally it walks up from the kernel's
start directory until it finds `data/raw`), then loads the starter CSV from
`data/raw/content_refresh_anonymized.csv` — which ships tracked in the repo. `capstone.ipynb` rebuilds the
baseline inline rather than depending on another notebook having been run, and then **asserts its numbers
match the committed receipt**, so a stale artifact fails loudly instead of silently.

**Verification status (2026-07-30).** All seven notebooks above were executed top to bottom with no
errors, and every number quoted in this report is reproduced by the cell outputs. They were executed as
extracted cell sequences (in order, shared namespace) rather than through a Jupyter kernel —
`jupyter`/`nbformat` are not installed locally — so the committed notebooks carry no stored outputs for
cells written in that pass. **Run all in Colab before submitting** so the notebooks ship with visible
outputs.

**Receipts.** `work/outputs/baseline_metrics.json` is committed and carries every precision@K in Section
3, plus its own honesty note in the `evaluation` field. `work/figures/*.svg` holds the two figures the
paper embeds. The ranked-queue CSV (`work/outputs/baseline_action_score.csv`, 30,000 rows) is
**deliberately gitignored** — `git check-ignore` confirms `work/**/*.csv` catches it — because datasets
never enter git; the metrics JSON is the committed trace instead.

**Sealed/holdout claim: none made.** No cell in this repo builds a sealed frame, and no metrics file
claims a blind evaluation. When ML-09 produces one, both the building cell and the metrics file it wrote
get committed, so "evaluated once, blind" is checkable from the repo rather than taken on faith.

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** — [flyrank.ai](https://flyrank.ai). Thanks to the FlyRank
team for the pseudonymized data release and the lane framing.

---

> **Claims checklist status.** All statements are *observed* / *measured* / *decision-support*. No causal
> claims — the dataset records no intervention. No "predicted Google's algorithm" — I modelled observable
> performance metrics in one pseudonymized portfolio. No client-identifying details. The base rate
> (0.5421) sits beside every precision figure. Small buckets are reported with their n (n=17 stale-visible
> pages, n=174 at 181+ days freshness, n=3 in the engagement band I refuse to quote as a rate). The
> reference-pipeline numbers (0.240 / 0.740 / AUC 0.750) are labelled as the repo's bundled results, not
> mine. Sections 4 and 5 say PENDING rather than borrowing a number, and no sealed-evaluation claim is
> made.
