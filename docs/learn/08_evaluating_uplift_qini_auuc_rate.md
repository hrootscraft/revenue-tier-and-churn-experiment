# 8. Evaluating uplift: Qini, AUUC, RATE/TOC and the calibration test

*Course video 18 (validating CATE estimates).* *Where it shows up in this repo:*
`src/tierlift/uplift/evaluate.py`, `results/uplift_benchmark.csv`, `results/qini_curves.csv`,
figures `05_qini_curves.png` and `06_uplift_benchmark.png`.

---

## 8.1 The difficulty

A classifier is scored against labels. A CATE model has no labels: nobody's true effect is
observed (write-up 1). So "RMSE against the truth" is impossible on real data. What we *can*
do is ask: **if we rank clients by the model's score, do the top-ranked clients really show a
larger treated-vs-holdout gap than the rest?** Every metric below is a version of that
question, computed on cross-fitted scores with the randomised holdout as the counterfactual.

## 8.2 Qini curve and Qini coefficient

Sort clients by score, highest first. Walk down the list. After the top *k* clients:

```
Qini(k) = Σ_{treated in top k} Y  −  Σ_{holdout in top k} Y × ( n_t(k) / n_c(k) )
```

The first term is the revenue of the treated clients in the top k; the second is the holdout
revenue in the top k, rescaled to the same number of clients. The difference is the
*incremental* revenue from treating the top k (Radcliffe, 2007). We divide by N so the y-axis
reads "incremental revenue index per client in the population".

* The curve ends at the overall ATE × N/N whatever the ranking: at k = N every client is in.
* A **random ranking** gives a straight line from 0 to that endpoint.
* A good model's curve rises *above* the line early (it put the responders first) and comes
  back to the endpoint.
* The **Qini coefficient** is the area between the curve and the random line
  (`qini_above_random`, averaged). Positive = better than random; negative = ranks backwards.

**A tiny example.** Six clients ranked by a model; each is treated (T) or holdout (C) with
revenue index y:

| rank | arm | y |
|---|---|---|
| 1 | T | 1.4 |
| 2 | C | 1.0 |
| 3 | T | 1.3 |
| 4 | T | 1.0 |
| 5 | C | 1.1 |
| 6 | C | 0.9 |

Cumulative treated sum, holdout sum, counts, and Qini (not yet divided by N):

| k | ΣY_T | ΣY_C | n_t | n_c | Qini(k) |
|---|---|---|---|---|---|
| 1 | 1.4 | 0 | 1 | 0 | 1.4 (no holdout yet; defined as ΣY_T) |
| 2 | 1.4 | 1.0 | 1 | 1 | 1.4 − 1.0·1 = 0.4 |
| 3 | 2.7 | 1.0 | 2 | 1 | 2.7 − 1.0·2 = 0.7 |
| 4 | 3.7 | 1.0 | 3 | 1 | 3.7 − 1.0·3 = 0.7 |
| 5 | 3.7 | 2.1 | 3 | 2 | 3.7 − 2.1·1.5 = 0.55 |
| 6 | 3.7 | 3.0 | 3 | 3 | 3.7 − 3.0·1 = 0.7 |

The endpoint 0.7 = 3·(1.233 − 1.0) = n_t × ATE. The curve peaked at k = 3–4 (the model put
the two responders, 1.4 and 1.3, at the top) and then flattened: client 4 added nothing.
Random targeting would have been the straight line to 0.7; the model's curve is above it, so
its Qini coefficient is positive. In this project the overall ATE is *negative*, so the
random line slopes down and a good model's curve rises first and falls later.

## 8.3 Uplift curve and AUUC

Same walk, but with *means* instead of sums: gain(k) = (mean_T(k) − mean_C(k)) × k/N. AUUC is
the area under it, again measured above the random line. Qini and AUUC almost always agree
in sign; Qini weights by the actual arm counts in the top k, AUUC assumes the arm ratio is
constant. We report both.

## 8.4 TOC and RATE (Yadlowsky et al., 2021)

The **Targeting Operator Characteristic** at a fraction q is

```
TOC(q) = ATE(top q by score) − ATE(everyone)
```

"how much better is the average effect among the top q than in the whole population?" A
model with no ranking skill has TOC ≈ 0 everywhere. The **RATE** (here AUTOC, area under TOC)
is a single number with a clean interpretation and, unlike Qini, an easy bootstrap standard
error. `results/uplift_benchmark.csv` reports `rate_autoc`, `rate_se` and a one-sided p-value
for "better than random". Course video 18 calls this the right way to *test* for useful
heterogeneity.

## 8.5 The BLP calibration test (Chernozhukov, Demirer, Duflo & Fernández-Val, 2018)

Ranking metrics say "the order is right". Calibration asks "is the *scale* right?" With a
known propensity e, regress

```
Y = a + b₁·(D − e) + b₂·(D − e)·(τ̂(X) − mean τ̂) + error
```

* b₁ estimates the ATE.
* b₂ is the slope of true effect on predicted effect. b₂ = 1: perfectly calibrated spread.
  b₂ > 1: the model is too timid (compresses heterogeneity). b₂ < 1: too bold. b₂ = 0: no
  heterogeneity signal at all (p-value in `blp_heterogeneity_p`).

In our benchmark the forest and the S-learner have b₂ ≈ 1.7–1.9 (they shrink effects toward
zero but rank well; the forest's heterogeneity is significant at p = 0.05, the S-learner's at
p = 0.04), the T/X/R/DR-learners have b₂ ≈ 0.5–0.7 (over-spread, and their heterogeneity is
not significant). Neither is "wrong" for targeting, which only needs the order; it matters if
you quote τ̂ as a dollar forecast.

## 8.6 Oracle metrics: only because this is a simulation

`data/truth.parquet` holds each client's true ITE, so `oracle_metrics()` also reports
PEHE (RMSE of τ̂ vs truth) and the rank correlation. Their only job is to **validate the
validators**. On the showcase run the forest is first on every metric, observed and oracle,
and the observed metrics agree with the oracle on the bottom three; they swap the S- and
X-learners, which are within noise of each other. `perfect_auuc` in
`results/uplift_ceiling.json` is the AUUC a model that *knew* the truth would score on the
observed data (0.036); the forest reaches 0.023, two thirds of it, with 3,500 clients.

Across five replications (`results/benchmark_montecarlo_summary.csv`, reduced settings) the
picture is sharper: the forest is first by rank correlation with the truth in *all five*, but
the observed AUUC leaderboard reorders from seed to seed (the S-learner is first by AUUC in
three of five; the AUUC standard deviation across seeds is as large as its mean). The oracle
metrics are stable because they see the truth for every client; the observed metrics see one
noisy outcome per client. Pitfall 2 below is the practical consequence.

## 8.7 Pitfalls

1. **In-sample scores.** A model scored on clients it trained on will "win" by memorising
   noise. Everything here is cross-fitted (4 folds × 2 repeats).
2. **One split, one seed.** Qini on 5,000 clients is noisy enough to reorder the middle of
   the leaderboard; RATE's bootstrap SE shows how much. Repeated cross-fitting halves the
   Monte-Carlo noise in the scores.
3. **Reading Qini in dollars.** The curves are in revenue-index units per client; multiply by
   pre-period revenue to talk money (the policy comparison does).
4. **Ranking ≠ calibration.** A model can rank perfectly and be off by a factor of three in
   scale.

## 8.8 Exercises

1. In the six-client example, swap the ranks of clients 4 and 5. Recompute the Qini curve.
   Did the area above the random line go up or down?
2. Why must the Qini curve end at the same point for every model?
3. Suppose the overall ATE is exactly zero. Describe the random line and what a good model's
   curve looks like. Can the Qini coefficient still be positive?
4. The BLP slope for a model is 2.5 with SE 1.1. Is there significant heterogeneity? Is the
   model calibrated? What would you do before quoting its τ̂ as "+$X per client"?

<details><summary>Answers</summary>

1. New order: T1.4, C1.0, T1.3, C1.1, T1.0, C0.9. Qini: 1.4, 0.4, 0.7, 2.7 − 2.1·1 = 0.6,
   3.7 − 2.1·1.5 = 0.55, 0.7. The curve dips earlier (0.6 at k = 4 vs 0.7): a holdout client
   with a high outcome was ranked above a treated non-responder, which looks like a worse
   ranking; area goes down slightly.
2. At k = N the top-k set is everyone, so Qini(N) = n_t·(mean_T − mean_C) regardless of
   order.
3. The random line is flat at zero. A good model's curve rises above zero (responders first),
   peaks, then returns to zero as the negative-effect clients are added. Yes: the area above
   the (flat) line is positive whenever the ranking separates positive from negative effects.
4. The slope is 2.3 SE from zero: significant heterogeneity. Not calibrated: the true spread
   is about 2.5× the predicted spread. Rescale τ̂ by the slope (or report ranks, not dollars)
   before quoting per-client effects.

</details>
