# 7. Causal trees and forests (and CausalForestDML)

*Course videos 10–11 and 15.* *Where it shows up in this repo:* `CausalForestDML` in
`learners.py`; figure `07_cate_vs_distance.png`; the policy tree in
`results/policy_tree.txt`.

---

## 7.1 A regression tree that splits on *effects*

An ordinary regression tree splits the feature space to make Y homogeneous within leaves. A
**causal tree** (Athey & Imbens, 2016) splits to make the *treatment effect* homogeneous within
leaves: in each leaf, the effect is the difference in means between treated and control
clients in that leaf, and the split criterion rewards leaves whose effects differ from each
other.

> **Intuition.** Picture the 10–20% band. A causal tree that splits on `distance_pct` at
> 0.10 and 0.20 puts the clients with a +30% lift in one leaf and the rest in others; inside
> each leaf the treated-minus-holdout difference is a clean, local ATE. The policy tree in
> `results/policy_tree.txt` does exactly this (its first split is on distance at 0.20).

## 7.2 Honesty: the one idea to remember

If the same clients are used to *choose* the splits and to *estimate* the leaf effects, the
leaf estimates are biased: the tree chose those splits precisely because they made the
within-leaf differences look large (it fitted the noise). **Honest estimation** splits the
sample in two: one half decides the tree structure, the other half estimates the leaf
effects. Each half is doing a job the other half cannot contaminate.

The cost is half the data for each job. The benefit is that leaf estimates have valid
confidence intervals and the tree does not hallucinate heterogeneity. Course video 10.

## 7.3 From a tree to a forest

A single tree is unstable: move a few clients and the splits change. A **causal forest**
(Wager & Athey, 2018) grows thousands of honest trees on subsamples and averages. But the
averaging is subtler than in a random forest. For a target point x, each tree says which
training clients share a leaf with x. Across trees, that gives a **weight** αᵢ(x) for every
training client: how often client *i* "looks like" x. The forest's CATE is then a *weighted*
local treatment-effect estimate:

```
τ̂(x) = Σᵢ αᵢ(x) (Yᵢ − m̂(Xᵢ)) (Dᵢ − ê(Xᵢ))  /  Σᵢ αᵢ(x) (Dᵢ − ê(Xᵢ))²
```

That is the R-learner's residual-on-residual regression (write-up 6), computed *locally*
with forest weights. The forest is an adaptive nearest-neighbour method: it learns which
features matter for heterogeneity and ignores the rest when deciding who is a neighbour.

## 7.4 CausalForestDML = orthogonalised causal forest

EconML's `CausalForestDML` does two things in order:

1. **DML stage.** Cross-fit m̂(x) = E[Y | X] with the GBM and ê(x) with the (known)
   propensity; form the residuals Ỹ = Y − m̂, D̃ = D − ê.
2. **Forest stage.** Grow an honest causal forest on (Ỹ, D̃) with X as the splitting
   features. Each tree's split criterion targets heterogeneity in the local slope of Ỹ on D̃.

So it is the R-loss of write-up 6 with a forest as the final learner. This is also what
`grf::causal_forest` does in R (course video 15: "confounding-robust forests"). Two
knobs matter here:

* `n_estimators=1000` and `min_samples_leaf=10`: many trees, small leaves; honesty keeps
  small leaves from over-fitting.
* `cv=5` cross-fitting for the nuisances; the forest's own subsampling gives it honest
  out-of-bag estimates and **pointwise confidence intervals** (`effect_interval`), which the
  meta-learners cannot provide.

## 7.5 A tiny honest tree by hand

Eight clients, one feature x ∈ {near, far}, RCT with e = 0.5. True effect: +4 for near, 0 for
far.

| id | x | D | Y |
|---|---|---|---|
| 1 | near | 1 | 14 |
| 2 | near | 1 | 15 |
| 3 | near | 0 | 10 |
| 4 | near | 0 | 11 |
| 5 | far | 1 | 10 |
| 6 | far | 1 | 11 |
| 7 | far | 0 | 10 |
| 8 | far | 0 | 10 |

**Dishonest tree (all 8 clients for both jobs).** Candidate split on x: near effect = 14.5 −
10.5 = +4; far effect = 10.5 − 10 = +0.5. Heterogeneity criterion (squared difference of leaf
effects, weighted) is large, so it splits on x and reports +4 / +0.5. Fine *here*, but
imagine x were random noise: the tree would still find *some* split that separates the
largest treated outcomes from the smallest control ones, and report spurious heterogeneity.

**Honest tree.** Use clients 1, 3, 5, 7 to choose the split (near: 14 − 10 = +4; far: 10 − 10
= 0 → split on x), then clients 2, 4, 6, 8 to estimate: near = 15 − 11 = +4; far = 11 − 10 =
+1. The estimates are noisier (two clients per leaf) but unbiased, because the clients that
produced them had no say in the split.

## 7.6 How it shows up in this repo

* `results/uplift_benchmark.csv`: the forest has the highest AUUC and Qini, the highest
  rank correlation with the true CATE, and the lowest PEHE. Its RATE is significantly
  positive.
* `07_cate_vs_distance.png` bins the forest's cross-fitted τ̂ by distance to the tier and
  overlays the true CATE: the goal-gradient hump at 10–20% is recovered.
* `results/policy_tree.txt`: a depth-2 doubly-robust policy tree (write-up 9). Its root split
  is "distance ≤ 0.20", then "maker ratio ≤ 0.44": taker-heavy clients pay more per dollar
  of volume, so their volume gain is worth more.

## 7.7 Pitfalls

1. **Dishonest trees "discover" heterogeneity in pure noise.** Always honest.
2. **Reading a single tree as the truth.** A single tree is a summary for communication;
   the forest is the estimate.
3. **Leaf size vs honesty.** Tiny leaves with honesty are fine; tiny leaves without honesty
   are the recipe for over-fitting.
4. **Forest CIs are pointwise**, not simultaneous: 95% at each x, not for the whole curve.

## 7.8 Exercises

1. In the eight-client example, swap client 2's outcome to 11. Recompute the honest leaf
   estimates. Which half of the data absorbed the change, and why does that matter?
2. Explain in one sentence why the forest weights αᵢ(x) sum to one over training clients
   and why a client in a different segment can still get a non-zero weight.
3. The policy tree's second split is on maker ratio. Using the fee table in
   `configs/default.yaml`, explain the economic reason a taker-heavy client's volume lift is
   worth more.
4. Why does `fit_crossfit` in `learners.py` still cross-fit the forest's *scores* (4 outer
   folds) even though the forest is already honest?

<details><summary>Answers</summary>

1. Estimation half: near = 11 − 11 = 0, far = +1. The split (chosen on 1, 3, 5, 7) is
   unchanged; only the estimate moved. Honesty keeps *structure* and *estimate* errors
   independent, which is what makes the leaf CIs valid.
2. Each tree puts x in exactly one leaf and spreads unit weight over the training clients in
   it; averaging over trees keeps the sum at one. A client from another segment shares a leaf
   whenever no tree found segment worth splitting on near x.
3. The blended rate is maker_ratio·maker_bps + (1 − maker_ratio)·taker_bps and taker fees are
   roughly twice maker fees, so each extra dollar of volume from a taker-heavy client earns
   more revenue; the same volume lift is worth more.
4. Honesty makes each *tree's* leaf estimates unbiased for the clients it trained on, but the
   forest's score for a training client still used that client's outcome in some trees'
   estimation halves. For a fair Qini comparison against learners that are not honest at
   all, every model's score must be strictly out-of-sample.

</details>
