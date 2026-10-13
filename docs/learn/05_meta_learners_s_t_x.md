# 5. Meta-learners: S, T and X

*Course video 9 (CATE overview).* *Where it shows up in this repo:*
`src/tierlift/uplift/learners.py`, `results/uplift_benchmark.csv`, figures
`05_qini_curves.png` and `06_uplift_benchmark.png`.

---

## 5.1 From "does it work" to "for whom"

The ATE answers "should we roll this out?" The CATE

```
τ(x) = E[Y(1) − Y(0) | X = x]
```

answers "roll it out *to whom*?" In this project the incentive loses money on average
(blanket rollout −2 to −3%) and makes money for clients 10–20% from a tier boundary (+11%).
Only a CATE can find that.

A **meta-learner** is a recipe that turns any supervised learner (here gradient-boosted trees)
into a CATE estimator. Künzel et al. (2019) named the S-, T- and X-learners. They are easy to
describe and each has a characteristic failure mode, which is exactly why benchmarking them is
informative.

## 5.2 The S-learner ("single")

Fit **one** model with the treatment indicator as just another feature:

```
μ̂(x, d) ≈ E[Y | X = x, D = d]
τ̂_S(x) = μ̂(x, 1) − μ̂(x, 0)
```

> **Intuition.** Ask one model "what happens to a client like this?" twice: once with
> `d = 1`, once with `d = 0`, and take the difference.

**Failure mode: regularisation bias.** The treatment indicator is one feature among twenty.
If treatment matters less than, say, client size, a regularised model (trees with a minimum
leaf size, a penalised linear model) may never split on `d`. Then τ̂_S(x) = 0 for everyone:
the S-learner is biased *toward no effect*. The more heterogeneous and the smaller the effect,
the worse it gets.

In our data the S-learner is the strongest meta-learner: on the showcase run it ties the
X-learner by AUUC (0.0185 vs 0.0188) and is second only to the causal forest at recovering
the true ranking (Spearman 0.66), and across five replications it is first by AUUC in three
(`results/benchmark_montecarlo_summary.csv`). The reason is that the outcome is the revenue
*index* (Y / pre-period revenue): size has been divided out, so treatment is a relatively
important feature. In dollar space it was the worst learner by far. Same algorithm, different
outcome scale; see 5.6.

## 5.3 The T-learner ("two")

Fit **two** models, one per arm:

```
μ̂₁(x) ≈ E[Y | X = x, D = 1]       (treated clients only)
μ̂₀(x) ≈ E[Y | X = x, D = 0]       (holdout clients only)
τ̂_T(x) = μ̂₁(x) − μ̂₀(x)
```

> **Intuition.** Learn the treated world and the untreated world separately; the effect is
> the gap between the two maps.

**Failure mode: difference of two noisy fits.** The two models never see each other. Each
over-fits in its own way, so τ̂_T inherits the noise of both and can show "heterogeneity"
that is just the two models disagreeing in regions with few data points. With a 30% holdout,
μ̂₀ is fitted on fewer clients and is the noisier of the two. In our benchmark the T-learner
is last.

## 5.4 The X-learner ("cross")

Three stages. Stage 1 is the T-learner. Stage 2 imputes each client's *individual* effect
using the other arm's model, then regresses those imputed effects on X:

```
D̃ᵢ = Yᵢ − μ̂₀(Xᵢ)        for treated clients   →  fit τ̂₁(x) on {(Xᵢ, D̃ᵢ)}
D̃ᵢ = μ̂₁(Xᵢ) − Yᵢ        for control clients   →  fit τ̂₀(x) on {(Xᵢ, D̃ᵢ)}
```

Stage 3 blends them with the propensity:

```
τ̂_X(x) = e(x) τ̂₀(x) + (1 − e(x)) τ̂₁(x)
```

> **Intuition.** For a treated client, "what would you have earned untreated?" is best
> answered by the model trained on the *many* untreated clients, and vice versa. Then the
> effect itself is smoothed by a regression, instead of being a raw difference. The blend
> weights trust τ̂₁ (built from treated clients' own outcomes) where there are few treated
> clients (e small) and τ̂₀ where there are few controls.

**Where it shines.** Unequal arm sizes (e.g. a 5% holdout). **Where it does not.** Balanced
arms with a non-smooth effect: the stage-2 regression over-smooths. Our 40/30 split is
close to balanced, and the X-learner lands mid-table.

## 5.5 A tiny worked example

Six clients, one feature x (distance to tier, in %), outcome index y. Effect is +0.3 only when
x ∈ [10, 20]:

| client | x | D | y |
|---|---|---|---|
| A | 5 | 1 | 1.02 |
| B | 15 | 1 | 1.31 |
| C | 25 | 1 | 1.00 |
| D | 5 | 0 | 1.00 |
| E | 15 | 0 | 1.01 |
| F | 25 | 0 | 0.99 |

* **T-learner** with "nearest-x" models: μ̂₁(15) = 1.31, μ̂₀(15) = 1.01, τ̂_T(15) = +0.30 ✔;
  τ̂_T(5) = +0.02, τ̂_T(25) = +0.01. Correct here only because every x has a client in each arm.
* **S-learner** with a single decision stump that may split on x *or* D. The best single
  split on x (x ≤ 10 vs > 10) reduces squared error more than the split on D (treated mean
  1.11 vs control 1.00), so the stump ignores D and τ̂_S ≡ 0. Regularisation bias in
  miniature.
* **X-learner** stage 2 for treated clients: D̃_A = 1.02 − 1.00 = 0.02, D̃_B = 1.31 − 1.01 =
  0.30, D̃_C = 1.00 − 0.99 = 0.01; a regression of D̃ on x with a bump at 15 recovers the
  shape, and because stage 2 regresses, a single noisy client moves the curve less than in
  the T-learner.

## 5.6 Why we model the revenue *index*

Dollar revenue spans four orders of magnitude. Any learner fitted on dollars spends its
capacity predicting *who is big*, not *who responds*; the treatment effect of a $5M client is
larger in dollars than the whole revenue of a $5K client. Dividing by pre-period revenue puts
every client on the same scale (1.0 = "same as last quarter"). It is CUPED's idea applied to
CATE estimation. Dollar effects come back by multiplying τ̂(x) by pre-period revenue
(`UpliftResult.tau_dollars`), which is what the policy comparison uses.

## 5.7 Cross-fitting the scores

Every score in `results/uplift_benchmark.csv` and every Qini curve is **out-of-fold**: each
client's τ̂ comes from a model that never saw that client's outcome (4 folds × 2 repeats,
averaged). Otherwise an over-fitted learner would "predict" the noise it memorised and win the
benchmark dishonestly. This is the sample-splitting theme of course section 2 applied to the
evaluation itself.

## 5.8 Pitfalls

1. **Comparing learners with different base models** tells you about the base models.
   `default_regressor()` is shared by all six.
2. **Reading τ̂ for a single client as that client's effect.** It is an estimate of the
   *average* effect for clients like them; the ITE is unknowable (write-up 1).
3. **Extrapolation.** A T-learner's μ̂₀ for a region with no holdout clients is a guess.
4. **Estimating the propensity in an RCT.** Use the known constant; an estimated e(x) only
   adds noise (`known_propensity()`).

## 5.9 Exercises

1. In the six-client example, change client B's outcome to 1.05. Recompute τ̂_T(15). What
   does this say about the T-learner's variance?
2. Explain why dividing the outcome by pre-period revenue helps the S-learner more than the
   T-learner.
3. The holdout is 30% of clients. Write the X-learner blend weight for a client with e = 0.57
   and say in words which stage-2 model it trusts more.
4. Suppose the incentive had a *constant* effect of +0.1 on the index for every client. Rank
   the three learners by how well you expect them to estimate τ(x), and justify.

<details><summary>Answers</summary>

1. τ̂_T(15) = 1.05 − 1.01 = +0.04: one client's noise moved the estimate from +0.30 to +0.04.
   With no smoothing across x, the T-learner's CATE is as noisy as the individual outcomes.
2. In dollar space the treatment feature competes with "size" features that explain almost
   all the variance, so a regularised single model ignores it (regularisation bias). The
   T-learner never has to choose whether to split on D. In index space size is gone and D
   becomes important, which rescues the S-learner specifically.
3. τ̂_X = 0.57·τ̂₀ + 0.43·τ̂₁: slightly more weight on τ̂₀, the model built from *control*
   clients' imputed effects, because treated clients are the larger group and their μ̂₁ is
   the better-estimated map.
4. S-learner best (a constant effect is one easy split on D, and shared structure across arms
   is fully used), X-learner next (stage 2 smooths a constant well), T-learner worst
   (difference of two independent fits invents heterogeneity that does not exist).

</details>
