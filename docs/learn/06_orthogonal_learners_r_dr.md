# 6. Orthogonalisation: the R-learner, the DR-learner and cross-fitting

*Course videos 12–14 (robust estimation of heterogeneity) and 16–17 (loss functions).*
*Where it shows up in this repo:* `R-learner` (`NonParamDML`) and `DR-learner` (`DRLearner`)
in `src/tierlift/uplift/learners.py`; `dr_scores()` in `policy.py`.

---

## 6.1 The problem the meta-learners share

S, T and X all estimate τ(x) *through* models of the outcome. If those models are a bit wrong
(they always are), the error leaks into τ̂ at full strength. Course section 5 asks: can we
set the problem up so that **small mistakes in the nuisance models cause only
second-order mistakes in τ̂?** The answer is Neyman orthogonality, and the two learners here
are its two classic implementations.

## 6.2 Robinson's partialling-out and the R-loss

Write the outcome as

```
Y = τ(X)·D + g(X) + ε,          E[ε | X, D] = 0
```

(g is the untreated baseline, τ the effect). Take conditional expectations given X:

```
E[Y | X] = τ(X)·e(X) + g(X)          where  m(X) := E[Y | X],  e(X) := E[D | X]
```

Subtract:

```
Y − m(X) = τ(X)·(D − e(X)) + ε
```

The baseline g(X) is **gone**. What is left is a regression of the *residualised* outcome on
the *residualised* treatment, with τ(X) as the slope. The **R-loss** (Nie & Wager, 2021) is
the squared error of that regression:

```
L_R(τ) = mean[ ( (Yᵢ − m̂(Xᵢ)) − τ(Xᵢ)·(Dᵢ − ê(Xᵢ)) )² ]
```

> **Intuition.** Remove from Y everything X can predict. Remove from D everything X can
> predict (in an RCT that is just the constant e). What remains of Y, divided by what
> remains of D, is the effect. The first step is CUPED on steroids; the second is why the
> estimate cannot be fooled by confounding through X.

**Why it is robust.** The error in τ̂ depends on the *product* of the errors in m̂ and ê, not on
either alone. If each nuisance model converges at rate n^(−1/4), the product converges at
n^(−1/2): as fast as if the nuisances were known. (Course video 14.) In an RCT ê is exactly
known, so the R-learner's error from m̂ is second-order automatically.

**In code.** `NonParamDML(model_y=GBM, model_t=known propensity, model_final=GBM)`. EconML
fits m̂ with cross-fitting, residualises, then fits the final GBM to the residual ratio with
weights (D − e)², which is the R-loss minimiser. `CausalForestDML` (write-up 7) is the same
R-loss with a forest as `model_final`.

## 6.3 The DR-learner

Take the AIPW score from write-up 4, per client:

```
Γᵢ = μ̂₁(Xᵢ) − μ̂₀(Xᵢ) + Dᵢ (Yᵢ − μ̂₁(Xᵢ)) / e  −  (1 − Dᵢ)(Yᵢ − μ̂₀(Xᵢ)) / (1 − e)
```

Γᵢ is a noisy, unbiased signal of client *i*'s effect: E[Γ | X = x] = τ(x). So regress Γ on X:

```
τ̂_DR(x) = regression of Γᵢ on Xᵢ        (Kennedy, 2020)
```

> **Intuition.** AIPW gives the *average* effect by averaging Γ. The DR-learner refuses to
> average; it asks a model to explain *how Γ varies with x*.

It inherits AIPW's double robustness pointwise, and the pseudo-outcomes Γᵢ are reused directly
by the policy evaluation in write-up 9 (`dr_scores()`).

## 6.4 Cross-fitting: the step you cannot skip

Both learners plug fitted nuisances (m̂, μ̂₀, μ̂₁) into a second-stage regression. If the
nuisance for client *i* was fitted *using* client *i*, a flexible model partially memorises Yᵢ:

* the residual Yᵢ − m̂(Xᵢ) is too small, so the R-loss sees less signal (bias toward zero);
* Γᵢ's correction term is too small, so the DR-learner's pseudo-outcomes shrink.

Cross-fitting (Chernozhukov et al., 2018): split into K folds; for fold k, fit the nuisances
on the other K−1 folds and evaluate on fold k. Every client's nuisance prediction comes from a
model that never saw them. `cv=5` in the learner specs is this. The *evaluation* in
write-up 8 adds a second, outer layer of cross-fitting for the same reason.

## 6.5 A tiny R-learner by hand

Four clients in an RCT with e = 0.5. One binary feature x. Baseline g(x) = 10 if x = 1 else 0
(a huge baseline difference); effect τ(x) = 2 if x = 1 else 0.

| client | x | D | Y |
|---|---|---|---|
| A | 1 | 1 | 12 |
| B | 1 | 0 | 10 |
| C | 0 | 1 | 0 |
| D | 0 | 0 | 0 |

m̂(x=1) = (12 + 10)/2 = 11; m̂(x=0) = 0. e = 0.5.

| client | Y − m̂ | D − e |
|---|---|---|
| A | +1 | +0.5 |
| B | −1 | −0.5 |
| C | 0 | +0.5 |
| D | 0 | −0.5 |

Regress (Y − m̂) on (D − e) separately for each x: slope at x = 1 is (+1)/(+0.5) = (−1)/(−0.5)
= **2** ✔; at x = 0 it is **0** ✔. The baseline of 10 never entered. A T-learner would also
get this right here; the difference appears once m̂ is estimated with error: the R-learner's
τ̂ is off by (error in m̂) × (error in ê) = (error in m̂) × 0, while the T-learner's is off by the
full error in μ̂₁ − μ̂₀.

## 6.6 How they did in the benchmark

`results/uplift_benchmark.csv`: on this data the two orthogonal learners sit mid-table on
AUUC, behind the forest and the S-learner, and their BLP calibration slope is below 1 (they
*over*-spread their predictions). The reason is not the loss function but the final stage: a
GBM regressing a very noisy target (residual ratios, or Γ) on 19 features with 3,500 clients
over-fits, and EconML's defaults do little to stop it. The forest is the same R-loss with
honest trees as the final stage, and it wins. The lesson of course video 15 in one line:
orthogonalisation protects you from nuisance error; it does not protect you from a noisy
final-stage learner.

## 6.7 Pitfalls

1. **Skipping cross-fitting** (see 6.4). The bias is toward "no heterogeneity", which is the
   most dangerous direction: it looks like a safe conclusion.
2. **Estimating e(x) in an RCT.** Known constant beats estimate.
3. **Over-flexible final stage.** The nuisance can be a big GBM; the final stage should be
   the *simpler* model (a shallow forest, a penalised linear model), because its target is
   noisy by construction.
4. **Reading τ̂ in index units as dollars.** Multiply by pre-period revenue first.

## 6.8 Exercises

1. In the four-client example, suppose m̂ is mis-estimated as 10.5 at x = 1 (instead of 11).
   Recompute the R-learner slope at x = 1. Then suppose instead that ê = 0.6 (wrong) with
   m̂ = 11. Recompute. Which mistake hurt?
2. Write the DR pseudo-outcome Γ for client A (μ̂₁(1) = 12, μ̂₀(1) = 10, e = 0.5) and for
   client B. Average them. What do you get and why is it not surprising?
3. Why does `dr_scores()` in `policy.py` re-use exactly the quantity from 6.3?
4. The R-loss weights each client by (Dᵢ − e)². In an RCT with e = 0.57, which arm gets the
   larger weight, and does that make sense?

<details><summary>Answers</summary>

1. m̂ = 10.5: residuals A +1.5, B −0.5; regress on (±0.5) with least squares: slope =
   (1.5·0.5 + (−0.5)(−0.5)) / (0.25 + 0.25) = (0.75 + 0.25)/0.5 = 2. Unharmed, because the
   two residuals shifted by the same +0.5 and D − e sums to zero. ê = 0.6: D − e is +0.4 and
   −0.6; slope = (1·0.4 + (−1)(−0.6))/(0.16 + 0.36) = 1.0/0.52 = 1.92. Small error; and with
   both wrong the errors multiply. In an RCT ê is exact, so only the harmless kind occurs.
2. Γ_A = (12 − 10) + 1·(12 − 12)/0.5 − 0 = 2; Γ_B = (12 − 10) + 0 − (10 − 10)/0.5 = 2.
   Average 2 = the true effect at x = 1. Not surprising: with perfect μ̂'s the correction
   terms vanish and Γ is just the regression estimate.
3. Because E[Γ | X] = τ(X) without any model of how τ varies, so mean(π(X)·Γ) is an unbiased
   estimate of the value of any targeting rule π, including rules learned from data.
4. (1 − 0.57)² = 0.18 for treated, 0.57² = 0.32 for controls: the smaller arm (holdout) gets
   more weight. Yes: the residualised treatment is further from zero for the rarer arm, so
   those clients carry more information about the slope.

</details>
