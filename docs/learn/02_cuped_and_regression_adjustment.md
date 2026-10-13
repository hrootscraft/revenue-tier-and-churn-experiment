# 2. CUPED and regression adjustment: the same effect, half the noise

*Where it shows up in this repo:* `src/tierlift/experiment/ate.py` (`cuped`, `lin_regression`,
`aipw`), `results/ate_incentive.csv`, figures `02_variance_reduction.png` and
`03_cuped_scatter.png`, and the ~50% variance cut in the README.

---

## 2.1 The problem: clients are wildly different sizes

Quarterly revenue per client in this data runs from a few hundred dollars to several million.
The incentive changes revenue by a few percent. So the *signal* (a few percent of each client's
revenue) is buried in *noise* (the fact that client A is a hedge fund and client B is a family
office). The difference in means is unbiased, but its standard error is huge, because most of
the variance in Y has nothing to do with treatment.

> **Intuition.** You want to know if a new running shoe makes people faster. You could compare
> the finishing times of two random groups. Or you could compare each runner's time *to their
> own previous race*. The second comparison is far less noisy, because "is this person a fast
> runner?" has been subtracted out. CUPED is "compare to their own previous race", done
> correctly.

## 2.2 CUPED in one formula

Let Y be the outcome (campaign-quarter revenue) and X a **pre-treatment** covariate that
predicts it (pre-period revenue). CUPED (Deng, Xu, Kohavi & Walker, 2013) replaces Y by

```
Y_adj = Y − θ (X − X̄),        θ = Cov(Y, X) / Var(X)
```

and then runs the ordinary difference in means on Y_adj.

Three things to notice:

1. **It does not change the expected effect.** X was fixed before randomisation, so its mean is
   the same in both arms. The term θ(X − X̄) has the same expectation in treatment and control,
   and cancels in the difference.
2. **It removes the predictable part of Y.** θ is the regression slope of Y on X. Y_adj is the
   residual (plus a constant), the part of Y that X could *not* predict.
3. **The variance shrinks by exactly ρ²**, where ρ = Corr(Y, X):

   ```
   Var(Y_adj) = Var(Y) − 2θ Cov(Y,X) + θ² Var(X) = Var(Y) − Cov(Y,X)²/Var(X) = Var(Y)(1 − ρ²)
   ```

   ρ = 0.71 in this experiment (`results/ate_incentive.csv`, column `rho`), so the variance
   falls by 0.71² ≈ 50%. Equivalent to doubling the sample size for free.

## 2.3 A six-client example by hand

Six clients, three treated (T) and three control (C). Pre-period revenue X and outcome Y in $K.
The incentive adds exactly +4 to every treated client (so the true effect is 4).

| client | arm | X (pre) | Y (post) |
|---|---|---|---|
| A | T | 20 | 26 |
| B | T | 60 | 65 |
| C | T | 40 | 45 |
| D | C | 30 | 31 |
| E | C | 50 | 52 |
| F | C | 10 | 11 |

**Difference in means.** Treated mean = (26+65+45)/3 = 45.33; control mean = (31+52+11)/3 =
31.33; DiM = **+14.0**. Far from the true +4, because the treated clients were bigger on
average (X̄_T = 40 vs X̄_C = 30).

**CUPED.** X̄ = 35. Regress Y on X across all six: the slope θ comes out at about 1.05
(Y rises roughly one-for-one with X). Adjusted outcomes Y_adj = Y − 1.05 (X − 35):

| client | arm | X − 35 | 1.05·(X−35) | Y_adj |
|---|---|---|---|---|
| A | T | −15 | −15.8 | 41.8 |
| B | T | +25 | +26.3 | 38.7 |
| C | T | +5 | +5.3 | 39.7 |
| D | C | −5 | −5.3 | 36.3 |
| E | C | +15 | +15.8 | 36.2 |
| F | C | −25 | −26.3 | 37.3 |

Adjusted DiM = (41.8+38.7+39.7)/3 − (36.3+36.2+37.3)/3 = 40.1 − 36.6 = **+3.5**. Much closer
to 4, and look at the spread: the Y_adj values sit within ±2 of each other inside each arm,
where the raw Y values ranged over 50. That tightness is the variance reduction.

(The small gap from 4 is because θ was estimated from six points that *contain* the treatment
effect; with 5,000 clients it vanishes. In the code θ is estimated on all clients pooled,
never per arm, which keeps it independent of the effect.)

## 2.4 Regression adjustment: CUPED with more covariates

CUPED is one regression slope. Why stop at one covariate? Lin (2013) showed that the OLS

```
Y = α + τ D + β'(X − X̄) + γ'(X − X̄)·D + ε
```

with **centred** covariates and **treatment-covariate interactions** gives an estimate τ̂ that
is (a) consistent for the ATE under randomisation whatever the true functional form, and (b)
never less precise than the difference in means, as long as you use heteroskedasticity-robust
(HC2) standard errors. The interaction terms are what make (b) true: without them, a covariate
that matters only in one arm can *increase* the variance.

CUPED is the special case with one covariate and no interaction. `lin_regression` in
`ate.py` uses the full feature matrix (`features.py`): segment, region, API share,
engagement, AUM flows, tenure, maker ratio. In `results/ate_incentive.csv` it beats CUPED
because some of this quarter's drift is predictable from last quarter's AUM flows, which
pre-period revenue cannot see.

AIPW (write-up 4) is the next step: let a gradient-boosted model do the prediction instead of
OLS, with cross-fitting so that over-fitting cannot bias the estimate.

## 2.5 How it shows up in this repo

`results/ate_incentive.csv`, columns `se` and `variance_vs_dim` (variance relative to the
difference in means):

| estimator | what it adjusts for | typical variance vs DiM |
|---|---|---|
| Difference in means | nothing | 100% |
| CUPED | pre-period revenue | ~50% |
| Lin regression (HC2) | all pre-treatment features, linearly | ~40% |
| AIPW (cross-fitted GBM) | all features, non-linearly | ~20% |

`results/power.csv` shows the design-time version: the minimum detectable effect with and
without CUPED, and `equivalent_sample_multiplier` = 1/(1 − ρ²) ≈ 2.

`results/montecarlo_summary.csv` verifies the standard errors: across 40 simulated
experiments, the spread of the CUPED estimates matches the SE the formula reports, and the 95%
CI covers the truth 95% of the time.

## 2.6 Pitfalls

1. **X must be pre-treatment.** Adjusting for anything the treatment could affect (volume in
   week 2, tier crossing) biases the estimate. This is the single most common CUPED bug in
   practice.
2. **θ from the pooled sample.** Estimating θ separately per arm changes the estimand subtly;
   the standard recipe pools.
3. **Relative effects.** The business wants "+11%", not "+$1,368 per client". Dividing the
   adjusted effect by the *raw* control mean is fine (the adjustment has mean zero), and the
   SE of the ratio needs the delta method (`Estimate.rel_se` in `ate.py`).
4. **Heavy tails.** ρ² is a population quantity; a few whales can make the sample ρ swing.
   The permutation test (write-up 3) does not care.

## 2.7 Exercises

1. If ρ = 0.9, by what factor does CUPED shrink the variance, and how many clients would a plain
   difference in means need to match CUPED's precision on 5,000?
2. Show algebraically that E[Y_adj | D=1] − E[Y_adj | D=0] = E[Y | D=1] − E[Y | D=0] when X is
   independent of D.
3. In the six-client example, suppose an analyst adjusts for Y itself in week 1 (post-launch).
   Without computing anything, explain the direction of the bias.
4. The holdout cell is 30% and the incentive cell 40%. Write the variance of the difference in
   means as a function of σ² and N, then of ρ² after CUPED.

<details><summary>Answers</summary>

1. 1 − 0.81 = 0.19: variance falls 81%. DiM would need 5,000 / 0.19 ≈ 26,000 clients.
2. E[θ(X − X̄) | D=1] = θ(E[X | D=1] − X̄) = θ(E[X] − X̄) = 0 by independence, and the same for
   D=0, so the adjustment cancels in the difference.
3. Week-1 revenue already contains part of the treatment effect, so adjusting for it subtracts
   part of the effect from the treated arm: bias towards zero.
4. Var(DiM) = σ²(1/0.4N + 1/0.3N) = σ²·5.83/N; after CUPED, σ²(1 − ρ²)·5.83/N.

</details>
