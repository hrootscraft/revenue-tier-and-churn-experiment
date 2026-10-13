# 4. Propensity scores, IPW and AIPW: double robustness

*Course videos 5–8.* *Where it shows up in this repo:* `aipw()` in
`src/tierlift/experiment/ate.py`, `src/tierlift/experiment/observational.py`,
`results/observational.csv`, figure `04_observational_bias.png`.

---

## 4.1 When you did not randomise

Suppose the incentive had simply been *launched*, with a sign-up button. Engaged, growing
clients near a tier boundary would sign up more. They would also have grown anyway. The naive
comparison of sign-ups vs non-sign-ups is biased (write-up 1, section 1.4).

The rescue is the **unconfoundedness** assumption:

```
(Y(0), Y(1)) ⟂ D | X
```

"Within clients with the same features X, who signed up is as good as random." It is an
assumption, not a fact; but if it holds, two families of estimators recover the ATE.

## 4.2 The propensity score

```
e(x) = P(D = 1 | X = x)
```

the probability of being treated given the features. Rosenbaum & Rubin (1983): if
unconfoundedness holds given X, it also holds given the single number e(X). So instead of
matching on 19 features you can balance on one.

In a **randomised** experiment e(x) is a known constant (0.4/0.7 = 0.57 for incentive vs
holdout here). That is what `known_propensity()` in `learners.py` encodes, and it is why the
RCT estimators cannot be biased by a wrong propensity model: there is no model.

## 4.3 Inverse propensity weighting (IPW)

Treated clients with a *low* propensity are rare in the treated arm but common in the
population; weight them up. Control clients with a *high* propensity likewise.

```
ATE_IPW = mean[ D·Y / e(X) ]  −  mean[ (1−D)·Y / (1 − e(X)) ]
```

> **Intuition.** Each treated client "stands in" for 1/e(x) clients like them, because only a
> fraction e(x) of such clients were treated. Weighting rebuilds the full population from the
> treated half and, separately, from the control half.

**An eight-client example.** Two feature groups of four clients each: "near tier", where
3 of 4 opted in (e = 0.75), and "far", where 1 of 4 did (e = 0.25). Near clients are bigger.
The incentive adds +10 to a near client and +2 to a far one, so the true ATE is (10 + 2)/2 =
**+6**.

| group | e(x) | treated outcomes | control outcomes |
|---|---|---|---|
| near | 0.75 | 100, 100, 100 | 90 |
| far | 0.25 | 50 | 48, 48, 48 |

Naive DiM: treated mean (300 + 50)/4 = 87.5; control mean (90 + 144)/4 = 58.5; DiM = **+29**.
Biased far upward, because the treated arm is mostly big near-tier clients.

IPW. Treated weights 1/e: near 1/0.75 = 1.33 each, far 1/0.25 = 4. Weighted treated mean =
(3·1.33·100 + 4·50) / (3·1.33 + 4) = (400 + 200)/8 = 75. Control weights 1/(1−e): near 4, far
1.33 each. Weighted control mean = (4·90 + 3·1.33·48)/8 = (360 + 192)/8 = 69. IPW ATE =
75 − 69 = **+6**. Exactly right: the weights rebuilt a population that is half near, half far
on *each* side.

Two practical notes. Weights of 4 on single clients show IPW's weakness: extreme propensities
give extreme weights and high variance. In code we clip e to [0.02, 0.98] and use the
**Hájek** (normalised) version, dividing each side by the sum of its weights rather than by n
(here the two coincide because the weights sum to 8).

## 4.4 Outcome regression

The other family: model E[Y | X, D=1] = μ₁(x) and E[Y | X, D=0] = μ₀(x), then

```
ATE_reg = mean[ μ₁(X) − μ₀(X) ]
```

Impute every client's missing potential outcome and average. Unbiased if the outcome model
is right. Biased if it is wrong (and OLS on a skewed outcome with 19 features is usually
somewhat wrong).

## 4.5 AIPW: the best of both

Augmented IPW (Robins, Rotnitzky & Zhao, 1994) uses the regression as the main estimate and
the IPW-weighted *residuals* as a correction:

```
ψ₁ᵢ = μ₁(Xᵢ) + Dᵢ (Yᵢ − μ₁(Xᵢ)) / e(Xᵢ)
ψ₀ᵢ = μ₀(Xᵢ) + (1−Dᵢ)(Yᵢ − μ₀(Xᵢ)) / (1 − e(Xᵢ))
ATE_AIPW = mean(ψ₁ − ψ₀)
```

**Double robustness.** If the outcome model is right, the residuals Y − μ have mean zero given
X, so the correction term averages to zero whatever e(x) is. If the propensity model is right,
the weighted residuals exactly fix whatever the outcome model got wrong. You need only *one* of
the two to be right. (Course video 8.)

**Efficiency.** With both roughly right, AIPW reaches the semiparametric efficiency bound:
no regular estimator has smaller variance. This is why in `results/ate_incentive.csv` AIPW
has the smallest SE, even though in an RCT its unbiasedness does not depend on the GBM at all.

**Cross-fitting.** μ₀, μ₁ (and e, if estimated) for client *i* must be fitted on data that
excludes *i*. Otherwise a flexible model memorises Yᵢ, the residual shrinks artificially, and
the correction term is biased. `aipw()` uses 5-fold cross-fitting (course video 4:
prediction vs estimation, and the "sample splitting" theme of section 2).

## 4.6 The experiment we ran on our own data

`observational.py` takes the *same* potential outcomes and lets clients opt in with

```
P(opt in | x) = sigmoid(−0.9 + 0.8·engagement_z + 0.9·aum_flow_z + 0.7·near_tier)
```

so uptake is confounded with growth (AUM flows drive next quarter's drift). Then four
estimators run on the opted-in-vs-not data. The true ATE of the incentive on these clients is
**−$1.8K per client** (−3% of revenue). `results/observational.csv` (showcase run):

| estimator | estimate | what happened |
|---|---|---|
| Naive difference in means | +$24.6K ± 2.2K | sign-ups were going to grow anyway: the bias is 13× the effect, with the wrong sign |
| Regression adjustment (linear) | −$3.9K ± 4.3K | covers the truth; wide because dollars are heavy-tailed |
| IPW (logistic propensity) | +$10.9K ± 3.1K | removes half the bias; clients with e(x) ≈ 0.05 get weight 20, and a few of them are whales |
| AIPW (cross-fitted) | +$0.6K ± 0.5K | removes 91% of the bias, but its CI *misses* the truth |

The column `true_ate` is the planted truth, `covers_truth` says whether the 95% CI contains
it. Figure `04_observational_bias.png` draws it.

**Why AIPW misses here, and what to do about it.** Double robustness is an asymptotic promise.
The correction term `d (y − μ̂₁) / ê` divides a dollar residual by a propensity; when a
$2M-revenue client lands in a cell with ê = 0.05, one client contributes a $40M/n term and the
influence-function SE does not know how rare that was. Two things help and both are in the
code: (i) the outcome models are fitted on the revenue *index* (y / pre-period revenue) and
scaled back, so the whales' residuals are small before the division; (ii) we repeat the whole
opt-in draw ten times (`results/observational_replication.csv`): across draws AIPW is
unbiased on average (mean bias −$0.1K, RMSE $2.3K, 90% coverage) and the naive estimator is
never within $10K of the truth. The honest lesson is that the observational estimate of a 3%
effect on a heavy-tailed outcome is fragile even when every confounder is measured; the RCT's
CUPED/AIPW estimates in `results/ate_incentive.csv` are both tighter and assumption-free.

## 4.7 Pitfalls

1. **Overlap / positivity.** If some clients have e(x) ≈ 0 or 1, no method can say anything
   about them; the weights explode or the regression extrapolates. Check the propensity
   histogram before trusting anything.
2. **Unconfoundedness is untestable.** AIPW is doubly robust to *model misspecification*, not
   to an unmeasured confounder. Randomisation is the only cure for that.
3. **Post-treatment covariates in X** break everything, exactly as in CUPED.
4. **Not cross-fitting** with flexible learners: the bias is subtle and in the direction of
   "no effect".

## 4.8 Exercises

1. In the eight-client example, compute the regression-adjustment estimate (section 4.4)
   using within-group means as μ₁(x) and μ₀(x). Does it also recover +6? Why?
2. Suppose the propensity model is perfect but the outcome model returns μ₁ = μ₀ = 0 for
   everyone. Simplify the AIPW formula and name the estimator you get.
3. Suppose the outcome model is perfect but the propensity model returns e = 0.5 for everyone
   (wrong). Argue that AIPW is still unbiased.
4. In an RCT with e = 0.57 known, which of the two "robustness legs" can ever be broken?
   What does the outcome model buy you then?

<details><summary>Answers</summary>

1. μ₁(near) = 100, μ₀(near) = 90, μ₁(far) = 50, μ₀(far) = 48. Imputed effects: +10 for each
   of the 4 near clients, +2 for each of the 4 far clients; mean = +6. Yes: within a group
   the opt-in is as good as random (unconfoundedness given "group"), so group means are the
   right μ's. With a *continuous* X you would need a model for μ, and that is where it can go
   wrong.
2. ψ₁ = D·Y/e, ψ₀ = (1−D)Y/(1−e): plain IPW.
3. With μ correct, E[Y − μ_d(X) | X, D = d] = 0, so each correction term has expectation
   zero whatever weight multiplies it. The estimate reduces to mean[μ₁ − μ₀], the correct
   regression estimate.
4. Only the outcome model can be wrong, and that cannot bias the estimate. The outcome model
   buys variance reduction: the better it predicts Y, the smaller the residuals and the SE.

</details>
