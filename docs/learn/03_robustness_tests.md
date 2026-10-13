# 3. Welch's t, Mann–Whitney, permutation: three questions, not one

*Where it shows up in this repo:* `src/tierlift/experiment/robustness.py`,
`results/robustness_incentive.csv`, `results/robustness_nudge.csv`,
`results/multiple_testing.json`.

---

## 3.1 Why three tests

Each test answers "could this difference be chance?", but under a different definition of
"difference" and a different set of assumptions. Revenue per client is **heavily right-skewed**
(a few whales), so the assumptions matter. If all three agree, the finding is solid. If they
disagree, the disagreement tells you *what kind* of effect you have.

| test | what it compares | assumes | sensitive to whales? |
|---|---|---|---|
| Welch's t | means | CLT for the two sample means; unequal variances OK | very |
| Mann–Whitney U | ranks ("is a random treated client ahead of a random control client?") | none about shape; tests stochastic dominance, not means | no |
| Permutation | means (or any statistic), under re-randomisation | only that assignment was random | somewhat, but exact |

## 3.2 Welch's t-test

```
t = (Ȳ₁ − Ȳ₀) / sqrt(s₁²/n₁ + s₀²/n₀)
```

The denominator is the standard error of the difference in means with each arm's own variance
(Student's t assumes equal variances, which treated and holdout arms rarely have). Welch's
degrees-of-freedom formula is ugly but the idea is simple: with n in the thousands the
reference distribution is essentially normal.

**When it fails.** The CLT needs the sample means to be approximately normal. With a skewed
outcome and n₀ = 114 holdout clients inside the 10–20% band, a single $500K client moving
between arms shifts the mean noticeably. The t-test's p-value is then too optimistic in one
direction and too pessimistic in the other.

## 3.3 Mann–Whitney U

Rank all clients together, ignore arms, then sum the ranks of the treated clients. Equivalent
and more intuitive: count, over all n₁·n₀ (treated, control) pairs, how often the treated
client has the larger outcome. U divided by n₁n₀ is the **probability of superiority** (the
`prob_superiority` column): 0.5 means no difference, 0.6 means a random treated client beats a
random control client 60% of the time.

**What it does not test.** Means. Two distributions can have the same median and different
means (one has a longer tail). The incentive in this project *lowers* the rate for everyone
and *raises* volume for some, so the median client can be slightly worse off while the mean
(dollar-weighted) is better: Mann–Whitney and Welch can legitimately point in opposite
directions. That is information, not a bug.

## 3.4 Permutation test (Fisher's exact logic)

Under the null "the incentive did nothing", every client's outcome would have been the same
whichever cell they landed in. So the only randomness is the assignment. Re-draw the
assignment many times (within strata, because that is how it was drawn), recompute the
statistic each time, and see where the real one falls:

```
p = ( 1 + #{ |T_perm| ≥ |T_obs| } ) / (B + 1)
```

Nothing is assumed about the distribution of Y. The test is exact for the sharp null and
honours the design (stratification) exactly.

**A five-client example.** Outcomes 3, 5, 8, 2, 9; two are treated: {8, 9}. Observed DiM =
8.5 − 3.33 = 5.17. There are C(5,2) = 10 possible treated pairs. The DiMs for each:

| treated pair | DiM |
|---|---|
| {3,5} | 4.0 − 6.33 = −2.33 |
| {3,8} | 5.5 − 5.33 = +0.17 |
| {3,2} | 2.5 − 7.33 = −4.83 |
| {3,9} | 6.0 − 5.0 = +1.0 |
| {5,8} | 6.5 − 4.67 = +1.83 |
| {5,2} | 3.5 − 6.67 = −3.17 |
| {5,9} | 7.0 − 4.33 = +2.67 |
| {8,2} | 5.0 − 5.67 = −0.67 |
| **{8,9}** | **+5.17** |
| {2,9} | 5.5 − 5.33 = +0.17 |

Only one assignment out of ten gives |DiM| ≥ 5.17, so p = 1/10 (two-sided). With five clients
you cannot get below 0.1 whatever happens: a useful reminder that p-values have a floor set
by the design.

## 3.5 Multiple cells: Holm's correction

Two treatment cells each compared with the holdout = two tests. At α = 0.05 each, the chance
of at least one false positive is 1 − 0.95² ≈ 10%. Holm's step-down: sort the p-values,
multiply the smallest by m, the next by m−1, ..., and enforce monotonicity. It controls the
family-wise error rate at 5% and is uniformly more powerful than Bonferroni.
`results/multiple_testing.json` reports both raw and Holm-adjusted p-values.

## 3.6 How it shows up in this repo

`results/robustness_incentive.csv` for the full-population incentive-vs-holdout contrast:
Welch and the permutation test agree on the mean effect (slightly negative overall); Mann–Whitney
reports a probability of superiority near 0.5. Inside the 10–20% band
(`results/targeting_band_grid.csv`) all three tests agree the effect is positive.

The permutation test is also the honest check on the Welch SE: `null_sd` (the SD of the
permutation distribution) should be close to the Welch denominator. When it is not, the CLT
approximation is strained.

## 3.7 Pitfalls

1. **Permuting across strata.** The design permuted within strata; the test must too, or it
   tests the wrong null.
2. **Reading Mann–Whitney as a test of means.** It is not; it is a test of ranks.
3. **Running many tests and reporting the best.** Pre-register the primary contrast
   (`analysis.primary_contrast` in the config) and correct the rest.
4. **"Not significant" ≠ "no effect".** With 114 holdout clients in a band, the MDE is large;
   `results/power.csv` says how large.

## 3.8 Exercises

1. In the five-client example, what is the smallest two-sided p-value a permutation test can
   report? Generalise to n clients with k treated.
2. Two arms: control outcomes {1, 1, 1, 1, 100}, treated outcomes {2, 2, 2, 2, 2}. Which arm has
   the higher mean? Which wins Mann–Whitney? What business situation does this resemble?
3. Raw p-values for nudge and incentive are 0.03 and 0.20. Compute the Holm-adjusted values.
4. Why does `permutation()` use `rng.permutation(d[g])` per stratum rather than shuffling the
   whole `d` vector?

<details><summary>Answers</summary>

1. 1/C(5,2) = 0.1 (if the observed assignment is the unique extreme). In general 1/C(n,k)
   (or 2/C(n,k) if its mirror image is also attainable).
2. Control mean 20.8 > treated 2; but treated beats control in 16 of 25 pairs, so
   Mann–Whitney favours treated. One whale in the control arm; e.g. a blanket incentive that
   helps most clients a little while one large client's rate cut dominates the dollars.
3. Sorted: 0.03 (×2 = 0.06), 0.20 (×1 = 0.20, and ≥ 0.06): adjusted 0.06 and 0.20. Neither
   significant at 5% after correction.
4. Shuffling within each stratum keeps each stratum's cell counts exactly as designed;
   shuffling globally would produce assignments the design could never have produced.

</details>
