# 11. Survival analysis for effect decay, and the ROI correction

*Where it shows up in this repo:* `src/tierlift/durability/survival.py`,
`src/tierlift/durability/roi.py`, `results/weekly_effect.csv`, `results/kaplan_meier.csv`,
`results/weekly_hazard.csv`, `results/cox.csv`, `results/roi_projection.csv`, figures
`11_weekly_effect.png`, `12_survival.png`, `13_cox_hazard_ratios.png`, `14_roi_projection.png`.

---

## 11.1 The question

The campaign quarter showed a lift. Finance wants an annual number. The lazy answer is
"multiply by four". If the effect fades (novelty, clients stop looking at the progress bar),
that answer is wrong, and it is wrong in the expensive direction. So: *does the effect last,
and how fast does it fade?*

## 11.2 View 1: the weekly effect curve

The holdout is kept for 26 weeks after the campaign, so for each of 39 weeks we can compute
the incentive-vs-holdout gap in volume (as a % of the holdout mean, after dividing each
client's volume by their own pre-period average to tame the whales). `results/weekly_effect.csv`.
If the effect decays geometrically, effect_t ≈ effect₁ × (1 − h)^(t−1), and a weighted
log-linear fit gives h (`fit_exponential_decay`).

**Honest verdict:** this curve is *noisy*. The population-average lift is only ~5% and the
weekly standard error is ~2.5 points, so the fitted decay has a wide CI. It is the right
sanity check and the wrong headline estimator. We need something per client.

## 11.3 View 2: survival analysis of engagement

Each treated client stays "engaged" with the campaign until they stop interacting with it:
the last week with a progress-bar view or email click (`lapse_week`). Once engagement lapses,
the volume lift is gone. Two features of this data make it a textbook survival problem:

* **Time-to-event.** The outcome is *when* something happens, not whether.
* **Right-censoring.** Clients still engaged at week 39 have no lapse week; we know only that
  it is > 39. Throwing them away (or pretending they lapsed at 39) biases everything. A naive
  "average weeks engaged" is wrong for this reason.

### Kaplan–Meier

S(t) = P(still engaged after week t). Estimate it week by week:

```
Ŝ(t) = Π_{j ≤ t} ( 1 − d_j / n_j )
```

d_j = clients who lapsed in week j; n_j = clients still engaged (and not censored) at the start
of week j. Censored clients count in n_j for every week they are observed and then drop out
without contributing a d.

**A tiny example.** Six clients, follow-up ends at week 4. Lapse weeks: A=1, B=2, C=2,
D=censored at 3 (data ends), E=4, F=censored at 4.

| week j | at risk n_j | lapsed d_j | 1 − d_j/n_j | Ŝ(j) |
|---|---|---|---|---|
| 1 | 6 | 1 | 5/6 | 0.833 |
| 2 | 5 | 2 | 3/5 | 0.500 |
| 3 | 3 | 0 | 1 | 0.500 |
| 4 | 2 | 1 | 1/2 | 0.250 |

D was at risk in weeks 1–3 and then left without lapsing; in week 4 only E and F remain. If
you had treated D and F as "lapsed at the end", Ŝ(4) would be 0 instead of 0.25.

### The weekly hazard

h(t) = d_t / n_t: the chance of lapsing in week t given engagement at its start. If h is
constant, S(t) = (1 − h)^t and decay is *memoryless*: the client's clock does not matter. The
pooled estimate Σd / Σn in `results/weekly_hazard.csv` is **≈ 1.9% per week** with a tight
SE, and figure 12's hazard bars are flat, so the memoryless model fits. (The simulator's
planted rate is 2.0%; the difference is sampling noise plus the fact that the planted hazard
varies by client, see 11.4.)

Consistency check: the aggregate curve (view 1) should decay at about the same rate:
(1 − 0.019)^38 ≈ 0.48, i.e. about half the week-1 effect left at week 39. It does, within its
wide CI.

## 11.4 Cox proportional hazards: who lapses fastest?

```
h_i(t) = h₀(t) · exp(β' xᵢ)
```

The baseline hazard h₀(t) is left unspecified (that is the "semi-parametric" part); β says
how each covariate multiplies the hazard. exp(β) is the **hazard ratio**. `results/cox.csv`,
standardised covariates:

* account manager: HR < 1 (lapse slower; someone is reminding them),
* API share: HR > 1 (algorithmic flow does not look at dashboards),
* engagement: HR < 1 (clients who log in keep looking at the progress bar),
* tenure: ≈ 1.

Figure 13. The point for the business: decay is not uniform, and a relationship manager touch
for high-API clients is where the durability can be bought.

## 11.5 The ROI correction

Naive annualisation: quarter lift × 4. Two things are wrong with it.

1. The quarter's average already contains 13 weeks of decay. The week-1 effect is larger
   than the quarter average by 1 / m(13), where

   ```
   m(H) = (1 − (1 − h)^H) / (h·H)        (average of (1−h)^t over H weeks, relative to week 1)
   ```

2. Over 52 weeks the average effect is m(52) of the week-1 effect, not 1.

So the true annual benefit is effect₁ × 52 × m(52), while the naive figure is effect₁ ×
m(13) × 52. The ratio is

```
naive / true = m(13) / m(52)
```

At h = 2%: m(13) = (1 − 0.98¹³)/(0.26) = 0.889, m(52) = (1 − 0.98⁵²)/(1.04) = 0.625, ratio
**1.42**. At our estimated 1.9%: about **1.40**. The naive annual benefit is ~40% too high.

The rate-compression cost does *not* decay the same way (a client who crossed keeps the
cheaper tier), so `roi_projection()` applies the decay only to the volume gain and keeps the
rate cost flat. `results/roi_projection.csv` reports both the benefit overstatement (~40%)
and the net overstatement (higher, since a non-decaying cost is subtracted from a decaying
benefit).

## 11.6 Pitfalls

1. **Censoring handled by deletion or by imputation at the end.** Both bias S(t); KM is the
   fix and it is one line of code.
2. **Detecting lapse from volume.** For most clients the lift (3–8%) is smaller than their
   weekly noise (20–35%); a "back to baseline for 3 weeks" rule fires on noise (we tried it:
   zero correlation with the true lapse week). Use product telemetry for the event and the
   aggregate curve as the volume-based check.
3. **Annualising with the quarter average as the starting point.** Even if you apply decay
   afterwards, you have double-counted the first quarter's decay (point 1 in 11.5).
4. **Proportional hazards is an assumption.** Check that the hazard ratio is roughly constant
   over time (e.g. stratified KM curves do not cross).

## 11.7 Exercises

1. In the six-client KM example, suppose D actually lapsed in week 3. Recompute Ŝ(3) and Ŝ(4).
2. With h = 3%/week, compute m(13), m(52) and the naive/true ratio. Does a faster decay make
   the naive number better or worse?
3. A hazard ratio of 1.12 per SD of API share (`results/cox.csv`): for a client 2 SD above
   the mean, how much faster do they lapse? What is their expected engagement half-life if
   the baseline is ln 2 / 0.019 ≈ 36 weeks?
4. The weekly-curve fit gives h = 2.4% ± 0.5% (`results/decay_fit.json`) while the survival
   estimate gives 1.93% ± 0.04%. Are they inconsistent? Which would you report and why?

<details><summary>Answers</summary>

1. Week 3: n = 3, d = 1, factor 2/3, Ŝ(3) = 0.5 × 2/3 = 0.333. Week 4: n = 2 (E, F), d = 1,
   Ŝ(4) = 0.333 × 0.5 = 0.167.
2. m(13) = (1 − 0.97¹³)/0.39 = (1 − 0.673)/0.39 = 0.839; m(52) = (1 − 0.97⁵²)/1.56 =
   (1 − 0.205)/1.56 = 0.510; ratio 1.65. Faster decay → bigger overstatement.
3. exp(2 × ln 1.12) = 1.12² = 1.25: 25% higher hazard, h ≈ 2.4%, half-life ≈ ln 2/0.024 ≈
   29 weeks instead of 36.
4. The curve's estimate is one SE above the survival estimate: compatible with noise. Report
   the survival estimate: it uses per-client event data and is 10× more precise; cite the
   curve as the model-free cross-check.

</details>
