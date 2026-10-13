# 10. Revenue decomposition: volume effect vs rate compression

*Where it shows up in this repo:* `src/tierlift/economics/decomposition.py`,
`src/tierlift/economics/rollout.py`, `results/decomposition_incentive.csv`,
`results/rollout.csv`, `results/targeting_band_grid.csv`, figures
`09_revenue_decomposition.png` and `10_rollout_comparison.png`.

---

## 10.1 Why "did revenue go up?" is the wrong first question

Exchange revenue is **volume × rate**. The Tier Accelerator does two things at once:

* it makes clients trade more (volume up, good), and
* it charges them less per dollar traded (rate down, bad), both through the half-step
  discount during the campaign and through clients reaching the next tier.

A single "revenue went up 2%" number hides whether that is a lot of extra volume at a slightly
lower price, or a little extra volume with a big giveaway. The two have opposite implications
for rollout. So split it.

## 10.2 The identity

For one client (or one group's totals), with V = volume and r = rate, before (0) and after (1):

```
ΔR = V₁r₁ − V₀r₀
   = (V₁ − V₀)·r₀             volume effect:  the extra volume, priced at the OLD rate
   + V₀·(r₁ − r₀)             rate effect:    the OLD volume, repriced at the new rate (negative)
   + (V₁ − V₀)(r₁ − r₀)       interaction:    the extra volume's own discount
```

It is an **exact identity**: multiply out the right-hand side and every cross term cancels.
`tests/test_economics_and_durability.py::test_decomposition_identity` checks it to 1e-12.
Economists know it as a Laspeyres-style price/quantity split; the only choice is which base
to price the volume change at (old rate here, which keeps "volume effect" meaning "what the
extra trading would have been worth at the price we used to charge").

## 10.3 A worked example

One client, quarterly volume $90M at a blended 9.0 bps. The incentive lifts volume 25% to
$112.5M and, because they cross $100M, the rate drops to 7.9 bps.

| | before | after |
|---|---|---|
| volume V | 90.0M | 112.5M |
| rate r | 9.0 bps | 7.9 bps |
| revenue V·r | $81.0K | $88.9K |

* Volume effect = (112.5 − 90)M × 9.0 bps = 22.5M × 0.0009 = **+$20.25K**
* Rate effect = 90M × (7.9 − 9.0) bps = 90M × (−0.00011) = **−$9.90K**
* Interaction = 22.5M × (−0.00011) = **−$2.48K**
* Total = 20.25 − 9.90 − 2.48 = **+$7.88K** ✔ (= 88.9 − 81.0)

Read it as a sentence: "the client's extra trading was worth $20K at the old price; we gave
back $10K of discount on the trading they were doing anyway and $2.5K on the new trading; net
+$8K." Now change the volume lift to 8% (a client who was going to cross anyway): volume
effect +$6.5K, rate effect −$9.9K, interaction −$0.8K, total **−$4.2K**. Same discount, not
enough new volume to pay for it. That is the within-10% bucket.

## 10.4 Doing it with experimental data

For a group of clients we do not see V₀ and V₁ for the *same* client. But inside a group the
incentive and holdout arms are exchangeable (randomisation), so:

* V₀ = mean volume of the holdout arm, V₁ = mean volume of the incentive arm,
* r₀ = holdout revenue / holdout volume (volume-weighted rate), r₁ likewise for the incentive
  arm.

`decompose_groups()` applies the identity to those four numbers per distance bucket. The
per-client numbers in `results/decomposition_incentive.csv` are what figure 09 plots.
`results/decomposition_true.csv` is the same calculation on the planted potential outcomes,
so you can see the sampling noise.

**What it shows** (`decomposition_incentive_adjusted.csv`, arm means adjusted with AIPW so
that whales landing in one arm do not drive the picture). The rate effect is negative
everywhere; the volume effect is what varies. Net is positive only for 10–20%, where a +30%
volume lift pays for a 10% rate cut (net +16% of revenue, truth +12%). Within 10% the lift is
≈ 0 (they cross anyway) and the rate cut is the biggest (they reach the next tier for sure):
net −8%. Beyond 20% the lift is +5% and the half-step discount applies to everyone: net −4%.

## 10.5 Blanket vs targeted rollout

A rollout policy S (a set of clients) has impact Σ_{i∈S} τᵢ. We estimate it two ways and they
agree:

1. **Experimental** (`rollout_table`): AIPW influence scores (write-up 4) are fitted once on
   the whole incentive-vs-holdout sample and averaged over the clients in S. Inside S the arms
   are still randomised (membership is pre-treatment), so the average is unbiased; because the
   outcome models saw all 3,500 clients rather than the 290 in the band, the standard error is
   half what refitting inside the band gives (`ContrastScores` in `rollout.py`; this is the
   same "full-sample nuisance, subgroup average" trick as the DR policy value). Showcase run:
   blanket **−3.4% [−6.0, −0.8]** (truth −3.0%); band **+12.5% [+4.3, +20.6]** (truth +11.8%).
   CUPED inside the band is reported alongside: +9.1% with a CI from −10% to +28%, which is
   the honest reason the AIPW machinery is there.
2. **Doubly robust policy value** (`compare_policies`, write-up 9): mean over S of the DR
   scores, in dollars per client.

`results/targeting_band_grid.csv` sweeps the band: 0–10% (−6.5%), 10–20% (+12.5%), 20–30%
(+1.8%), 30–50% (−5.7%), 50%+ (−4.9%, CI excludes zero). Only 10–20% is positive with a CI
that excludes zero. The sweet spot is a *band*, not a point, and it is where the headline
targeted-rollout number comes from.

## 10.6 Monte Carlo check

`results/montecarlo_summary.csv`: across 40 simulated experiments, the true band effect
averages +11.0% and the AIPW estimate +13.0% (SD 5.5%, mean reported SE 4.7%, 95% CI coverage
95%); blanket −2.9% true, −2.7% estimated (coverage 95%). The band's point estimate runs
about two points high on average because the relative effect is a ratio (effect / holdout
mean of a small, heavy-tailed group), a known small-sample bias of ratio estimators; the CI
coverage is right. The run behind the README's figures (seed 13) is the replication whose band
estimate is closest to the median of the 40 estimates (a rule that never looks at the truth),
so the figures show a typical run rather than a lucky one.

## 10.7 Pitfalls

1. **Comparing revenue before vs after** for the treated clients only. Market-wide shocks
   (the seasonal rally in weeks 10–13 is planted for exactly this reason) would be attributed
   to the campaign. Always treated vs holdout, same period.
2. **Using the simple average rate** instead of the volume-weighted one. The identity then
   fails to add up.
3. **Reporting "+11%" without saying of what.** It is the net revenue change *of the targeted
   clients*, about 8% of the book. Scaled to the whole book it is smaller; the comparison with
   blanket (−3% on 100% of the book) is the decision-relevant one.
4. **Forgetting the rate cut persists.** A client who crossed a tier keeps the lower rate
   next quarter; write-up 11 keeps the rate cost un-decayed in the projection.

## 10.8 Exercises

1. Redo the worked example with the volume lift at 15%. Is the net positive? Find the
   break-even lift (where total = 0) algebraically.
2. Show that pricing the volume change at the *new* rate instead moves the interaction term
   into the volume effect. Does the total change?
3. A group has holdout mean volume $50M at 10 bps and incentive-arm mean volume $60M at 9
   bps. Compute the three components and the net % change.
4. Why can the decomposition be applied to the *difference between arms* at all? What
   assumption makes V₁ − V₀ (arm means) estimate the causal volume effect?

<details><summary>Answers</summary>

1. V₁ = 103.5M: volume +$12.15K, rate −$9.90K, interaction −$1.49K, total +$0.77K: barely.
   Break-even: V₀(1+g)·r₁ = V₀·r₀ → 1 + g = r₀/r₁ = 9.0/7.9 → g = 13.9%.
2. ΔR = (V₁ − V₀)r₁ + V₀(r₁ − r₀): two terms, the interaction absorbed into the first. The
   total is the same (it is an identity); only the labelling changes.
3. Volume: 10M × 0.0010 = +$10K; rate: 50M × (−0.0001) = −$5K; interaction: 10M × (−0.0001) =
   −$1K; total +$4K on a base of $50K = +8%.
4. Randomisation: E[V | holdout] = E[V(0)] and E[V | incentive] = E[V(1)] within the group,
   so the arm means estimate the potential-outcome means and their difference is the
   average causal effect on volume.

</details>
